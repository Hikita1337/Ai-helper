# model.py
import numpy as np
import pandas as pd
import json
import time
import logging
from collections import defaultdict, Counter

logger = logging.getLogger("ai_assistant.model")

class AIAssistant:
    def __init__(self):
        # Храним историю как DataFrame с колонками:
        # game_id, crash, color_bucket (опционально), deposit_sum, num_players
        # bets -> список словарей (user_id, name, amount, auto)
        self.history_df = pd.DataFrame()
        # user frequency for bot detection
        self.user_counts = Counter()
        # simple aggregated stats
        self.crash_values = []  # list of floats
        self.color_counts = Counter()
        self.games_index = set()
        self.last_log_time = 0

    def load_history(self, path):
        """
        Ожидает JSON array or NDJSON with objects like:
        { "game_id":..., "crash":..., "bets":[...], "deposit_sum":..., "num_players":... }
        """
        logger.info("Загружаю историю из " + path)
        # try to read as json array
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        raw = raw.strip()
        try:
            data = json.loads(raw)
        except Exception:
            # try line-by-line
            data = []
            for line in raw.splitlines():
                line = line.strip()
                if not line: 
                    continue
                data.append(json.loads(line))
        rows = []
        for item in data:
            gid = item.get("game_id") or item.get("id")
            if gid is None:
                continue
            if gid in self.games_index:
                continue
            self.games_index.add(gid)
            crash = float(item.get("crash", 0))
            bets = item.get("bets", [])
            deposit_sum = item.get("deposit_sum", None)
            num_players = item.get("num_players", None)
            bucket = item.get("color_bucket", None)
            rows.append({
                "game_id": gid,
                "crash": crash,
                "bets": bets,
                "deposit_sum": deposit_sum,
                "num_players": num_players,
                "color_bucket": bucket
            })
            # update user counts
            for b in bets:
                uid = b.get("user_id")
                if uid is not None:
                    self.user_counts[uid] += 1
            self.crash_values.append(crash)
            if bucket:
                self.color_counts[bucket] += 1
        self.history_df = pd.DataFrame(rows)
        logger.info(f"Loaded {len(self.history_df)} games; unique users: {len(self.user_counts)}")

    def history_count(self):
        return len(self.history_df)

    def detect_bots_in_snapshot(self, bets):
        """
        Простая эвристика: если user_id уже встречается часто в истории -> бот.
        Возвращает fraction of bets by bots, and list of detected user_ids.
        """
        if not bets:
            return 0.0, []
        bot_ids = []
        total_amount = 0.0
        bot_amount = 0.0
        for b in bets:
            uid = b.get("user_id")
            amt = float(b.get("amount", 0) or 0)
            total_amount += amt
            if uid is None:
                continue
            # threshold: если пользователь встречался > X в истории, считаем как бот
            # X динамический: sqrt of history size scaled
            hist_size = max(1, len(self.history_df))
            threshold = max(5, int(np.sqrt(hist_size)))  # экспериментальная эвристика
            if self.user_counts.get(uid, 0) >= threshold:
                bot_ids.append(uid)
                bot_amount += amt
        frac_amount = (bot_amount / total_amount) if total_amount > 0 else 0.0
        frac_count = (len(bot_ids) / max(1, len(bets)))
        # use money-weighted fraction
        return frac_amount, bot_ids

    def _crash_percentiles(self, tail=None):
        if not self.crash_values:
            return {}
        arr = np.array(self.crash_values)
        if tail:
            arr = arr[-tail:]
        return {
            "p50": float(np.percentile(arr, 50)),
            "p60": float(np.percentile(arr, 60)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "count": len(arr)
        }

    def predict_and_log(self, payload):
        """
        Основная функция предсказания — ничего не возвращает, только логирует прогноз.
        payload: dict {game_id, bets, deposit_sum, num_players, meta}
        """
        start = time.time()
        game_id = payload.get("game_id")
        bets = payload.get("bets", [])
        deposit_sum = payload.get("deposit_sum", None)
        num_players = payload.get("num_players", len(bets))
        # detect bots
        bot_frac_money, bot_ids = self.detect_bots_in_snapshot(bets)

        # compute simple summary features
        amounts = [float(b.get("amount", 0) or 0) for b in bets]
        autos = [float(b.get("auto", 0) or 0) for b in bets if b.get("auto") is not None]
        total = sum(amounts)
        avg_bet = (total / len(amounts)) if amounts else 0.0
        avg_auto = (np.mean(autos) if autos else 1.0)

        # base percentiles on recent tail if available
        percentiles_recent = self._crash_percentiles(tail=2000)  # tail size configurable
        percentiles_all = self._crash_percentiles()

        # Strategy to produce three coefficients:
        # safe ~ p60, medium ~ p75, risky ~ p90, but scaled toward current avg_auto and bot presence
        base_safe = percentiles_recent.get("p60") or percentiles_all.get("p60") or 1.2
        base_med  = percentiles_recent.get("p75") or percentiles_all.get("p75") or 1.5
        base_risk = percentiles_recent.get("p90") or percentiles_all.get("p90") or 2.0

        # adjust based on avg_auto: if many players target high autos, increase suggestions slightly
        adjust_factor = 1.0 + (avg_auto - 1.0) * 0.05  # small adjustment
        # if large bot fraction (money), reduce safe and confidence
        bot_penalty = 1.0 - min(0.5, bot_frac_money * 0.8)

        safe = max(1.01, round(base_safe * adjust_factor * bot_penalty, 2))
        med  = max(safe + 0.01, round(base_med * adjust_factor * bot_penalty, 2))
        risk = max(med + 0.01, round(base_risk * adjust_factor * bot_penalty, 2))

        # point prediction: a weighted median between p60 and p75 based on bot_frac_money
        point = round(safe * (1 - bot_frac_money) * 0.6 + med * (bot_frac_money) * 0.4, 2)

        # confidence: function of history size + snapshot size + low bot fraction
        hist_factor = min(1.0, len(self.crash_values) / 50000)  # saturate at 50k
        snap_factor = min(1.0, max(1, len(bets)) / 30)  # more bets => more confident
        bot_factor = max(0.2, 1.0 - bot_frac_money)  # if many bots, confidence drops
        confidence = int(round( (0.3*hist_factor + 0.5*snap_factor + 0.2*bot_factor) * 100 ))
        confidence = max(5, min(99, confidence))

        # Recommended stake percent of bank guideline (simple heuristic)
        # safe -> small percent, medium -> moderate, risk -> only small bankroll
        rec_safe_pct = round( max(0.5, 2.0 * (confidence/100) ), 2 )  # e.g. 1.6% for conf 80
        rec_med_pct = round( max(0.5, rec_safe_pct * 2.0 ), 2)
        rec_risk_pct = round( max(0.3, rec_safe_pct * 4.0 ), 2)

        # Determine likely color probabilities using histogram of crashes
        # We'll estimate probabilities for buckets using historical data
        color_probs = self.estimate_color_probs()

        # Log full analysis in russian to console
        logger.info(f"=== PREDICT (game {game_id}) ===")
        logger.info(f"Игроков в снимке: {len(bets)}; суммарно: {total:.4f}; avg bet: {avg_bet:.4f}; avg_auto: {avg_auto:.2f}")
        logger.info(f"Доля денег от ботов (эвристика): {bot_frac_money:.3f}; обнаруженных bot ids: {len(bot_ids)}")
        logger.info(f"Рекомендации (confidence {confidence}%):")
        logger.info(f"  Безопасный коэффициент: {safe}")
        logger.info(f"  Средний коэффициент: {med}")
        logger.info(f"  Рисковый коэффициент: {risk}")
        logger.info(f"  Точный прогноз: {point}")
        logger.info(f"  Рекомендуемый % от банка: безопасно:{rec_safe_pct}%, средне:{rec_med_pct}%, риск:{rec_risk_pct}%")
        logger.info(f"  Цветовые вероятности (прибл.): {color_probs}")
        logger.info(f"  Исторический объём игр: {len(self.crash_values)}")
        logger.info(f"=== END PREDICT (game {game_id}) in {time.time()-start:.3f}s ===")

        # NOTE: we intentionally do NOT return the analysis in HTTP body (per your request).
        # The assistant only logs to console. If later you want it in response, we can change.

    def process_feedback(self, game_id, crash, persist=False, bets=None, deposit_sum=None, num_players=None):
        """
        Добавляет игру в историю, обновляет статистики и user_counts (если bets переданы).
        Если persist==True, записывает обновленный history в файл 'history_updated.json'.
        """
        if game_id in self.games_index:
            logger.info(f"Feedback: игра {game_id} уже в истории — обновляю (перезапись).")
        self.games_index.add(game_id)
        self.crash_values.append(float(crash))
        # append a minimal row; bets optional
        row = {
            "game_id": game_id,
            "crash": float(crash),
            "bets": bets or [],
            "deposit_sum": deposit_sum,
            "num_players": num_players,
            "color_bucket": None
        }
        # update user counts
        for b in row["bets"]:
            uid = b.get("user_id")
            if uid is not None:
                self.user_counts[uid] += 1
        self.history_df = pd.concat([self.history_df, pd.DataFrame([row])], ignore_index=True)
        logger.info(f"Feedback: добавлена игра {game_id}, crash={crash}. Всего игр: {len(self.crash_values)}")
        if persist:
            self._persist_history("history_updated.json")

    def _persist_history(self, path):
        # simple dump to NDJSON to avoid memory spikes
        with open(path, "w", encoding="utf-8") as f:
            for _, row in self.history_df.iterrows():
                obj = {
                    "game_id": int(row["game_id"]),
                    "crash": float(row["crash"]),
                    "bets": row["bets"],
                    "deposit_sum": row.get("deposit_sum"),
                    "num_players": row.get("num_players"),
                    "color_bucket": row.get("color_bucket")
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        logger.info(f"Persisted history to {path}")

    def estimate_color_probs(self):
        """
        Простейшая оценка вероятностей по цветам на основе crash_values.
        Цветовая схема:
            red: <1.2
            blue: [1.2,2)
            pink: [2,4)
            green: [4,8)
            yellow: [8,25)
            gradient: >=25
        """
        if not self.crash_values:
            return {}
        arr = np.array(self.crash_values)
        buckets = {
            "red": ((arr < 1.2).sum()),
            "blue": (((arr >= 1.2) & (arr < 2)).sum()),
            "pink": (((arr >= 2) & (arr < 4)).sum()),
            "green": (((arr >= 4) & (arr < 8)).sum()),
            "yellow": (((arr >= 8) & (arr < 25)).sum()),
            "gradient": ((arr >= 25).sum()),
        }
        total = float(len(arr))
        probs = {k: round(v/total, 3) for k, v in buckets.items()}
        return probs
