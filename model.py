# model.py
import numpy as np
import pandas as pd
import json
import logging
import requests
import time
from collections import Counter

logger = logging.getLogger("ai_assistant.model")

class AIAssistant:
    def __init__(self):
        self.history_df = pd.DataFrame()
        self.user_counts = Counter()
        self.crash_values = []
        self.color_counts = Counter()
        self.games_index = set()

    # -------------------- Загрузка истории --------------------
    def load_history(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.load_history_from_list(data)

    def load_history_from_list(self, games_list):
        rows = []
        for item in games_list:
            gid = item.get("game_id") or item.get("id")
            if gid is None or gid in self.games_index:
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

    # -------------------- Подгрузка с URL --------------------
def load_json_from_url(url):
    logger.info(f"Скачиваю данные с {url}")
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()

# -------------------- Детекция ботов --------------------
    def detect_bots_in_snapshot(self, bets):
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
            threshold = max(5, int(np.sqrt(max(1, len(self.history_df)))))
            if self.user_counts.get(uid, 0) >= threshold:
                bot_ids.append(uid)
                bot_amount += amt
        frac_amount = (bot_amount / total_amount) if total_amount > 0 else 0.0
        return frac_amount, bot_ids

# -------------------- Основной прогноз --------------------
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
        start = time.time()
        game_id = payload.get("game_id")
        bets = payload.get("bets", [])
        deposit_sum = payload.get("deposit_sum", None)
        num_players = payload.get("num_players", len(bets))
        bot_frac_money, bot_ids = self.detect_bots_in_snapshot(bets)

        amounts = [float(b.get("amount", 0) or 0) for b in bets]
        autos = [float(b.get("auto", 0) or 0) for b in bets if b.get("auto") is not None]
        total = sum(amounts)
        avg_auto = np.mean(autos) if autos else 1.0

        percentiles_recent = self._crash_percentiles(tail=2000)
        percentiles_all = self._crash_percentiles()

        base_safe = percentiles_recent.get("p60") or percentiles_all.get("p60") or 1.2
        base_med  = percentiles_recent.get("p75") or percentiles_all.get("p75") or 1.5
        base_risk = percentiles_recent.get("p90") or percentiles_all.get("p90") or 2.0

        adjust_factor = 1.0 + (avg_auto - 1.0) * 0.05
        bot_penalty = 1.0 - min(0.5, bot_frac_money * 0.8)

        safe = max(1.01, round(base_safe * adjust_factor * bot_penalty, 2))
        med  = max(safe + 0.01, round(base_med * adjust_factor * bot_penalty, 2))
        risk = max(med + 0.01, round(base_risk * adjust_factor * bot_penalty, 2))

        point = round(safe * (1 - bot_frac_money) * 0.6 + med * (bot_frac_money) * 0.4, 2)

        hist_factor = min(1.0, len(self.crash_values) / 50000)
        snap_factor = min(1.0, max(1, len(bets)) / 30)
        bot_factor = max(0.2, 1.0 - bot_frac_money)
        confidence = int(round((0.3*hist_factor + 0.5*snap_factor + 0.2*bot_factor)*100))
        confidence = max(5, min(99, confidence))

        rec_safe_pct = round(max(0.5, 2.0*(confidence/100)), 2)
        rec_med_pct = round(max(0.5, rec_safe_pct*2.0), 2)
        rec_risk_pct = round(max(0.3, rec_safe_pct*4.0), 2)

        color_probs = self.estimate_color_probs()

        logger.info(f"=== PREDICT (game {game_id}) ===")
        logger.info(f"Игроков в снимке: {len(bets)}; суммарно: {total:.4f}; avg_auto: {avg_auto:.2f}")
        logger.info(f"Доля денег от ботов: {bot_frac_money:.3f}; обнаружено bot_ids: {len(bot_ids)}")
        logger.info(f"Рекомендации (confidence {confidence}%): безопасный:{safe}, средний:{med}, рисковый:{risk}, точный:{point}")
        logger.info(f"Рекомендуемый % от банка: безопасно:{rec_safe_pct}%, средне:{rec_med_pct}%, риск:{rec_risk_pct}%")
        logger.info(f"Цветовые вероятности: {color_probs}")
        logger.info(f"Исторический объём игр: {len(self.crash_values)}")
        logger.info(f"=== END PREDICT (game {game_id}) in {time.time()-start:.3f}s ===")

    def process_feedback(self, game_id, crash, persist=False, bets=None, deposit_sum=None, num_players=None):
        if game_id in self.games_index:
            logger.info(f"Feedback: игра {game_id} уже в истории — обновляю.")
        self.games_index.add(game_id)
        self.crash_values.append(float(crash))
        row = {
            "game_id": game_id,
            "crash": float(crash),
            "bets": bets or [],
            "deposit_sum": deposit_sum,
            "num_players": num_players,
            "color_bucket": None
        }
        for b in row["bets"]:
            uid = b.get("user_id")
            if uid is not None:
                self.user_counts[uid] += 1
        self.history_df = pd.concat([self.history_df, pd.DataFrame([row])], ignore_index=True)
        logger.info(f"Feedback: добавлена игра {game_id}, crash={crash}. Всего игр: {len(self.crash_values)}")
        if persist:
            self._persist_history("history_updated.json")

    def _persist_history(self, path):
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
        return {k: round(v/total, 3) for k, v in buckets.items()}