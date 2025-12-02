import numpy as np
import pandas as pd
from collections import Counter, deque
import logging
from lightgbm import LGBMRegressor

logger = logging.getLogger("ai_assistant.model")


class AIAssistant:
    def __init__(self):
        self.history_df = pd.DataFrame()
        self.crash_values = []
        self.games_index = set()
        self.user_counts = Counter()
        self.color_counts = Counter()
        self.bot_patterns = {}  # uid -> история ставок
        self.pred_log = deque(maxlen=1000)  # лог предиктов
        self.model_safe = LGBMRegressor()
        self.model_med = LGBMRegressor()
        self.model_risk = LGBMRegressor()
        self.pending_feedback = []

    # -------------------- Загрузка истории --------------------
    def load_history_from_list(self, games_list):
        rows = []
        for item in games_list:
            gid = item.get("game_id") or item.get("id")
            if gid in self.games_index:
                continue
            self.games_index.add(gid)
            crash = float(item.get("crash") or 0)
            bets = item.get("bets") or []
            deposit_sum = item.get("deposit_sum")
            num_players = item.get("num_players")
            color_bucket = item.get("color_bucket")
            rows.append({
                "game_id": gid,
                "crash": crash,
                "bets": bets,
                "deposit_sum": deposit_sum,
                "num_players": num_players,
                "color_bucket": color_bucket
            })
            self.crash_values.append(crash)
            if color_bucket:
                self.color_counts[color_bucket] += 1
            for b in bets:
                uid = b.get("user_id")
                if uid is not None:
                    self.user_counts[uid] += 1
                    if uid not in self.bot_patterns:
                        self.bot_patterns[uid] = []
                    self.bot_patterns[uid].append({
                        "amount": float(b.get("amount") or 0),
                        "auto": float(b.get("auto") or 0),
                        "crash": crash
                    })
        self.history_df = pd.DataFrame(rows)
        logger.info(f"Loaded {len(self.history_df)} games; unique users: {len(self.user_counts)}")

    # -------------------- Детекция ботов --------------------
    def detect_bots_in_snapshot(self, bets):
        if not bets:
            return 0.0, []
        bot_ids = []
        total_amount = 0.0
        bot_amount = 0.0
        for b in bets:
            uid = b.get("user_id")
            amt = float(b.get("amount") or 0)
            total_amount += amt
            if uid is None:
                continue
            threshold = max(5, int(np.sqrt(max(1, len(self.history_df)))))
            if self.user_counts.get(uid, 0) >= threshold:
                bot_ids.append(uid)
                bot_amount += amt
        frac_amount = (bot_amount / total_amount) if total_amount > 0 else 0.0
        return frac_amount, bot_ids

    # -------------------- Основной предикт --------------------
    def predict_and_log(self, payload):
        import time
        start = time.time()
        game_id = payload.get("game_id")
        bets = payload.get("bets") or []
        bot_frac_money, bot_ids = self.detect_bots_in_snapshot(bets)

        total_bets = sum(float(b.get("amount") or 0) for b in bets)
        num_bets = len(bets)
        avg_auto = np.mean([float(b.get("auto") or 0) for b in bets if b.get("auto") is not None] or [1.0])
        features = np.array([[bot_frac_money, num_bets, total_bets, avg_auto]])

        try:
            safe = float(self.model_safe.predict(features))
            med = float(self.model_med.predict(features))
            risk = float(self.model_risk.predict(features))
        except:
            safe = 1.2
            med = 1.5
            risk = 2.0

        recommended_pct = max(0.5, 2.0*(1-bot_frac_money))

        self.pred_log.append({
            "game_id": game_id,
            "safe": safe,
            "med": med,
            "risk": risk,
            "recommended_pct": recommended_pct,
            "bot_frac_money": bot_frac_money,
            "num_bets": num_bets,
            "total_bets": total_bets,
            "timestamp": time.time()
        })

        logger.info(f"Predicted for game {game_id}: safe={safe}, med={med}, risk={risk}, recommended_pct={recommended_pct:.2f}")
        logger.info(f"=== END PREDICT in {time.time()-start:.3f}s ===")

    # -------------------- Обратная связь и онлайн-обучение --------------------
    def process_feedback(self, game_id, crash, bets=None, deposit_sum=None, num_players=None, fast_game=False):
        self.games_index.add(game_id)
        self.crash_values.append(float(crash))
        row = {
            "game_id": game_id,
            "crash": float(crash),
            "bets": bets or [],
            "deposit_sum": deposit_sum,
            "num_players": num_players,
            "fast_game": fast_game
        }

        for b in row["bets"]:
            uid = b.get("user_id")
            if uid is not None:
                self.user_counts[uid] += 1
                if uid not in self.bot_patterns:
                    self.bot_patterns[uid] = []
                self.bot_patterns[uid].append({
                    "amount": float(b.get("amount") or 0),
                    "auto": float(b.get("auto") or 0),
                    "crash": crash
                })

        self.history_df = pd.concat([self.history_df, pd.DataFrame([row])], ignore_index=True)
        self.pending_feedback.append(row)

        if len(self.pending_feedback) >= 50:
            self._online_train()
            self.pending_feedback.clear()

        if fast_game:
            logger.info(f"Fast game {game_id} processed without visual prediction")
        else:
            logger.info(f"Feedback added for game {game_id}, crash={crash}")

    # -------------------- Онлайн-обучение --------------------
    def _online_train(self):
        if len(self.history_df) < 50:
            return
        features, safe_targets, med_targets, risk_targets = [], [], [], []
        for row in self.history_df[-50:].itertuples():
            bets = row.bets or []
            bot_frac, _ = self.detect_bots_in_snapshot(bets)
            total_bets = sum(float(b.get("amount") or 0) for b in bets)
            num_bets = len(bets)
            avg_auto = np.mean([float(b.get("auto") or 0) for b in bets if b.get("auto") is not None] or [1.0])
            features.append([bot_frac, num_bets, total_bets, avg_auto])
            safe_targets.append(np.percentile([row.crash], 50))
            med_targets.append(np.percentile([row.crash], 75))
            risk_targets.append(np.percentile([row.crash], 90))

        X = np.array(features)
        self.model_safe.fit(X, np.array(safe_targets))
        self.model_med.fit(X, np.array(med_targets))
        self.model_risk.fit(X, np.array(risk_targets))
        logger.info("Online training completed for last 50 games.")

    # -------------------- Анализ цветов --------------------
    def estimate_color_probs(self):
        if not self.crash_values:
            return {}
        arr = np.array(self.crash_values)
        buckets = {
            "red": (arr < 1.2).sum(),
            "blue": ((arr >= 1.2) & (arr < 2)).sum(),
            "pink": ((arr >= 2) & (arr < 4)).sum(),
            "green": ((arr >= 4) & (arr < 8)).sum(),
            "yellow": ((arr >= 8) & (arr < 25)).sum(),
            "gradient": (arr >= 25).sum()
        }
        total = float(len(arr))
        return {k: round(v/total, 3) for k, v in buckets.items()}

    # -------------------- Получение лога предиктов --------------------
    def get_pred_log(self, limit=20):
        return list(self.pred_log)[-limit:]