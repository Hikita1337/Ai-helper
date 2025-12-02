# model.py
import time
import numpy as np
import pandas as pd
from collections import Counter, deque, defaultdict
import logging
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger("ai_assistant.model")
logger.setLevel(logging.INFO)

class AIAssistant:
    def __init__(self,
                 color_seq_len=50,
                 pred_log_len=1000,
                 pending_threshold=50,
                 retrain_min_minutes=10):
        # История/статистика
        self.history_df = pd.DataFrame()
        self.crash_values = []
        self.games_index = set()

        # Пользователи / боты / паттерны
        self.user_counts = Counter()           # сколько раз встречался uid
        self.bot_patterns = defaultdict(list)  # uid -> список {amount, auto, crash}
        self.color_counts = Counter()
        self.color_sequence = deque(maxlen=color_seq_len)

        # Лог предсказаний
        self.pred_log = deque(maxlen=pred_log_len)

        # Модели LightGBM
        self.model_safe = LGBMRegressor(n_estimators=100, verbose=-1)
        self.model_med  = LGBMRegressor(n_estimators=100, verbose=-1)
        self.model_risk = LGBMRegressor(n_estimators=100, verbose=-1)

        # Онлайн-обучение / throttle
        self.pending_feedback = []
        self.pending_threshold = pending_threshold
        self.last_trained_at = 0
        self.retrain_min_seconds = retrain_min_minutes * 60

        # Метрики истории тренировки
        self.last_metrics = {}

    # -------------------- Экспорт/импорт состояния --------------------
    def export_state(self):
        state = {
            "crash_values": self.crash_values,
            "user_counts": dict(self.user_counts),
            "bot_patterns": {k: v for k, v in self.bot_patterns.items()},
            "color_sequence": list(self.color_sequence),
            "history_index": list(self.games_index),
            "last_trained_at": self.last_trained_at,
            "last_metrics": self.last_metrics
        }
        return state

    def load_state(self, state: dict):
        self.crash_values = list(state.get("crash_values", []))
        self.user_counts = Counter(state.get("user_counts", {}))
        bp = state.get("bot_patterns", {})
        self.bot_patterns = defaultdict(list, {int(k): v for k, v in bp.items()})
        self.color_sequence = deque(state.get("color_sequence", []), maxlen=self.color_sequence.maxlen)
        self.games_index = set(state.get("history_index", []))
        self.last_trained_at = state.get("last_trained_at", 0)
        self.last_metrics = state.get("last_metrics", {})
        logger.info("State loaded into assistant")

    # -------------------- Загрузка истории --------------------
    def load_history_from_list(self, games_list):
        rows = []
        count_added = 0
        for item in games_list:
            gid = item.get("game_id") or item.get("id")
            if gid is None or gid in self.games_index:
                continue
            self.games_index.add(gid)
            crash = float(item.get("crash") or 0)
            bets = item.get("bets") or []
            deposit_sum = item.get("deposit_sum")
            num_players = item.get("num_players")

            bucket = item.get("color_bucket") or self._bucket_by_crash(crash)
            rows.append({
                "game_id": gid,
                "crash": crash,
                "bets": bets,
                "deposit_sum": deposit_sum,
                "num_players": num_players,
                "color_bucket": bucket
            })

            # update stats
            self.crash_values.append(crash)
            if bucket:
                self.color_counts[bucket] += 1
                self.color_sequence.append(bucket)

            for b in bets:
                uid = b.get("user_id")
                if uid is not None:
                    self.user_counts[uid] += 1
                    self.bot_patterns[int(uid)].append({
                        "amount": float(b.get("amount") or 0),
                        "auto": float(b.get("auto") or 0) if b.get("auto") is not None else 0.0,
                        "crash": crash
                    })
            count_added += 1

        if rows:
            self.history_df = pd.concat([self.history_df, pd.DataFrame(rows)], ignore_index=True)
        logger.info(f"Loaded {count_added} new games; total games: {len(self.history_df)}")

    # -------------------- Базовая детекция ботов --------------------
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
            # порог: пользователь считается "подозрительным" если он встречался >= threshold раз
            threshold = max(5, int(np.sqrt(max(1, len(self.history_df)))))
            if self.user_counts.get(int(uid), 0) >= threshold:
                bot_ids.append(int(uid))
                bot_amount += amt
        frac_amount = (bot_amount / total_amount) if total_amount > 0 else 0.0
        return frac_amount, bot_ids

    # -------------------- Цветовая логика --------------------
    @staticmethod
    def _bucket_by_crash(crash):
        try:
            c = float(crash)
        except:
            return None
        if c < 1.2: return "red"
        if c < 2.0: return "blue"
        if c < 4.0: return "pink"
        if c < 8.0: return "green"
        if c < 25.0: return "yellow"
        return "gradient"

    def _color_features(self):
        categories = ["red","blue","pink","green","yellow","gradient"]
        counts = Counter(self.color_sequence)
        total = max(1, len(self.color_sequence))
        freq = [counts.get(c,0)/total for c in categories]

        # transitions 6x6 flattened
        transitions = Counter()
        seq = list(self.color_sequence)
        for i in range(len(seq)-1):
            transitions[(seq[i], seq[i+1])] += 1
        total_trans = max(1, sum(transitions.values()))
        trans_feats = []
        for a in categories:
            for b in categories:
                trans_feats.append(transitions.get((a,b),0)/total_trans)
        return freq + trans_feats

    # -------------------- Фичи по ботам (агрегированная) --------------------
    def _bot_aggregate_features(self, bot_ids):
        if not bot_ids:
            return [0.0]*6  # avg_auto, std_auto, avg_amount, std_amount, avg_frac_of_game, num_bots
        autos = []
        amts = []
        frac_in_game = []
        for uid in bot_ids:
            pats = self.bot_patterns.get(int(uid), [])
            if not pats:
                continue
            autos_local = [p.get("auto",0) for p in pats]
            amts_local  = [p.get("amount",0) for p in pats]
            autos += autos_local
            amts += amts_local
            # доля игр где этот бот участвовал — approximate by count / total_games
            frac_in_game.append(len(pats) / max(1, len(self.history_df)))
        if not autos:
            return [0.0]*6
        return [
            float(np.mean(autos)),
            float(np.std(autos)),
            float(np.mean(amts)),
            float(np.std(amts)),
            float(np.mean(frac_in_game)) if frac_in_game else 0.0,
            float(len(bot_ids))
        ]

    # -------------------- Подготовка фичей для одной игры --------------------
    def _make_features_for_game(self, bets):
        bot_frac, bot_ids = self.detect_bots_in_snapshot(bets)
        total_bets = sum(float(b.get("amount") or 0) for b in bets)
        num_bets = len(bets)
        autos = [float(b.get("auto") or 0) for b in bets if b.get("auto") is not None]
        avg_auto = float(np.mean(autos)) if autos else 1.0

        color_feats = self._color_features()
        bot_feats = self._bot_aggregate_features(bot_ids)
        base = [bot_frac, num_bets, total_bets, avg_auto]
        return np.array(base + bot_feats + color_feats, dtype=float)

    # -------------------- Предикт и логирование --------------------
    def predict_and_log(self, payload):
        t0 = time.time()
    game_id = payload.get("game_id")
    bets = payload.get("bets") or []
    fast_game = payload.get("meta", {}).get("fast_game", False)

    X = self._make_features_for_game(bets).reshape(1,-1)
    try:
        safe = float(self.model_safe.predict(X))
        med = float(self.model_med.predict(X))
        risk = float(self.model_risk.predict(X))
    except Exception:
        safe, med, risk = 1.2, 1.5, 2.0

    bot_frac, _ = self.detect_bots_in_snapshot(bets)
    recommended_pct = max(0.5, 2.0*(1 - bot_frac))

    # ---------------- Цветовой прогноз и уверенность ----------------
    if self.color_sequence:
        color_counts = Counter(self.color_sequence)
        predicted_color, count = color_counts.most_common(1)[0]
        color_confidence = count / sum(color_counts.values())
    else:
        predicted_color = "unknown"
        color_confidence = 0.0

    # ---------------- Запись в лог ----------------
    self.pred_log.append({
        "game_id": game_id,
        "safe": round(safe,2),
        "med": round(med,2),
        "risk": round(risk,2),
        "recommended_pct": round(recommended_pct,2),
        "bot_frac_money": round(bot_frac,3),
        "num_bets": len(bets),
        "timestamp": time.time(),
        "fast_game": bool(fast_game),
        "predicted_color": predicted_color,          
        "color_confidence": round(color_confidence,3)
    })

    logger.info(f"PREDICT game={game_id} safe={safe} med={med} risk={risk} in {time.time()-t0:.3f}s")
    
    def get_pred_log(self, limit=20):
        return list(self.pred_log)[-limit:]

    # -------------------- Обратная связь и онлайн-обучение --------------------
    def process_feedback(self, game_id, crash, bets=None, deposit_sum=None, num_players=None, fast_game=False):
        if game_id in self.games_index:
            logger.debug(f"Feedback for known game {game_id}")
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

        # update user counts and bot patterns
        for b in row["bets"]:
            uid = b.get("user_id")
            if uid is not None:
                uid = int(uid)
                self.user_counts[uid] += 1
                self.bot_patterns[uid].append({
                    "amount": float(b.get("amount") or 0),
                    "auto": float(b.get("auto") or 0) if b.get("auto") is not None else 0.0,
                    "crash": float(crash)
                })

        self.history_df = pd.concat([self.history_df, pd.DataFrame([row])], ignore_index=True)
        bucket = row.get("color_bucket") or self._bucket_by_crash(crash)
        if bucket:
            self.color_sequence.append(bucket)

        # enqueue feedback for retrain
        self.pending_feedback.append(row)

        # throttle training: либо накоплено достаточно, либо прошло времени с последнего тренига
        now = time.time()
        if (len(self.pending_feedback) >= self.pending_threshold and
            (now - self.last_trained_at) > self.retrain_min_seconds):
            try:
                self._online_train()
                self.pending_feedback.clear()
                self.last_trained_at = time.time()
            except Exception as e:
                logger.exception("Online train failed: %s", e)

        if fast_game:
            logger.info(f"Fast game {game_id} processed (no visible predict)")
        else:
            logger.info(f"Feedback processed game {game_id}, crash={crash}")

    # -------------------- Онлайн-обучение (batch на последних N игр) --------------------
    def _online_train(self, window=200):
        # Требуем минимум данных
        if len(self.history_df) < 50:
            logger.info("Not enough data to train")
            return

        # Берём последние window игр (или все если меньше)
        df = self.history_df.tail(window).reset_index(drop=True)
        features = []
        safe_targets = []
        med_targets = []
        risk_targets = []

        # build dataset
        for r in df.itertuples():
            bets = r.bets or []
            X = self._make_features_for_game(bets)
            features.append(X)
            # targets
            crash_val = float(r.crash)
            safe_targets.append(np.percentile([crash_val], 50))
            med_targets.append(np.percentile([crash_val], 75))
            risk_targets.append(np.percentile([crash_val], 90))

        X = np.vstack(features)
        y_safe = np.array(safe_targets)
        y_med = np.array(med_targets)
        y_risk = np.array(risk_targets)

        # Fit models (lightweight)
        try:
            self.model_safe.fit(X, y_safe)
            self.model_med.fit(X, y_med)
            self.model_risk.fit(X, y_risk)

            # Metrics on train window
            preds_safe = self.model_safe.predict(X)
            mae_safe = float(mean_absolute_error(y_safe, preds_safe))
            rmse_safe = float(np.sqrt(mean_squared_error(y_safe, preds_safe)))

            self.last_metrics = {
                "trained_on": int(time.time()),
                "window": window,
                "mae_safe": mae_safe,
                "rmse_safe": rmse_safe,
                "samples": len(X)
            }
            logger.info(f"Online training done: samples={len(X)}, mae_safe={mae_safe:.4f}, rmse_safe={rmse_safe:.4f}")
        except Exception as e:
            logger.exception("Training failed: %s", e)

    # -------------------- Экспериментальные утилиты --------------------
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
        return {k: round(v/total, 3) for k,v in buckets.items()}

    # Получение простых метрик
    def get_status(self):
        return {
            "games_loaded": len(self.history_df),
            "unique_users": len(self.user_counts),
            "pending_feedback": len(self.pending_feedback),
            "last_trained_at": self.last_trained_at,
            "last_metrics": self.last_metrics
        }