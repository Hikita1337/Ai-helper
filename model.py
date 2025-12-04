import os
import time
import math
import json
import logging
from collections import deque, defaultdict
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
import ably

logger = logging.getLogger("ai_assistant.model")
logger.setLevel(logging.INFO)


def clamp(v, a, b):
    return max(a, min(b, v))


def assign_color(crash_value: float) -> str:
    if crash_value < 1.2:
        return "red"
    elif crash_value < 2:
        return "blue"
    elif crash_value < 4:
        return "pink"
    elif crash_value < 8:
        return "green"
    elif crash_value < 25:
        return "yellow"
    else:
        return "gradient"


class AIAssistant:
    def __init__(self,
                 color_seq_len: int = 4000,
                 pred_log_len: int = 2000,
                 pending_threshold: int = 50,
                 retrain_min_minutes: int = 10,
                 max_history_records: int = 50000,
                 max_training_buffer: int = 50000,
                 ably_channel: ably.RealtimeChannel | None = None):
        self.color_seq_len = color_seq_len
        self.pred_log_len = pred_log_len
        self.pending_threshold = pending_threshold
        self.retrain_min_seconds = retrain_min_minutes * 60

        self.color_sequence = deque(maxlen=color_seq_len)
        self.history_deque = deque(maxlen=max_history_records)
        self.pred_log = deque(maxlen=pred_log_len)

        self.user_stats = defaultdict(lambda: {
            "count": 0,
            "total_amount": 0.0,
            "avg_auto": 0.0,
            "last_active": 0,
            "wins": 0,
            "losses": 0,
            "total_win": 0.0,
            "nickname": ""
        })

        self.bot_set = set()

        self.model_med = LGBMRegressor(n_estimators=80, verbose=-1)
        self.training_buffer = deque(maxlen=max_training_buffer)

        self.last_trained_at = 0
        self.last_metrics = {}
        self.total_processed_games = 0

        self.ably_channel = ably_channel

        logger.info("AIAssistant initialized")

    # -------------------------
    # State backup/restore
    # -------------------------
    def export_state(self) -> Dict[str, Any]:
        return {
            "pred_log": list(self.pred_log),
            "last_metrics": self.last_metrics,
            "total_processed_games": int(self.total_processed_games),
            "bot_list": {uid: self.user_stats.get(uid, {}).get("nickname", "") for uid in self.bot_set},
            "user_summary_count": len(self.user_stats),
            "color_tail": list(self.color_sequence)[-1000:],
            "timestamp": time.time()
        }

    def load_state(self, state: Dict[str, Any]):
        try:
            self.pred_log = deque(state.get("pred_log", []), maxlen=self.pred_log_len)
            self.last_metrics = state.get("last_metrics", {})
            self.total_processed_games = int(state.get("total_processed_games", 0))
            bot_list = state.get("bot_list", {})
            self.bot_set = set(bot_list.keys())
            for uid, nick in bot_list.items():
                self.user_stats[uid]["nickname"] = nick
            self.color_sequence = deque(state.get("color_tail", []), maxlen=self.color_seq_len)
            logger.info("State loaded into assistant")
        except Exception as e:
            logger.exception("Failed to load state: %s", e)

    # -------------------------
    # Utilities
    # -------------------------
    def _record_user_bet(self, user_id: int, amount: float, auto: float, taken_coef: float | None, ts: float, nickname: str | None = None):
        st = self.user_stats[user_id]
        st["count"] += 1
        st["total_amount"] += float(amount or 0.0)
        prev_avg = st["avg_auto"]
        st["avg_auto"] = ((prev_avg * (st["count"] - 1)) + (auto or 0.0)) / st["count"]
        st["last_active"] = max(st["last_active"], ts)
        if taken_coef:
            st["wins"] += 1
            st["total_win"] += max(0.0, (float(taken_coef) - 1.0) * float(amount))
        else:
            st["losses"] += 1
        if nickname:
            st["nickname"] = nickname

    def _features_from_game(self, game: Dict[str, Any]) -> Dict[str, float]:
        num_players = float(game.get("num_players", 0)) or 0.0
        bets = game.get("bets", [])
        if isinstance(bets, str):
            try:
                bets = json.loads(bets)
            except Exception:
                bets = []
        total_amount = 0.0
        avg_auto = 0.0
        bot_count = 0
        for b in bets:
            amt = float(b.get("amount", 0.0) or 0.0)
            total_amount += amt
            auto = float(b.get("auto", 0.0) or 0.0)
            avg_auto += auto
            if b.get("user_id") in self.bot_set:
                bot_count += 1
        avg_auto = (avg_auto / len(bets)) if bets else 0.0
        bot_fraction = (bot_count / len(bets)) if bets else 0.0
        color = game.get("color_bucket", "")
        color_index = (hash(color) % 100) if color else 0
        return {
            "avg_auto": float(avg_auto),
            "total_amount": float(total_amount),
            "num_players": float(num_players),
            "bot_fraction": float(bot_fraction),
            "color_index": float(color_index)
        }

    def _success_rate(self, coef_key: str) -> float:
        successes = 0
        total = 0
        for p in self.pred_log:
            pred = p.get(coef_key)
            actual = p.get("crash_actual")
            if pred is not None and actual is not None:
                if pred <= actual:
                    successes += 1
                total += 1
        return round((successes / total) * 100, 1) if total else 0.0

    # -------------------------
    # History loader
    # -------------------------
    def load_history_from_list(self, games_list: List[Dict[str, Any]]):
        added = 0
        ts_now = time.time()
        for g in games_list:
            try:
                game_id = g.get("game_id") or g.get("id")
                bets = g.get("bets", [])
                if isinstance(bets, str):
                    try:
                        bets = json.loads(bets)
                    except Exception:
                        bets = []
                g["bets"] = bets

                crash_val = g.get("crash")
                color_bucket = assign_color(float(crash_val)) if crash_val not in (None, "", "null") else None

                compact = {
                    "game_id": int(game_id) if game_id is not None else None,
                    "crash": float(crash_val) if crash_val not in (None, "", "null") else None,
                    "color_bucket": color_bucket,
                    "num_players": int(g.get("num_players") or 0)
                }
                self.history_deque.append(compact)
                self.total_processed_games += 1

                if color_bucket:
                    self.color_sequence.append(color_bucket)

                for b in bets:
                    uid = b.get("user_id")
                    amt = float(b.get("amount", 0.0) or 0.0)
                    auto = float(b.get("auto", 0.0) or 0.0)
                    nickname = b.get("nickname")
                    taken_coef = b.get("coefficient") if b.get("coefficient") not in (None, "", "null") else None
                    if taken_coef is None and auto and crash_val is not None and crash_val >= auto:
                        taken_coef = auto
                    self._record_user_bet(uid, amt, auto, taken_coef, ts_now, nickname=nickname)

                if compact["crash"] is not None:
                    features = self._features_from_game({"bets": bets,
                                                         "num_players": compact["num_players"],
                                                         "color_bucket": compact["color_bucket"]})
                    sample = {"features": features, "target": float(compact["crash"])}
                    self.training_buffer.append(sample)
                added += 1
            except Exception as e:
                logger.exception("Failed to process game in load_history_from_list: %s", e)
        logger.info("Loaded %d new games; total games: %d", added, self.total_processed_games)

        if len(self.training_buffer) >= self.pending_threshold and (time.time() - self.last_trained_at) > self.retrain_min_seconds:
            self._online_train()

    # -------------------------
    # Prediction API
    # -------------------------
    def predict_and_log(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            game_id = payload.get("game_id")
            bets = payload.get("bets", [])
            if isinstance(bets, str):
                try:
                    bets = json.loads(bets)
                except Exception:
                    bets = []

            features = self._features_from_game({"bets": bets,
                                                "num_players": len(bets),
                                                "color_bucket": None})
            med_pred = None
            try:
                med_pred = float(self.model_med.predict(pd.DataFrame([features]))[0])
                med_pred = max(1.0, med_pred)
            except Exception:
                hist_crashes = [h["crash"] for h in list(self.history_deque) if h.get("crash")]
                hist_mean = float(np.mean(hist_crashes)) if hist_crashes else 1.5
                med_pred = clamp((features["avg_auto"] * 0.6 + hist_mean * 0.4), 1.01, 1000.0)

            safe = clamp(med_pred * 0.88, 1.01, med_pred)
            risk = clamp(med_pred * 1.20, med_pred, med_pred * 3)
            recommended = clamp(safe + (med_pred - safe) * 0.5, safe, risk)

            recommended_pct = clamp(0.05 / max(recommended - 1.0, 0.01), 0.005, 0.2)

            result = {
                "game_id": game_id,
                "safe": round(float(safe), 3),
                "med": round(float(med_pred), 3),
                "risk": round(float(risk), 3),
                "recommended": round(float(recommended), 3),
                "recommended_pct": float(round(recommended_pct, 4)),
                "ts": time.time(),
                "crash_actual": None
            }

            # сохраняем предсказание
            self.pred_log.append(result)
            if len(self.pred_log) > 75:
                self.pred_log.popleft()

            # обновление success_rate и avg_error
            for coef in ["safe", "med", "risk", "recommended"]:
                result[f"success_rate_{coef}"] = self._success_rate(coef)
                avg_err = self.last_metrics.get(f"avg_error_{coef}", None)
                result[f"avg_error_{coef}"] = round(avg_err, 4) if avg_err is not None else None

            if self.ably_channel:
                try:
                    self.ably_channel.publish("new_prediction", result)
                except Exception as e:
                    logger.exception("Ably publish failed: %s", e)

            return result
        except Exception as e:
            logger.exception("predict_and_log failed: %s", e)
            raise

    # -------------------------
    # Feedback & online learning
    # -------------------------
    def process_feedback(self, game_id: int, crash: float, bets: List[Dict[str, Any]] | None = None):
        ts_now = time.time()
        if bets:
            for b in bets:
                uid = b.get("user_id")
                amt = float(b.get("amount", 0.0) or 0.0)
                auto = float(b.get("auto", 0.0) or 0.0)
                nickname = b.get("nickname")
                taken_coef = b.get("coefficient") if b.get("coefficient") not in (None, "", "null") else None
                if taken_coef is None and auto and crash is not None and crash >= auto:
                    taken_coef = auto
                self._record_user_bet(uid, amt, auto, taken_coef, ts_now, nickname=nickname)

        features = self._features_from_game({"bets": bets or [], "num_players": len(bets or []), "color_bucket": None})
        self.training_buffer.append({"features": features, "target": float(crash)})

        # Обновление фактического значения и ошибок
        last_pred = None
        for p in reversed(self.pred_log):
            if p.get("game_id") == game_id:
                last_pred = p
                break
        if last_pred:
            last_pred["crash_actual"] = crash
            for coef_name in ["safe", "med", "risk", "recommended"]:
                last_val = last_pred.get(coef_name)
                if last_val:
                    error = abs(last_val - crash) / max(1.0, crash)
                    prev = self.last_metrics.get(f"avg_error_{coef_name}", 0.0)
                    count = self.last_metrics.get(f"count_error_{coef_name}", 0)
                    new_avg = (prev * count + error) / (count + 1)
                    self.last_metrics[f"avg_error_{coef_name}"] = new_avg
                    self.last_metrics[f"count_error_{coef_name}"] = count + 1

        if len(self.training_buffer) >= self.pending_threshold and (time.time() - self.last_trained_at) > self.retrain_min_seconds:
            self._online_train()

    # -------------------------
    # Online training
    # -------------------------
    def _online_train(self):
        if not self.training_buffer:
            return
        df = pd.DataFrame([{"target": s["target"], **s["features"]} for s in self.training_buffer])
        X = df.drop(columns=["target"])
        y = df["target"]
        if len(y) >= 5:  # минимум 5 примеров
            self.model_med.fit(X, y)
            self.last_trained_at = time.time()
            logger.info("Online training completed on %d samples", len(y))
        self.training_buffer.clear()

    # -------------------------
    # Status & bot management
    # -------------------------
    def get_pred_log(self, limit: int = 20):
        return list(self.pred_log)[-limit:]

    def get_status(self):
        return {
            "total_games": self.total_processed_games,
            "pred_log_len": len(self.pred_log),
            "last_trained_at": self.last_trained_at,
            "bot_count": len(self.bot_set)
        }

    def mark_user_as_bot(self, user_id: int):
        self.bot_set.add(user_id)

    def unmark_user_as_bot(self, user_id: int):
        self.bot_set.discard(user_id)