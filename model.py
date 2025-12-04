# model.py
import os
import time
import math
import json
import joblib
import logging
from collections import deque, Counter, defaultdict
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

logger = logging.getLogger("ai_assistant.model")
logger.setLevel(logging.INFO)


def clamp(v, a, b):
    return max(a, min(b, v))


class AIAssistant:
    """
    Лёгкая, ресурс-ориентированная реализация помощника.
    Основные цели:
      - не раздувать память (ограниченные deque'ы для истории/предсказаний/трейна),
      - давать рабочие предсказания без тяжёлого ML;
      - сохранять компактное состояние через export_state/load_state.
    """

    def __init__(self,
                 color_seq_len: int = 4000,
                 pred_log_len: int = 2000,
                 pending_threshold: int = 50,
                 retrain_min_minutes: int = 10,
                 max_history_records: int = 50000,
                 max_training_buffer: int = 50000):
        # Параметры
        self.color_seq_len = color_seq_len
        self.pred_log_len = pred_log_len
        self.pending_threshold = pending_threshold
        self.retrain_min_seconds = retrain_min_minutes * 60

        # Ограниченные буферы
        self.color_sequence = deque(maxlen=color_seq_len)            # последовательность цветов (для pattern)
        self.history_deque = deque(maxlen=max_history_records)       # recent games (compact dicts)
        self.pred_log = deque(maxlen=pred_log_len)                  # последние предсказания

        # Пользовательская статистика (агрегаты)
        # user_stats[user_id] = {'count', 'total_amount', 'avg_auto', 'last_active', 'wins', 'losses', 'total_win'}
        self.user_stats = defaultdict(lambda: {
            "count": 0,
            "total_amount": 0.0,
            "avg_auto": 0.0,
            "last_active": 0,
            "wins": 0,
            "losses": 0,
            "total_win": 0.0
        })

        # Набор выявленных ботов (user_id)
        self.bot_set = set()

        # Модели (light-weight)
        # модель предсказывает численное значение crash
        self.model_med = LGBMRegressor(n_estimators=80, verbose=-1)
        self.model_safe = None  # можем сохранять дополнительные модели при необходимости
        self.model_risk = None

        # Буфер для онлайн-обучения: список dict {'features':..., 'target': crash}
        self.training_buffer = deque(maxlen=max_training_buffer)

        # Метрики/статистика
        self.last_trained_at = 0
        self.last_metrics = {}
        self.total_processed_games = 0

        # Доп. параметры
        self.max_active_users_to_cache = 5000  # можно менять
        self.min_train_samples = 200           # сколько примеров нужно для обучения

        logger.info("AIAssistant initialized")

    # -------------------------
    # State (backup/restore)
    # -------------------------
    def export_state(self) -> Dict[str, Any]:
        """
        Возвращает компактный JSON-сериализуемый снимок состояния.
        Не встраивает большие массивы данных.
        """
        state = {
            "pred_log": list(self.pred_log),  # последние предсказания (целые/float -> json ok)
            "last_metrics": self.last_metrics,
            "total_processed_games": int(self.total_processed_games),
            # бот-лист (ограниченный размер)
            "bot_list": list(sorted(list(self.bot_set)))[:5000],
            # user summary (только ключи и небольшая статистика для быстрого восстановления)
            "user_summary_count": len(self.user_stats),
            # color sequence tail (оставим компактно — последние N цветов)
            "color_tail": list(self.color_sequence)[-1000:],  # 1000 последних цветов
            "timestamp": time.time()
        }
        return state

    def load_state(self, state: Dict[str, Any]):
        """
        Восстанавливает минимальное состояние.
        Не пытаемся восстановить тяжёлые структуры — они будут собраны при работе.
        """
        try:
            pred_log = state.get("pred_log", [])
            self.pred_log = deque(pred_log, maxlen=self.pred_log_len)
            self.last_metrics = state.get("last_metrics", {})
            self.total_processed_games = int(state.get("total_processed_games", 0))
            bot_list = state.get("bot_list", [])
            self.bot_set = set(bot_list)
            color_tail = state.get("color_tail", [])
            self.color_sequence = deque(color_tail, maxlen=self.color_seq_len)
            logger.info("State loaded into assistant")
        except Exception as e:
            logger.exception("Failed to load state: %s", e)

    # -------------------------
    # Utilities: features / scoring
    # -------------------------
    def _record_user_bet(self, user_id: int, amount: float, auto: float, taken_coef: float | None, ts: float):
        st = self.user_stats[user_id]
        st["count"] += 1
        st["total_amount"] += float(amount or 0.0)
        # incremental avg
        prev_avg = st["avg_auto"]
        st["avg_auto"] = ((prev_avg * (st["count"] - 1)) + (auto or 0.0)) / st["count"]
        st["last_active"] = max(st["last_active"], ts)
        if taken_coef:
            st["wins"] += 1
            st["total_win"] += max(0.0, (float(taken_coef) - 1.0) * float(amount))
        else:
            st["losses"] += 1

    def _features_from_game(self, game: Dict[str, Any]) -> Dict[str, float]:
        """
        Собирает простые признаки из записи игры для предсказания:
          - avg_auto, total_amount, num_players, bot_fraction, color_index
        """
        num_players = float(game.get("num_players", 0)) or 0.0
        # bets может быть списком или строкой JSON
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
            uid = b.get("user_id")
            if uid in self.bot_set:
                bot_count += 1
        avg_auto = (avg_auto / len(bets)) if bets else 0.0
        bot_fraction = (bot_count / len(bets)) if bets else 0.0

        color = game.get("color_bucket", "")
        # map color to small integer: deterministic hash
        color_index = (hash(color) % 100) if color else 0

        return {
            "avg_auto": float(avg_auto),
            "total_amount": float(total_amount),
            "num_players": float(num_players),
            "bot_fraction": float(bot_fraction),
            "color_index": float(color_index)
        }

    # -------------------------
    # Loading history (batch)
    # -------------------------
    def load_history_from_list(self, games_list: List[Dict[str, Any]]):
        """
        Обработка блока игр (вызывается из main.load_history_files).
        Мы:
          - собираем компактную историю (ограниченная deque),
          - обновляем статистики по пользователям,
          - добавляем в буфер обучения примеры (если есть crash),
          - отмечаем цветовую последовательность.
        """
        added = 0
        ts_now = time.time()
        for g in games_list:
            try:
                game_id = g.get("game_id") or g.get("id")
                # нормализуем bets
                bets = g.get("bets", [])
                if isinstance(bets, str):
                    try:
                        bets = json.loads(bets)
                    except Exception:
                        bets = []
                g["bets"] = bets

                # add to history (compact)
                compact = {
                    "game_id": int(game_id) if game_id is not None else None,
                    "crash": float(g["crash"]) if g.get("crash") not in (None, "", "null") else None,
                    "color_bucket": g.get("color_bucket"),
                    "num_players": int(g.get("num_players") or 0)
                }
                self.history_deque.append(compact)
                self.total_processed_games += 1
                # colors
                if compact["color_bucket"]:
                    self.color_sequence.append(compact["color_bucket"])

                # update user stats from bets
                for b in bets:
                    uid = b.get("user_id")
                    amt = float(b.get("amount", 0.0) or 0.0)
                    auto = float(b.get("auto", 0.0) or 0.0)
                    # taken_coef may be present later (in feedback), here unknown -> None
                    self._record_user_bet(uid, amt, auto, None, ts_now)

                # if crash present, add example to training buffer
                if compact["crash"] is not None:
                    features = self._features_from_game({"bets": bets,
                                                        "num_players": compact["num_players"],
                                                        "color_bucket": compact["color_bucket"]})
                    sample = {
                        "features": features,
                        "target": float(compact["crash"])
                    }
                    self.training_buffer.append(sample)
                added += 1
            except Exception as e:
                logger.exception("Failed to process game in load_history_from_list: %s", e)
        logger.info("Loaded %d new games; total games: %d", added, self.total_processed_games)

        # Опционально: триггерим обучение, если накопилось достаточно
        if len(self.training_buffer) >= self.pending_threshold and (time.time() - self.last_trained_at) > self.retrain_min_seconds:
            try:
                self._online_train()
            except Exception as e:
                logger.exception("Online train failed: %s", e)

    # -------------------------
    # Prediction API
    # -------------------------
    def predict_and_log(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        payload: dict с полями game_id, bets (список)
        Возвращает и логирует предсказание в self.pred_log.
        Формирует: safe, med, risk, recommended_pct.
        """
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
            # Predict med
            med_pred = None
            try:
                # модель может быть не обучена
                med_pred = float(self.model_med.predict(pd.DataFrame([features]))[0])
                med_pred = max(1.0, med_pred)
            except Exception:
                # fallback: простой эвристический прогноз: weighted avg of avg_auto и mean crash history
                hist_crashes = [h["crash"] for h in list(self.history_deque) if h.get("crash")]
                hist_mean = float(np.mean(hist_crashes)) if hist_crashes else 1.5
                med_pred = clamp((features["avg_auto"] * 0.6 + hist_mean * 0.4), 1.01, 1000.0)

            # derive safe/risk heuristics
            safe = clamp(med_pred * 0.88, 1.01, med_pred)   # чуть осторожнее
            risk = clamp(med_pred * 1.20, med_pred, med_pred * 3)

            # recommended_pct: чем выше predicted med and farther from 1, тем меньше рекомендуемый процент.
            # простая формула: base = 0.05, scale inversely by (med_pred - 1)
            if med_pred - 1.0 <= 0:
                recommended_pct = 0.01
            else:
                recommended_pct = clamp(0.05 / (med_pred - 1.0), 0.005, 0.2)

            result = {
                "game_id": game_id,
                "safe": round(float(safe), 3),
                "med": round(float(med_pred), 3),
                "risk": round(float(risk), 3),
                "recommended_pct": float(round(recommended_pct, 4)),
                "ts": time.time()
            }

            # log
            self.pred_log.append(result)
            return result
        except Exception as e:
            logger.exception("predict_and_log failed: %s", e)
            raise

    # -------------------------
    # Feedback & online learning
    # -------------------------
    def process_feedback(self, game_id: int, crash: float, bets: List[Dict[str, Any]] | None = None):
        """
        Добавляет факт завершившейся игры (game_id, crash) в буфер обучения и пересчитывает user stats (дополняет taken_coef).
        """
        try:
            ts_now = time.time()
            # Уточняем user_stats: пометим у участников их фактический результат
            if bets:
                for b in bets:
                    uid = b.get("user_id")
                    amt = float(b.get("amount", 0.0) or 0.0)
                    auto = float(b.get("auto", 0.0) or 0.0)
                    # taken coefficient: если у игрока есть поле 'coefficient' or 'coefficientAuto' or 'coefficient' maybe None
                    taken_coef = b.get("coefficient") if b.get("coefficient") not in (None, "", "null") else None
                    # если нет taken_coef и crash >= auto, можно infer they cashed at auto
                    if taken_coef is None and auto and crash is not None and crash >= auto:
                        taken_coef = auto
                    self._record_user_bet(uid, amt, auto, taken_coef, ts_now)

            # Добавляем пример в training_buffer
            # features вычислим по bets
            features = self._features_from_game({"bets": bets or [], "num_players": len(bets or []), "color_bucket": None})
            sample = {"features": features, "target": float(crash)}
            self.training_buffer.append(sample)

            # update metrics (simple)
            # compute last prediction for same game_id if exists and accumulate error
            last_pred = None
            for p in reversed(self.pred_log):
                if p.get("game_id") == game_id:
                    last_pred = p
                    break
            if last_pred:
                error = abs(last_pred["med"] - crash) / max(1.0, crash)
                # rolling metric
                prev = self.last_metrics.get("avg_error", 0.0)
                count = self.last_metrics.get("count_error", 0)
                new_avg = (prev * count + error) / (count + 1)
                self.last_metrics["avg_error"] = new_avg
                self.last_metrics["count_error"] = count + 1

            # возможно запуск обучения
            if len(self.training_buffer) >= self.pending_threshold and (time.time() - self.last_trained_at) > self.retrain_min_seconds:
                try:
                    self._online_train()
                except Exception as e:
                    logger.exception("process_feedback: _online_train failed: %s", e)
        except Exception as e:
            logger.exception("process_feedback failed: %s", e)
            raise

    def _online_train(self):
        """
        Быстрое и лёгкое обучение на собранных примерах в training_buffer.
        Обучаем одну модель (model_med). Храним простые метрики.
        """
        try:
            n = len(self.training_buffer)
            if n < self.min_train_samples:
                logger.info("Not enough samples to train: %d/%d", n, self.min_train_samples)
                return

            logger.info("Starting online train on %d samples", n)
            # подготовим датафрейм
            feats = []
            targets = []
            for s in self.training_buffer:
                feats.append(s["features"])
                targets.append(s["target"])
            df = pd.DataFrame(feats)
            y = np.array(targets, dtype=float)

            # simple train/test split for metrics
            idx = int(len(df) * 0.8)
            if idx < 20:
                idx = max(1, int(len(df) * 0.7))
            X_train = df.iloc[:idx, :].fillna(0.0)
            y_train = y[:idx]
            X_val = df.iloc[idx:, :].fillna(0.0)
            y_val = y[idx:]

            # train
            self.model_med = LGBMRegressor(n_estimators=100, verbose=-1)
            self.model_med.fit(X_train, y_train)

            # metrics
            y_pred = self.model_med.predict(X_val) if len(y_val) > 0 else self.model_med.predict(X_train)
            rmse = float(np.sqrt(np.mean((y_pred - (y_val if len(y_val) > 0 else y_train)) ** 2)))
            mae = float(np.mean(np.abs(y_pred - (y_val if len(y_val) > 0 else y_train))))
            self.last_metrics.update({
                "rmse": rmse,
                "mae": mae,
                "trained_samples": int(len(df)),
                "trained_at": time.time()
            })
            self.last_trained_at = time.time()
            logger.info("Online training finished. RMSE=%.4f MAE=%.4f samples=%d", rmse, mae, len(df))

            # optionally persist model to disk (keeps backup small; files stored locally)
            try:
                fname = "model_med.joblib"
                joblib.dump(self.model_med, fname, compress=3)
                self.last_metrics["model_file"] = fname
            except Exception as e:
                logger.warning("Failed to save model to disk: %s", e)

        except Exception as e:
            logger.exception("_online_train failed: %s", e)
            raise

    # -------------------------
    # Introspection helpers
    # -------------------------
    def get_pred_log(self, limit: int = 20):
        try:
            items = list(self.pred_log)[-limit:]
            # ensure JSON-serializable
            def norm(x):
                if isinstance(x, float):
                    return float(round(x, 6))
                return x
            return [{k: (norm(v) if not isinstance(v, (list, dict)) else v) for k, v in it.items()} for it in items]
        except Exception:
            return []

    def get_status(self):
        return {
            "total_games": int(self.total_processed_games),
            "pred_log_len": len(self.pred_log),
            "total_users": int(len(self.user_stats)),
            "total_bots": int(len(self.bot_set)),
            "training_buffer": int(len(self.training_buffer)),
            "last_metrics": self.last_metrics
        }

    # -------------------------
    # Bot / user helpers (API)
    # -------------------------
    def mark_user_as_bot(self, user_id: int):
        self.bot_set.add(user_id)
        logger.info("Marked user %s as bot", user_id)

    def unmark_user_as_bot(self, user_id: int):
        if user_id in self.bot_set:
            self.bot_set.remove(user_id)
            logger.info("Unmarked user %s as bot", user_id)