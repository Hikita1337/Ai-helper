# model.py
import time
import numpy as np
import pandas as pd
from collections import Counter, deque
import logging

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

logger = logging.getLogger("ai_assistant.model")


class AIAssistant:
    def __init__(self):
        # Основные структуры
        self.history_df = pd.DataFrame()
        self.crash_values = []
        self.games_index = set()
        self.user_counts = Counter()
        self.color_counts = Counter()
        self.color_sequence = deque(maxlen=50)   # последние N цветов
        self.pred_log = deque(maxlen=1000)

        # Поддержка паттернов ботов: uid -> list of {amount, auto, crash}
        self.bot_patterns = {}

        # Модели (LightGBM) — могут быть None, если пакет не установлен
        self.model_safe = LGBMRegressor() if LGBMRegressor else None
        self.model_med = LGBMRegressor() if LGBMRegressor else None
        self.model_risk = LGBMRegressor() if LGBMRegressor else None

        # Очередь для батчевого онлайн-обучения
        self.pending_feedback = []

    # -------------------- Утилиты состояния --------------------
    def export_state(self):
        """
        Экспорт минимального состояния для бэкапа.
        Не сохраняем целиком history_df (можно сохранять частично при желании).
        """
        return {
            "history_tail": self.history_df.tail(20000).to_dict(orient="records"),
            "crash_values_tail": list(self.crash_values[-20000:]),
            "games_index": list(self.games_index)[-20000:],
            "user_counts": dict(self.user_counts),
            "bot_patterns": self.bot_patterns,
            "color_sequence": list(self.color_sequence),
            "pred_log": list(self.pred_log)[-1000:]
        }

    def load_state(self, state: dict):
        """
        Загружает состояние, полученное из export_state.
        """
        try:
            hist = state.get("history_tail", [])
            if hist:
                self.history_df = pd.DataFrame(hist)
            else:
                self.history_df = pd.DataFrame()

            self.crash_values = state.get("crash_values_tail", []) or []
            self.games_index = set(state.get("games_index", []) or [])
            self.user_counts = Counter(state.get("user_counts", {}) or {})
            self.bot_patterns = state.get("bot_patterns", {}) or {}
            seq = state.get("color_sequence", []) or []
            self.color_sequence = deque(seq, maxlen=50)
            self.pred_log = deque(state.get("pred_log", []) or [], maxlen=1000)
            logger.info("State loaded into assistant")
        except Exception as e:
            logger.exception("Failed to load state: %s", e)

    def history_count(self):
        return int(self.history_df.shape[0])

    def get_pred_log(self, limit: int = 20):
        return list(self.pred_log)[-limit:]

    # -------------------- Загрузка истории --------------------
    def load_history_from_list(self, games_list):
        """
        Принимает список игр (dict). Обновляет history_df, user_counts, bot_patterns, color_sequence.
        Обрабатывает блоки: не очищает всё, просто дописывает новые записи.
        """
        rows = []
        for item in games_list:
            gid = item.get("game_id") or item.get("id")
            if gid is None:
                continue
            if gid in self.games_index:
                continue
            self.games_index.add(gid)

            # безошибочное чтение полей
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

            # обновляем базовые агрегаты
            self.crash_values.append(crash)
            if color_bucket:
                self.color_counts[color_bucket] += 1
                self.color_sequence.append(color_bucket)

            # обновляем user_counts и паттерны
            for b in bets:
                uid = b.get("user_id")
                amt = float(b.get("amount") or 0)
                auto = b.get("auto") if b.get("auto") is not None else b.get("coefficientAuto") if b.get("coefficientAuto") is not None else 1.0
                if uid is not None:
                    self.user_counts[uid] += 1
                    if uid not in self.bot_patterns:
                        self.bot_patterns[uid] = []
                    self.bot_patterns[uid].append({"amount": amt, "auto": float(auto), "crash": crash})

        if rows:
            self.history_df = pd.concat([self.history_df, pd.DataFrame(rows)], ignore_index=True)
        logger.info("Loaded %d games; total history %d", len(rows), self.history_count())

    # -------------------- Детекция ботов --------------------
    def detect_bots_in_snapshot(self, bets):
        """
        Возвращает (frac_amount_from_detected_bots, list_of_bot_ids)
        Правило обнаружения — если пользователь встречается часто (user_counts >= threshold).
        """
        if not bets:
            return 0.0, []

        bot_ids = []
        total_amount = 0.0
        bot_amount = 0.0

        threshold = max(5, int(np.sqrt(max(1, len(self.history_df)))))  # адаптивный порог

        for b in bets:
            uid = b.get("user_id")
            amt = float(b.get("amount") or 0)
            total_amount += amt
            if uid is None:
                continue
            if self.user_counts.get(uid, 0) >= threshold:
                bot_ids.append(uid)
                bot_amount += amt

        frac_amount = (bot_amount / total_amount) if total_amount > 0 else 0.0
        return frac_amount, bot_ids

    # -------------------- Создание признаков цветов --------------------
    def _color_features(self):
        """
        Возвращает вектор: частоты цветов + нормализованные частоты переходов (6x6).
        Итог: 6 + 36 = 42 признака.
        """
        colors = ["red", "blue", "pink", "green", "yellow", "gradient"]
        counts = Counter(self.color_sequence)
        total = max(1, len(self.color_sequence))
        freq_features = [counts.get(c, 0) / total for c in colors]

        # переходы
        transitions = Counter()
        seq = list(self.color_sequence)
        for i in range(len(seq) - 1):
            transitions[(seq[i], seq[i + 1])] += 1
        total_trans = max(1, sum(transitions.values()))
        transition_features = [transitions.get((a, b), 0) / total_trans for a in colors for b in colors]

        return freq_features + transition_features

    # -------------------- Фичи основной --------------------
    def _make_features(self, bets):
        bot_frac, bot_ids = self.detect_bots_in_snapshot(bets)
        total_bets = sum(float(b.get("amount") or 0) for b in bets) if bets else 0.0
        num_bets = len(bets) if bets else 0
        autos = [float(b.get("auto") or b.get("coefficientAuto") or 1.0) for b in bets if (b.get("auto") is not None or b.get("coefficientAuto") is not None)]
        avg_auto = float(np.mean(autos)) if autos else 1.0

        color_feats = self._color_features()
        base = [bot_frac, num_bets, total_bets, avg_auto]
        return np.array(base + color_feats).reshape(1, -1)

    # -------------------- Предикт и логирование --------------------
    def predict_and_log(self, payload):
        """
        payload: dict с keys game_id, bets, deposit_sum, num_players, meta...
        Записываем предсказание в pred_log (не отображаем fast games здесь).
        """
        start = time.time()
        game_id = payload.get("game_id")
        bets = payload.get("bets") or []
        try:
            X = self._make_features(bets)
            if self.model_safe and self.model_med and self.model_risk:
                safe = float(self.model_safe.predict(X)[0])
                med = float(self.model_med.predict(X)[0])
                risk = float(self.model_risk.predict(X)[0])
            else:
                # fallback значения
                safe, med, risk = 1.2, 1.5, 2.0
        except Exception as e:
            logger.debug("Predict error, fallback defaults: %s", e)
            safe, med, risk = 1.2, 1.5, 2.0

        bot_frac, _ = self.detect_bots_in_snapshot(bets)
        total_bets = sum(float(b.get("amount") or 0) for b in bets) if bets else 0.0
        num_bets = len(bets) if bets else 0
        recommended_pct = max(0.5, round(2.0 * (1 - bot_frac), 2))

        entry = {
            "ts": time.time(),
            "game_id": game_id,
            "safe": float(safe),
            "med": float(med),
            "risk": float(risk),
            "recommended_pct": recommended_pct,
            "bot_frac_money": round(bot_frac, 4),
            "num_bets": int(num_bets),
            "total_bets": float(total_bets)
        }
        self.pred_log.append(entry)
        logger.info("Predicted game=%s safe=%.2f med=%.2f risk=%.2f pct=%s", game_id, safe, med, risk, recommended_pct)
        logger.debug("Predict took %.3fs", time.time() - start)

    # -------------------- Обратная связь / онлайн-обучение --------------------
    def process_feedback(self, game_id, crash, bets=None, deposit_sum=None, num_players=None, fast_game=False):
        """
        Добавляем игру в историю, обновляем user_counts и bot_patterns; откладываем для онлайн-обучения.
        fast_game — флаг, указывающий, что предикт не показывался визуально (быстрая игра).
        """
        try:
            if game_id in self.games_index:
                logger.debug("Feedback: game %s already exists, updating", game_id)
            self.games_index.add(game_id)

            crash_f = float(crash)
            self.crash_values.append(crash_f)

            row = {
                "game_id": game_id,
                "crash": crash_f,
                "bets": bets or [],
                "deposit_sum": deposit_sum,
                "num_players": num_players,
                "fast_game": bool(fast_game)
            }

            # обновляем статистику пользователей и паттерны
            for b in (row["bets"] or []):
                uid = b.get("user_id")
                amt = float(b.get("amount") or 0)
                auto = b.get("auto") if b.get("auto") is not None else b.get("coefficientAuto") if b.get("coefficientAuto") is not None else 1.0
                if uid is not None:
                    self.user_counts[uid] += 1
                    if uid not in self.bot_patterns:
                        self.bot_patterns[uid] = []
                    self.bot_patterns[uid].append({"amount": amt, "auto": float(auto), "crash": crash_f})

            # добавляем в history_df
            self.history_df = pd.concat([self.history_df, pd.DataFrame([row])], ignore_index=True)

            # Если есть цвет — обновляем последовательность
            color_bucket = row.get("color_bucket") or None
            if color_bucket:
                self.color_sequence.append(color_bucket)

            # очередь для обучения
            self.pending_feedback.append((row["bets"] or [], crash_f, fast_game))

            # если накоплено достаточно — обучаем
            if len(self.pending_feedback) >= 50:
                try:
                    self._online_train()
                except Exception as e:
                    logger.exception("Online train failed: %s", e)
                finally:
                    self.pending_feedback.clear()

            if fast_game:
                logger.info("Fast game %s processed (no visual predict)", game_id)
            else:
                logger.info("Feedback processed for game %s crash=%.2f", game_id, crash_f)

        except Exception as e:
            logger.exception("process_feedback error: %s", e)

    # -------------------- Онлайн обучение (батч) --------------------
    def _online_train(self):
        """
        Подготавливаем признаки из pending_feedback и дообучаем LightGBM модели.
        Простая целевая генерация: используем реальные краши и берём percentiles как таргеты.
        """
        if not (self.model_safe and self.model_med and self.model_risk):
            logger.warning("LightGBM not available — skipping online train")
            return

        if len(self.pending_feedback) < 5:
            logger.info("Not enough pending feedback to train")
            return

        X = []
        y_safe = []
        y_med = []
        y_risk = []

        # формируем X и таргеты
        for bets, crash, fast_flag in self.pending_feedback:
            feats = self._make_features(bets)
            X.append(feats.flatten().tolist())
            # простая схема целей: p50/p75/p90 на основе краша
            y_safe.append(float(crash) * 0.6)   # приблизительный conservative
            y_med.append(float(crash) * 0.9)
            y_risk.append(float(crash) * 1.1)

        X = np.array(X)
        try:
            self.model_safe.fit(X, np.array(y_safe))
            self.model_med.fit(X, np.array(y_med))
            self.model_risk.fit(X, np.array(y_risk))
            logger.info("Online training finished on %d samples", len(X))
        except Exception as e:
            logger.exception("Training error: %s", e)