"""
Метрики/аналитика: функции оценки качества прогнозов, базовый расчёт recommended pct и
рекомендованных коэффициентов. Полное сохранение состояния через BackupManager.
"""

from typing import List, Dict, Any
import logging
from utils import calculate_net_win, crash_to_color

logger = logging.getLogger("ai_assistant.analytics")
logger.setLevel(logging.INFO)


class Analytics:
    def __init__(self):
        # Полное состояние аналитики
        self.analytics_state = {
            "pred_records": [],      # все записи прогнозов
            "user_results": {},      # словарь user_id -> результат
            "color_annotations": {}  # дополнительная аналитика по цветам
        }
        self.ready_for_backup = True  # атрибут для BackupManager
        logger.info("Analytics module initialized")

    # -------------------------
    # Метрики и расчет ошибок
    # -------------------------
    @staticmethod
    def compute_avg_relative_error(preds: List[float], actuals: List[float]) -> float:
        errors = []
        for p, a in zip(preds, actuals):
            try:
                a = float(a)
                p = float(p)
                if a == 0:
                    continue
                errors.append(abs(p - a) / a)
            except Exception:
                continue
        if not errors:
            return float("nan")
        return sum(errors) / len(errors)

    @staticmethod
    def compute_hit_rate(preds: List[float], actuals: List[float]) -> float:
        hits = 0
        total = 0
        for p, a in zip(preds, actuals):
            try:
                if float(p) <= float(a):
                    hits += 1
                total += 1
            except Exception:
                continue
        return hits / total if total else float("nan")

    def evaluate_predictions_batch(self, pred_records: List[Dict[str, Any]]) -> Dict[str, float]:
        safe_preds, med_preds, risk_preds, actuals = [], [], [], []
        for r in pred_records:
            try:
                safe_preds.append(float(r.get("pred_safe", 0)))
                med_preds.append(float(r.get("pred_med", 0)))
                risk_preds.append(float(r.get("pred_risk", 0)))
                actuals.append(float(r.get("actual", 0)))
            except Exception:
                continue

        return {
            "avg_rel_error_safe": self.compute_avg_relative_error(safe_preds, actuals),
            "avg_rel_error_med": self.compute_avg_relative_error(med_preds, actuals),
            "avg_rel_error_risk": self.compute_avg_relative_error(risk_preds, actuals),
            "hit_rate_safe": self.compute_hit_rate(safe_preds, actuals),
            "hit_rate_med": self.compute_hit_rate(med_preds, actuals),
            "hit_rate_risk": self.compute_hit_rate(risk_preds, actuals),
        }

    def recommend_percent_from_risk(self, risk_score: float) -> float:
        base = 0.05
        pct = max(0.005, base * (1.0 - risk_score))
        return pct

    # -------------------------
    # Новые функции для анализа ставок
    # -------------------------
    def compute_net_wins(self, user_results: List[Dict[str, Any]]) -> Dict[int, float]:
        net_wins = {}
        for u in user_results:
            uid = u.get("user_id")
            coef = u.get("coefficient")
            amt = u.get("amount", 0.0)
            net_wins[uid] = calculate_net_win(amt, coef)
        self.analytics_state["user_results"] = net_wins
        return net_wins

    def annotate_crash_colors(self, crash: float) -> str:
        color = crash_to_color(crash)
        self.analytics_state["color_annotations"][crash] = color
        return color

    # -------------------------
    # Полное сохранение/загрузка состояния
    # -------------------------
    def export_state(self) -> Dict[str, Any]:
        """
        Экспорт всего состояния аналитики для BackupManager.
        """
        return {
            "analytics_state": self.analytics_state.copy(),
            "timestamp": time.time()
        }

    def load_state(self, state: Dict[str, Any]):
        """
        Восстановление состояния аналитики из бэкапа.
        """
        try:
            self.analytics_state = state.get("analytics_state", {}).copy()
            logger.info("Analytics state loaded successfully")
        except Exception as e:
            logger.exception("Failed to load analytics state: %s", e)