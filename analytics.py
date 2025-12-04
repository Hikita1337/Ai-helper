"""
Метрики/аналитика: функции оценки качества прогнозов, базовый расчёт recommended pct и
рекомендованных коэффициентов.
"""

from typing import List, Dict, Any
import logging

from utils import calculate_net_win, crash_to_color

logger = logging.getLogger("ai_assistant.analytics")


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


def evaluate_predictions_batch(pred_records: List[Dict[str, Any]]) -> Dict[str, float]:
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
        "avg_rel_error_safe": compute_avg_relative_error(safe_preds, actuals),
        "avg_rel_error_med": compute_avg_relative_error(med_preds, actuals),
        "avg_rel_error_risk": compute_avg_relative_error(risk_preds, actuals),
        "hit_rate_safe": compute_hit_rate(safe_preds, actuals),
        "hit_rate_med": compute_hit_rate(med_preds, actuals),
        "hit_rate_risk": compute_hit_rate(risk_preds, actuals),
    }


def recommend_percent_from_risk(risk_score: float) -> float:
    base = 0.05
    pct = max(0.005, base * (1.0 - risk_score))
    return pct


# -------------------------
# Новые функции для анализа ставок
# -------------------------
def compute_net_wins(user_results: List[Dict[str, Any]]) -> Dict[int, float]:
    """
    Возвращает словарь user_id -> чистый выигрыш
    Если coefficient=None, игрок проиграл
    """
    net_wins = {}
    for u in user_results:
        uid = u.get("user_id")
        coef = u.get("coefficient")
        amt = u.get("amount", 0.0)  # предполагаем, что в user_results есть amount
        net_wins[uid] = calculate_net_win(amt, coef)
    return net_wins


def annotate_crash_colors(crash: float) -> str:
    """
    Преобразует crash в цвет по шкале:
      1.00-1.19: красный, 1.2-1.99: синий, 2-3.99: розовый, 4-7.99: зеленый, 8-24.99: желтый, 25+: градиент
    """
    return crash_to_color(crash)