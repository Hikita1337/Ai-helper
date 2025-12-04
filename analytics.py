"""
Метрики/аналитика: функции оценки качества прогнозов, базовый расчёт recommended pct и
рекомендованных коэффициентов (stub / простая версия — затем можно усложнять).
"""

from typing import List, Dict, Any
import math
import logging

logger = logging.getLogger("ai_assistant.analytics")


def compute_avg_relative_error(preds: List[float], actuals: List[float]) -> float:
    """
    Средняя относительная погрешность: mean(|pred-actual|/actual).
    Если actual == 0 — пропускаем.
    Возвращаем число в диапазоне 0..inf (0 — идеально).
    """
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
    """
    Доля предсказаний, где pred <= actual (т.е. предсказание 'безопасно').
    """
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
    """
    Ожидаемый формат pred_records: [{ "pred_safe": x, "pred_med": y, "pred_risk": z, "actual": a }, ...]
    Возвращает словарь метрик: avg_error, hit_rate_safe, hit_rate_med, hit_rate_risk
    """
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
    """
    Простейшая формула: чем выше риск_score (0..1) — тем ниже рекомендуемый процент.
    risk_score: 0.0 (very safe) .. 1.0 (very risky)
    Возвращает pct (0.0..1.0)
    """
    base = 0.05  # 5% базовая для безопасного
    pct = max(0.005, base * (1.0 - risk_score))  # не ниже 0.5%
    return pct