# ai_helper.py
import os
import json
import math
from collections import Counter, deque, defaultdict
from flask import Flask, request, jsonify
import numpy as np

# ========== Настройки ==========
HISTORICAL_FILE = os.environ.get("HISTORICAL_FILE", "games_merged.json")
PORT = int(os.environ.get("PORT", 5000))

# Параметры анализа
MAX_SEQ = 5                       # максимальная длина последовательностей цветов
MIN_BOT_BETS = 50                 # порог для пометки юзера как "бот" (по историке)
BANK_PERCENT_BASE = 0.03          # базовый % от банка, если ничего подозрительного
BOT_RISK_PENALTY = 0.5            # как сильно уменьшаем % при сильном бот-давлении
SAFE_QUANTILE = 0.25
MID_QUANTILE = 0.50
HIGH_QUANTILE = 0.75

# ========== Вспомогательные ==========
def color_of(crash):
    c = float(crash)
    if 1 <= c <= 1.19:
        return "red"
    if 1.2 <= c <= 1.99:
        return "blue"
    if 2 <= c <= 3.99:
        return "pink"
    if 4 <= c <= 7.99:
        return "green"
    if 8 <= c <= 24.99:
        return "yellow"
    return "gradient"

def percentile(arr, q):
    if not arr:
        return None
    return float(np.percentile(arr, q * 100))

# ========== Хранение данных в памяти ==========
games = []                         # список {id, crash, color, ...}
color_history = []                 # список цветов в порядке исторических игр
seq_counters = {l: Counter() for l in range(1, MAX_SEQ+1)}
crash_by_color = defaultdict(list) # color -> list of crashes

# профиль пользователей: user_id -> {'count': int, 'autos': Counter(), 'amount': total_amount}
user_profiles = defaultdict(lambda: {"count":0, "autos": Counter(), "amount":0.0})

# быстрый буфер последних цветов (для предсказания)
recent_colors = deque(maxlen=MAX_SEQ)

# ========== Загрузка исторической базы ==========
def load_historical(path):
    if not os.path.exists(path):
        print("Historical file not found:", path)
        return
    print("Loading historical file:", path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for g in data:
        try:
            crash = float(g.get("crash"))
        except Exception:
            continue
        color = color_of(crash)
        games.append({"id": g.get("id"), "crash": crash, "color": color, "bets": g.get("bets")})
        color_history.append(color)
        crash_by_color[color].append(crash)
        # update sequence counters after building full list
    # build seq counters
    for i in range(len(color_history)):
        for l in range(1, MAX_SEQ+1):
            if i + l <= len(color_history):
                seq = tuple(color_history[i:i+l])
                seq_counters[l][seq] += 1
    # build user profiles from bets in history if present
    for g in data:
        bets = g.get("bets") or []
        for b in bets:
            try:
                uid = b.get("user_id")
                auto = float(b.get("auto", 0))
                amt = float(b.get("amount", 0) or 0)
            except Exception:
                continue
            profile = user_profiles[uid]
            profile["count"] += 1
            profile["autos"][round(auto,2)] += 1
            profile["amount"] += amt
    recent_colors.extend(color_history[-MAX_SEQ:])
    print(f"Loaded {len(games)} games, {len(user_profiles)} users profiled.")

# ========== Utility: detect bots heuristically ==========
def is_bot_profile(profile):
    # простая эвристика: много ставок в истории OR очень repetitive autos OR tiny user_id numeric names omitted
    if profile["count"] >= MIN_BOT_BETS:
        return True
    # if top auto frequency is large fraction
    total = profile["count"]
    if total >= 5:
        top = profile["autos"].most_common(1)[0][1]
        if top / total >= 0.8:
            return True
    return False

def top_auto_of_profile(profile):
    if not profile["autos"]:
        return None
    return profile["autos"].most_common(1)[0][0]

# ========== Sequence prediction ==========
def predict_next_color_by_sequence(recent_colors_list):
    # Try longest match first: find sequences ending with recent_colors_list and return most common continuation
    for l in range(MAX_SEQ, 0, -1):
        if len(recent_colors_list) < l:
            continue
        key = tuple(recent_colors_list[-l:])
        # Find sequences of length l+1 where prefix == key
        candidate = Counter()
        for seq, cnt in seq_counters.get(l+1, {}).items():
            if seq[:-1] == key:
                candidate[seq[-1]] += cnt
        if candidate:
            return candidate.most_common(1)[0][0]
    # fallback: most frequent color globally
    global_counts = Counter()
    for col, arr in crash_by_color.items():
        global_counts[col] = len(arr)
    if not global_counts:
        return None
    return global_counts.most_common(1)[0][0]

# ========== Core prediction logic ==========
def compute_prediction(current_bets, bank_amount=None):
    # update temporary stats from incoming bets (but do not persist permanently unless desired)
    # classify current bets: count bots among current bettors by historical profile and by heuristics (frequent bettors)
    current_count = len(current_bets)
    if current_count == 0:
        return {"error":"no current bets"}

    # update user profiles online (we will persist these updates into user_profiles)
    bot_votes = 0
    total_bet_amount = 0.0
    weighted_auto_sum = 0.0
    weighted_count = 0.0

    for b in current_bets:
        uid = b.get("user_id")
        auto = float(b.get("auto", 1.0))
        amt = float(b.get("amount", 0) or 0)
        # update profile
        profile = user_profiles[uid]
        profile["count"] += 1
        profile["autos"][round(auto,2)] += 1
        profile["amount"] += amt

        total_bet_amount += amt
        weighted_auto_sum += auto * max(amt, 0.01)
        weighted_count += max(amt, 0.01)

        # detect if this user looks like bot (quick heuristic using updated profile)
        if is_bot_profile(profile):
            bot_votes += 1

    # determine bot pressure: fraction of bettors that are bots
    bot_fraction = bot_votes / current_count

    # baseline avg auto (weighted)
    avg_auto = (weighted_auto_sum / weighted_count) if weighted_count else np.mean([b.get("auto",1.0) for b in current_bets])

    # sequence-based color prediction
    next_color = predict_next_color_by_sequence(list(recent_colors))

    # distribution-based crash percentiles for predicted color (fallback to global)
    target_color = next_color or ("all")
    if target_color in crash_by_color and crash_by_color[target_color]:
        arr = crash_by_color[target_color]
    else:
        # global array
        arr = []
        for v in crash_by_color.values():
            arr.extend(v)
    if not arr:
        # fallback simple heuristic based on avg_auto
        safe = max(1.01, round(avg_auto * 0.9, 2))
        mid = round(avg_auto, 2)
        high = round(avg_auto * 1.15 + 0.01, 2)
    else:
        safe = round(percentile(arr, SAFE_QUANTILE), 2)
        mid = round(percentile(arr, MID_QUANTILE), 2)
        high = round(percentile(arr, HIGH_QUANTILE), 2)

    # Bank percent recommendation: reduce if many bots or if many large bets concentrated
    suggested_percent = BANK_PERCENT_BASE
    # if a lot of total bet amount relative to bank, be more conservative
    if bank_amount and bank_amount > 0:
        exposure = total_bet_amount / bank_amount
        # exposure high -> reduce percent
        if exposure > 0.05:
            suggested_percent *= 0.6
        if exposure > 0.1:
            suggested_percent *= 0.5
    # apply bot penalty
    if bot_fraction > 0.2:
        suggested_percent *= (1 - BOT_RISK_PENALTY * bot_fraction)  # decrease with bot fraction
    suggested_percent = max(0.001, round(suggested_percent, 4))

    # update recent colors buffer using last seen (we don't have final result yet), don't mutate historical seq_counters here
    # but we can compute diagnostics: last N colors
    diagnostics = {
        "bot_fraction": round(bot_fraction, 3),
        "current_bets_count": current_count,
        "avg_auto_weighted": round(avg_auto, 3),
        "predicted_color": next_color,
        "safe_crash": safe,
        "mid_crash": mid,
        "high_crash": high,
        "suggested_bank_percent": suggested_percent,
        "historical_games_count": len(games)
    }

    return diagnostics

# ========== Flask API ==========
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    payload = request.get_json(force=True)
    bets = payload.get("bets", [])
    bank_amount = payload.get("bank")   # optional, numeric
    # normalize bets: list of dicts with keys user_id, auto, amount
    norm_bets = []
    for b in bets:
        try:
            norm_bets.append({
                "user_id": b.get("user_id"),
                "auto": float(b.get("auto", 1.0)),
                "amount": float(b.get("amount", 0) or 0)
            })
        except Exception:
            continue

    result = compute_prediction(norm_bets, bank_amount=bank_amount)

    # after prediction, we also update rolling sequences if the payload optionally includes 'last_game_color' or 'last_game_crash'
    last_game_crash = payload.get("last_game_crash")
    last_game_color = None
    if last_game_crash is not None:
        try:
            last_game_color = color_of(float(last_game_crash))
        except Exception:
            last_game_color = None
    elif payload.get("last_game_color"):
        last_game_color = payload.get("last_game_color")

    if last_game_color:
        # push into recent buffer and update seq_counters and crash_by_color only if payload includes last_game_crash value explicitly
        recent_colors.append(last_game_color)
        # optionally update seq counters for lengths
        # We only increment sequence counters for local stats (to adapt model online)
        for l in range(1, MAX_SEQ+1):
            if len(recent_colors) >= l:
                seq = tuple(list(recent_colors)[-l:])
                seq_counters[l][seq] += 1

    return jsonify(result)

@app.route("/stats", methods=["GET"])
def stats():
    # basic diagnostics endpoint
    top_colors = {c: len(arr) for c, arr in crash_by_color.items()}
    # identify top bot user_ids (by profile count)
    bot_candidates = []
    for uid, prof in user_profiles.items():
        if is_bot_profile(prof):
            bot_candidates.append({"user_id": uid, "count": prof["count"], "top_auto": top_auto_of_profile(prof)})
    bot_candidates_sorted = sorted(bot_candidates, key=lambda x: -x["count"])[:50]
    return jsonify({
        "historical_games": len(games),
        "color_counts": top_colors,
        "top_bot_candidates": bot_candidates_sorted,
        "recent_colors": list(recent_colors)
    })

# ========== Start ==========
if __name__ == "__main__":
    load_historical(HISTORICAL_FILE)
    print("Starting AI helper on port", PORT)
    app.run(host="0.0.0.0", port=PORT)