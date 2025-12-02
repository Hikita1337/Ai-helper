#!/usr/bin/env python3
# ai_agent.py
# Лёгкий AI-помощник для прогнозов по парсеру игры crash.
# Требования: Python 3.9+, Flask
# Запуск: python ai_agent.py
# Endpoints:
#  POST /load   { "url": "<http or local path to json array>" }
#  POST /predict  { "players":[{ "user_id":..., "name":"...", "amount":..., "auto":... }], "game_id": 6233375, "bank": 100.0 }
#  POST /result { "game_id":..., "crash": ... }
#  GET  /status  -> краткая статистика
#
# Форматы входа совпадают с тем, что у тебя: "bets" items: user_id, name, amount, auto
# Цвета в исторической базе: "red","blue","pink","green","yellow","gradient"
#
from flask import Flask, request, jsonify
import json, math, time, threading, urllib.request
from collections import Counter, defaultdict, deque

app = Flask(__name__)

# -----------------------
# Конфигурация / память
# -----------------------
HISTORY = []           # список dict с полями: game_id, crash, color_bucket, bets, deposit_sum, num_players, created_at(optional)
TOTAL_LOCK = threading.Lock()

# Быстрая аггрегация
TOTAL_GAMES = 0
CRASHES = []           # список float
COLOR_SEQ = []         # последовательность color_bucket по порядку (для частых цепочек)
USER_BET_COUNTS = Counter()  # как часто встречается username
USER_TOTAL_AMOUNT = defaultdict(float)

# Порог для выделения «ботов» по частоте
BOT_FREQ_THRESHOLD = 300    # если юзер в истории > threshold — помечаем как бот (значение ориентировочное)
# Limiting bet fraction
MAX_BANK_FRACTION = 0.20    # никогда не рекомендуем >20% от банка

# Пороговые множители для safe/medium/risky
LEVEL_PROBS = {
    "safe": 0.90,
    "medium": 0.70,
    "risky": 0.50
}

# Предопределённые пороги коэффициентов, которые мы проверяем
CHECK_COEFS = [1.2, 1.5, 2, 3, 4, 5, 8, 10, 25]

# -----------------------
# Утилиты
# -----------------------
def recalc_indexes():
    """Пересчитать вспомогательные структуры на основе HISTORY (должно вызываться под lock)."""
    global TOTAL_GAMES, CRASHES, COLOR_SEQ, USER_BET_COUNTS, USER_TOTAL_AMOUNT
    TOTAL_GAMES = len(HISTORY)
    CRASHES = [h['crash'] for h in HISTORY if isinstance(h.get('crash'), (int,float))]
    COLOR_SEQ = [h.get('color_bucket') for h in HISTORY if h.get('color_bucket')]
    USER_BET_COUNTS = Counter()
    USER_TOTAL_AMOUNT = defaultdict(float)
    for h in HISTORY:
        bets = h.get('bets') or []
        for b in bets:
            name = b.get('name') or str(b.get('user_id'))
            USER_BET_COUNTS[name] += 1
            try:
                USER_TOTAL_AMOUNT[name] += float(b.get('amount') or 0)
            except:
                pass

def load_json_from_url_or_file(path_or_url):
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        with urllib.request.urlopen(path_or_url, timeout=30) as r:
            raw = r.read()
            # try decode as utf-8
            text = raw.decode('utf-8', errors='ignore')
            return json.loads(text)
    else:
        with open(path_or_url, 'r', encoding='utf-8') as f:
            return json.load(f)

def safediv(a,b):
    try:
        return a/b
    except:
        return 0.0

def prob_crash_ge(threshold):
    """Оценка вероятности (историческая) что crash >= threshold."""
    if not CRASHES: return 0.0
    count = sum(1 for c in CRASHES if c >= threshold)
    return count/len(CRASHES)

def most_common_color_sequences(k=3, window=3):
    """Ищем частые подпоследовательности цветов длины window."""
    seqs = Counter()
    s = [c for c in COLOR_SEQ if c]
    for i in range(len(s)-window+1):
        seqs[tuple(s[i:i+window])] += 1
    return seqs.most_common(k)

def detect_bots_from_players(players):
    """Определим какая доля суммы сейчас — от вероятных ботов (по истории)."""
    if not players: return 0.0, []
    bot_sum = 0.0
    total = 0.0
    bots_found = []
    for p in players:
        name = p.get('name') or str(p.get('user_id'))
        amt = float(p.get('amount') or 0)
        total += amt
        if USER_BET_COUNTS.get(name,0) >= BOT_FREQ_THRESHOLD:
            bot_sum += amt
            bots_found.append(name)
    return safediv(bot_sum, total) if total>0 else 0.0, bots_found

def kelly_fraction(p, r):
    """Классический Kelly: f* = (p*(r)-1)/(r-1). r — множитель (коэффициент)."""
    if r <= 1: return 0.0
    f = (p*(r) - 1.0) / (r - 1.0)
    # уверенно ограничим:
    if f < 0: f = 0.0
    if f > MAX_BANK_FRACTION: f = MAX_BANK_FRACTION
    return f

def recommend_for_bets(players, bank = 100.0):
    """
    Основная логика прогнозирования:
    - считаем долю бот-сумм среди текущих ставок,
    - по историческим данным считаем вероятность crash >= T для CHECK_COEFS,
    - возможно скорректируем эти вероятности в зависимости от bot_fraction (консервативная коррекция),
    - формируем safe/medium/risky: максимальный коэффициент, для которого p >= threshold (safe/medium/risky),
    - считаем "точечный" коэффициент — к примеру коэффициент с наибольшим ожидаемым Kelly-ставкой.
    """
    bot_frac, bots_list = detect_bots_from_players(players)
    # historical probs
    probs = {c: prob_crash_ge(c) for c in CHECK_COEFS}

    # корректировка вероятностей в зависимости от bot_frac.
    # Если много ботов — считаем что риск чуть выше, уменьшаем p на небольшую величину.
    adjust = 0.0
    if bot_frac >= 0.6:
        adjust = -0.09
    elif bot_frac >= 0.3:
        adjust = -0.05
    elif bot_frac >= 0.15:
        adjust = -0.02
    # clip and apply
    for c in probs:
        p = probs[c] + adjust
        if p < 0: p = 0.0
        if p > 1: p = 1.0
        probs[c] = p

    # pick recommended coefs for each level
    recommended = {}
    for level, p_threshold in LEVEL_PROBS.items():
        # найти максимальный коэффициент t where probs[t] >= p_threshold
        opts = [c for c in CHECK_COEFS if probs.get(c,0) >= p_threshold]
        recommended[level] = max(opts) if opts else None

    # Точечный коэффициент: максимально выгодный с точки зрения Kelly (с учётом estimated p)
    best_coef = None
    best_kelly = 0.0
    for c in CHECK_COEFS:
        p = probs[c]
        kf = kelly_fraction(p, c)
        if kf > best_kelly:
            best_kelly = kf
            best_coef = c

    # Рассчитать рекомендованные доли от банка для уровней (консервативно):
    bank_fracs = {}
    for level in ['safe','medium','risky']:
        coef = recommended[level]
        if coef is None:
            bank_fracs[level] = 0.0
            continue
        p = probs[coef]
        f = kelly_fraction(p, coef)
        # дополнительно склоняем в сторону консерватизма: safe *0.5, medium *0.8, risky *1.0
        factor = {'safe': 0.5, 'medium': 0.8, 'risky': 1.0}[level]
        bank_fracs[level] = min(MAX_BANK_FRACTION, f * factor)
    # Точечный % от банка
    point_bank_frac = min(MAX_BANK_FRACTION, best_kelly)

    explanation = {
        "total_history_games": TOTAL_GAMES,
        "bot_fraction_in_current_bets": round(bot_frac, 3),
        "bots_detected_example": bots_list[:10],
        "note_on_adjust": f"Probabilities adjusted by {adjust:+.3f} due to bot_fraction"
    }

    return {
        "probs": {str(c): round(probs[c],3) for c in sorted(probs.keys())},
        "recommended_coeffs": recommended,
        "bank_fracs": {k: round(v,4) for k,v in bank_fracs.items()},
        "point_coef": best_coef,
        "point_bank_frac": round(point_bank_frac,4),
        "explanation": explanation
    }

# -----------------------
# Endpoints
# -----------------------
@app.route('/load', methods=['POST'])
def load_endpoint():
    """
    POST /load
    body: { "url": "<http(s) or local path>" }
    Загружает исторический JSON (массив объектов) и инициализирует память.
    """
    body = request.get_json(force=True)
    if not body or 'url' not in body:
        return jsonify({"ok": False, "error": "url required in JSON body"}), 400
    url = body['url']
    try:
        arr = load_json_from_url_or_file(url)
    except Exception as e:
        return jsonify({"ok": False, "error": f"load failed: {str(e)}"}), 500

    if not isinstance(arr, list):
        return jsonify({"ok": False, "error": "expected JSON array of game objects"}), 400

    # Normalize and push into HISTORY
    with TOTAL_LOCK:
        HISTORY.clear()
        for it in arr:
            # minimal normalization: we expect keys: id/game_id, crash, color_bucket, bets[], deposit_sum, num_players
            gid = it.get('game_id') or it.get('id') or it.get('gameId')
            crash = None
            try:
                crash = float(it.get('crash')) if it.get('crash') is not None else None
            except:
                crash = None
            rec = {
                "game_id": int(gid) if gid is not None else None,
                "crash": crash,
                "color_bucket": it.get('color_bucket') or it.get('color') or it.get('bucket'),
                "bets": it.get('bets') or it.get('bets_json') or [],
                "deposit_sum": it.get('deposit_sum') or it.get('deposit') or None,
                "num_players": it.get('num_players') or it.get('players') or None,
                "raw": None
            }
            HISTORY.append(rec)
        recalc_indexes()
    return jsonify({"ok": True, "loaded": len(HISTORY)})

@app.route('/status', methods=['GET'])
def status():
    with TOTAL_LOCK:
        top_bots = USER_BET_COUNTS.most_common(10)
        seqs = most_common_color_sequences(5, window=3)
        return jsonify({
            "total_games": TOTAL_GAMES,
            "last_5_crashes": CRASHES[-5:],
            "top_bots_example": top_bots,
            "top_color_triplets": [[list(k), v] for k,v in seqs]
        })

@app.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict
    body:
    {
      "players": [ { "user_id":..., "name":"...", "amount":..., "auto":... }, ... ],
      "game_id": 6233375,
      "bank": 100.0   # optional, для расчёта доли от банка
    }
    """
    body = request.get_json(force=True)
    if not body:
        return jsonify({"ok": False, "error": "JSON body required"}), 400
    players = body.get('players') or []
    bank = float(body.get('bank', 100.0) or 100.0)
    with TOTAL_LOCK:
        res = recommend_for_bets(players, bank)
    # Вернём человечески читабельный ответ на русском + структуры
    human = {
        "Рекомендация (safe/medium/risky)": {
            "safe": {
                "коэффициент": res['recommended_coeffs']['safe'],
                "риск_в%_банка": round(100*res['bank_fracs']['safe'], 3),
                "вероятность_успеха": None if res['recommended_coeffs']['safe'] is None else res['probs'].get(str(res['recommended_coeffs']['safe']))
            },
            "medium": {
                "коэффициент": res['recommended_coeffs']['medium'],
                "риск_в%_банка": round(100*res['bank_fracs']['medium'], 3),
                "вероятность_успеха": None if res['recommended_coeffs']['medium'] is None else res['probs'].get(str(res['recommended_coeffs']['medium']))
            },
            "risky": {
                "коэффициент": res['recommended_coeffs']['risky'],
                "риск_в%_банка": round(100*res['bank_fracs']['risky'], 3),
                "вероятность_успеха": None if res['recommended_coeffs']['risky'] is None else res['probs'].get(str(res['recommended_coeffs']['risky']))
            }
        },
        "Точечный_коэффициент": {
            "coef": res['point_coef'],
            "suggested_%_of_bank": round(100*res['point_bank_frac'], 3),
            "note": "точечный коэффициент выбрался как максимизирующий Kelly"
        },
        "объяснение": res['explanation'],
        "raw_probs": res['probs']
    }
    return jsonify({"ok": True, "result": human})

@app.route('/result', methods=['POST'])
def result():
    """
    POST /result
    body: { "game_id": 6233375, "crash": 1.52, "color_bucket": "blue", "bets": [ ... optional ... ] }
    Добавляет результат в локальную историю и пересчитывает индексы.
    """
    body = request.get_json(force=True)
    if not body or 'game_id' not in body or 'crash' not in body:
        return jsonify({"ok": False, "error": "game_id and crash required"}), 400
    game_id = int(body['game_id'])
    crash = float(body['crash'])
    cb = body.get('color_bucket')
    bets = body.get('bets') or []
    with TOTAL_LOCK:
        HISTORY.append({
            "game_id": game_id,
            "crash": crash,
            "color_bucket": cb,
            "bets": bets
        })
        recalc_indexes()
    return jsonify({"ok": True, "total_games": TOTAL_GAMES})

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    print("AI assistant starting... open /status to check")
    # Optionally, if environment variable AUTO_LOAD_URL present, try loading
    import os
    auto = os.environ.get('AUTO_LOAD_URL')
    if auto:
        try:
            print("Auto-loading", auto)
            arr = load_json_from_url_or_file(auto)
            if isinstance(arr, list):
                with TOTAL_LOCK:
                    HISTORY.clear()
                    for it in arr:
                        gid = it.get('game_id') or it.get('id') or it.get('gameId')
                        try:
                            crash = float(it.get('crash')) if it.get('crash') is not None else None
                        except:
                            crash = None
                        HISTORY.append({
                            "game_id": int(gid) if gid is not None else None,
                            "crash": crash,
                            "color_bucket": it.get('color_bucket') or it.get('color'),
                            "bets": it.get('bets') or []
                        })
                    recalc_indexes()
                    print("Loaded", len(HISTORY), "games")
        except Exception as e:
            print("Auto-load failed:", e)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 3001)))