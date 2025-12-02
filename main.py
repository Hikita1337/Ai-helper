# main.py
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import os
import logging
import threading
import time
import requests
import numpy as np
import json
from model import AIAssistant

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai_assistant")

PORT = int(os.getenv("PORT", 8000))
SELF_URL = os.getenv("SELF_URL")  # для keep-alive
app = FastAPI(title="Crash AI Assistant")
assistant = AIAssistant()

# ===== Keep-alive поток =====
def keep_alive():
    if not SELF_URL:
        logger.warning("SELF_URL не задан, keep-alive не будет работать")
        return
    while True:
        try:
            resp = requests.get(f"{SELF_URL}/healthz", timeout=5)
            logger.info(f"Keep-alive ping OK: {resp.status_code}")
        except Exception as e:
            logger.warning(f"Keep-alive error: {e}")
        time.sleep(240 + 120 * np.random.rand())

threading.Thread(target=keep_alive, daemon=True).start()

# ===== Модели для API =====
class BetsPayload(BaseModel):
    game_id: int
    num_players: int | None = None
    deposit_sum: float | None = None
    bets: list
    meta: dict | None = {}

class FeedbackPayload(BaseModel):
    game_id: int
    crash: float

class LoadGamesPayload(BaseModel):
    url: str

# ===== Эндпоинты =====
@app.post("/predict", status_code=204)
async def predict(payload: BetsPayload, request: Request):
    try:
        assistant.predict_and_log(payload.dict())
        return
    except Exception as e:
        logger.exception("Ошибка в /predict")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def feedback(payload: FeedbackPayload):
    try:
        assistant.process_feedback(payload.game_id, payload.crash)
        return {"status": "ok"}
    except Exception as e:
        logger.exception("Ошибка в /feedback")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

# ===== Новый эндпоинт для загрузки игр через ссылку =====
@app.post("/load_games")
async def load_games(payload: LoadGamesPayload):
    try:
        resp = requests.get(payload.url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # Приведение bets к списку словарей
        for game in data:
            if "bets" in game and isinstance(game["bets"], str):
                try:
                    game["bets"] = json.loads(game["bets"])
                except Exception as e:
                    logger.warning(f"Не удалось распарсить bets для game_id {game.get('game_id')}: {e}")
                    game["bets"] = []

        assistant.load_history_from_list(data)
        logger.info(f"Игры загружены! Всего в истории: {assistant.history_count()}")
        return {"status": "ok", "games_loaded": assistant.history_count()}

    except Exception as e:
        logger.exception("Ошибка при загрузке игр")
        raise HTTPException(status_code=500, detail=str(e))