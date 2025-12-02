# main.py
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import os
import logging
import threading
import time
import requests
import numpy as np  # добавляем, чтобы keep_alive работал
from model import AIAssistant

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai_assistant")

# Настройки через переменные среды
DATA_FILE = os.getenv("GAMES_FILE", "")  # путь к JSON с историей
PERSIST_ON_UPDATE = os.getenv("PERSIST_ON_UPDATE", "false").lower() == "true"
PORT = int(os.getenv("PORT", 8000))
SELF_URL = os.getenv("SELF_URL")  # URL своего Render-сервера

app = FastAPI(title="Crash AI Assistant")
assistant = AIAssistant()

if DATA_FILE:
    try:
        assistant.load_history(DATA_FILE)
        logger.info(f"История загружена из {DATA_FILE} (игр: {assistant.history_count()})")
    except Exception as e:
        logger.warning(f"Не удалось загрузить {DATA_FILE}: {e}")

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
        # случайная пауза 4–6 минут
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

# ===== Роуты =====
@app.post("/predict", status_code=204)
async def predict(payload: BetsPayload, request: Request):
    try:
        assistant.predict_and_log(payload.dict())
    except Exception as e:
        logger.exception("Ошибка в /predict")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def feedback(payload: FeedbackPayload):
    try:
        assistant.process_feedback(payload.game_id, payload.crash, persist=PERSIST_ON_UPDATE)
        return {"status": "ok"}
    except Exception as e:
        logger.exception("Ошибка в /feedback")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/")
async def root():
    # логируем количество загруженных игр
    count = assistant.history_count()
    logger.info(f"GET / — всего загружено игр: {count}")
    return {"message": f"AI Assistant is running — загружено игр: {count}"}