# main.py
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import os
import logging
import threading
import time
import requests
import gdown  # добавляем gdown для скачки с Google Drive
from model import AIAssistant

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai_assistant")

# Настройки через переменные среды
GAMES_FILE_URL = os.getenv("GAMES_FILE", "")  # ссылка на JSON на Google Drive
PERSIST_ON_UPDATE = os.getenv("PERSIST_ON_UPDATE", "false").lower() == "true"
PORT = int(os.getenv("PORT", 8000))
SELF_URL = os.getenv("SELF_URL")

app = FastAPI(title="Crash AI Assistant")
assistant = AIAssistant()

# ===== Загрузка истории =====
if GAMES_FILE_URL:
    try:
        local_file = "games.json"
        logger.info(f"Скачиваем историю с Google Drive: {GAMES_FILE_URL}")
        gdown.download(GAMES_FILE_URL, local_file, quiet=False)
        assistant.load_history(local_file)
        logger.info(f"История загружена из {local_file} (игр: {assistant.history_count()})")
    except Exception as e:
        logger.warning(f"Не удалось загрузить историю: {e}")

# ===== Keep-alive поток =====
def keep_alive():
    if not SELF_URL:
        logger.warning("SELF_URL не задан, keep-alive не будет работать")
        return
    import numpy as np  # np нужен для генерации случайной задержки
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