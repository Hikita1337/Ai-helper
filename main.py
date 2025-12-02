# main.py
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import os
import logging
import requests
from model import AIAssistant

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai_assistant")

# Настройки через переменные среды
DATA_FILE = os.getenv("GAMES_FILE", "")  # локальный путь, если есть
DATA_URL = os.getenv("GAMES_FILE_URL", "")  # прямая ссылка на Google Drive
PERSIST_ON_UPDATE = os.getenv("PERSIST_ON_UPDATE", "false").lower() == "true"
PORT = int(os.getenv("PORT", 8000))

app = FastAPI(title="Crash AI Assistant")
assistant = AIAssistant()

# Загрузка истории
try:
    if DATA_FILE and os.path.exists(DATA_FILE):
        assistant.load_history(DATA_FILE)
        logger.info(f"История загружена из {DATA_FILE} (игр: {assistant.history_count()})")
    elif DATA_URL:
        logger.info(f"Скачиваем историю с {DATA_URL} ...")
        r = requests.get(DATA_URL)
        r.raise_for_status()
        data = r.json()
        assistant.load_history_from_list(data)
        logger.info(f"История загружена с URL (игр: {assistant.history_count()})")
    else:
        logger.warning("Нет источника данных для истории. Будет пустая база.")
except Exception as e:
    logger.warning(f"Не удалось загрузить историю: {e}")

class BetsPayload(BaseModel):
    game_id: int
    num_players: int | None = None
    deposit_sum: float | None = None
    bets: list  # список объектов {user_id, name, amount, auto}
    meta: dict | None = {}  # доп. поля

class FeedbackPayload(BaseModel):
    game_id: int
    crash: float

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
        assistant.process_feedback(payload.game_id, payload.crash, persist=PERSIST_ON_UPDATE)
        return {"status": "ok"}
    except Exception as e:
        logger.exception("Ошибка в /feedback")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}