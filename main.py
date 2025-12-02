# main.py
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import os
import logging
from model import AIAssistant, load_json_from_url

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai_assistant")

DATA_FILE = os.getenv("GAMES_FILE", "")  # локальный файл JSON
GAMES_FILE_URL = os.getenv("GAMES_FILE_URL", "")  # ссылка на JSON с Google Drive
PERSIST_ON_UPDATE = os.getenv("PERSIST_ON_UPDATE", "false").lower() == "true"
PORT = int(os.getenv("PORT", 8000))

app = FastAPI(title="Crash AI Assistant")

assistant = AIAssistant()

# Загружаем историю
if DATA_FILE:
    try:
        assistant.load_history(DATA_FILE)
        logger.info(f"История загружена из {DATA_FILE} (игр: {assistant.history_count()})")
    except Exception as e:
        logger.warning(f"Не удалось загрузить {DATA_FILE}: {e}")
elif GAMES_FILE_URL:
    try:
        games_list = load_json_from_url(GAMES_FILE_URL)
        assistant.load_history_from_list(games_list)
        logger.info(f"История загружена с URL {GAMES_FILE_URL} (игр: {assistant.history_count()})")
    except Exception as e:
        logger.warning(f"Не удалось загрузить историю с URL: {e}")

# -------------------- Pydantic модели --------------------
class BetsPayload(BaseModel):
    game_id: int
    num_players: int | None = None
    deposit_sum: float | None = None
    bets: list
    meta: dict | None = {}

class FeedbackPayload(BaseModel):
    game_id: int
    crash: float

# -------------------- Endpoints --------------------
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