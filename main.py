# main.py
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import os
import logging
from model import AIAssistant

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai_assistant")

# Настройки через переменные среды
DATA_FILE = os.getenv("GAMES_FILE", "")  # путь к JSON с историей (укажешь позже)
PERSIST_ON_UPDATE = os.getenv("PERSIST_ON_UPDATE", "false").lower() == "true"
PORT = int(os.getenv("PORT", 8000))

app = FastAPI(title="Crash AI Assistant")

assistant = AIAssistant()
if DATA_FILE:
    try:
        assistant.load_history(DATA_FILE)
        logger.info(f"История загружена из {DATA_FILE} (игр: {assistant.history_count()})")
    except Exception as e:
        logger.warning(f"Не удалось загрузить {DATA_FILE}: {e}")

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
    """
    Принимает snapshot ставок, не отвечает телу парсера данными — логирует прогноз.
    Возвращает 204 No Content (парсеру не нужно тело).
    """
    try:
        assistant.predict_and_log(payload.dict())
        return
    except Exception as e:
        logger.exception("Ошибка в /predict")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def feedback(payload: FeedbackPayload):
    """
    Приходит после окончания игры с реальным crash.
    Добавляем игру в память (history) и обновляем статистики.
    Возвращаем простой JSON {status: ok}.
    """
    try:
        assistant.process_feedback(payload.game_id, payload.crash, persist=PERSIST_ON_UPDATE)
        return {"status": "ok"}
    except Exception as e:
        logger.exception("Ошибка в /feedback")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
