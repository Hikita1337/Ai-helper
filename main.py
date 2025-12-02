from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import os
import logging
import threading
import time
import requests
from model import AIAssistant
from mega import Mega
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai_assistant")

PORT = int(os.getenv("PORT", 8000))
SELF_URL = os.getenv("SELF_URL")

app = FastAPI(title="Crash AI Assistant")
assistant = AIAssistant()

# ======== Mega загрузка файла =========
MEGA_EMAIL = os.getenv("MEGA_EMAIL")
MEGA_PASSWORD = os.getenv("MEGA_PASSWORD")
MEGA_FOLDER = "crashAi_backup"
MEGA_FILE = "crash_23k.json"

def load_history_from_mega():
    try:
        mega = Mega()
        m = mega.login(MEGA_EMAIL, MEGA_PASSWORD)
        logger.info("Вход в Mega успешен")
        file = m.find(MEGA_FILE)
        if not file:
            logger.warning(f"Файл {MEGA_FILE} не найден в Mega")
            return
        local_path = MEGA_FILE
        logger.info(f"Скачиваем {MEGA_FILE} с Mega...")
        m.download(file, dest_filename=local_path)
        logger.info("Файл скачан, начинаем обработку")
        # Загружаем и обрабатываем игры блоками по 5000 для экономии RAM
        with open(local_path, "r") as f:
            data = json.load(f)
            block_size = 5000
            for i in range(0, len(data), block_size):
                block = data[i:i+block_size]
                assistant.load_history_from_list(block)
                logger.info(f"Загружен блок {i}-{i+len(block)} игр")
        logger.info("История игр полностью загружена из Mega")
    except Exception as e:
        logger.exception(f"Ошибка загрузки истории из Mega: {e}")

# Запускаем в отдельном потоке, чтобы не блокировать API
threading.Thread(target=load_history_from_mega, daemon=True).start()

# ======== KEEP-ALIVE =========
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

# ===== Pydantic Models =====
class BetsPayload(BaseModel):
    game_id: int
    bets: list
    num_players: int | None = None
    deposit_sum: float | None = None
    meta: dict | None = None

class FeedbackPayload(BaseModel):
    game_id: int
    crash: float
    bets: list | None = None
    deposit_sum: float | None = None
    num_players: int | None = None

class LoadGamesPayload(BaseModel):
    url: str

# ===== ENDPOINTS =====
@app.post("/predict", status_code=204)
async def predict(payload: BetsPayload, request: Request):
    try:
        assistant.predict_and_log(payload.model_dump())
    except Exception as e:
        logger.exception("Ошибка в /predict")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def feedback(payload: FeedbackPayload):
    try:
        fast_game = False
        if payload.bets and payload.crash:
            fast_game = True

        assistant.process_feedback(
            game_id=payload.game_id,
            crash=payload.crash,
            bets=payload.bets,
            deposit_sum=payload.deposit_sum,
            num_players=payload.num_players,
            fast_game=fast_game
        )

        if fast_game:
            logger.info(f"Быстрая игра {payload.game_id}, визуальный предикт пропущен")
            return {"status": "ok", "fast_game": True}

        return {"status": "ok"}

    except Exception as e:
        logger.exception("Ошибка в /feedback")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/logs")
async def get_logs(limit: int = 20):
    return {"logs": assistant.get_pred_log(limit)}