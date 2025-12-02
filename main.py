from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import os
import logging
import threading
import time
import requests
import json
import signal
from mega import Mega
from model import AIAssistant

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai_assistant")

PORT = int(os.getenv("PORT", 8000))
SELF_URL = os.getenv("SELF_URL")

MEGA_EMAIL = os.getenv("MEGA_EMAIL")
MEGA_PASSWORD = os.getenv("MEGA_PASSWORD")
MEGA_FOLDER = "crashAi_backup"

assistant = AIAssistant()

app = FastAPI(title="Crash AI Assistant")

# ===== MEGA =====
mega = Mega()
m = None
FOLDER = None


def mega_connect():
    global m, FOLDER
    if m is None:
        m = mega.login(MEGA_EMAIL, MEGA_PASSWORD)
        FOLDER = m.find(MEGA_FOLDER)
        if not FOLDER:
            FOLDER = m.create_folder(MEGA_FOLDER)
        logger.info("Mega: connected")


def mega_find(name):
    mega_connect()
    file = m.find(name)
    if isinstance(file, list):
        return file[0] if file else None
    return file


# ===== Backup filenames =====
STATE_FILE = "assistant_state.json"
HISTORY_FILE = "crash_history.json"
OLD_SUFFIX = "_old"


# ===== State Management =====
def save_state():
    try:
        mega_connect()
        logger.info("Saving assistant state to Mega...")

        # 1. Удаляем корзину
        trash = m.get_files_in_node(m.get_trash_folder())
        for t in trash.values():
            m.delete(t)

        # 2. Переименовываем текущий state -> old
        f = mega_find(STATE_FILE)
        if f:
            m.rename(f, STATE_FILE + OLD_SUFFIX)

        # 3. Генерируем новый state локально
        with open(STATE_FILE, "w") as f:
            json.dump(assistant.export_state(), f)

        # 4. Загружаем новый state
        m.upload(STATE_FILE, FOLDER)

        # 5. Старый переносим в корзину
        f_old = mega_find(STATE_FILE + OLD_SUFFIX)
        if f_old:
            m.move(f_old, m.get_trash_folder())

        logger.info("State saved successfully")

    except Exception as e:
        logger.error(f"State backup failed: {e}")


def load_state():
    try:
        mega_connect()
        f = mega_find(STATE_FILE)
        if not f:
            return False
        m.download(f, STATE_FILE)
        with open(STATE_FILE) as file:
            assistant.load_state(json.load(file))
        logger.info("State restored successfully")
        return True
    except Exception as e:
        logger.error(f"State restore error: {e}")
        return False


# ===== History Storage =====
def save_history():
    try:
        mega_connect()
        logger.info("Saving history to Mega...")

        data = assistant.export_history()

        with open(HISTORY_FILE, "w") as f:
            json.dump(data, f)

        m.upload(HISTORY_FILE, FOLDER, ignore=True)
        logger.info("History saved successfully")

    except Exception as e:
        logger.error(f"History backup failed: {e}")


def load_history():
    try:
        mega_connect()
        file = mega_find(HISTORY_FILE)
        if not file:
            return False
        m.download(file, HISTORY_FILE)

        with open(HISTORY_FILE) as f:
            data = json.load(f)

        for i in range(0, len(data), 5000):
            assistant.load_history_from_list(data[i:i + 5000])
        logger.info("History loaded successfully")
        return True
    except Exception as e:
        logger.error(f"History load failed: {e}")
        return False


# ===== Auto-backup Thread =====
def auto_backup():
    while True:
        save_state()
        time.sleep(900)  # каждые 15 минут


threading.Thread(target=auto_backup, daemon=True).start()


# ===== Graceful Shutdown =====
def shutdown_handler(*args):
    logger.warning("Shutdown detected, saving state...")
    save_state()
    time.sleep(1)
    os._exit(0)


signal.signal(signal.SIGTERM, shutdown_handler)
signal.signal(signal.SIGINT, shutdown_handler)


# ===== Restore On Startup =====
def earliest_recovery():
    if load_state():
        logger.info("Recovered STATE")
    if load_history():
        logger.info("Recovered GAME HISTORY")


earliest_recovery()


# ========== API ==========
class BetsPayload(BaseModel):
    game_id: int
    bets: list


class FeedbackPayload(BaseModel):
    game_id: int
    crash: float
    bets: list | None = None


@app.post("/predict")
async def predict(payload: BetsPayload):
    try:
        assistant.predict_and_log(payload.model_dump())
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/feedback")
async def feedback(payload: FeedbackPayload):
    try:
        assistant.process_feedback(payload.game_id, payload.crash, payload.bets)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/state/save")
async def save_state_api():
    save_state()
    return {"status": "saved"}


@app.get("/state/load")
async def load_state_api():
    load_state()
    return {"status": "loaded"}


@app.get("/history/save")
async def save_history_api():
    save_history()
    return {"status": "saved"}


@app.get("/history/load")
async def load_history_api():
    load_history()
    return {"status": "loaded"}


@app.get("/logs")
async def logs(limit: int = 20):
    return {"logs": assistant.get_pred_log(limit)}


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}