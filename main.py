from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
import threading
import time
import requests
import json
from mega import Mega
from model import AIAssistant

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai_assistant.main")

PORT = int(os.getenv("PORT", 8000))
SELF_URL = os.getenv("SELF_URL")

MEGA_EMAIL = os.getenv("MEGA_EMAIL")
MEGA_PASSWORD = os.getenv("MEGA_PASSWORD")
MEGA_FOLDER = "crashAi_backup"

assistant = AIAssistant()

mega = Mega()
m = None
FOLDER = None


# ====================== Mega ======================
def mega_connect():
    global m, FOLDER
    if m is None:
        m = mega.login(MEGA_EMAIL, MEGA_PASSWORD)
        FOLDER = m.find(MEGA_FOLDER)
        if not FOLDER:
            FOLDER = m.create_folder(MEGA_FOLDER)
        logger.info("Mega: connected")


def mega_find_file(name: str):
    mega_connect()
    files = m.find(name)
    return files[0] if files else None


# ====================== BACKUP ======================
BACKUP_NAME = "assistant_backup.json"
OLD_BACKUP_NAME = "assistant_backup_old.json"


def save_backup():
    try:
        mega_connect()

        # Clean trash before backup
        trash = m.get_files_in_node(m.get_trash_folder())
        for t in trash.values():
            m.delete(t)

        # Rename previous backup
        file = mega_find_file(BACKUP_NAME)
        if file:
            m.rename(file, OLD_BACKUP_NAME)

        # Save new backup locally
        with open(BACKUP_NAME, "w") as f:
            json.dump(assistant.export_state(), f)

        # Upload
        logger.info("Uploading new backup to Mega...")
        m.upload(BACKUP_NAME, FOLDER)

        # Move old backup to trash
        file_old = mega_find_file(OLD_BACKUP_NAME)
        if file_old:
            m.move(file_old, m.get_trash_folder())
            logger.info("Old backup moved to Trash")

        logger.info("Backup updated successfully")

    except Exception as e:
        logger.error(f"Backup failed: {e}")


def backup_loop():
    while True:
        save_backup()
        time.sleep(3600)


threading.Thread(target=backup_loop, daemon=True).start()


def restore_backup():
    try:
        mega_connect()
        file = mega_find_file(BACKUP_NAME)
        if not file:
            logger.warning("No backups found")
            return
        m.download(file, BACKUP_NAME)
        with open(BACKUP_NAME) as f:
            assistant.load_state(json.load(f))
        logger.info("Assistant state restored")
    except Exception as e:
        logger.error(f"Restore error: {e}")


restore_backup()


# ====================== History Load ======================
def load_big_history(filename="crash_23k.json"):
    try:
        file = mega_find_file(filename)
        if not file:
            return

        logger.info("Downloading history from Mega...")
        m.download(file, filename)

        with open(filename) as f:
            data = json.load(f)

        block = 5000
        for i in range(0, len(data), block):
            assistant.load_history_from_list(data[i:i+block])
            logger.info(f"Loaded block {i}-{min(i+block, len(data))}")

        logger.info("Full history loaded successfully")

    except Exception as e:
        logger.error(f"History load error: {e}")


threading.Thread(target=load_big_history, daemon=True).start()


# ====================== KEEP ALIVE ======================
def keep_alive():
    if not SELF_URL:
        return
    while True:
        try:
            requests.get(f"{SELF_URL}/healthz", timeout=5)
        except:
            pass
        time.sleep(240)


threading.Thread(target=keep_alive, daemon=True).start()


# ====================== API ======================
app = FastAPI(title="Crash AI Assistant")


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


@app.get("/logs")
async def logs(limit: int = 20):
    return {"logs": assistant.get_pred_log(limit)}


@app.get("/status")
async def status():
    return assistant.get_status()


@app.post("/train/trigger")
async def manual_train():
    try:
        assistant._online_train()
        return {"status": "trained", "metrics": assistant.last_metrics}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}