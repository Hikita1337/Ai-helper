from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
import asyncio
import requests
import json
from ably import AblyRest
from model import AIAssistant
from mega import Mega  # классическая библиотека mega.py

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai_assistant.main")

PORT = int(os.getenv("PORT", 8000))
SELF_URL = os.getenv("SELF_URL")
ABLY_API_KEY = os.getenv("ABLY_API_KEY")

MEGA_EMAIL = os.getenv("MEGA_EMAIL")
MEGA_PASSWORD = os.getenv("MEGA_PASSWORD")
MEGA_FOLDER = "crashAi_backup"

assistant = AIAssistant()
ably_client = AblyRest(ABLY_API_KEY)
ably_channel = ably_client.channels.get("crash_ai_hud")

# ====================== Mega ======================
mega_client: Mega | None = None
mega_logged_in = None
FOLDER_ID: str | None = None

def mega_connect():
    global mega_client, mega_logged_in, FOLDER_ID
    if mega_client is None:
        mega_client = Mega()
        mega_logged_in = mega_client.login(MEGA_EMAIL, MEGA_PASSWORD)
        folders = mega_logged_in.get_files()
        for k, v in folders.items():
            if v.get('a') and v['a']['n'] == MEGA_FOLDER and v['t'] == 1:  # папка
                FOLDER_ID = k
                break
        if not FOLDER_ID:
            folder = mega_logged_in.create_folder(MEGA_FOLDER)
            FOLDER_ID = folder['f']
        logger.info("Mega: connected")

def mega_find_file(name: str):
    mega_connect()
    files = mega_logged_in.get_files()
    for k, v in files.items():
        if v.get('a') and v['a']['n'] == name:
            return k
    return None

def mega_upload_file(local_path: str):
    mega_connect()
    mega_logged_in.upload(local_path, dest=FOLDER_ID)
    logger.info(f"Uploaded {local_path} to Mega")

def mega_download_file(remote_name: str, local_path: str):
    mega_connect()
    file_id = mega_find_file(remote_name)
    if file_id:
        mega_logged_in.download(file_id, local_path)
        logger.info(f"Downloaded {remote_name} from Mega")

def mega_delete_file(name: str):
    file_id = mega_find_file(name)
    if file_id:
        mega_logged_in.delete(file_id)
        logger.info(f"Deleted {name} from Mega")

def mega_rename_file(old_name: str, new_name: str):
    file_id = mega_find_file(old_name)
    if file_id:
        mega_logged_in.rename(file_id, new_name)
        logger.info(f"Renamed {old_name} -> {new_name}")

# ====================== BACKUP ======================
BACKUP_NAME = "assistant_backup.json"
OLD_BACKUP_NAME = "assistant_backup_old.json"

def save_backup():
    try:
        mega_connect()

        # Переименовываем предыдущий бэкап
        old_file_id = mega_find_file(BACKUP_NAME)
        if old_file_id:
            mega_rename_file(BACKUP_NAME, OLD_BACKUP_NAME)

        # Сохраняем новый бэкап локально
        with open(BACKUP_NAME, "w") as f:
            json.dump(assistant.export_state(), f)

        # Загружаем на Mega
        mega_upload_file(BACKUP_NAME)

        # Перемещаем старый бэкап в корзину
        old_file_id = mega_find_file(OLD_BACKUP_NAME)
        if old_file_id:
            mega_logged_in.trash(old_file_id)
            logger.info("Old backup moved to Trash")

        logger.info("Backup updated successfully")
    except Exception as e:
        logger.error(f"Backup failed: {e}")

def save_backup_loop():
    while True:
        save_backup()
        asyncio.sleep(3600)

def restore_backup():
    try:
        file_id = mega_find_file(BACKUP_NAME)
        if not file_id:
            logger.warning("No backups found")
            return
        mega_download_file(BACKUP_NAME, BACKUP_NAME)
        with open(BACKUP_NAME) as f:
            assistant.load_state(json.load(f))
        logger.info("Assistant state restored")
    except Exception as e:
        logger.error(f"Restore error: {e}")

# ====================== History Load ======================
def load_big_history(filename="crash_23k.json"):
    try:
        file_id = mega_find_file(filename)
        if not file_id:
            return

        logger.info("Downloading history from Mega...")
        mega_download_file(filename, filename)

        with open(filename) as f:
            data = json.load(f)

        block = 5000
        for i in range(0, len(data), block):
            assistant.load_history_from_list(data[i:i+block])
            logger.info(f"Loaded block {i}-{min(i+block, len(data))}")

        logger.info("Full history loaded successfully")
    except Exception as e:
        logger.error(f"History load error: {e}")

# ====================== KEEP ALIVE ======================
async def keep_alive_loop():
    if not SELF_URL:
        return
    while True:
        try:
            requests.get(f"{SELF_URL}/healthz", timeout=5)
        except:
            pass
        await asyncio.sleep(240)

# ====================== API ======================
app = FastAPI(title="Crash AI Assistant")

class BetsPayload(BaseModel):
    game_id: int
    bets: list

class FeedbackPayload(BaseModel):
    game_id: int
    crash: float
    bets: list | None = None

@app.on_event("startup")
async def startup_event():
    restore_backup()
    # Асинхронные циклы
    asyncio.create_task(asyncio.to_thread(save_backup_loop))
    asyncio.create_task(asyncio.to_thread(load_big_history))
    asyncio.create_task(keep_alive_loop())

@app.post("/predict")
async def predict(payload: BetsPayload):
    try:
        assistant.predict_and_log(payload.model_dump())
        last_pred = assistant.pred_log[-1] if assistant.pred_log else None
        if last_pred:
            hud_data = {
                "game_id": last_pred["game_id"],
                "safe": last_pred["safe"],
                "med": last_pred["med"],
                "risk": last_pred["risk"],
                "recommended_pct": last_pred["recommended_pct"]
            }
            ably_channel.publish("prediction", hud_data)
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