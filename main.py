from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
import asyncio
import requests
import json
from ably import AblyRest
from model import AIAssistant
from mega import Mega  # для версии 2.x mega-lite

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
m: MegaLite | None = None
FOLDER: str | None = None

async def mega_connect():
    global m, FOLDER
    if m is None:
        m = MegaLite()
        await m.login(MEGA_EMAIL, MEGA_PASSWORD)
        FOLDER = await m.find(MEGA_FOLDER)
        if not FOLDER:
            FOLDER = await m.create_folder(MEGA_FOLDER)
        logger.info("Mega: connected")

async def mega_find_file(name: str):
    await mega_connect()
    files = await m.find(name)
    return files[0] if files else None

async def mega_upload_file(local_path: str):
    await mega_connect()
    await m.upload(local_path, FOLDER)
    logger.info(f"Uploaded {local_path} to Mega")

async def mega_download_file(remote_name: str, local_path: str):
    await mega_connect()
    file = await mega_find_file(remote_name)
    if file:
        await m.download(file, local_path)
        logger.info(f"Downloaded {remote_name} from Mega")

async def mega_delete_file(name: str):
    file = await mega_find_file(name)
    if file:
        await m.delete(file)
        logger.info(f"Deleted {name} from Mega")

# ====================== BACKUP ======================
BACKUP_NAME = "assistant_backup.json"
OLD_BACKUP_NAME = "assistant_backup_old.json"

async def save_backup():
    try:
        await mega_connect()

        # Очистка корзины
        trash = await m.get_files_in_node(await m.get_trash_folder())
        for t in trash.values():
            await m.delete(t)

        # Переименовываем предыдущий бэкап
        file = await mega_find_file(BACKUP_NAME)
        if file:
            await m.rename(file, OLD_BACKUP_NAME)

        # Сохраняем новый бэкап локально
        with open(BACKUP_NAME, "w") as f:
            json.dump(assistant.export_state(), f)

        # Загружаем на Mega
        await mega_upload_file(BACKUP_NAME)

        # Перемещаем старый бэкап в корзину
        file_old = await mega_find_file(OLD_BACKUP_NAME)
        if file_old:
            await m.move(file_old, await m.get_trash_folder())
            logger.info("Old backup moved to Trash")

        logger.info("Backup updated successfully")
    except Exception as e:
        logger.error(f"Backup failed: {e}")

async def save_backup_loop():
    while True:
        await save_backup()
        await asyncio.sleep(3600)

async def restore_backup():
    try:
        file = await mega_find_file(BACKUP_NAME)
        if not file:
            logger.warning("No backups found")
            return
        await mega_download_file(BACKUP_NAME, BACKUP_NAME)
        with open(BACKUP_NAME) as f:
            assistant.load_state(json.load(f))
        logger.info("Assistant state restored")
    except Exception as e:
        logger.error(f"Restore error: {e}")

# ====================== History Load ======================
async def load_big_history(filename="crash_23k.json"):
    try:
        file = await mega_find_file(filename)
        if not file:
            return

        logger.info("Downloading history from Mega...")
        await mega_download_file(filename, filename)

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
    asyncio.create_task(save_backup_loop())
    asyncio.create_task(load_big_history())
    asyncio.create_task(keep_alive_loop())
    await restore_backup()

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