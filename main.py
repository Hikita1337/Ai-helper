from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
import asyncio
import aiohttp
import json
from dotenv import load_dotenv
from ably import AblyRest
from model import AIAssistant
from mega.client import Mega

load_dotenv()
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

async def mega_connect():
    global mega_client, mega_logged_in, FOLDER_ID
    if mega_client is None:
        mega_client = Mega()
        logger.info("Logging in to Mega...")
        mega_logged_in = await mega_client.login(MEGA_EMAIL, MEGA_PASSWORD)

        nodes = await mega_logged_in.get_files()
        # ищем папку для бэкапов
        for node_id, node in nodes.items():
            if node.get("name") == MEGA_FOLDER and node.get("type") == 1:
                FOLDER_ID = node_id
                break

        if not FOLDER_ID:
            folder = await mega_logged_in.create_folder(MEGA_FOLDER)
            FOLDER_ID = folder["id"]

        logger.info(f"Mega connected, folder ID: {FOLDER_ID}")


async def mega_find_file(name: str):
    await mega_connect()
    nodes = await mega_logged_in.get_files()
    for node_id, node in nodes.items():
        if node.get("name") == name:
            return node["id"]
    return None

async def mega_upload_file(local_path: str):
    await mega_connect()
    await mega_logged_in.upload(local_path, dest=FOLDER_ID)
    logger.info(f"Uploaded {local_path} to Mega")

async def mega_download_file(remote_name: str, local_path: str):
    await mega_connect()
    file_id = await mega_find_file(remote_name)
    if file_id:
        await mega_logged_in.download(file_id, local_path)
        logger.info(f"Downloaded {remote_name} from Mega")
        return True
    return False

async def mega_delete_file(name: str):
    file_id = await mega_find_file(name)
    if file_id:
        await mega_logged_in.delete(file_id)
        logger.info(f"Deleted {name} from Mega")

async def mega_rename_file(old_name: str, new_name: str):
    file_id = await mega_find_file(old_name)
    if file_id:
        await mega_logged_in.rename(file_id, new_name)
        logger.info(f"Renamed {old_name} -> {new_name}")

# ====================== Backup ======================
BACKUP_NAME = "assistant_backup.json"
OLD_BACKUP_NAME = "assistant_backup_old.json"

async def save_backup():
    try:
        await mega_connect()
        old_file_id = await mega_find_file(BACKUP_NAME)
        if old_file_id:
            await mega_rename_file(BACKUP_NAME, OLD_BACKUP_NAME)

        with open(BACKUP_NAME, "w") as f:
            json.dump(assistant.export_state(), f)

        await mega_upload_file(BACKUP_NAME)

        old_file_id = await mega_find_file(OLD_BACKUP_NAME)
        if old_file_id:
            await mega_logged_in.delete(old_file_id)
            logger.info("Old backup removed after upload")

        logger.info("Backup updated successfully")
    except Exception as e:
        logger.error(f"Backup failed: {e}")

async def restore_backup():
    try:
        downloaded = await mega_download_file(BACKUP_NAME, BACKUP_NAME)
        if downloaded:
            with open(BACKUP_NAME) as f:
                state = json.load(f)
            assistant.load_state(state)
            logger.info("Assistant state restored from backup")
        else:
            logger.warning("No backup found to restore")
    except Exception as e:
        logger.error(f"Restore backup failed: {e}")

async def save_backup_loop():
    while True:
        await save_backup()
        await asyncio.sleep(3600)  # строго раз в час

# ====================== History Load ======================
CRASH_HISTORY_FILES = ["crash_23k.json"]

async def load_history_files(files=CRASH_HISTORY_FILES):
    await mega_connect()
    for filename in files:
        logger.info(f"Processing history file: {filename}")
        file_downloaded = await mega_download_file(filename, filename)
        if not file_downloaded:
            logger.warning(f"File {filename} not found in Mega")
            continue

        with open(filename) as f:
            data = json.load(f)

        block = 7000
        for i in range(0, len(data), block):
            assistant.load_history_from_list(data[i:i+block])
            logger.info(f"Loaded block {i}-{min(i+block, len(data))} from {filename}")

        os.remove(filename)
        logger.info(f"File {filename} removed from local storage")

# ====================== Keep Alive ======================
async def keep_alive_loop():
    if not SELF_URL:
        return
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                await session.get(f"{SELF_URL}/healthz", timeout=5)
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
    await restore_backup()
    await load_history_files()
    asyncio.create_task(save_backup_loop())
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