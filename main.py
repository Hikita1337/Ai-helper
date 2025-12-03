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
import yadisk  # pip install yadisk

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai_assistant.main")

PORT = int(os.getenv("PORT", 8000))
SELF_URL = os.getenv("SELF_URL")
ABLY_API_KEY = os.getenv("ABLY_API_KEY")
YANDEX_OAUTH_TOKEN = os.getenv("YANDEX_OAUTH_TOKEN")

assistant = AIAssistant()
ably_client = AblyRest(ABLY_API_KEY)
ably_channel = ably_client.channels.get("crash_ai_hud")

# ====================== Yandex Disk ======================
yadisk_client: yadisk.YaDisk | None = None
yandex_queue: asyncio.Queue = asyncio.Queue()
yandex_worker_task: asyncio.Task | None = None

async def run_yandex_task(func, *args, **kwargs):
    fut = asyncio.get_event_loop().create_future()
    await yandex_queue.put((func, args, kwargs, fut))
    return await fut

async def yandex_worker():
    logger.info("Yandex-Disk worker started")
    while True:
        func, args, kwargs, fut = await yandex_queue.get()
        try:
            result = func(*args, **kwargs)
            fut.set_result(result)
        except Exception as e:
            if not fut.done():
                fut.set_exception(e)
        finally:
            yandex_queue.task_done()

def ensure_yadisk():
    global yadisk_client
    if yadisk_client is None:
        if not YANDEX_OAUTH_TOKEN:
            raise RuntimeError("YANDEX_OAUTH_TOKEN not set")
        yadisk_client = yadisk.YaDisk(token=YANDEX_OAUTH_TOKEN)
        if not yadisk_client.check_token():
            raise RuntimeError("Invalid Yandex OAuth token")

# --- High-level wrappers ---
async def yandex_upload(local_path: str, remote_path: str):
    ensure_yadisk()
    return await run_yandex_task(yadisk_client.upload, local_path, remote_path, overwrite=True)

async def yandex_download(remote_path: str, local_path: str):
    ensure_yadisk()
    return await run_yandex_task(yadisk_client.download, remote_path, local_path)

async def yandex_rm(remote_path: str):
    ensure_yadisk()
    return await run_yandex_task(yadisk_client.remove, remote_path, permanently=True)

async def yandex_mv(src: str, dst: str):
    ensure_yadisk()
    return await run_yandex_task(yadisk_client.move, src, dst, overwrite=True)

async def yandex_ls():
    ensure_yadisk()
    return await run_yandex_task(lambda: list(yadisk_client.listdir("/")))

async def yandex_find(filename: str):
    try:
        items = await yandex_ls()
        for item in items:
            if item.name == filename:
                return "/" + filename
        return None
    except Exception as e:
        logger.error("Yandex find error: %s", e)
        return None

# ====================== Backup logic ======================
BACKUP_NAME = "assistant_backup.json"
OLD_BACKUP_NAME = "assistant_backup_old.json"

async def save_backup():
    try:
        logger.info("Saving backup...")

        # удаляем старый старый бэкап
        old_old_path = await yandex_find(OLD_BACKUP_NAME)
        if old_old_path:
            try:
                await yandex_rm(old_old_path)
                logger.info("Deleted old-old backup %s", old_old_path)
            except Exception as e:
                logger.warning("Failed to delete old-old backup: %s", e)

        # переименовываем текущий бэкап в OLD
        current_path = await yandex_find(BACKUP_NAME)
        if current_path:
            try:
                await yandex_mv(current_path, "/" + OLD_BACKUP_NAME)
                logger.info("Renamed current backup to old: %s -> %s", current_path, OLD_BACKUP_NAME)
            except Exception as e:
                logger.warning("Failed to rename current backup: %s", e)

        # создаём новый локальный бэкап
        with open(BACKUP_NAME, "w") as f:
            json.dump(assistant.export_state(), f)

        # загружаем новый бэкап
        await yandex_upload(BACKUP_NAME, "/" + BACKUP_NAME)
        logger.info("Backup uploaded successfully")

    except Exception as e:
        logger.error("Backup failed: %s", e)

async def restore_backup():
    try:
        logger.info("Restoring backup if exists...")
        remote = await yandex_find(BACKUP_NAME)
        if remote:
            await yandex_download(remote, BACKUP_NAME)
            with open(BACKUP_NAME) as f:
                state = json.load(f)
            assistant.load_state(state)
            logger.info("Assistant state restored from backup")
        else:
            logger.warning("No backup found to restore")
    except Exception as e:
        logger.error("Restore backup failed: %s", e)

async def save_backup_loop():
    while True:
        try:
            await save_backup()
        except Exception as e:
            logger.error("save_backup_loop error: %s", e)
        await asyncio.sleep(3600)

# ====================== History Load ======================
CRASH_HISTORY_FILES = ["crash_23k.json"]

async def load_history_files(files=CRASH_HISTORY_FILES):
    await asyncio.sleep(0.1)
    for filename in files:
        logger.info(f"Processing history file: {filename}")
        remote = await yandex_find(filename)
        if not remote:
            logger.warning(f"File {filename} not found in Yandex Disk")
            continue
        try:
            await yandex_download(remote, filename)
            with open(filename) as f:
                data = json.load(f)
            block = 7000
            for i in range(0, len(data), block):
                assistant.load_history_from_list(data[i:i+block])
                logger.info(f"Loaded block {i}-{min(i+block, len(data))} from {filename}")
            os.remove(filename)
            logger.info(f"File {filename} removed from local storage")
        except Exception as e:
            logger.error("Error processing history file %s: %s", filename, e)

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
    global yandex_worker_task
    if yandex_worker_task is None:
        yandex_worker_task = asyncio.create_task(yandex_worker())

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