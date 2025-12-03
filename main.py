from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
import asyncio
import aiohttp
import aiofiles
import json
from dotenv import load_dotenv
from ably import AblyRest
from model import AIAssistant
import yadisk  # pip install yadisk
import ijson  # pip install ijson
import io


load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai_assistant.main")

PORT = int(os.getenv("PORT", 8000))
SELF_URL = os.getenv("SELF_URL")
ABLY_API_KEY = os.getenv("ABLY_API_KEY")
YANDEX_ACCESS_TOKEN = os.getenv("YANDEX_ACCESS_TOKEN")
CRASH_HISTORY_FILES = ["crash_23k.json"]

assistant = AIAssistant()
ably_client = AblyRest(ABLY_API_KEY)
ably_channel = ably_client.channels.get("crash_ai_hud")

# ====================== Yandex Disk ======================
yadisk_client = yadisk.YaDisk(token=YANDEX_ACCESS_TOKEN)
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
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            fut.set_result(result)
        except Exception as e:
            if not fut.done():
                fut.set_exception(e)
        finally:
            yandex_queue.task_done()

async def yandex_upload(local_path: str, remote_path: str):
    return await run_yandex_task(yadisk_client.upload, local_path, remote_path, overwrite=True)

async def yandex_ls():
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

async def yandex_download_stream(remote_path: str, chunk_size: int = 1024*1024):
    """Асинхронный генератор блоков данных из файла на Яндекс.Диске"""
    download_url = await run_yandex_task(yadisk_client.get_download_link, remote_path)
    async with aiohttp.ClientSession() as session:
        async with session.get(download_url) as resp:
            resp.raise_for_status()
            async for chunk in resp.content.iter_chunked(chunk_size):
                yield chunk

# ====================== Backup logic ======================
BACKUP_NAME = "assistant_backup.json"
OLD_BACKUP_NAME = "assistant_backup_old.json"

async def save_backup():
    try:
        logger.info("Saving backup...")
        old_old_path = await yandex_find(OLD_BACKUP_NAME)
        if old_old_path:
            try:
                await run_yandex_task(yadisk_client.remove, old_old_path, permanently=True)
                logger.info("Deleted old-old backup %s", old_old_path)
            except Exception as e:
                logger.warning("Failed to delete old-old backup: %s", e)

        current_path = await yandex_find(BACKUP_NAME)
        if current_path:
            try:
                await run_yandex_task(yadisk_client.move, current_path, "/" + OLD_BACKUP_NAME, overwrite=True)
                logger.info("Renamed current backup to old: %s -> %s", current_path, OLD_BACKUP_NAME)
            except Exception as e:
                logger.warning("Failed to rename current backup: %s", e)

        with open(BACKUP_NAME, "w") as f:
            json.dump(assistant.export_state(), f)
        await yandex_upload(BACKUP_NAME, "/" + BACKUP_NAME)
        logger.info("Backup uploaded successfully")
    except Exception as e:
        logger.error("Backup failed: %s", e)

async def restore_backup():
    try:
        logger.info("Restoring backup if exists...")
        remote = await yandex_find(BACKUP_NAME)
        if remote:
            local_path = BACKUP_NAME
            async with aiofiles.open(local_path, "wb") as f:
                async for block in yandex_download_stream(remote):
                    await f.write(block)
            with open(local_path) as f:
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
async def stream_json_from_yadisk(remote_path: str):
    """
    Асинхронно итерируем по JSON-массиву с Яндекс.Диска.
    Возвращает dict для каждой записи в массиве.
    """
    download_url = await run_yandex_task(yadisk_client.get_download_link, remote_path)
    async with aiohttp.ClientSession() as session:
        async with session.get(download_url) as resp:
            resp.raise_for_status()
            # resp.content — это async-байтовый поток, ijson.items_async умеет его читать
            async for record in ijson.items_async(resp.content, "item"):
                yield record


async def load_history_files(files=CRASH_HISTORY_FILES, block_records=7000):
    """
    Асинхронно обрабатывает JSON-файлы истории, блоками.
    Каждый record — dict, как в твоем примере JSON.
    """
    for filename in files:
        flag_file = filename + "_processed_flag.json"
        flag_remote = await yandex_find(flag_file)
        if flag_remote:
            logger.info(f"File {filename} already processed. Skipping.")
            continue

        logger.info(f"Processing history file: {filename}")
        remote = await yandex_find(filename)
        if not remote:
            logger.warning(f"File {filename} not found in Yandex Disk")
            continue

        try:
            batch = []
            async for record in stream_json_from_yadisk(remote):
                # record здесь dict, можно спокойно делать record.get("game_id") и т.д.
                batch.append(record)
                if len(batch) >= block_records:
                    assistant.load_history_from_list(batch)
                    logger.info(f"Loaded block of {len(batch)} records from {filename}")
                    batch.clear()

            if batch:
                assistant.load_history_from_list(batch)
                logger.info(f"Loaded final block of {len(batch)} records from {filename}")

            # Создаём флаг, что обработка завершена
            async with aiofiles.open(flag_file, "w") as f:
                await f.write(json.dumps({"processed": True}))
            await yandex_upload(flag_file, "/" + flag_file)
            logger.info(f"Processing of {filename} finished. Flag uploaded.")

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