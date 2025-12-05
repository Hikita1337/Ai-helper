from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
import asyncio
import json
from dotenv import load_dotenv
import ijson
import ably
import time

from config import (
    PORT, SELF_URL, ABLY_API_KEY, YANDEX_ACCESS_TOKEN,
    CRASH_HISTORY_FILES, BLOCK_RECORDS, PRED_LOG_LEN, BACKUP_INTERVAL_SECONDS
)
from utils import yandex_download_to_file, yandex_find, yandex_worker, yandex_worker_task
from bots_manager import BotsManager
from backup_manager import BackupManager
from model import AIAssistant
from analytics import Analytics

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai_assistant.main")

app = FastAPI(title="Crash AI Assistant")

# -------------------- Ably setup --------------------
ably_client = ably.RestClient(ABLY_API_KEY)
ably_channel = ably_client.channels.get("ABLU-TAI")  # канал реального времени

# -------------------- Core objects --------------------
assistant = AIAssistant(ably_channel=ably_channel)
bots_mgr = BotsManager()
analytics_module = Analytics()

# -------------------- Backup Manager --------------------
backup_mgr = BackupManager({
    "assistant": assistant,
    "bots": bots_mgr,
    "analytics": analytics_module
})
assistant.attach_backup_manager(backup_mgr)

# -------------------- API Payloads --------------------
class BetsPayload(BaseModel):
    game_id: int
    deposit_sum: float
    num_players: int
    bets: list

class FeedbackPayload(BaseModel):
    game_id: int
    crash: float
    bets: list | None = None

# -------------------- History loader --------------------
async def process_history_file(remote_path: str, filename: str, block_records: int = BLOCK_RECORDS):
    logger.info("Processing history file: %s (remote: %s)", filename, remote_path)
    local_path = filename
    try:
        await yandex_download_to_file(remote_path, local_path)

        def iter_items(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                for item in ijson.items(f, "item"):
                    yield item

        batch = []
        async for item in async_iter_from_thread(iter_items, local_path):
            if isinstance(item, dict) and "bets" in item and isinstance(item["bets"], str):
                try:
                    item["bets"] = json.loads(item["bets"])
                except Exception:
                    item["bets"] = []
            batch.append(item)
            if len(batch) >= block_records:
                if hasattr(assistant, "load_history_from_list"):
                    assistant.load_history_from_list(batch)
                batch.clear()

        if batch:
            if hasattr(assistant, "load_history_from_list"):
                assistant.load_history_from_list(batch)

        try:
            os.remove(local_path)
        except Exception:
            pass

        logger.info("Finished processing %s", filename)
    except Exception as e:
        logger.exception("Error processing history file %s: %s", filename, e)

async def async_iter_from_thread(generator_fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    q = asyncio.Queue()

    def run():
        try:
            for v in generator_fn(*args, **kwargs):
                loop.call_soon_threadsafe(q.put_nowait, v)
        except Exception as e:
            loop.call_soon_threadsafe(q.put_nowait, e)
        finally:
            loop.call_soon_threadsafe(q.put_nowait, None)

    await loop.run_in_executor(None, run)

    while True:
        item = await q.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        yield item

async def load_history_files(files=CRASH_HISTORY_FILES, block_records=BLOCK_RECORDS):
    for filename in files:
        remote = await yandex_find(filename)
        if not remote:
            logger.warning("History file %s not found on Yandex", filename)
            continue
        await process_history_file(remote, filename, block_records=block_records)

# -------------------- Periodic backup worker --------------------
async def periodic_backup_worker(interval: int = BACKUP_INTERVAL_SECONDS):
    while True:
        await asyncio.sleep(interval)
        try:
            if getattr(assistant, "ready_for_backup", True):
                await assistant.save_full_backup()
            else:
                logger.info("Backup postponed: assistant still processing")
        except Exception as e:
            logger.exception("Periodic backup failed: %s", e)

# -------------------- Startup --------------------
@app.on_event("startup")
async def startup_event():
    global yandex_worker_task
    if yandex_worker_task is None:
        from utils import yandex_worker as _yworker
        yandex_worker_task = asyncio.create_task(_yworker())

    await bots_mgr.load_from_disk()
    await backup_mgr.start_worker()
    await backup_mgr.restore_backup()  # восстановление полного состояния
    asyncio.create_task(load_history_files())
    asyncio.create_task(periodic_backup_worker(BACKUP_INTERVAL_SECONDS))
    logger.info("Startup complete")

# -------------------- Feedback endpoint --------------------
@app.post("/feedback")
async def feedback(payload: FeedbackPayload):
    try:
        if hasattr(assistant, "process_feedback"):
            assistant.process_feedback(payload.game_id, payload.crash, payload.bets)
            return {"status": "ok"}
        else:
            raise RuntimeError("assistant has no process_feedback")
    except Exception as e:
        logger.exception("feedback failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

# -------------------- Monitoring endpoints --------------------
@app.get("/logs")
async def logs(limit: int = 20):
    try:
        get_pred_log = getattr(assistant, "get_pred_log", None)
        if callable(get_pred_log):
            return {"logs": get_pred_log(limit)}
        else:
            pl = list(getattr(assistant, "pred_log", []))
            return {"logs": pl[-limit:]}
    except Exception as e:
        logger.exception("logs error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def status():
    try:
        get_status = getattr(assistant, "get_status", None)
        if callable(get_status):
            st = get_status()
        else:
            st = {
                "total_games": len(getattr(assistant, "games_index", [])),
                "pred_log_len": len(getattr(assistant, "pred_log", []))
            }
        st.update(bots_mgr.summary())
        return st
    except Exception as e:
        logger.exception("status error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/bots")
async def bots_list():
    return bots_mgr.summary()

@app.post("/bots/mark/{user_id}")
async def mark_bot_endpoint(user_id: str, info: dict | None = None):
    try:
        await bots_mgr.mark_bot(user_id, info or {})
        await bots_mgr.save_to_disk()
        return {"status": "ok"}
    except Exception as e:
        logger.exception("mark_bot failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bots/unmark/{user_id}")
async def unmark_bot_endpoint(user_id: str):
    try:
        await bots_mgr.unmark_bot(user_id)
        await bots_mgr.save_to_disk()
        return {"status": "ok"}
    except Exception as e:
        logger.exception("unmark_bot failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics_sample(limit: int = 100):
    try:
        pl = list(getattr(assistant, "pred_log", []))
        data = pl[-limit:]
        metrics = analytics_module.evaluate_predictions_batch(data)
        return {"metrics": metrics}
    except Exception as e:
        logger.exception("metrics error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))