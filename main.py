from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
import asyncio
import json
import ijson
import io
from ably import AblyRealtime
import time
from collections import defaultdict

from config import (
    PORT, SELF_URL, ABLY_API_KEY, YANDEX_ACCESS_TOKEN,
    CRASH_HISTORY_FILES, BLOCK_RECORDS, PRED_LOG_LEN, BACKUP_INTERVAL_SECONDS
)
from utils import (
    yandex_download_stream, yandex_find, yandex_worker, yandex_worker_task,
    ensure_dir
)
from bots_manager import BotsManager
from backup_manager import BackupManager
from model import AIAssistant
from analytics import Analytics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai_assistant.main")

app = FastAPI(title="Crash AI Assistant")

# -------------------- Ably setup --------------------
ably_client = AblyRealtime(ABLY_API_KEY)
ably_channel = ably_client.channels.get("ABLU-TAI")

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

# -------------------- Flag для обработки истории --------------------
FLAG_FILE = "history_processed_flag.json"

async def is_history_processed() -> bool:
    """Проверяет наличие флага, что история уже обработана."""
    if os.path.exists(FLAG_FILE):
        try:
            with open(FLAG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("processed", False)
        except Exception:
            return False
    return False

async def set_history_processed_flag():
    """Создаёт флаг о том, что история обработана."""
    try:
        with open(FLAG_FILE, "w", encoding="utf-8") as f:
            json.dump({"processed": True, "timestamp": time.time()}, f)
    except Exception as e:
        logger.exception("Не удалось установить флаг обработки истории: %s", e)

# -------------------- History loader --------------------
async def process_history_file(remote_path: str, block_records: int = BLOCK_RECORDS):
    logger.info("Начало обработки истории с файла: %s", remote_path)
    batch = []
    unique_users = set()
    total_processed = 0
    leftover = ""  # остаток неполного JSON

    try:
        async for chunk in yandex_download_stream(remote_path):
            chunk_str = chunk.decode("utf-8")
            chunk_data = leftover + chunk_str
            stream = io.StringIO(chunk_data)
            parser = ijson.items(stream, "item")
            consumed_chars = 0

            for item in parser:
                if isinstance(item, dict) and "bets" in item and isinstance(item["bets"], str):
                    try:
                        item["bets"] = json.loads(item["bets"])
                    except Exception:
                        item["bets"] = []

                for b in item.get("bets", []):
                    uid = b.get("user_id")
                    if uid is not None:
                        unique_users.add(uid)

                batch.append(item)
                total_processed += 1
                consumed_chars = stream.tell()

                if len(batch) >= block_records:
                    if hasattr(assistant, "load_history_from_list"):
                        assistant.load_history_from_list(batch)
                    logger.info("Обработан блок из %d игр", len(batch))
                    batch.clear()

            leftover = chunk_data[consumed_chars:]

        if leftover.strip():
            stream = io.StringIO(leftover)
            parser = ijson.items(stream, "item")
            for item in parser:
                if isinstance(item, dict) and "bets" in item and isinstance(item["bets"], str):
                    try:
                        item["bets"] = json.loads(item["bets"])
                    except Exception:
                        item["bets"] = []

                for b in item.get("bets", []):
                    uid = b.get("user_id")
                    if uid is not None:
                        unique_users.add(uid)

                batch.append(item)
                total_processed += 1

        if batch:
            if hasattr(assistant, "load_history_from_list"):
                assistant.load_history_from_list(batch)
            logger.info("Обработан финальный блок из %d игр", len(batch))

        logger.info("Обработка истории завершена: всего игр %d, уникальных пользователей %d",
                    total_processed, len(unique_users))

        # Ставим флаг
        await set_history_processed_flag()

    except Exception as e:
        logger.exception("Ошибка обработки файла истории %s: %s", remote_path, e)

async def load_history_files(files=CRASH_HISTORY_FILES, block_records=BLOCK_RECORDS):
    for filename in files:
        remote = await yandex_find(filename)
        if not remote:
            logger.warning("Файл истории %s не найден на Яндекс.Диске", filename)
            continue
        await process_history_file(remote, block_records=block_records)

# -------------------- Periodic backup worker --------------------
async def periodic_backup_worker(interval: int = BACKUP_INTERVAL_SECONDS):
    while True:
        await asyncio.sleep(interval)
        try:
            if getattr(assistant, "ready_for_backup", True):
                await assistant.save_full_backup()
            else:
                logger.info("Бэкап отложен: ассистент ещё обрабатывает данные")
        except Exception as e:
            logger.exception("Ошибка периодического бэкапа: %s", e)

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

    if not await is_history_processed():
        logger.info("История ещё не обработана. Начинаем обработку...")
        asyncio.create_task(load_history_files())
    else:
        logger.info("История уже обработана. Пропуск обработки файла.")

    asyncio.create_task(periodic_backup_worker(BACKUP_INTERVAL_SECONDS))
    logger.info("Старт приложения завершён")

# -------------------- API endpoints --------------------
@app.post("/feedback")
async def feedback(payload: FeedbackPayload):
    try:
        if hasattr(assistant, "process_feedback"):
            assistant.process_feedback(payload.game_id, payload.crash, payload.bets)
            return {"status": "ok"}
        else:
            raise RuntimeError("Ассистент не имеет метода process_feedback")
    except Exception as e:
        logger.exception("Ошибка /feedback: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

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
        logger.exception("Ошибка /logs: %s", e)
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
        logger.exception("Ошибка /status: %s", e)
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
        logger.exception("Ошибка /bots/mark: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bots/unmark/{user_id}")
async def unmark_bot_endpoint(user_id: str):
    try:
        await bots_mgr.unmark_bot(user_id)
        await bots_mgr.save_to_disk()
        return {"status": "ok"}
    except Exception as e:
        logger.exception("Ошибка /bots/unmark: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics_sample(limit: int = 100):
    try:
        pl = list(getattr(assistant, "pred_log", []))
        data = pl[-limit:]
        metrics = analytics_module.evaluate_predictions_batch(data)
        return {"metrics": metrics}
    except Exception as e:
        logger.exception("Ошибка /metrics: %s", e)
        raise HTTPException(status_code=500, detail=str(e))