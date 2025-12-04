"""
Основной FastAPI приложение.
Подключаем BotsManager, utils, analytics; обрабатываем старые файлы истории,
потоково загружаем большие JSON и парсим их батчами.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
import asyncio
import aiofiles
import json
from dotenv import load_dotenv
import ijson

from config import (
    PORT, SELF_URL, ABLY_API_KEY, YANDEX_ACCESS_TOKEN,
    BACKUP_NAME, BACKUP_FOLDER, CRASH_HISTORY_FILES, BLOCK_RECORDS, BACKUP_PERIOD_SECONDS, PRED_LOG_LEN
)
from utils import (
    yandex_download_to_file, yandex_find, yandex_upload, yandex_download_stream,
    yandex_worker, yandex_queue, yandex_worker_task, yandex_get_download_link, run_yandex_task, yadisk_client
)
from bots_manager import BotsManager
from analytics import evaluate_predictions_batch

# model import (пользовательский AIAssistant)
from model import AIAssistant

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai_assistant.main")

app = FastAPI(title="Crash AI Assistant")

assistant = AIAssistant()
bots_mgr = BotsManager()

# -------------------- API Payloads --------------------
class BetsPayload(BaseModel):
    game_id: int
    bets: list


class FeedbackPayload(BaseModel):
    game_id: int
    crash: float
    bets: list | None = None


# -------------------- Backup helpers --------------------
async def save_backup():
    """
    Сохраняем snapshot состояния помощника (ассоциация state -> файл).
    Чтобы уменьшить размер — рекомендуем сохранять только минимальное состояние:
    - модели (можно сериализовать в отдельные файлы)
    - индекс уже обработанных game_id'ов (games_index)
    - предсказания (pred_log) — ограниченный по длине
    - bots metadata (мы их сохраняем отдельно через BotsManager)
    """
    try:
        logger.info("Saving backup...")
        snapshot = {
            "games_index": list(getattr(assistant, "games_index", [])),
            # pred_log может быть deque — привести к list
            "pred_log": list(getattr(assistant, "pred_log", [])[-PRED_LOG_LEN:]),
            # иные небольшие состояния, если есть (например модели можно сохранять отдельно)
            "meta": {
                "saved_at": asyncio.get_event_loop().time()
            }
        }
        # write locally
        with open(BACKUP_NAME, "w", encoding="utf-8") as f:
            json.dump(snapshot, f)
        # upload
        await yandex_upload(BACKUP_NAME, BACKUP_FOLDER.rstrip("/") + "/" + BACKUP_NAME)
        logger.info("Backup uploaded successfully")
    except Exception as e:
        logger.exception("save_backup failed: %s", e)


async def restore_backup():
    """
    Скачиваем backup если есть и загружаем в assistant
    """
    try:
        logger.info("Restoring backup if exists...")
        remote = await yandex_find(BACKUP_NAME)
        if not remote:
            logger.warning("No backup found")
            return
        local = BACKUP_NAME
        await yandex_download_to_file(remote, local)
        with open(local, "r", encoding="utf-8") as f:
            state = json.load(f)
        # ожидаем, что у assistant есть метод load_state или аналог
        if hasattr(assistant, "load_state"):
            assistant.load_state(state)
            logger.info("Assistant state restored from backup")
        else:
            logger.warning("Assistant has no load_state method, skipping restore")
    except Exception as e:
        logger.exception("restore_backup failed: %s", e)


async def save_backup_loop():
    while True:
        try:
            await save_backup()
        except Exception as e:
            logger.exception("save_backup_loop error: %s", e)
        await asyncio.sleep(BACKUP_PERIOD_SECONDS)


# -------------------- History loader --------------------
async def process_history_file(remote_path: str, filename: str, block_records: int = BLOCK_RECORDS):
    """
    Скачиваем remote_path локально и парсим JSON массив поточно через ijson (в threadpool),
    обрабатываем батчами.
    """
    logger.info("Processing history file: %s (remote: %s)", filename, remote_path)
    local_path = filename
    try:
        # скачиваем локально потоково
        await yandex_download_to_file(remote_path, local_path)

        # парсинг ijson.items в отдельном потоке (чтобы не блокировать event loop)
        def iter_items(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                for item in ijson.items(f, "item"):
                    yield item

        batch = []
        async for item in async_iter_from_thread(iter_items, local_path):
            # преобразуем bets поле если это строка
            if isinstance(item, dict) and "bets" in item and isinstance(item["bets"], str):
                try:
                    item["bets"] = json.loads(item["bets"])
                except Exception:
                    item["bets"] = []
            batch.append(item)
            if len(batch) >= block_records:
                # отправляем в assistant
                if hasattr(assistant, "load_history_from_list"):
                    assistant.load_history_from_list(batch)
                else:
                    logger.warning("assistant has no load_history_from_list")
                logger.info("Loaded block of %d records from %s", len(batch), filename)
                batch.clear()

        # остаток
        if batch:
            if hasattr(assistant, "load_history_from_list"):
                assistant.load_history_from_list(batch)
            logger.info("Loaded final block of %d records from %s", len(batch), filename)

        # отметка: можно загружать флаг на диск (bots_mgr будет это делать)
        # удаляем локальный файл
        try:
            os.remove(local_path)
        except Exception:
            pass

        logger.info("Finished processing %s", filename)
    except Exception as e:
        logger.exception("Error processing history file %s: %s", filename, e)


async def async_iter_from_thread(generator_fn, *args, **kwargs):
    """
    Обёртка: запускает generator_fn в threadpool и возвращает async-генератор.
    generator_fn должен возвращать итератор (yield) синхронно.
    """
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

    # запускаем в threadpool
    await loop.run_in_executor(None, run)

    while True:
        item = await q.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        yield item


async def load_history_files(files=CRASH_HISTORY_FILES, block_records=BLOCK_RECORDS):
    # загрузка списка history files, пропуск тех, что уже помечены/обработаны
    for filename in files:
        # check remote presence
        remote = await yandex_find(filename)
        if not remote:
            logger.warning("History file %s not found on Yandex", filename)
            continue
        await process_history_file(remote, filename, block_records=block_records)


# -------------------- Startup / Endpoints --------------------
@app.on_event("startup")
async def startup_event():
    # стартуем yandex worker
    global yandex_worker_task
    if yandex_worker_task is None:
        from utils import yandex_worker as _yworker
        yandex_worker_task = asyncio.create_task(_yworker())

    # загружаем bots data
    await bots_mgr.load_from_disk()

    # восстанавливаем backup
    await restore_backup()

    # грузим history (первичная загрузка)
    # делаем в background task, чтобы стартап не заблокировался слишком долго
    asyncio.create_task(load_history_files())

    # периодические задачи
    asyncio.create_task(save_backup_loop())
    logger.info("Startup complete")


@app.post("/predict")
async def predict(payload: BetsPayload):
    try:
        if hasattr(assistant, "predict_and_log"):
            assistant.predict_and_log(payload.model_dump())
            last_pred = assistant.pred_log[-1] if getattr(assistant, "pred_log", None) else None
            # publish to ably if configured (user code had ably usage)
            # user handles Ably in their code — мы не публикуем здесь напрямую
            return {"status": "ok", "prediction": last_pred}
        else:
            raise RuntimeError("assistant has no predict_and_log")
    except Exception as e:
        logger.exception("predict failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


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


@app.get("/logs")
async def logs(limit: int = 20):
    try:
        get_pred_log = getattr(assistant, "get_pred_log", None)
        if callable(get_pred_log):
            return {"logs": get_pred_log(limit)}
        else:
            # fallback: if assistant.pred_log exists
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
            # minimal fallback
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
    """
    Пример endpoint'а, который собирает последние limit предсказаний из assistant.pred_log
    и возвращает базовые метрики (avg_error, hit rates).
    """
    try:
        pl = list(getattr(assistant, "pred_log", []))
        data = pl[-limit:]
        # ожидаем, что каждый pred в log содержит pred_safe/pred_med/pred_risk и actual
        metrics = evaluate_predictions_batch(data)
        return {"metrics": metrics}
    except Exception as e:
        logger.exception("metrics error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))