import asyncio
import logging
import aiohttp
import aiofiles
import os
from typing import AsyncGenerator
import yadisk
from config import YANDEX_ACCESS_TOKEN, DOWNLOAD_CHUNK

logger = logging.getLogger("ai_assistant.utils")

# Инициализация клиента yadisk (синхронный)
yadisk_client = yadisk.YaDisk(token=YANDEX_ACCESS_TOKEN)

yandex_queue: asyncio.Queue = asyncio.Queue()
yandex_worker_task: asyncio.Task | None = None

async def run_yandex_task(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    fut = loop.create_future()
    await yandex_queue.put((func, args, kwargs, fut))
    return await fut

async def yandex_worker():
    logger.info("Запущен Yandex-Disk worker")
    while True:
        func, args, kwargs, fut = await yandex_queue.get()
        try:
            result = await asyncio.get_event_loop().run_in_executor(None, lambda: func(*args, **kwargs))
            if not fut.done():
                fut.set_result(result)
        except Exception as e:
            logger.exception("Ошибка выполнения задачи Yandex worker: %s", e)
            if not fut.done():
                fut.set_exception(e)
        finally:
            yandex_queue.task_done()

# --- обёртки ---
async def yandex_ls(path: str = "/"):
    return await run_yandex_task(lambda: list(yadisk_client.listdir(path)))

async def yandex_find(filename: str, path: str = "/"):
    try:
        items = await yandex_ls(path)
        for item in items:
            if getattr(item, "name", None) == filename or getattr(item, "path", "").endswith("/" + filename):
                return getattr(item, "path", "/" + filename)
        return None
    except Exception as e:
        logger.exception("Ошибка поиска файла на Яндекс.Диске: %s", e)
        return None

async def yandex_upload(local_path: str, remote_path: str, overwrite: bool = True):
    return await run_yandex_task(yadisk_client.upload, local_path, remote_path, overwrite=overwrite)

async def yandex_remove(remote_path: str, permanently: bool = True):
    return await run_yandex_task(yadisk_client.remove, remote_path, permanently=permanently)

async def yandex_move(src: str, dst: str, overwrite: bool = True):
    return await run_yandex_task(yadisk_client.move, src, dst, overwrite=overwrite)

async def yandex_get_download_link(remote_path: str) -> str:
    return await run_yandex_task(yadisk_client.get_download_link, remote_path)

async def yandex_download_stream(remote_path: str, chunk_size: int = DOWNLOAD_CHUNK) -> AsyncGenerator[bytes, None]:
    download_url = await yandex_get_download_link(remote_path)
    if not download_url:
        raise FileNotFoundError(f"Не удалось получить ссылку для скачивания {remote_path}")
    async with aiohttp.ClientSession() as session:
        async with session.get(download_url) as resp:
            resp.raise_for_status()
            async for chunk in resp.content.iter_chunked(chunk_size):
                yield chunk

async def yandex_download_to_file(remote_path: str, local_path: str, chunk_size: int = DOWNLOAD_CHUNK):
    async with aiofiles.open(local_path, "wb") as f:
        async for block in yandex_download_stream(remote_path, chunk_size=chunk_size):
            await f.write(block)
    return local_path

# -------------------------
# Вспомогательные функции
# -------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def calculate_net_win(amount: float, coefficient: float | None) -> float:
    if coefficient is None:
        return 0.0
    return max(0.0, (float(coefficient) - 1.0) * float(amount))

def crash_to_color(crash: float) -> str:
    if crash < 1.2:
        return "red"
    elif crash < 2.0:
        return "blue"
    elif crash < 4.0:
        return "pink"
    elif crash < 8.0:
        return "green"
    elif crash < 25.0:
        return "yellow"
    else:
        return "gradient"

def parse_bets_input(game: dict) -> list[dict]:
    bets = game.get("bets", [])
    if isinstance(bets, str):
        import json
        try:
            bets = json.loads(bets)
        except Exception:
            bets = []
    return [
        {
            "user_id": b.get("user_id"),
            "nickname": b.get("nickname"),
            "amount": float(b.get("amount", 0.0) or 0.0),
            "coefficient_auto": float(b.get("coefficient_auto", 0.0) or 0.0)
        }
        for b in bets
    ]