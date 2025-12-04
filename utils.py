"""
Утилиты для взаимодействия с Яндекс.Диском, async-обертки и небольшие помощники.
"""

import asyncio
import logging
import aiohttp
import aiofiles
import os
from typing import AsyncGenerator
import yadisk

from config import YANDEX_ACCESS_TOKEN, DOWNLOAD_CHUNK

logger = logging.getLogger("ai_assistant.utils")

# Инициализация клиента yadisk (синхронный клиент)
yadisk_client = yadisk.YaDisk(token=YANDEX_ACCESS_TOKEN)

# очередь для последовательного выполнения синхронных операций yadisk в background task
yandex_queue: asyncio.Queue = asyncio.Queue()
yandex_worker_task: asyncio.Task | None = None


async def run_yandex_task(func, *args, **kwargs):
    """
    Помещаем синхронную функцию yadisk в очередь worker'а для последовательного выполнения.
    Возвращает результат или бросает исключение.
    """
    loop = asyncio.get_event_loop()
    fut = loop.create_future()
    await yandex_queue.put((func, args, kwargs, fut))
    return await fut


async def yandex_worker():
    logger.info("Yandex-Disk worker started")
    while True:
        func, args, kwargs, fut = await yandex_queue.get()
        try:
            # выполняем синхронную функцию в threadpool, чтобы не блокировать event loop
            result = await asyncio.get_event_loop().run_in_executor(None, lambda: func(*args, **kwargs))
            if not fut.done():
                fut.set_result(result)
        except Exception as e:
            logger.exception("Yandex worker task failed: %s", e)
            if not fut.done():
                fut.set_exception(e)
        finally:
            yandex_queue.task_done()


# --- обёртки для common операций yadisk ---
async def yandex_ls(path: str = "/"):
    return await run_yandex_task(lambda: list(yadisk_client.listdir(path)))


async def yandex_find(filename: str, path: str = "/"):
    """
    Ищет файл в указанной папке (по name). Возвращает путь '/filename' или None.
    NOTE: простая реализация: листинг корня. При необходимости расширить до рекурсивного поиска.
    """
    try:
        items = await yandex_ls(path)
        for item in items:
            # item — объект yadisk.YaDiskFile (имеет attribute name и path)
            if getattr(item, "name", None) == filename or getattr(item, "path", "").endswith("/" + filename):
                # вернуть полный путь
                return getattr(item, "path", "/" + filename)
        return None
    except Exception as e:
        logger.exception("yandex_find error: %s", e)
        return None


async def yandex_upload(local_path: str, remote_path: str, overwrite: bool = True):
    return await run_yandex_task(yadisk_client.upload, local_path, remote_path, overwrite=overwrite)


async def yandex_remove(remote_path: str, permanently: bool = True):
    return await run_yandex_task(yadisk_client.remove, remote_path, permanently=permanently)


async def yandex_move(src: str, dst: str, overwrite: bool = True):
    return await run_yandex_task(yadisk_client.move, src, dst, overwrite=overwrite)


async def yandex_get_download_link(remote_path: str) -> str:
    """
    Возвращает прямую ссылку на файл (yadisk.get_download_link)
    """
    return await run_yandex_task(yadisk_client.get_download_link, remote_path)


async def yandex_download_stream(remote_path: str, chunk_size: int = DOWNLOAD_CHUNK) -> AsyncGenerator[bytes, None]:
    """
    Асинхронный генератор блоков данных из файла на Яндекс.Диске.
    Возвращает raw bytes.
    """
    download_url = await yandex_get_download_link(remote_path)
    if not download_url:
        raise FileNotFoundError(f"Can't obtain download link for {remote_path}")

    async with aiohttp.ClientSession() as session:
        async with session.get(download_url) as resp:
            resp.raise_for_status()
            async for chunk in resp.content.iter_chunked(chunk_size):
                yield chunk


async def yandex_download_to_file(remote_path: str, local_path: str, chunk_size: int = DOWNLOAD_CHUNK):
    """
    Скачиваем удалённый файл в локальный файл потоково.
    """
    async with aiofiles.open(local_path, "wb") as f:
        async for block in yandex_download_stream(remote_path, chunk_size=chunk_size):
            await f.write(block)
    return local_path


# Небольшие помощники
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)