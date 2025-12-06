"""
Утилиты для взаимодействия с Яндекс.Диском, потоковое чтение JSON и вспомогательные функции.
"""

import asyncio
import logging
import aiohttp
import aiofiles
import os
import json
import codecs
from typing import AsyncGenerator, Any, Iterable, Optional
import yadisk

from config import YANDEX_ACCESS_TOKEN, DOWNLOAD_CHUNK

logger = logging.getLogger("ai_assistant.utils")

# Инициализация клиента yadisk (синхронный)
yadisk_client = yadisk.YaDisk(token=YANDEX_ACCESS_TOKEN)

# Очередь для последовательного выполнения синхронных операций yadisk в background task
yandex_queue: asyncio.Queue = asyncio.Queue()
yandex_worker_task: asyncio.Task | None = None


async def run_yandex_task(func, *args, **kwargs):
    """Запускает синхронную функцию yadisk в отдельном потоке через очередь"""
    loop = asyncio.get_event_loop()
    fut = loop.create_future()
    await yandex_queue.put((func, args, kwargs, fut))
    return await fut


async def yandex_worker():
    """Фоновый воркер: последовательно выполняет синхронные вызовы yadisk в executor."""
    logger.info("Рабочий поток Яндекс.Диск запущен")
    while True:
        func, args, kwargs, fut = await yandex_queue.get()
        try:
            result = await asyncio.get_event_loop().run_in_executor(None, lambda: func(*args, **kwargs))
            if not fut.done():
                fut.set_result(result)
        except Exception as e:
            logger.exception("Ошибка в задаче Yandex worker: %s", e)
            if not fut.done():
                fut.set_exception(e)
        finally:
            yandex_queue.task_done()


# --- Обёртки для основных операций yadisk ---
async def yandex_ls(path: str = "/") -> list:
    """Список объектов в каталоге на Яндекс.Диске. Возвращает список словарей/объектов."""
    try:
        res = await run_yandex_task(lambda: yadisk_client.listdir(path))
        # Иногда yadisk возвращает generator/iterable, иногда list
        if isinstance(res, Iterable) and not isinstance(res, (str, bytes)):
            return list(res)
        return [res]
    except Exception as e:
        logger.exception("Ошибка yandex_ls(%s): %s", path, e)
        return []


async def yandex_find(filename: str, path: str = "/") -> Optional[str]:
    """
    Находит путь файла на Яндекс.Диске.
    - filename может быть либо точным именем (например "full_backup.json"), либо путём.
    - возвращает путь вида '/full_backup.json' или None.
    """
    try:
        # если filename уже начинается с '/', считаем это полным путём и пытаемся проверить его существование
        if filename.startswith("/"):
            try:
                exists = await run_yandex_task(lambda: yadisk_client.exists(filename))
                return filename if exists else None
            except Exception:
                # fallback на перечисление
                pass

        items = await yandex_ls(path)
        for item in items:
            # item может быть dict или объект; пробуем безопасно достать name/path
            name = None
            pth = None
            if isinstance(item, dict):
                name = item.get("name") or item.get("display_name")
                pth = item.get("path")
            else:
                # объект yadisk: пробуем атрибуты
                name = getattr(item, "name", None)
                pth = getattr(item, "path", None)

            if name == filename:
                return pth or ("/" + filename)
            if isinstance(pth, str) and pth.endswith("/" + filename):
                return pth
        # Попытка найти рекурсивно в корне (если указан корень)
        if path == "/":
            # перечислим еще корневые файлы (в большинстве случаев уже были)
            return None
        return None
    except Exception as e:
        logger.exception("Ошибка поиска файла на Яндекс.Диске: %s", e)
        return None


async def yandex_upload(local_path: str, remote_path: str, overwrite: bool = True):
    """
    Загружает локальный файл на Яндекс.Диск.
    remote_path — путь на Диске (например '/full_backup.json' или '/backups/full_backup.json').
    """
    try:
        return await run_yandex_task(yadisk_client.upload, local_path, remote_path, overwrite=overwrite)
    except Exception as e:
        logger.exception("Ошибка загрузки на Яндекс.Диск (%s -> %s): %s", local_path, remote_path, e)
        raise


async def yandex_remove(remote_path: str, permanently: bool = True):
    """Удаляет файл/папку на Яндекс.Диске."""
    try:
        return await run_yandex_task(yadisk_client.remove, remote_path, permanently=permanently)
    except Exception as e:
        logger.exception("Ошибка удаления на Яндекс.Диске (%s): %s", remote_path, e)
        raise


# Алиас для совместимости (иногда модуль использует yandex_delete)
async def yandex_delete(remote_path: str, permanently: bool = True):
    return await yandex_remove(remote_path, permanently=permanently)


async def yandex_move(src: str, dst: str, overwrite: bool = True):
    try:
        return await run_yandex_task(yadisk_client.move, src, dst, overwrite=overwrite)
    except Exception as e:
        logger.exception("Ошибка перемещения на Яндекс.Диске (%s -> %s): %s", src, dst, e)
        raise


async def yandex_get_download_link(remote_path: str) -> str:
    """Получает публичную ссылку для скачивания (внутренняя ссылка yadisk)."""
    try:
        return await run_yandex_task(yadisk_client.get_download_link, remote_path)
    except Exception as e:
        logger.exception("Ошибка получения ссылки для %s: %s", remote_path, e)
        raise


# --- Потоковое скачивание байтов из файла на Яндекс.Диске ---
async def yandex_download_stream(remote_path: str, chunk_size: int = DOWNLOAD_CHUNK) -> AsyncGenerator[bytes, None]:
    """Возвращает поток байтов из файла на Яндекс.Диске"""
    download_url = await yandex_get_download_link(remote_path)
    if not download_url:
        raise FileNotFoundError(f"Не удалось получить ссылку для {remote_path}")
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(download_url) as resp:
            resp.raise_for_status()
            async for chunk in resp.content.iter_chunked(chunk_size):
                yield chunk


async def yandex_download_to_file(remote_path: str, local_path: str, chunk_size: int = DOWNLOAD_CHUNK):
    """Скачивает файл целиком на диск (по чанкам). Возвращает локальный путь."""
    # Создаём директорию для локального пути, если нужно
    parent = os.path.dirname(local_path)
    if parent and parent != "/":
        os.makedirs(parent, exist_ok=True)

    async with aiofiles.open(local_path, "wb") as f:
        async for block in yandex_download_stream(remote_path, chunk_size=chunk_size):
            await f.write(block)
    logger.info("Файл %s скачан на диск", local_path)
    return local_path


# --- Потоковое скачивание JSON чанками с корректным UTF-8 ---
async def yandex_download_stream_json(remote_path: str, chunk_size: int = DOWNLOAD_CHUNK) -> AsyncGenerator[Any, None]:
    """
    Потоковый парсер JSON-массива очень большого размера.
    Работает даже если массив > 10ГБ.
    Выдаёт элементы массива один за другим.
    Особенности:
      - устойчив к тому, что объекты разделены запятыми между чанками;
      - корректно декодирует UTF-8 через incremental decoder;
      - безопасно обрабатывает хвосты/непарные данные.
    """
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    inside_array = False
    decoder_obj = json.JSONDecoder()

    async for chunk in yandex_download_stream(remote_path, chunk_size=chunk_size):
        # chunk — bytes
        try:
            text = decoder.decode(chunk)
        except Exception:
            # Если декодер по какой-то причине не может декодировать — попробуем с 'ignore'
            try:
                text = chunk.decode("utf-8", errors="ignore")
            except Exception:
                text = ""
        buffer += text

        i = 0
        while i < len(buffer):
            # Ждём начало массива [
            if not inside_array:
                pos = buffer.find("[", i)
                if pos == -1:
                    # По безопасности держим небольшую часть хвоста (на случай, если '[' придёт в следующем чанке)
                    buffer = buffer[-1024:]
                    break
                inside_array = True
                i = pos + 1
                continue

            # Пропускаем пробелы и запятые
            while i < len(buffer) and buffer[i] in " \n\r\t,":
                i += 1

            # Если дошли до конца массива — выходим
            if i < len(buffer) and buffer[i] == "]":
                return

            # Пытаемся распарсить элемент (объект должен начинаться с '{' или '[')
            try:
                obj, idx = decoder_obj.raw_decode(buffer, i)
                yield obj
                i = idx
                # После успешного чтения продолжаем (может быть следующий объект в буфере)
            except json.JSONDecodeError:
                # Нужны дополнительные данные: сдвигаем буфер на текущую позицию и ждём следующий chunk
                buffer = buffer[i:]
                break

        else:
            # если цикл завершился без break — очищаем буфер, оставляя небольшой хвост
            buffer = buffer[-1024:]

    # Завершаем декодер, чтобы собрать остаток
    try:
        tail = decoder.decode(b"", final=True)
    except Exception:
        tail = ""

    if tail:
        buffer += tail

    # Финальная попытка распарсить оставшиеся элементы в буфере
    buffer = buffer.strip()
    if not buffer:
        return

    # Если остался массив целиком — возможно в маленьком файле
    if buffer.startswith("["):
        try:
            arr = json.loads(buffer)
            if isinstance(arr, list):
                for e in arr:
                    yield e
            return
        except Exception:
            # пробуем вытащить элементы по одному
            pass

    # Попытка вытащить по одному
    i = 0
    while i < len(buffer):
        while i < len(buffer) and buffer[i] in " \n\r\t,":
            i += 1
        if i < len(buffer) and buffer[i] == "]":
            break
        try:
            obj, idx = decoder_obj.raw_decode(buffer, i)
            yield obj
            i = idx
        except Exception:
            break


# --- Помощники ---
def ensure_dir(path: str):
    """Создаёт директорию, если её нет. Не пытаемся создавать корень '/'."""
    if not path:
        return
    # не создавать корень, если указан '/'
    if path == "/":
        return
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        logger.exception("Не удалось создать директорию %s: %s", path, e)


def calculate_net_win(amount: float, coefficient: float | None) -> float:
    """
    Рассчитывает чистый выигрыш игрока:
      - если coefficient = None -> проиграл -> 0
      - иначе -> (coefficient - 1) * amount
    """
    if coefficient is None:
        return 0.0
    return max(0.0, (float(coefficient) - 1.0) * float(amount))


def crash_to_color(crash: float) -> str:
    """Преобразует значение краша в цвет"""
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
    """
    Преобразует raw input из парсера в список ставок с нужными ключами:
    {
      "user_id": int,
      "nickname": str,
      "amount": float,
      "coefficient_auto": float
    }
    """
    bets = game.get("bets", [])
    if isinstance(bets, str):
        try:
            bets = json.loads(bets)
        except Exception:
            bets = []
    return [
        {
            "user_id": b.get("user_id"),
            "nickname": b.get("nickname") or b.get("name"),
            "amount": float(b.get("amount", 0.0) or 0.0),
            "coefficient_auto": float(b.get("auto", b.get("coefficient_auto", 0.0)) or 0.0)
        }
        for b in bets if isinstance(b, dict)
    ]