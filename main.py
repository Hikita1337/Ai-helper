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
import shlex

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai_assistant.main")

PORT = int(os.getenv("PORT", 8000))
SELF_URL = os.getenv("SELF_URL")
ABLY_API_KEY = os.getenv("ABLY_API_KEY")

MEGA_EMAIL = os.getenv("MEGA_EMAIL")
MEGA_PASSWORD = os.getenv("MEGA_PASSWORD")
MEGA_CMD_PATH = os.getenv("MEGA_CMD_PATH", "mega")  # путь к бинарю mega (можно оставить "mega" если в PATH)

assistant = AIAssistant()
ably_client = AblyRest(ABLY_API_KEY)
ably_channel = ably_client.channels.get("crash_ai_hud")

# ====================== Mega-CMD via subprocess ======================
# Очередь команд для последовательного выполнения (чтобы Mega-CMD не конфликтовал)
mega_cmd_queue: asyncio.Queue = asyncio.Queue()
mega_worker_task: asyncio.Task | None = None

async def run_mega_cmd_raw(args: list[str], timeout: int = 120) -> tuple[int, str, str]:
    """Запустить mega команду как подпроцесс; возвращает (returncode, stdout, stderr)."""
    cmd = [MEGA_CMD_PATH] + args
    logger.debug("Run mega cmd: %s", " ".join(shlex.quote(x) for x in cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(f"mega command timeout: {' '.join(cmd)}")
    stdout = out.decode(errors="ignore") if out else ""
    stderr = err.decode(errors="ignore") if err else ""
    logger.debug("Mega stdout: %s", stdout.strip())
    logger.debug("Mega stderr: %s", stderr.strip())
    return proc.returncode, stdout, stderr

async def mega_worker():
    """Worker, который последовательно выполняет команды из очереди."""
    logger.info("Mega worker started")
    while True:
        item = await mega_cmd_queue.get()
        cmd_args, fut, timeout = item
        try:
            rc, out, err = await run_mega_cmd_raw(cmd_args, timeout=timeout)
            if rc != 0:
                fut.set_exception(RuntimeError(f"mega cmd failed rc={rc}, err={err.strip()}"))
            else:
                fut.set_result(out)
        except Exception as e:
            if not fut.done():
                fut.set_exception(e)
        finally:
            mega_cmd_queue.task_done()

async def enqueue_mega(cmd_args: list[str], timeout: int = 120) -> str:
    """Поместить команду в очередь и дождаться результата (stdout) или исключения."""
    fut = asyncio.get_event_loop().create_future()
    await mega_cmd_queue.put((cmd_args, fut, timeout))
    return await fut

# high-level обертки --------------------------------------------------------
async def mega_login():
    """Войти в Mega через CLI (использует MEGA_EMAIL/MEGA_PASSWORD)."""
    if not MEGA_EMAIL or not MEGA_PASSWORD:
        logger.warning("MEGA_EMAIL/MEGA_PASSWORD not set — пропускаем логин")
        return
    try:
        out = await enqueue_mega(["login", MEGA_EMAIL, MEGA_PASSWORD], timeout=60)
        logger.info("Mega login result: %s", out.strip())
    except Exception as e:
        logger.error("Mega login failed: %s", e)
        # не падаем — приложение всё равно должно стартовать

async def mega_ls_all() -> str:
    """Получить рекурсивный список файлов (text)."""
    # -R рекурсивно; может быть медленно для большого аккаунта
    return await enqueue_mega(["ls", "-R", "/"], timeout=120)

def parse_ls_for_paths(ls_text: str) -> list[str]:
    """
    Простейший парсер вывода `mega ls -R /`.
    Он ищет строки, похожие на пути и имена.
    Формат `mega-ls` может отличаться между версиями; этот парсер пытается извлечь имена файлов.
    Возвращает список строк (пути вида /path/to/file).
    """
    lines = ls_text.splitlines()
    paths = []
    current_dir = ""
    for line in lines:
        line = line.rstrip()
        if not line:
            continue
        # директория: заканчивается на ':' e.g. /some/folder:
        if line.endswith(":"):
            current_dir = line[:-1]
            if not current_dir.startswith("/"):
                current_dir = "/" + current_dir.lstrip("/")
            continue
        # строка файла/папки — берем имя первого слова (обычно)
        # Простейшая логика: имя — последнее слово
        parts = line.split()
        if not parts:
            continue
        name = parts[-1]
        # игнорируем служебные строки
        if name in (".", ".."):
            continue
        full = current_dir.rstrip("/") + "/" + name if current_dir else "/" + name
        paths.append(full)
    return paths

async def mega_ls_find(name: str) -> str | None:
    """
    Найти путь к файлу с точным именем `name`. Возвращает первый найденный remote-path (например /crash_23k.json).
    Если не найден — None.
    """
    try:
        out = await mega_ls_all()
    except Exception as e:
        logger.error("mega_ls_all failed: %s", e)
        return None
    paths = parse_ls_for_paths(out)
    # Ищем точное совпадение по имени (последний сегмент)
    for p in paths:
        if p.rstrip("/").split("/")[-1] == name:
            logger.debug("Found file %s -> %s", name, p)
            return p
    logger.debug("File %s not found in parsed ls output", name)
    return None

async def mega_put(local_path: str, remote_path: str | None = None):
    """
    Загрузить local_path на Mega.
    Если remote_path is None — загрузит в корень (оставит имя файла).
    Если remote_path задан — используется как путь /path/on/mega (если папки нет — mega создаст).
    """
    if remote_path:
        args = ["put", local_path, remote_path]
    else:
        args = ["put", local_path]
    return await enqueue_mega(args, timeout=600)

async def mega_get(remote_path: str, local_path: str):
    """Скачать файл remote_path (например /crash_23k.json) в local_path."""
    return await enqueue_mega(["get", remote_path, local_path], timeout=600)

async def mega_rm(remote_path: str):
    """Удалить файл/путь на Mega."""
    return await enqueue_mega(["rm", remote_path], timeout=120)

async def mega_mkdir(remote_path: str):
    return await enqueue_mega(["mkdir", remote_path], timeout=60)

async def mega_mv(src: str, dst: str):
    return await enqueue_mega(["mv", src, dst], timeout=120)

async def mega_emptytrash():
    return await enqueue_mega(["trash", "empty"], timeout=60)

# ====================== Backup logic (uses above wrappers) ======================
BACKUP_NAME = "assistant_backup.json"
OLD_BACKUP_NAME = "assistant_backup_old.json"

async def save_backup():
    try:
        logger.info("Saving backup...")
        # найти текущий бэкап
        existing = await mega_ls_find(BACKUP_NAME)
        if existing:
            # переместить текущий бэкап (пометить старым: mv /assistant_backup.json /assistant_backup_old.json)
            # если уже есть OLD — удалить OLD (переместим, затем удалим OLD из корзины)
            old_existing = await mega_ls_find(OLD_BACKUP_NAME)
            if old_existing:
                # удаляем старый (поместится в корзину)
                try:
                    await mega_rm(old_existing)
                    logger.info("Removed previous old backup %s", old_existing)
                except Exception as e:
                    logger.warning("Failed to remove previous old backup: %s", e)
            try:
                await mega_mv(existing, "/" + OLD_BACKUP_NAME)
                logger.info("Renamed current backup %s -> %s", existing, OLD_BACKUP_NAME)
            except Exception as e:
                logger.warning("Failed to mv existing backup: %s", e)

        # сохраняем текущий state в локальный файл
        with open(BACKUP_NAME, "w") as f:
            json.dump(assistant.export_state(), f)

        # загружаем новый бэкап
        # загружаем в корень: mega put assistant_backup.json /
        await mega_put(BACKUP_NAME, "/" + BACKUP_NAME)
        logger.info("Uploaded %s to Mega", BACKUP_NAME)

        # очистка корзины (необязательная): оставим в корзине OLD; если хочешь полностью стирать — раскомментируй:
        # await mega_emptytrash()

        logger.info("Backup updated successfully")
    except Exception as e:
        logger.error(f"Backup failed: {e}")

async def restore_backup():
    try:
        logger.info("Restoring backup if exists...")
        remote = await mega_ls_find(BACKUP_NAME)
        if remote:
            await mega_get(remote, BACKUP_NAME)
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
        try:
            await save_backup()
        except Exception as e:
            logger.error("save_backup_loop error: %s", e)
        await asyncio.sleep(3600)  # строго раз в час

# ====================== History Load ======================
CRASH_HISTORY_FILES = ["crash_23k.json"]

async def load_history_files(files=CRASH_HISTORY_FILES):
    await asyncio.sleep(0.1)  # небольшой сдвиг, чтобы worker стартовал
    for filename in files:
        logger.info(f"Processing history file: {filename}")
        remote = await mega_ls_find(filename)
        if not remote:
            logger.warning(f"File {filename} not found in Mega")
            continue
        try:
            # скачиваем и обрабатываем локально
            await mega_get(remote, filename)
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
    global mega_worker_task
    # старт worker'а очереди
    if mega_worker_task is None:
        mega_worker_task = asyncio.create_task(mega_worker())

    # логинимся (через очередь), затем пытаемся восстановить стейт и загрузить истории
    await mega_login()
    await restore_backup()
    await load_history_files()

    # запуски периодических задач
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