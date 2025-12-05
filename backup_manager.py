import asyncio
import json
import logging
import os
import time
from typing import Any, Dict

from config import BACKUP_FOLDER, FULL_BACKUP_FILE, BACKUP_INTERVAL_SECONDS
from utils import yandex_upload, ensure_dir, yandex_find, yandex_remove

logger = logging.getLogger("ai_assistant.backup")
logger.setLevel(logging.INFO)

class BackupManager:
    def __init__(self, assistant_modules: Dict[str, Any]):
        self.modules = assistant_modules
        self.local_backup_path = os.path.join(BACKUP_FOLDER, FULL_BACKUP_FILE)
        self.remote_backup_paths = {
            "current": "/" + FULL_BACKUP_FILE,
            "old": "/" + FULL_BACKUP_FILE + ".old",
            "old_old": "/" + FULL_BACKUP_FILE + ".old.old"
        }
        self._lock = asyncio.Lock()
        self._backup_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._last_backup_time = 0

    async def start_worker(self):
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._backup_loop())
            logger.info("Backup worker started")

    async def queue_backup(self):
        await self._backup_queue.put(True)
        logger.info("Backup queued")

    async def _backup_loop(self):
        while True:
            await self._backup_queue.get()
            try:
                await self._wait_ready_for_backup()
                now = time.time()
                if now - self._last_backup_time >= BACKUP_INTERVAL_SECONDS:
                    await self._perform_backup()
                    self._last_backup_time = now
                else:
                    logger.info("Backup skipped: period not reached yet")
            except Exception as e:
                logger.exception("Backup worker failed: %s", e)
            finally:
                self._backup_queue.task_done()

    async def _wait_ready_for_backup(self, timeout: int = 60):
        start_time = time.time()
        while True:
            all_ready = all(getattr(module, "ready_for_backup", True) for module in self.modules.values())
            if all_ready:
                break
            if time.time() - start_time > timeout:
                logger.warning("Timeout waiting for modules to be ready, forcing backup")
                break
            await asyncio.sleep(1)

    async def _perform_backup(self):
        async with self._lock:
            ensure_dir(BACKUP_FOLDER)
            state = {}
            for name, module in self.modules.items():
                if hasattr(module, "export_state"):
                    try:
                        state[name] = module.export_state()
                        logger.info("Module '%s' state exported", name)
                    except Exception as e:
                        logger.exception("Failed to export state for %s: %s", name, e)
            state["_meta"] = {"timestamp": time.time()}

            # Сохраняем локально
            try:
                with open(self.local_backup_path, "w", encoding="utf-8") as f:
                    json.dump(state, f, ensure_ascii=False, indent=2)
                logger.info("Backup saved locally: %s", self.local_backup_path)
            except Exception as e:
                logger.exception("Failed to save local backup: %s", e)

            # Ротация и загрузка на Яндекс.Диск
            try:
                # 1. Удаляем самый старый, если существует
                old_old_remote = self.remote_backup_paths["old_old"]
                if await yandex_find(old_old_remote):
                    await yandex_delete(old_old_remote)
                    logger.info("Deleted old_old backup: %s", old_old_remote)

                # 2. Переименовываем backup.old в backup.old.old
                old_remote = self.remote_backup_paths["old"]
                if await yandex_find(old_remote):
                    await yandex_upload(self.local_backup_path, self.remote_backup_paths["old_old"], overwrite=True)
                    await yandex_delete(old_remote)
                    logger.info("Rotated backup.old -> backup.old.old")

                # 3. Переименовываем текущий backup в backup.old
                current_remote = self.remote_backup_paths["current"]
                if await yandex_find(current_remote):
                    await yandex_upload(self.local_backup_path, old_remote, overwrite=True)
                    await yandex_delete(current_remote)
                    logger.info("Rotated backup -> backup.old")

                # 4. Загружаем новый backup
                await yandex_upload(self.local_backup_path, current_remote, overwrite=True)
                logger.info("Uploaded new backup to Yandex.Disk: %s", current_remote)

            except Exception as e:
                logger.exception("Failed to rotate/upload backup on Yandex.Disk: %s", e)

    async def restore_backup(self):
        async with self._lock:
            try:
                remote_file = await yandex_find(self.remote_backup_paths["current"])
                if remote_file:
                    await yandex_download_to_file(remote_file, self.local_backup_path)
                if not os.path.exists(self.local_backup_path):
                    logger.warning("No local backup found at %s", self.local_backup_path)
                    return
                with open(self.local_backup_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                for name, module in self.modules.items():
                    mod_state = state.get(name)
                    if mod_state and hasattr(module, "load_state"):
                        try:
                            module.load_state(mod_state)
                            logger.info("Module '%s' state restored", name)
                        except Exception as e:
                            logger.exception("Failed to load state for %s: %s", name, e)
                logger.info("Backup restored successfully")
            except Exception as e:
                logger.exception("Failed to restore backup: %s", e)