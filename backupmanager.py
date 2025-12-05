"""
BackupManager.py

Отвечает за полный бэкап состояния AI-помощника:
- сохранение и восстановление состояния всех модулей (модель, аналитика, боты)
- работа с локальными файлами и Яндекс.Диском
- асинхронная очередь, чтобы основной процесс не блокировался
- контролируемый таймер для выполнения бэкапа раз в час или по запросу
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict

from config import BACKUP_FOLDER, FULL_BACKUP_FILE, BACKUP_PERIOD_SECONDS
from utils import yandex_upload, yandex_download_to_file, ensure_dir, yandex_find

logger = logging.getLogger("ai_assistant.backup")
logger.setLevel(logging.INFO)

class BackupManager:
    def __init__(self, assistant_modules: Dict[str, Any]):
        """
        assistant_modules: словарь вида {"assistant": модель, "analytics": аналитика, "bots": менеджер_ботов}
        """
        self.modules = assistant_modules
        self.local_backup_path = os.path.join(BACKUP_FOLDER, FULL_BACKUP_FILE)
        self.remote_backup_path = "/" + FULL_BACKUP_FILE
        self._lock = asyncio.Lock()
        self._backup_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._last_backup_time = 0

    async def start_worker(self):
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._backup_loop())
            logger.info("Backup worker started")

    async def queue_backup(self):
        """
        Запланировать бэкап. Реально выполнится только при соблюдении таймера.
        """
        await self._backup_queue.put(True)
        logger.info("Backup queued")

    async def _backup_loop(self):
        """
        Основной цикл бэкапа: проверяет таймер и готовность модулей перед сохранением.
        """
        while True:
            await self._backup_queue.get()
            try:
                await self._wait_ready_for_backup()
                now = time.time()
                if now - self._last_backup_time >= BACKUP_PERIOD_SECONDS:
                    await self._perform_backup()
                    self._last_backup_time = now
                else:
                    logger.info("Backup skipped: period not reached yet")
            except Exception as e:
                logger.exception("Backup worker failed: %s", e)
            finally:
                self._backup_queue.task_done()

    async def _wait_ready_for_backup(self, timeout: int = 60):
        """
        Ждём, пока все модули сигнализируют, что можно делать бэкап.
        Если модуль не имеет атрибута `ready_for_backup`, считаем его готовым.
        """
        start_time = time.time()
        while True:
            all_ready = True
            for module in self.modules.values():
                ready = getattr(module, "ready_for_backup", True)
                if not ready:
                    all_ready = False
                    break
            if all_ready:
                break
            if time.time() - start_time > timeout:
                logger.warning("Timeout waiting for modules to be ready for backup, forcing backup")
                break
            await asyncio.sleep(1)

    async def _perform_backup(self):
        """
        Выполняет фактический бэкап: экспорт состояния модулей, сохранение локально и на Яндекс.Диск.
        """
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

            # Загружаем на Яндекс.Диск
            try:
                remote_file = await yandex_find(FULL_BACKUP_FILE)
                if not remote_file:
                    remote_file = self.remote_backup_path
                await yandex_upload(self.local_backup_path, remote_file, overwrite=True)
                logger.info("Backup uploaded to Yandex.Disk: %s", remote_file)
            except Exception as e:
                logger.exception("Failed to upload backup to Yandex.Disk: %s", e)

    async def restore_backup(self):
        """
        Восстанавливает полное состояние всех модулей из бэкапа (локально или с Яндекс.Диска).
        """
        async with self._lock:
            try:
                remote_file = await yandex_find(FULL_BACKUP_FILE)
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