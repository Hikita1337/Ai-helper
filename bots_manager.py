"""
Управление информацией о ботах и активных пользователях:
- хранение в памяти ограниченного набора активных пользователей
- хранение выявленных ботов (стратегии) (в памяти + persisted на Yandex Disk)
- поддержка подсчёта чистого выигрыша и анализа коэффициентов
- полное сохранение состояния для BackupManager
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
from config import BOTS_FILE, BACKUP_FOLDER, MAX_ACTIVE_USERS, MAX_BOTS_IN_MEMORY, BOT_INACTIVE_DAYS
from utils import yandex_find, yandex_download_to_file, yandex_upload, calculate_net_win

logger = logging.getLogger("ai_assistant.bots")


class BotsManager:
    def __init__(self):
        self.active_users: Dict[str, Dict[str, Any]] = {}
        self.bots: Dict[str, Dict[str, Any]] = {}

        self.local_cache = "assistant_bots_local.json"
        self.remote_path = BACKUP_FOLDER.rstrip("/") + "/" + BOTS_FILE

        self._lock = asyncio.Lock()
        self.ready_for_backup = True  # для BackupManager

        # BackupManager
        self.backup_manager = None

    # -------------------- Подключение BackupManager --------------------
    def attach_backup_manager(self, manager):
        self.backup_manager = manager

    async def queue_backup(self):
        if self.backup_manager:
            await self.backup_manager.queue_backup()

    # -------------------- Загрузка и сохранение --------------------
    async def load_from_disk(self):
        async with self._lock:
            try:
                remote = await yandex_find(BOTS_FILE)
                if not remote:
                    logger.info("No bots file on disk (%s). Starting fresh.", BOTS_FILE)
                    return
                await yandex_download_to_file(remote, self.local_cache)
                with open(self.local_cache, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                self.bots = payload.get("bots", {})
                self.active_users = payload.get("active_users", {})
                logger.info("BotsManager: loaded %d bots, %d active users from disk",
                            len(self.bots), len(self.active_users))
            except Exception as e:
                logger.exception("BotsManager.load_from_disk failed: %s", e)

    async def save_to_disk(self):
        async with self._lock:
            try:
                payload = {
                    "bots": self.bots,
                    "active_users": self.active_users,
                    "meta": {
                        "saved_at": datetime.utcnow().isoformat()
                    }
                }
                with open(self.local_cache, "w", encoding="utf-8") as f:
                    json.dump(payload, f)
                await yandex_upload(self.local_cache, self.remote_path, overwrite=True)
                logger.info("BotsManager: saved bots file to %s", self.remote_path)
            except Exception as e:
                logger.exception("BotsManager.save_to_disk failed: %s", e)

    # -------------------- Работа с ботами и активными пользователями --------------------
    async def mark_bot(self, user_id: str, info: dict):
        async with self._lock:
            now = datetime.utcnow().isoformat()
            entry = self.bots.get(user_id, {})
            entry.update(info)
            entry.setdefault("first_seen", now)
            entry["last_active"] = now
            entry["confirmed"] = True
            self.bots[user_id] = entry

            self.active_users.pop(user_id, None)

            if len(self.bots) > MAX_BOTS_IN_MEMORY:
                sorted_bots = sorted(self.bots.items(), key=lambda kv: kv[1].get("last_active", ""))
                to_remove = [k for k, _ in sorted_bots[:len(self.bots)-MAX_BOTS_IN_MEMORY]]
                for k in to_remove:
                    self.bots.pop(k, None)

            await self.queue_backup()

    async def unmark_bot(self, user_id: str):
        async with self._lock:
            self.bots.pop(user_id, None)
            await self.queue_backup()

    async def touch_active_user(self, user_id: str, info: dict | None = None):
        async with self._lock:
            now = datetime.utcnow().isoformat()
            entry = self.active_users.get(user_id, {})
            entry["last_seen"] = now
            if info:
                entry.update(info)
                coef = info.get("coefficient")
                amt = info.get("amount", 0.0)
                entry["net_win"] = calculate_net_win(amt, coef)
            self.active_users[user_id] = entry

            if len(self.active_users) > MAX_ACTIVE_USERS:
                sorted_users = sorted(self.active_users.items(), key=lambda kv: kv[1].get("last_seen", ""))
                to_remove = [k for k, _ in sorted_users[:len(self.active_users)-MAX_ACTIVE_USERS]]
                for k in to_remove:
                    self.active_users.pop(k, None)

            await self.queue_backup()

    async def prune_inactive_bots(self):
        async with self._lock:
            cutoff = datetime.utcnow() - timedelta(days=BOT_INACTIVE_DAYS)
            for uid, info in list(self.bots.items()):
                last = info.get("last_active")
                if last:
                    try:
                        dt = datetime.fromisoformat(last)
                        if dt < cutoff:
                            info["active"] = False
                    except Exception:
                        info["active"] = False

    # -------------------- Получение информации --------------------
    def get_bot(self, user_id: str):
        return self.bots.get(user_id)

    def get_active_user(self, user_id: str):
        return self.active_users.get(user_id)

    def summary(self):
        return {
            "bots_in_memory": len(self.bots),
            "active_users_in_memory": len(self.active_users)
        }

    # -------------------- Полный бэкап состояния --------------------
    def export_state(self) -> dict:
        """Возвращает полное состояние для BackupManager"""
        return {
            "bots": self.bots.copy(),
            "active_users": self.active_users.copy(),
            "meta": {
                "local_cache": self.local_cache,
                "remote_path": self.remote_path
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    def load_state(self, state: dict):
        """Восстанавливает полное состояние из BackupManager"""
        self.bots = state.get("bots", {}).copy()
        self.active_users = state.get("active_users", {}).copy()
        meta = state.get("meta", {})
        self.local_cache = meta.get("local_cache", self.local_cache)
        self.remote_path = meta.get("remote_path", self.remote_path)