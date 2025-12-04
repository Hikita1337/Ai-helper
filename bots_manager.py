"""
Управление информацией о ботах и активных пользователях:
- хранение в памяти ограниченного набора активных пользователей
- хранение выявленных ботов (стратегии) (в памяти + persisted на Yandex Disk)
- асинхронная загрузка/сохранение файла с ботами на Яндекс.Диск
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

from config import BOTS_FILE, BACKUP_FOLDER, MAX_ACTIVE_USERS, MAX_BOTS_IN_MEMORY, BOT_INACTIVE_DAYS
from utils import yandex_find, yandex_download_to_file, yandex_upload, run_yandex_task, yadisk_client

logger = logging.getLogger("ai_assistant.bots")


class BotsManager:
    def __init__(self):
        # in-memory structures
        # active_users: user_id -> {last_seen_ts: iso, summary_fields...}
        self.active_users: Dict[str, Dict[str, Any]] = {}
        # bots: user_id -> {strategy:..., last_active: iso, metadata...}
        self.bots: Dict[str, Dict[str, Any]] = {}

        # local cache file name
        self.local_cache = "assistant_bots_local.json"
        self.remote_path = BACKUP_FOLDER.rstrip("/") + "/" + BOTS_FILE

        # lock for async safety
        self._lock = asyncio.Lock()

    async def load_from_disk(self):
        """
        Загрузить файл bots с Яндекс.Диска в память (если есть).
        """
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
                logger.info("BotsManager: loaded %d bots, %d active users from disk", len(self.bots), len(self.active_users))
            except Exception as e:
                logger.exception("BotsManager.load_from_disk failed: %s", e)

    async def save_to_disk(self):
        """
        Сохраняем текущие данные ботов/активных пользователей на диск (overwrite).
        """
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

    async def mark_bot(self, user_id: str, info: dict):
        """
        Пометить пользователя как бота и записать стратегию/мета.
        """
        async with self._lock:
            now = datetime.utcnow().isoformat()
            entry = self.bots.get(user_id, {})
            entry.update(info)
            entry.setdefault("first_seen", now)
            entry["last_active"] = now
            entry["confirmed"] = True
            self.bots[user_id] = entry

            # удалить из active_users если есть (мы держим минимальный индекс)
            if user_id in self.active_users:
                self.active_users.pop(user_id, None)

            # trim bots in memory if needed
            if len(self.bots) > MAX_BOTS_IN_MEMORY:
                # простая эвикция по last_active (самые старые)
                sorted_bots = sorted(self.bots.items(), key=lambda kv: kv[1].get("last_active", ""))
                # удаляем старые
                to_remove = [k for k, _ in sorted_bots[:len(self.bots)-MAX_BOTS_IN_MEMORY]]
                for k in to_remove:
                    self.bots.pop(k, None)

    async def unmark_bot(self, user_id: str):
        """
        Снять метку бота (пользователь — не бот).
        """
        async with self._lock:
            if user_id in self.bots:
                self.bots.pop(user_id, None)

    async def touch_active_user(self, user_id: str, info: dict | None = None):
        """
        Обновить метаданные активного пользователя (закладываем лимит по MAX_ACTIVE_USERS).
        info: доп. данные (например: last_bet, avg_auto и т.п.)
        """
        async with self._lock:
            now = datetime.utcnow().isoformat()
            entry = self.active_users.get(user_id, {})
            entry["last_seen"] = now
            if info:
                entry.update(info)
            self.active_users[user_id] = entry

            # enforce size
            if len(self.active_users) > MAX_ACTIVE_USERS:
                # удалить самых старых
                sorted_users = sorted(self.active_users.items(), key=lambda kv: kv[1].get("last_seen", ""))
                to_remove = [k for k, _ in sorted_users[:len(self.active_users)-MAX_ACTIVE_USERS]]
                for k in to_remove:
                    self.active_users.pop(k, None)

    async def prune_inactive_bots(self):
        """
        Пометить бота как неактивного (оставить в списке, но поставить flag),
        если не было активности более BOT_INACTIVE_DAYS.
        """
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
                        # если формат невалидный — пометим неактивным
                        info["active"] = False

    def get_bot(self, user_id: str):
        return self.bots.get(user_id)

    def get_active_user(self, user_id: str):
        return self.active_users.get(user_id)

    def summary(self):
        return {
            "bots_in_memory": len(self.bots),
            "active_users_in_memory": len(self.active_users)
        }
