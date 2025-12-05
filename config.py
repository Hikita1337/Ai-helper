# Конфигурация / константы
import os
from datetime import timedelta

PORT = int(os.getenv("PORT", 8000))
SELF_URL = os.getenv("SELF_URL")
ABLY_API_KEY = os.getenv("ABLY_API_KEY")
YANDEX_ACCESS_TOKEN = os.getenv("YANDEX_ACCESS_TOKEN")

# Файлы/пути на Yandex Disk
BACKUP_NAME = "assistant_backup.json"
BACKUP_FOLDER = "/"  # корень, можно сменить
BOTS_FILE = "assistant_bots.json"         # хранит: bots + active_users метаданные
CRASH_HISTORY_FILES = ["crash_23k.json"]  # список имен файлов для первичной загрузки
FULL_BACKUP_FILE = os.path.join(BACKUP_FOLDER, "full_backup.json")

# Параметры обработки
BLOCK_RECORDS = 7000          # сколько записей обрабатывать за один "батч"
DOWNLOAD_CHUNK = 1024 * 1024  # 1MB chunks for download
MAX_ACTIVE_USERS = 5000       # сколько держать в памяти активных пользователей
MAX_BOTS_IN_MEMORY = 1000     # сколько ботов держать в памяти
BOT_INACTIVE_DAYS = 3         # через сколько дней помечать бота неактивным
BACKUP_INTERVAL_SECONDS = 3600  # как часто делать бэкап состояния

# Ограничения логирования/метрик
PRED_LOG_LEN = 5000