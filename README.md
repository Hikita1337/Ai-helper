AI Assistant for Crash (FastAPI)
================================

Файлы проекта:
- main.py
- model.py
- requirements.txt

Запуск локально:
1) Установить зависимости:
   python -m pip install -r requirements.txt

2) Поднять сервер (port = 8000 по умолчанию):
   uvicorn main:app --host 0.0.0.0 --port 8000

Запуск на Render:
- Указать Start Command:
    uvicorn main:app --host 0.0.0.0 --port $PORT
- Опционально задать ENV:
    GAMES_FILE=/path/to/games.json
    PERSIST_ON_UPDATE=true

API:
1) POST /predict  (payload JSON)
   - Пример тела:
     {
       "game_id": 6234000,
       "num_players": 21,
       "deposit_sum": 54.12,
       "bets": [
         {"user_id": 123, "name": "foo", "amount": 2.5, "auto": 1.5},
         {"user_id": 456, "name": "botA", "amount": 10.0, "auto": 1.02}
       ],
       "meta": {}
     }
   - Сервер возвращает 204 No Content.
   - Аналитика печатается в лог (консоль) на русском.

2) POST /feedback
   - Пример тела:
     { "game_id": 6234000, "crash": 1.53 }
   - Сервер вернёт { "status": "ok" } и добавит игру в память.

Как интегрировать:
- Парсер посылает snapshot ставок на /predict (как только закрылись ставки).
- После завершения игры парсер посылает /feedback с реальным crash.
- Остальная логика — ассистент ведёт память и логирует прогнозы.
