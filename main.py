from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
import threading
import time
import requests
from model import AIAssistant

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai_assistant")

PORT = int(os.getenv("PORT", 8000))
SELF_URL = os.getenv("SELF_URL")

app = FastAPI(title="Crash AI Assistant")
assistant = AIAssistant()

# ===== KEEP-ALIVE =====
def keep_alive():
    if not SELF_URL:
        logger.warning("SELF_URL не задан, keep-alive не будет работать")
        return
    while True:
        try:
            resp = requests.get(f"{SELF_URL}/healthz", timeout=5)
            logger.info(f"Keep-alive ping OK: {resp.status_code}")
        except Exception as e:
            logger.warning(f"Keep-alive error: {e}")
        time.sleep(240 + 120 * np.random.rand())

threading.Thread(target=keep_alive, daemon=True).start()

# ===== Pydantic Models =====
class BetsPayload(BaseModel):
    game_id: int
    bets: list
    num_players: int | None = None
    deposit_sum: float | None = None
    meta: dict | None = None
    color_bucket: str | None = None

class FeedbackPayload(BaseModel):
    game_id: int
    crash: float
    bets: list | None = None
    deposit_sum: float | None = None
    num_players: int | None = None
    color_bucket: str | None = None

class LoadGamesPayload(BaseModel):
    url: str

# ===== ENDPOINTS =====
@app.post("/predict", status_code=204)
async def predict(payload: BetsPayload):
    try:
        assistant.predict_and_log(payload.model_dump())
    except Exception as e:
        logger.exception("Ошибка в /predict")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def feedback(payload: FeedbackPayload):
    try:
        fast_game = False
        if payload.bets and payload.crash:
            fast_game = True

        assistant.process_feedback(
            game_id=payload.game_id,
            crash=payload.crash,
            bets=payload.bets,
            deposit_sum=payload.deposit_sum,
            num_players=payload.num_players,
            fast_game=fast_game,
            color_bucket=payload.color_bucket
        )

        if fast_game:
            logger.info(f"Быстрая игра {payload.game_id}, визуальный предикт пропущен")
            return {"status": "ok", "fast_game": True}

        return {"status": "ok"}

    except Exception as e:
        logger.exception("Ошибка в /feedback")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.post("/load_games")
async def load_games(payload: LoadGamesPayload):
    try:
        resp = requests.get(payload.url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        for game in data:
            if "bets" in game and isinstance(game["bets"], str):
                try:
                    import json
                    game["bets"] = json.loads(game["bets"])
                except Exception as e:
                    logger.warning(f"Не удалось распарсить bets для game_id {game.get('game_id')}: {e}")
                    game["bets"] = []

        assistant.load_history_from_list(data)
        logger.info(f"Игры загружены! Всего в истории: {assistant.history_df.shape[0]}")
        return {"status": "ok", "games_loaded": assistant.history_df.shape[0]}

    except Exception as e:
        logger.exception("Ошибка при загрузке игр")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/logs")
async def get_logs(limit: int = 20):
    return {"logs": assistant.get_pred_log(limit=limit)}

# ===== RUN =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)