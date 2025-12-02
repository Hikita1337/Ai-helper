def process_feedback(self, game_id, crash, bets=None, deposit_sum=None, num_players=None, fast_game=False):
    # Обновляем историю и веса всегда
    self.games_index.add(game_id)
    self.crash_values.append(float(crash))
    row = {
        "game_id": game_id,
        "crash": float(crash),
        "bets": bets or [],
        "deposit_sum": deposit_sum,
        "num_players": num_players,
        "color_bucket": None,
        "fast_game": fast_game
    }

    # Обновляем статистику пользователей
    for b in row["bets"]:
        uid = b.get("user_id")
        if uid is not None:
            self.user_counts[uid] += 1

    self.history_df = pd.concat([self.history_df, pd.DataFrame([row])], ignore_index=True)

    if fast_game:
        logger.info(f"Быстрая игра {game_id} обработана без визуального предикта")
    else:
        logger.info(f"Feedback: добавлена игра {game_id}, crash={crash}. Всего игр: {len(self.crash_values)}")