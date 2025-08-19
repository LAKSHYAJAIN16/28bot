#!/usr/bin/env python3
"""
Run MCTS games with configurable settings.

Edit the CONFIG section below and run:
  python run_mcts_games.py

Logs:
  - Per-game logs: logs/game28/mcts_games/
  - JSONL summaries: training.log (project root)
"""

from __future__ import annotations

# ===== CONFIG =====
NUM_GAMES: int = 100           # how many games to play
ITERATIONS: int = 10000        # MCTS iterations per move
CONCURRENT: bool = False      # run games in parallel using multiple processes
MAX_WORKERS: int | None = None  # None => auto (cpu_count)


def main() -> None:
    if CONCURRENT:
        from mcts.runner import play_games_concurrent
        print(f"Running {NUM_GAMES} games concurrently (iterations={ITERATIONS}, max_workers={MAX_WORKERS})...")
        play_games_concurrent(num_games=NUM_GAMES, iterations=ITERATIONS, max_workers=MAX_WORKERS)
    else:
        from mcts.runner import play_games
        print(f"Running {NUM_GAMES} games (iterations={ITERATIONS})...")
        play_games(num_games=NUM_GAMES, iterations=ITERATIONS)


if __name__ == "__main__":
    main()


