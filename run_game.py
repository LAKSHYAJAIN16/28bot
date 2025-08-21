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
NUM_GAMES: int = 2000              # how many games to play (reduced for testing)
ITERATIONS: int = 3000            # MCTS iterations per move (increased for better play)
CONCURRENT: bool = False         # run games in parallel using multiple processes
MAX_WORKERS: int | None = None    # None => auto (cpu_count)
SEARCH_MODE: str = "regular"    # "regular" or "long_bias" (using long_bias for better strategy)

# ===== MCTS GAMEPLAY PARAMETERS =====
# These control the quality of gameplay decisions
MCTS_SAMPLES: int = 16           # Number of ISMCTS samples (reduced for stability)
MCTS_ITERS_PER_SAMPLE: int = 70  # Iterations per sample (reduced for stability)
MCTS_C_PUCT: float = 1.0         # Exploration constant (balanced)

# ===== BIDDING PARAMETERS =====
# These control the bidding agent's simulation quality
BIDDING_SAMPLES: int = 8         # Number of samples for bidding (reduced for stability)
BIDDING_ITERATIONS: int = 100    # MCTS iterations for bidding (reduced for stability)
BIDDING_STAGE2_ITERATIONS: int = 200  # Heavy evaluation iterations (reduced for stability)

# ===== ROLLOUT PARAMETERS =====
# These control the ISMCTS rollout quality
ROLLOUT_ITERATIONS: int = 100      # Iterations for ISMCTS rollout evaluation (reduced for stability)


def main() -> None:
    # Create a config dict with all the MCTS parameters
    mcts_config = {
        'mcts_samples': MCTS_SAMPLES,
        'mcts_iters_per_sample': MCTS_ITERS_PER_SAMPLE,
        'mcts_c_puct': MCTS_C_PUCT,
        'bidding_samples': BIDDING_SAMPLES,
        'bidding_iterations': BIDDING_ITERATIONS,
        'bidding_stage2_iterations': BIDDING_STAGE2_ITERATIONS,
        'rollout_iterations': ROLLOUT_ITERATIONS,
    }
    
    if CONCURRENT:
        from mcts.runner import play_games_concurrent
        print(f"Running {NUM_GAMES} games concurrently (iterations={ITERATIONS}, max_workers={MAX_WORKERS}, mode={SEARCH_MODE})...")
        print(f"MCTS Config: {mcts_config}")
        play_games_concurrent(num_games=NUM_GAMES, iterations=ITERATIONS, max_workers=MAX_WORKERS, search_mode=SEARCH_MODE, mcts_config=mcts_config)
    else:
        from mcts.runner import play_games
        print(f"Running {NUM_GAMES} games (iterations={ITERATIONS}, mode={SEARCH_MODE})...")
        print(f"MCTS Config: {mcts_config}")
        play_games(num_games=NUM_GAMES, iterations=ITERATIONS, search_mode=SEARCH_MODE, mcts_config=mcts_config)


if __name__ == "__main__":
    main()


