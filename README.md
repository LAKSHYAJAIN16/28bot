# 28 (Twenty-Eight) ISMCTS Bot

The only way to beat my grandfather at 28 (an indian card game) was to create a strong, modular AI built around an incomplete monte carlo search tree.

### Highlights
- Conservative, human-like bidding with two-stage suit evaluation and multiprocessing.
- Precomputation pipeline that precomputes the ideal bids for certain hands (canonization that takes total permutations down to just ~2k)
- Enhanced Monte Carlo Search Tree that is void-aware (keeps track of the cards not played yet), and is information imperfect (AI does not know the cards that the other players have)
- Optional long-horizon “long_bias” search mode for better trump timing and resource conservation, reward structure designed to mimic human play.
- Detailed Logs + JSONL summaries, explanations for bid actions.
---


## Project Structure
### Internal Components
- `mcts/constants.py` – Suits, ranks, points, card utilities;
- `mcts/env28.py` – Game environment (auction, play phases, rules, scoring, stakes); belief tracking
- `mcts/mcts_core.py` – Core Monte Carlo functionality, components of tree
- `mcts/ismcts.py` – Better MCTS (void-aware, better trump timing)
- `mcts/mcts_bidding.py` – MonteCarloBiddingAgent (two-stage, parallel suit eval; uses precomputed cache)
- `mcts/policy.py` – In-game move policy, wrapper around core libraries for ease of use
- `mcts/runner.py` – Run single/multiple games
- `mcts/log_utils.py` – Logger utilities.

### Directly Accessible
- `bet_advisor.py` – Bidding logic; input hand and it will give you the optimial bid
- `precompute_bets.py` – Precomputation code to generate the optimial bids for the hand.
- `run_game.py` – Entry-point to simulate games with a simple config block
- `precomputed_bets.jsonl` – Append-only store of suit stats and recommendations per canonical hand
---

## Running Games
Edit `run_game.py`:
- `NUM_GAMES`: number of games to simulate
- `ITERATIONS`: MCTS iterations per move
- `CONCURRENT`: run games in parallel processes
- `MAX_WORKERS`: cap process count (None = auto)
- `SEARCH_MODE`: `"regular"` (default behavior) or `"long_bias"` (more long-term bias)

When run, you’ll see per-game console output tee’d to log files and a final summary. Concurrency uses `ProcessPoolExecutor` for performance and safety.
---

## Logs and Where to Find Them
- Per-game logs: `logs/game28/mcts_games/game_<id>_<timestamp>.log`
  - Contains everything that prints to the console for that game
- Structured JSONL events: appended to `training.log` (project root)
  - Records `game_start` and `game_end` metadata (stakes, bidder, trump, etc.)
---

## Bidding (MonteCarloBiddingAgent)

Designed for accuracy and speed:

- Stage 1 (quick screen)
  - For each suit in the 4-card hand, run a few partial ISMCTS simulations in parallel
  - Compute the 30th percentile (p30) and mean performance.

- Stage 2 (heavy eval)
  - Keep the top suits from Stage 1
  - Run more samples/iterations with full playouts
  - Record per-suit stats: avg, p30, std

- Decision rule
  - Require: p30 ≥ (min_allowed − 1) AND (avg − 0.5·std) ≥ (min_allowed − 1)
  - Proposed bid = floor(max(p30, avg − 1)), clipped to [min_allowed, 28]
  - Stakes double at ≥ 20 (environment enforces this)

- Precomputed shortcut
  - Looks up `precomputed_bets.jsonl` via a suit-canonical key and, if present, uses those stats directly
  - Gameplay and the precompute script both enrich this file over time

- Trump choice
  - After winning the auction, pick the suit that is estimated to yield the most points
---

## Gameplay
- Determinization
  - Data used by the Monte Carlo Search
    - Known voids (`void_suits_by_player`) observed during play (recognizing what suits players are most likely to 'trump')
    - Belief bias from `lead_suit_counts` (players are more likely long in suits their team has led)
  - Long-bias variant leads to more conservative play, keeping points till the end.

- Propagation through the tree : `void_suits_by_player` and `lead_suit_counts` are copied into envs created during expansion and rollout so the search consistently uses inferred information

- Priors and rollout
  - Regular mode: weighted mix of rank strength, card points, and immediate trick win prob
  - Long-bias mode: adds bias to develop long non-trump suits, preserve trump early, and avoid exposing trump for low-value tables
  - Heuristic rollout favors dumping points when partner is winning; only trumps off-suit when points justify exposure

Parameters are accessible via `run_game.py` (`ITERATIONS`) and `mcts/policy.py` (ISMCTS samples per move). The long-bias mode can be toggled globally via `SEARCH_MODE`.

---

## Precomputation (`precompute_bets.py`)
- Canonicalization: hands that differ only by a suit relabel are equivalent; we evaluate a single representative per S4 class
  - Cuts work by ~15–20× in practice (max 24×). C(32,4) ≈ 36k distinct sets shrink to ~2k canonical keys
- Append-only: loads existing JSONL and skips already-computed hands
- Two-stage evaluation: mirrors in-game pipeline with heavier budgets for top suits
- Progress debug: prints visited count, unique totals, dedup ratio, elapsed time, and rates

---

## Performance Tips
- Increase `ITERATIONS` for stronger play (diminishing returns apply)
- Use `CONCURRENT=True` in `run_game.py` to run many games in parallel
- Keep `precomputed_bets.jsonl` growing (precompute script + runtime enrichment) to accelerate bidding
- Choose `SEARCH_MODE="long_bias"` for improved trump timing and resource conservation
- For offline precomputation, run on multicore machines; consider splitting the combo space manually if needed

---

## Extending / Other Games
The engine layout is reusable for related trick-taking games: 29, Court Piece (Rang), Tarneeb/Hokm, Euchre, Call Break, Spades, Hearts, Oh Hell, and more. Bridge/Skat/Pinochle are longer-term targets that benefit from the same ISMCTS/belief pipeline.

---

## Troubleshooting
- Empty or tiny `precomputed_bets.jsonl`? Run `precompute_bets.py` and simulate games to enrich
- Logs not written? Ensure `logs/game28/mcts_games` exists (it’s auto-created) and the process has write permissions
- Performance slow? Lower `ITERATIONS`, reduce Stage 2 budgets, or enable precomputed shortcut

---

## Notes
This codebase currently uses handcrafted priors/rollouts. It is structured to be upgraded to a learned policy+value network guiding ISMCTS (with batching/caching) without large refactors.