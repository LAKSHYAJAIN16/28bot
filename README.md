# 28 (Twenty-Eight) ISMCTS Bot

The only way to beat my grandfather at 28 (an indian card game) was to create a strong, modular AI built around an incomplete monte carlo search tree.

### Highlights
- Conservative, human-like bidding with two-stage suit evaluation and multiprocessing.
- Precomputation pipeline that precomputes the ideal bids for certain hands (canonization that takes total permutations down to just ~2k)
- Enhanced Monte Carlo Search Tree that is void-aware (keeps track of the cards not played yet), and is information imperfect (AI does not know the cards that the other players have)
- Optional long-horizon “long_bias” search mode for better trump timing and resource conservation, reward structure designed to mimic human play.
- Detailed Logs + JSONL summaries, explanations for bid actions.
---

## Quickstart

1) Environment
- Python 3.10+
- Recommended: create a virtualenv

2) Install packages (minimal)
- Pure-Python core; optional PyTorch if you later add neural nets

3) Run games
```bash
python run_game.py
```
Edit the CONFIG block at the top to set number of games, MCTS iterations, concurrency, and search mode.

4) Bet advisor (interactive)
```bash
python bet_advisor.py
```
Enter your 4 cards and the current high bid; it will recommend a bid and trump with detailed analysis.

5) Precompute bids (append-only)
```bash
python precompute_bets.py
```
This will append new canonical 4-card hands to `precomputed_bets.jsonl`, skipping any already present.

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
- `bet_advisor.py` – Interactive advisor; input hand and it will give you the optimial bid
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

Pipeline designed for accuracy and speed:

- Stage 1 (quick screening + filtering)
  - For each suit present in the 4-card hand, run a few partial simulations parallely, using `multithreading`
  - Compute 30th percentile (p30) performance along with mean performance for better evaluation. 

- Stage 2 (heavy evaluation)
  - Select top suits based on results from quick screening.
  - Use more samples/iterations and full playouts
  - Compute per-suit stats: avg points, bottom 30 performance, standard deviation.

- Decision rule
  - Thresholds: require p30 >= (min_allowed − 1) AND (avg − 0.5·std) >= (min_allowed − 1)
  - Proposed bid = floor(max(p30, avg−1)), clipped to [min_allowed, 28]
  - Stakes doubling at ≥ 20

- Precomputation Dictionary
  - Before simulating, the agent looks up `precomputed_bets.jsonl`
  - If stats are found, it can compute a bid immediately with identical thresholds, skipping simulations
  - All bidding computations are automatically added to the `precomputed_bets.jsonl`

- Trump choice
  - After winning auction, trump is selected based on the suit leading to the most points.
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
- We store/evaluate only one representative per equivalence class (S4 orbit). This cuts work by roughly 15–20× in practice (upper bound 24×), since most hands have no nontrivial symmetries.
- basically C(4,32) goes down to 2k

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