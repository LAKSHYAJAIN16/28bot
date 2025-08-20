# 28 (Twenty-Eight) Bot

### AI designed to beat my grandfather at 28 (it was the only way). 28bot is a strong, modular AI built that uses an incomplete monte carlo search tree and a neural network.

### Highlights
- Conservative, human-like bidding with accurate two-stage evaluation and multiprocessing.
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
- `SEARCH_MODE`: `"regular"` (default behavior) or  `"long_bias"` (more long-term bias)
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

## NNEt Training (optional)

- What it is
  - `mcts/train_mcts_nnet.py` sketches a self-play training loop for a policy+value model to guide ISMCTS.
  - Self-play supplies targets: visit-count policies (for bidding/play) and final value (stakes applied).

- How to run (example)
  - `python -m mcts.train_mcts_nnet` or `python mcts/train_mcts_nnet.py`
  - Configure epochs, batch size, device (GPU if available), AMP, and optional `torch.compile` inside the script.

- Outputs and logs
  - Checkpoints like `pv_cycle_*.pt`
  - Metrics in `training.log`; you can wire TensorBoard if desired

- Integration
  - Swap handcrafted priors/rollouts for network priors/values in `mcts_core` once trained (code is structured for drop-in).

---

## Precomputation (`precompute_bets.py`)
- Canonicalization: hands that differ only by a suit relabel are equivalent; we evaluate a single representative per S4 class
  - Cuts work by ~15–20× in practice (max 24×). C(32,4) ≈ 36k distinct sets shrink to ~2k canonical keys
- Append-only: loads existing JSONL and skips already-computed hands
- Two-stage evaluation: mirrors in-game pipeline with heavier budgets for top suits
- Progress debug: prints visited count, unique totals, dedup ratio, elapsed time, and rates
