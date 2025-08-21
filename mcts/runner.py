import os
import sys
import itertools
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from .env28 import TwentyEightEnv
from .policy import policy_move
from .log_utils import log_event as _log_game_event, Tee as _TeeStdout, open_game_log as _open_game_log
from .constants import SUITS, RANKS, card_suit


LOG_DIR_GAMES = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "logs", "game28", "mcts_games"))

# Precompute sink (append-only JSONL) and in-memory de-dup set using suit-permutation-invariant keys
_PRECOMP_OUT_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "precomputed_bets.jsonl"))
_precomp_seen_keys: set[tuple[str, ...]] | None = None


def _rank_key(card: str) -> int:
    # card like 'JH' or '10D'
    rank = card[:-1]
    return RANKS.index(rank)


def _canonical_key(cards: list[str]) -> tuple[str, ...]:
    # Suit-permutation invariance: map suits by all permutations and pick lexicographically smallest normalized tuple
    best: tuple[str, ...] | None = None
    for perm in itertools.permutations(SUITS):
        suit_map = {s: perm[i] for i, s in enumerate(SUITS)}
        mapped = [c[:-1] + suit_map[c[-1]] for c in cards]
        mapped_sorted = sorted(mapped, key=lambda c: (_rank_key(c), SUITS.index(card_suit(c))))
        tup = tuple(mapped_sorted)
        if best is None or tup < best:
            best = tup
    return best if best is not None else tuple(sorted(cards))


def _ensure_precomp_seen_loaded() -> None:
    global _precomp_seen_keys
    if _precomp_seen_keys is not None:
        return
    _precomp_seen_keys = set()
    try:
        with open(_PRECOMP_OUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    four = obj.get("four_cards") or obj.get("four") or []
                    if four and isinstance(four, list):
                        key = _canonical_key(four)
                        _precomp_seen_keys.add(key)
                except Exception:
                    continue
    except FileNotFoundError:
        # no file yet
        pass


def _maybe_append_precomputed(
    four_cards: list[str],
    bidder_bid: int | None = None,
    trump: str | None = None,
    dbg: dict | None = None,
) -> None:
    _ensure_precomp_seen_loaded()
    assert _precomp_seen_keys is not None
    key = _canonical_key(four_cards)
    if key in _precomp_seen_keys:
        return
    suit_stats: dict[str, dict[str, float]] = {}
    analysis: dict | None = None
    # Fast path: use provided debug from in-game bidder if available
    if isinstance(dbg, dict) and dbg.get("suit_to_points"):
        stp = dbg.get("suit_to_points") or {}
        for s, pts in stp.items():
            n = max(1, len(pts))
            mean = sum(pts)/n
            p30 = sorted(pts)[max(0, min(len(pts)-1, int(0.3*len(pts))))] if pts else 0.0
            var = sum((v-mean)**2 for v in pts)/n if pts else 0.0
            suit_stats[s] = {"avg": mean, "p30": p30, "std": var**0.5}
        analysis = dbg
        bid = bidder_bid
    else:
        # Compute stats and recommendation using advisor's heavy settings at baseline (current_high=0)
        try:
            from bet_advisor import (
                simulate_points_for_suit_ismcts,
                propose_bid_ismcts,
            )
        except Exception:
            return
        present_suits = [s for s in SUITS if any(card_suit(c) == s for c in four_cards)]
        stage1: dict[str, tuple[float, float]] = {}
        for s in present_suits:
            pts = simulate_points_for_suit_ismcts(four_cards, s, my_seat=0, first_player=0,
                                                  num_samples=2, base_iterations=80, playout_tricks=4)
            n = max(1, len(pts))
            p30 = sorted(pts)[max(0, min(len(pts)-1, int(0.3*len(pts))))] if pts else 0.0
            stage1[s] = (sum(pts)/n, p30)
        top = sorted(present_suits, key=lambda s: (stage1[s][1], stage1[s][0]))[-max(1, 2):]
        suit_to_points: dict[str, list[float]] = {}
        for s in present_suits:
            if s in top:
                pts = simulate_points_for_suit_ismcts(four_cards, s, my_seat=0, first_player=0,
                                                      num_samples=6, base_iterations=220)
            else:
                pts = simulate_points_for_suit_ismcts(four_cards, s, my_seat=0, first_player=0,
                                                      num_samples=2, base_iterations=80, playout_tricks=4)
            suit_to_points[s] = pts
            n = max(1, len(pts))
            mean = sum(pts)/n
            p30 = sorted(pts)[max(0, min(len(pts)-1, int(0.3*len(pts))))] if pts else 0.0
            var = sum((v-mean)**2 for v in pts)/n if pts else 0.0
            suit_stats[s] = {"avg": mean, "p30": p30, "std": var**0.5}
        bid, trump, _suit_stats_unused, analysis = propose_bid_ismcts(
            four_cards, current_high_bid=0, num_samples=6, base_iterations=220
        )
    rec = {
        "four_cards": sorted(four_cards, key=lambda c: (card_suit(c), c)),
        "suit_stats": suit_stats,
        "recommendation": {"bid": bidder_bid if bidder_bid is not None else bid, "trump": trump},
        "analysis": analysis,
    }
    try:
        with open(_PRECOMP_OUT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        _precomp_seen_keys.add(key)
    except Exception:
        pass


def play_games(num_games=3, iterations=50, search_mode: str = "regular", mcts_config: dict = None):
    results_all = []
    series_game_score = [0, 0]
    for g in range(1, num_games + 1):
        os.makedirs(LOG_DIR_GAMES, exist_ok=True)
        game_log_path, ts = _open_game_log(g)
        _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
        with open(game_log_path, "a", encoding="utf-8") as _fh:
            sys.stdout = _TeeStdout(_fh, _orig_stdout)
            sys.stderr = _TeeStdout(_fh, _orig_stderr)
            try:
                print(f"\n===== GAME {g} ({ts}) =====")
                first_player = 0
                env = TwentyEightEnv()
                env.debug = True
                state = env.reset(initial_trump=None, first_player=first_player, mcts_config=mcts_config)
                _log_game_event(
                    "game_start",
                    {
                        "game_id": g,
                        "first_player": first_player,
                        "first_four_hands": env.first_four_hands,
                        "bidder": env.bidder,
                        "bid_value": getattr(env, "bid_value", 16),
                        "stakes": getattr(env, "round_stakes", 1),
                        "trump": env.trump_suit,
                        "phase": state.get("phase"),
                        "concealed_trump_card": env.face_down_trump_card,
                        "log_file": os.path.relpath(
                            game_log_path,
                            os.path.normpath(os.path.join(os.path.dirname(__file__), "..")),
                        ),
                        "timestamp": ts,
                    },
                )
                done = False
                for i, hand in enumerate(env.hands):
                    print(f"Player {i} hand : ", hand)
                for i in range(4):
                    print(f"Player {i} (auction 4 cards): {env.first_four_hands[i]}")
                print(
                    f"Auction winner (bidder): Player {env.bidder} with bid {getattr(env,'bid_value',16)}"
                )
                print(f"Bidder sets concealed trump suit: {state['trump']}")
                print(
                    f"Phase: {state['phase']}, bidder concealed card: {env.face_down_trump_card}"
                )
                print("")
                while not done:
                    current_player = state["turn"]
                    try:
                        move = policy_move(env, iterations, search_mode, mcts_config)
                        state, _, done, winner, trick_points = env.step(move)
                        print(f"Player {current_player} plays {move}")
                        if winner is not None:
                            print(f"Player {winner} won the hand: {trick_points} points\n")
                    except ValueError as e:
                        if "Game is already finished" in str(e) or "Game is finished" in str(e) or "No valid moves available" in str(e):
                            print(f"Game finished: {e}")
                            done = True
                            break
                        else:
                            print(f"Error during gameplay: {e}")
                            # Try to continue with a fallback move
                            current_hand = state["hands"][state["turn"]]
                            if current_hand:
                                move = current_hand[0]  # Just play the first card
                                state, _, done, winner, trick_points = env.step(move)
                                print(f"Player {current_player} plays {move} (fallback)")
                                if winner is not None:
                                    print(f"Player {winner} won the hand: {trick_points} points\n")
                            else:
                                print("No cards available, ending game")
                                done = True
                                break
                    if getattr(env, "invalid_round", False):
                        print("Round declared invalid: trump never exposed by end of 7th trick.")
                        break
                print(
                    f"Game {g} final points: Team A={state['scores'][0]}, Team B={state['scores'][1]}"
                )
                series_game_score[0] += state["game_score"][0]
                series_game_score[1] += state["game_score"][1]
                print(
                    f"Series cumulative game score so far: Team A={series_game_score[0]}, Team B={series_game_score[1]}"
                )
                results_all.append(state["scores"])
                _log_game_event(
                    "game_end",
                    {
                        "game_id": g,
                        "scores": state["scores"],
                        "game_score_delta": state["game_score"],
                        "stakes": getattr(env, "round_stakes", 1),
                        "invalid_round": getattr(env, "invalid_round", False),
                        "exposure_trick_index": getattr(env, "exposure_trick_index", None),
                        "last_exposer": getattr(env, "last_exposer", None),
                        "log_file": os.path.relpath(
                            game_log_path,
                            os.path.normpath(os.path.join(os.path.dirname(__file__), "..")),
                        ),
                        "timestamp": ts,
                    },
                )
                print(f"Log saved to: {game_log_path}")
            finally:
                sys.stdout = _orig_stdout
                sys.stderr = _orig_stderr
    print("\nAll game results:", results_all)
    print(
        f"Final series cumulative game score: Team A={series_game_score[0]}, Team B={series_game_score[1]}"
    )


def run_single_game(game_id: int, iterations: int, first_player: int = 0, search_mode: str = "regular"):
    os.makedirs(LOG_DIR_GAMES, exist_ok=True)
    game_log_path, ts = _open_game_log(game_id)
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    with open(game_log_path, "a", encoding="utf-8") as _fh:
        sys.stdout = _TeeStdout(_fh, _orig_stdout)
        sys.stderr = _TeeStdout(_fh, _orig_stderr)
        try:
            print(f"\n===== GAME {game_id} ({ts}) =====")
            env = TwentyEightEnv()
            env.debug = True
            state = env.reset(initial_trump=None, first_player=first_player)
            _log_game_event(
                "game_start",
                {
                    "game_id": game_id,
                    "first_player": first_player,
                    "first_four_hands": env.first_four_hands,
                    "bidder": env.bidder,
                    "bid_value": getattr(env, "bid_value", 16),
                    "stakes": getattr(env, "round_stakes", 1),
                    "trump": env.trump_suit,
                    "phase": state.get("phase"),
                    "concealed_trump_card": env.face_down_trump_card,
                    "log_file": os.path.relpath(
                        game_log_path,
                        os.path.normpath(os.path.join(os.path.dirname(__file__), "..")),
                    ),
                    "timestamp": ts,
                },
            )
            for i, hand in enumerate(env.hands):
                print(f"Player {i} hand : ", hand)
            for i in range(4):
                print(f"Player {i} (auction 4 cards): {env.first_four_hands[i]}")
            print(
                f"Auction winner (bidder): Player {env.bidder} with bid {getattr(env,'bid_value',16)}"
            )
            print(f"Bidder sets concealed trump suit: {state['trump']}")
            print(
                f"Phase: {state['phase']}, bidder concealed card: {env.face_down_trump_card}"
            )
            print("")
            done = False
            while not done:
                current_player = state["turn"]
                try:
                    move = policy_move(env, iterations, search_mode)
                    state, _, done, winner, trick_points = env.step(move)
                    print(f"Player {current_player} plays {move}")
                except ValueError as e:
                    if "Game is already finished" in str(e):
                        print(f"Game finished: {e}")
                        break
                    else:
                        raise
                if winner is not None:
                    print(f"Player {winner} won the hand: {trick_points} points\n")
                if getattr(env, "invalid_round", False):
                    print("Round declared invalid: trump never exposed by end of 7th trick.")
                    break
            print(
                f"Game {game_id} final points: Team A={state['scores'][0]}, Team B={state['scores'][1]}"
            )
            _log_game_event(
                "game_end",
                {
                    "game_id": game_id,
                    "scores": state["scores"],
                    "game_score_delta": state["game_score"],
                    "stakes": getattr(env, "round_stakes", 1),
                    "invalid_round": getattr(env, "invalid_round", False),
                    "exposure_trick_index": getattr(env, "exposure_trick_index", None),
                    "last_exposer": getattr(env, "last_exposer", None),
                    "log_file": os.path.relpath(
                        game_log_path,
                        os.path.normpath(os.path.join(os.path.dirname(__file__), "..")),
                    ),
                    "timestamp": ts,
                },
            )
            print(f"Log saved to: {game_log_path}")
            return state["scores"], state["game_score"]
        finally:
            sys.stdout = _orig_stdout
            sys.stderr = _orig_stderr


def play_games_concurrent(num_games: int = 4, iterations: int = 50, max_workers: int | None = None, search_mode: str = "regular", mcts_config: dict = None):
    if max_workers is None:
        try:
            max_workers = max(1, os.cpu_count() or 1)
        except Exception:
            max_workers = 1
    print(f"Spawning up to {max_workers} worker processes for {num_games} games...")
    results: dict[int, tuple[list[int], list[int]]] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_game, g, iterations, 0, search_mode): g for g in range(1, num_games + 1)}
        for fut in as_completed(futures):
            g = futures[fut]
            try:
                scores, game_score = fut.result()
                results[g] = (scores, game_score)
                print(f"Game {g} finished. Scores={scores}, game_score_delta={game_score}")
            except Exception as e:
                print(f"Game {g} failed: {e}")
    ordered = [results[g][0] for g in sorted(results.keys())]
    series = [0, 0]
    for g in sorted(results.keys()):
        gs = results[g][1]
        series[0] += gs[0]
        series[1] += gs[1]
    print("\nAll game results (ordered):", ordered)
    print(f"Final series cumulative game score: Team A={series[0]}, Team B={series[1]}")
    return ordered, series


