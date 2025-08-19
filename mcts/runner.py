import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from .env28 import TwentyEightEnv
from .policy import policy_move
from .log_utils import log_event as _log_game_event, Tee as _TeeStdout, open_game_log as _open_game_log


LOG_DIR_GAMES = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "logs", "game28", "mcts_games"))


def play_games(num_games=3, iterations=50):
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
                state = env.reset(initial_trump=None, first_player=first_player)
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
                    move = policy_move(env, iterations)
                    state, _, done, winner, trick_points = env.step(move)
                    print(f"Player {current_player} plays {move}")
                    if winner is not None:
                        print(f"Player {winner} won the hand: {trick_points} points\n")
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


def run_single_game(game_id: int, iterations: int, first_player: int = 0):
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
                move = policy_move(env, iterations)
                state, _, done, winner, trick_points = env.step(move)
                print(f"Player {current_player} plays {move}")
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


def play_games_concurrent(num_games: int = 4, iterations: int = 50, max_workers: int | None = None):
    if max_workers is None:
        try:
            max_workers = max(1, os.cpu_count() or 1)
        except Exception:
            max_workers = 1
    print(f"Spawning up to {max_workers} worker processes for {num_games} games...")
    results: dict[int, tuple[list[int], list[int]]] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_game, g, iterations, 0): g for g in range(1, num_games + 1)}
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


