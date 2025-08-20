import copy
from .ismcts import ismcts_plan
from .mcts_core import simulate_from_state, SearchConfig


def policy_move(env, iterations, search_mode: str = "regular"):
    state = env.get_state()
    
    # Check if game is done
    if state.get("done", False) or all(len(hand) == 0 for hand in state.get("hands", [])):
        raise ValueError("Game is already finished")
    
    # Heavier ISMCTS planning; disable heuristic override to trust search
    samples = 16
    iters_per_sample = max(10, iterations // 3)
    cfg = SearchConfig(mode=search_mode)
    move, _ = ismcts_plan(env, state, iterations=iters_per_sample, samples=samples, config=cfg)
    return move


