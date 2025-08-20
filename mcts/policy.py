import copy
from .ismcts import ismcts_plan
from .mcts_core import simulate_from_state, SearchConfig


def policy_move(env, iterations, search_mode: str = "regular", mcts_config: dict = None):
    state = env.get_state()
    
    # Check if game is done
    if state.get("done", False) or all(len(hand) == 0 for hand in state.get("hands", [])):
        raise ValueError("Game is already finished")
    
    # Use MCTS config if provided, otherwise use defaults
    if mcts_config:
        samples = mcts_config.get('mcts_samples', 24)
        iters_per_sample = mcts_config.get('mcts_iters_per_sample', max(20, iterations // 4))
        c_puct = mcts_config.get('mcts_c_puct', 1.5)
    else:
        samples = 24  # Increased from 16 for better exploration
        iters_per_sample = max(20, iterations // 4)  # More iterations per sample
        c_puct = 1.5  # Increased exploration
    
    cfg = SearchConfig(mode=search_mode, c_puct=c_puct)
    move, _ = ismcts_plan(env, state, iterations=iters_per_sample, samples=samples, config=cfg)
    return move


