import copy
from .ismcts import ismcts_plan
from .mcts_core import simulate_from_state


def policy_move(env, iterations):
    state = env.get_state()
    # Heavier ISMCTS planning; disable heuristic override to trust search
    samples = 16
    iters_per_sample = max(10, iterations // 3)
    move, _ = ismcts_plan(env, state, iterations=iters_per_sample, samples=samples)
    return move


