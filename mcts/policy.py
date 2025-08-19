import copy
from .ismcts import ismcts_plan
from .mcts_core import simulate_from_state


def policy_move(env, iterations):
    state = env.get_state()
    move, _ = ismcts_plan(env, state, iterations=max(10, iterations // 5), samples=8)
    hand = env.hands[state["turn"]]
    valid = env.valid_moves(hand)
    if not valid:
        return move
    scored = []
    for a in valid:
        sim = copy.deepcopy(env)
        sim.debug = False
        sim.step(a)
        score = simulate_from_state(sim, sim.get_state())
        scored.append((score, a))
    best_heur = max(scored, key=lambda x: x[0])[1]
    mcts_score = next((s for s, a2 in scored if a2 == move), None)
    best_score = max(s for s, _ in scored)
    if mcts_score is None or (best_score - mcts_score) > 2:
        return best_heur
    return move


