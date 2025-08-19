import copy
import random
from .constants import FULL_DECK, SUITS
from .mcts_core import mcts_plan


def determinize_state_for_player(state, perspective_player):
    det_state = copy.deepcopy(state)
    hands = det_state["hands"]
    my_hand = hands[perspective_player][:]
    try:
        deck = FULL_DECK
    except NameError:
        deck = [r + s for r in ["7", "8", "9", "10", "J", "Q", "K", "A"] for s in SUITS]
    pool = set(deck)
    for c in my_hand:
        pool.discard(c)
    for _, c in det_state.get("current_trick", []):
        pool.discard(c)
    trump_suit = det_state.get("trump", None)
    if det_state.get("phase", "concealed") == "concealed" and trump_suit in SUITS:
        trump_candidates = [c for c in pool if c.endswith(trump_suit)]
        sampled_face_down = random.choice(trump_candidates) if trump_candidates else None
        det_state["face_down_trump"] = sampled_face_down
        if sampled_face_down:
            pool.discard(sampled_face_down)
    remaining = list(pool)
    random.shuffle(remaining)
    idx = 0
    for p in range(4):
        if p == perspective_player:
            hands[p] = my_hand[:]
        else:
            target = len(hands[p])
            assigned = remaining[idx : idx + target]
            hands[p] = assigned[:]
            idx += target
    return det_state


def ismcts_plan(env, state, iterations=50, samples=8):
    acting_player = state["turn"]
    aggregate = {}
    last_best = None
    for _ in range(max(1, samples)):
        det_state = determinize_state_for_player(state, acting_player)
        d_env = copy.deepcopy(env)
        d_env.hands = copy.deepcopy(det_state["hands"])
        d_env.current_trick = copy.deepcopy(det_state["current_trick"])
        d_env.scores = det_state["scores"][:]
        d_env.turn = det_state["turn"]
        d_env.trump_suit = det_state["trump"]
        d_env.phase = det_state.get("phase", getattr(d_env, "phase", "revealed"))
        d_env.bidder = det_state.get("bidder", getattr(d_env, "bidder", 0))
        d_env.face_down_trump_card = det_state.get("face_down_trump", getattr(d_env, "face_down_trump_card", None))
        d_env.bid_value = det_state.get("bid_value", getattr(d_env, "bid_value", 16))
        d_env.last_exposer = det_state.get("last_exposer", getattr(d_env, "last_exposer", None))
        d_env.exposure_trick_index = det_state.get("exposure_trick_index", getattr(d_env, "exposure_trick_index", None))
        d_env.debug = False
        best, pi = mcts_plan(d_env, det_state, iterations)
        last_best = best
        for a, p in pi.items():
            aggregate[a] = aggregate.get(a, 0.0) + p
    if not aggregate:
        if last_best is not None:
            return last_best, {}
        hand = state["hands"][state["turn"]]
        valid = env.valid_moves(hand)
        return (valid[0] if valid else None), {}
    total = sum(aggregate.values())
    pi_agg = {a: (w / total) for a, w in aggregate.items()} if total > 0 else aggregate
    best_action = max(pi_agg.items(), key=lambda kv: kv[1])[0]
    return best_action, pi_agg


