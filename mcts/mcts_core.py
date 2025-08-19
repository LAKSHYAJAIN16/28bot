import copy
import math
import random
from dataclasses import dataclass
from .constants import RANKS, FULL_DECK, SUITS, card_suit, card_value, rank_index, trick_rank_index


class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = copy.deepcopy(state)
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.prior = 0.0

    def uct_score(self, c_puct=1.0):
        if self.parent is None:
            return 0.0
        exploitation = (self.value / self.visits) if self.visits > 0 else 0.0
        exploration = c_puct * self.prior * math.sqrt(max(1, self.parent.visits)) / (1 + self.visits)
        return exploitation + exploration

    def best_child(self):
        return max(self.children, key=lambda c: c.visits)


def select(node, c_puct: float = 1.0):
    while node.children:
        node = max(node.children, key=lambda c: c.uct_score(c_puct))
    return node


@dataclass
class SearchConfig:
    # mode: "regular" uses classic short-horizon prior; "long_bias" uses long-game bias
    mode: str = "regular"
    c_puct: float = 1.0


def _winner_so_far(sim_env):
    if not sim_env.current_trick:
        return None, None
    lead_suit = card_suit(sim_env.current_trick[0][1])
    winning_card, winner = sim_env.current_trick[0][1], sim_env.current_trick[0][0]
    if sim_env.phase == "concealed":
        for player, card in sim_env.current_trick[1:]:
            if card_suit(card) == lead_suit and trick_rank_index(card) > trick_rank_index(winning_card):
                winning_card, winner = card, player
    else:
        for player, card in sim_env.current_trick[1:]:
            win_suit = card_suit(winning_card)
            if card_suit(card) == win_suit and trick_rank_index(card) > trick_rank_index(winning_card):
                winning_card, winner = card, player
            elif sim_env.trump_suit is not None and card_suit(card) == sim_env.trump_suit and win_suit != sim_env.trump_suit:
                winning_card, winner = card, player
    return winner, winning_card


def heuristic_rollout(sim_env):
    done = False
    while not done:
        hand = sim_env.hands[sim_env.turn]
        valid = sim_env.valid_moves(hand)
        if not valid:
            break
        if sim_env.current_trick:
            lead_suit = card_suit(sim_env.current_trick[0][1])
            winner_so_far, win_card = _winner_so_far(sim_env)
            my_team = 0 if sim_env.turn % 2 == 0 else 1
            leader_team = (winner_so_far % 2) if winner_so_far is not None else None
            same_suit = [c for c in valid if card_suit(c) == lead_suit]
            trumps = [c for c in valid if card_suit(c) == sim_env.trump_suit]
            non_points = [c for c in valid if card_value(c) == 0]
            if leader_team is not None and leader_team == my_team:
                if same_suit:
                    action = min(same_suit, key=rank_index)
                else:
                    action = min(non_points, key=rank_index) if non_points else min(valid, key=rank_index)
            else:
                chosen = None
                if same_suit:
                    higher = [c for c in same_suit if rank_index(c) > rank_index(win_card)] if win_card and card_suit(win_card) == lead_suit else same_suit
                    if higher:
                        chosen = min(higher, key=rank_index)
                if chosen is None:
                    if trumps and (sim_env.phase == "revealed" or not same_suit):
                        if win_card and card_suit(win_card) == sim_env.trump_suit:
                            over = [c for c in trumps if rank_index(c) > rank_index(win_card)]
                            chosen = min(over, key=rank_index) if over else min(trumps, key=rank_index)
                        else:
                            chosen = min(trumps, key=rank_index)
                if chosen is None:
                    chosen = min(non_points, key=rank_index) if non_points else min(valid, key=rank_index)
                action = chosen
        else:
            non_trump = [c for c in valid if card_suit(c) != sim_env.trump_suit]
            lead_pool = non_trump if non_trump else valid
            suits = {}
            for c in lead_pool:
                suits.setdefault(card_suit(c), []).append(c)
            best_suit = max(suits.keys(), key=lambda s: (len(suits[s]), max(rank_index(c) for c in suits[s])))
            action = max(suits[best_suit], key=rank_index)
        _, _, done, _, _ = sim_env.step(action)
    return sim_env.scores[0] - sim_env.scores[1]


def estimate_trick_win_prob(env, acting_player, action, samples=10):
    # Allow ultra-fast mode for bulk/offline computations
    if hasattr(env, "quick_eval") and env.quick_eval:
        samples = min(samples, 2)
    wins = 0
    for _ in range(samples):
        sim = copy.deepcopy(env)
        sim.debug = False
        _ = sim.step(action)
        while sim.current_trick:
            hand = sim.hands[sim.turn]
            valid = sim.valid_moves(hand)
            if not valid:
                break
            rand_action = random.choice(valid)
            _, _, _, winner, _ = sim.step(rand_action)
        winner = sim.last_trick_winner
        if winner is None:
            continue
        if winner % 2 == acting_player % 2:
            wins += 1
    return wins / max(1, samples)


def _compute_action_prior_regular(env, action):
    current_player = env.turn
    base = rank_index(action) / (len(RANKS) - 1)
    points = card_value(action) / 3.0
    is_trump = (card_suit(action) == env.trump_suit)
    if env.current_trick:
        win_prob = estimate_trick_win_prob(env, current_player, action, samples=4)
        prior = 0.5 * win_prob + 0.25 * base + 0.25 * points
        if env.phase == "concealed":
            lead_suit = card_suit(env.current_trick[0][1])
            has_lead = any(card_suit(c) == lead_suit for c in env.hands[current_player])
            if not has_lead and is_trump and win_prob < 0.45:
                prior *= 0.4
    else:
        hand = env.hands[current_player]
        suit_len = sum(1 for c in hand if card_suit(c) == card_suit(action))
        prior = 0.45 * base + 0.25 * points + 0.3 * (suit_len / 8.0)
        if env.phase == "concealed" and is_trump:
            prior *= 0.5
    return max(0.01, min(1.0, prior))


def _compute_action_prior_longbias(env, action):
    current_player = env.turn
    base = rank_index(action) / (len(RANKS) - 1)
    points = card_value(action) / 3.0
    is_trump = (card_suit(action) == env.trump_suit)
    def long_game_bias() -> float:
        bias = 0.0
        if not env.current_trick:
            suit_len = sum(1 for c in env.hands[current_player] if card_suit(c) == card_suit(action))
            if not is_trump:
                bias += 0.2 * (suit_len / 8.0)
            if env.phase == "concealed" and is_trump:
                bias -= 0.2
        else:
            pts_on_table = sum(card_value(c) for _, c in env.current_trick)
            if is_trump and env.phase == "concealed" and pts_on_table < 2:
                bias -= 0.15
            else:
                bias += 0.05
        return max(-0.3, min(0.3, bias))
    if env.current_trick:
        win_prob = estimate_trick_win_prob(env, current_player, action, samples=4)
        prior = 0.3 * win_prob + 0.25 * base + 0.25 * points + 0.2 * (long_game_bias() + 0.3)
        if env.phase == "concealed":
            lead_suit = card_suit(env.current_trick[0][1])
            has_lead = any(card_suit(c) == lead_suit for c in env.hands[current_player])
            if not has_lead and is_trump and win_prob < 0.45:
                prior *= 0.4
    else:
        hand = env.hands[current_player]
        suit_len = sum(1 for c in hand if card_suit(c) == card_suit(action))
        prior = 0.35 * base + 0.25 * points + 0.2 * (suit_len / 8.0) + 0.2 * (long_game_bias() + 0.3)
        if env.phase == "concealed" and is_trump:
            prior *= 0.5
    return max(0.01, min(1.0, prior))


def compute_action_prior(env, action, config: SearchConfig | None = None):
    cfg = config or SearchConfig()
    if cfg.mode == "long_bias":
        return _compute_action_prior_longbias(env, action)
    # default: regular
    return _compute_action_prior_regular(env, action)


def expand(node, env, config: SearchConfig | None = None):
    hand = node.state["hands"][node.state["turn"]]
    tried = [c.action for c in node.children]
    temp_env = copy.deepcopy(env)
    temp_env.hands = copy.deepcopy(node.state["hands"])
    temp_env.current_trick = copy.deepcopy(node.state["current_trick"])
    temp_env.scores = node.state["scores"][:]
    temp_env.turn = node.state["turn"]
    temp_env.trump_suit = node.state["trump"]
    temp_env.phase = node.state.get("phase", getattr(temp_env, "phase", "revealed"))
    temp_env.bidder = node.state.get("bidder", getattr(temp_env, "bidder", 0))
    temp_env.face_down_trump_card = node.state.get("face_down_trump", getattr(temp_env, "face_down_trump_card", None))
    temp_env.bid_value = node.state.get("bid_value", getattr(temp_env, "bid_value", 16))
    temp_env.last_exposer = node.state.get("last_exposer", getattr(temp_env, "last_exposer", None))
    temp_env.exposure_trick_index = node.state.get("exposure_trick_index", getattr(temp_env, "exposure_trick_index", None))
    temp_env.debug = False
    valid = temp_env.valid_moves(hand)
    if not valid:
        return None
    priors = {a: compute_action_prior(temp_env, a, config) for a in valid}
    max_prior = max(priors.values()) if priors else 1.0
    for action in valid:
        if action not in tried:
            sim_env = copy.deepcopy(env)
            sim_env.hands = copy.deepcopy(node.state["hands"])
            sim_env.current_trick = copy.deepcopy(node.state["current_trick"])
            sim_env.scores = node.state["scores"][:]
            sim_env.turn = node.state["turn"]
            sim_env.trump_suit = node.state["trump"]
            sim_env.phase = node.state.get("phase", getattr(sim_env, "phase", "revealed"))
            sim_env.bidder = node.state.get("bidder", getattr(sim_env, "bidder", 0))
            sim_env.face_down_trump_card = node.state.get("face_down_trump", getattr(sim_env, "face_down_trump_card", None))
            sim_env.bid_value = node.state.get("bid_value", getattr(sim_env, "bid_value", 16))
            sim_env.last_exposer = node.state.get("last_exposer", getattr(sim_env, "last_exposer", None))
            sim_env.exposure_trick_index = node.state.get("exposure_trick_index", getattr(sim_env, "exposure_trick_index", None))
            sim_env.debug = False
            state, _, _, _, _ = sim_env.step(action)
            child = MCTSNode(state, parent=node, action=action)
            child.prior = (priors.get(action, 0.01) / max_prior) if max_prior > 0 else 0.01
            node.children.append(child)
            return child
    return None


def simulate_from_state(env, state):
    sim_env = copy.deepcopy(env)
    sim_env.hands = copy.deepcopy(state["hands"])
    sim_env.current_trick = copy.deepcopy(state["current_trick"])
    sim_env.scores = state["scores"][:]
    sim_env.turn = state["turn"]
    sim_env.trump_suit = state["trump"]
    sim_env.phase = state.get("phase", getattr(sim_env, "phase", "revealed"))
    sim_env.bidder = state.get("bidder", getattr(sim_env, "bidder", 0))
    sim_env.face_down_trump_card = state.get("face_down_trump", getattr(sim_env, "face_down_trump_card", None))
    sim_env.bid_value = state.get("bid_value", getattr(sim_env, "bid_value", 16))
    sim_env.last_exposer = state.get("last_exposer", getattr(sim_env, "last_exposer", None))
    sim_env.exposure_trick_index = state.get("exposure_trick_index", getattr(sim_env, "exposure_trick_index", None))
    sim_env.debug = False
    return heuristic_rollout(sim_env)


def mcts_search(env, state, iterations=50, config: SearchConfig | None = None):
    root = MCTSNode(state)
    for _ in range(iterations):
        node = select(root, (config.c_puct if config else 1.0))
        child = expand(node, env, config)
        if child is None:
            child = node
        reward = simulate_from_state(env, child.state)
        backpropagate(child, reward)
    return root.best_child().action


def backpropagate(node, reward):
    while node:
        node.visits += 1
        node.value += reward
        node = node.parent


def mcts_plan(env, state, iterations=50, config: SearchConfig | None = None):
    root = MCTSNode(state)
    for _ in range(iterations):
        node = select(root, (config.c_puct if config else 1.0))
        child = expand(node, env, config)
        if child is None:
            child = node
        reward = simulate_from_state(env, child.state)
        backpropagate(child, reward)
    valid = env.valid_moves(state["hands"][state["turn"]])
    visits = {a: 0 for a in valid}
    for ch in root.children:
        if ch.action in visits:
            visits[ch.action] = ch.visits
    total = sum(visits.values())
    if total == 0:
        for a in visits:
            visits[a] = 1
        total = len(visits)
    pi = {a: v / total for a, v in visits.items()}
    best_action = max(pi.items(), key=lambda kv: kv[1])[0]
    return best_action, pi


