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
    """Improved strategic rollout with better trump management, suit development, and team coordination."""
    done = False
    while not done:
        hand = sim_env.hands[sim_env.turn]
        valid = sim_env.valid_moves(hand)
        if not valid:
            # If no valid moves and hand is empty, game is finished
            if not hand:
                break
            # If hand has cards but no valid moves, something is wrong
            raise ValueError(f"No valid moves for hand {hand} at turn {sim_env.turn}")
        
        # Calculate game state metrics
        tricks_remaining = sum(len(h) for h in sim_env.hands) // 4
        my_team = 0 if sim_env.turn % 2 == 0 else 1
        opponent_team = 1 - my_team
        
        if sim_env.current_trick:
            action = _choose_card_during_trick(sim_env, valid, my_team, tricks_remaining)
        else:
            action = _choose_opening_lead(sim_env, valid, my_team, tricks_remaining)
        
        _, _, done, _, _ = sim_env.step(action)
    
    return sim_env.scores[0] - sim_env.scores[1]


def _choose_card_during_trick(sim_env, valid, my_team, tricks_remaining):
    """Strategic card selection during a trick."""
    lead_suit = card_suit(sim_env.current_trick[0][1])
    winner_so_far, win_card = _winner_so_far(sim_env)
    leader_team = (winner_so_far % 2) if winner_so_far is not None else None
    
    same_suit = [c for c in valid if card_suit(c) == lead_suit]
    trumps = [c for c in valid if card_suit(c) == sim_env.trump_suit]
    non_points = [c for c in valid if card_value(c) == 0]
    high_points = [c for c in valid if card_value(c) > 0]
    
    # Partner is winning - coordinate to maximize team score
    if leader_team == my_team:
        return _play_when_partner_winning(sim_env, valid, same_suit, high_points, non_points, win_card)
    
    # Opponent is winning - try to win or minimize loss
    else:
        return _play_when_opponent_winning(sim_env, valid, same_suit, trumps, non_points, high_points, win_card, lead_suit, tricks_remaining)


def _play_when_partner_winning(sim_env, valid, same_suit, high_points, non_points, win_card):
    """Strategic play when partner is winning the trick."""
    if same_suit:
        # Partner winning with lead suit - play lowest to conserve high cards
        return min(same_suit, key=rank_index)
    else:
        # Partner winning, we're off-suit - dump points safely
        if high_points:
            # Play highest point card to maximize score
            return max(high_points, key=rank_index)
        else:
            # No points to dump - play lowest non-point card
            return min(non_points, key=rank_index) if non_points else min(valid, key=rank_index)


def _play_when_opponent_winning(sim_env, valid, same_suit, trumps, non_points, high_points, win_card, lead_suit, tricks_remaining):
    """Strategic play when opponent is winning the trick."""
    pts_on_table = sum(card_value(c) for _, c in sim_env.current_trick)
    
    # Try to win with same suit first
    if same_suit:
        higher = [c for c in same_suit if rank_index(c) > rank_index(win_card)] if win_card and card_suit(win_card) == lead_suit else same_suit
        if higher:
            # Can win with same suit - play lowest winning card
            return min(higher, key=rank_index)
    
    # Consider trumping strategically
    if trumps and _should_trump(sim_env, pts_on_table, tricks_remaining, same_suit):
        return _choose_trump_card(sim_env, trumps, win_card)
    
    # Can't win - minimize loss
    return _minimize_loss(sim_env, valid, non_points, high_points, pts_on_table)


def _should_trump(sim_env, pts_on_table, tricks_remaining, same_suit):
    """Determine if trumping is strategically sound."""
    # Always trump if revealed phase
    if sim_env.phase == "revealed":
        return True
    
    # Trump if off-suit and points justify it
    if not same_suit and pts_on_table >= 2:
        return True
    
    # Trump if it's late game and we need to control the hand
    if tricks_remaining <= 2 and pts_on_table >= 1:
        return True
    
    # Trump if we have strong trump control and points are high
    if pts_on_table >= 3:
        return True
    
    return False


def _choose_trump_card(sim_env, trumps, win_card):
    """Choose the best trump card to play."""
    if win_card and card_suit(win_card) == sim_env.trump_suit:
        # Need to beat trump - play lowest trump that can win
        over = [c for c in trumps if rank_index(c) > rank_index(win_card)]
        if over:
            return min(over, key=rank_index)
    
    # Play lowest trump to conserve high trumps
    return min(trumps, key=rank_index)


def _minimize_loss(sim_env, valid, non_points, high_points, pts_on_table):
    """Minimize loss when we can't win the trick."""
    # If no points on table, play highest non-point card to avoid winning
    if pts_on_table == 0:
        if non_points:
            return max(non_points, key=rank_index)
        else:
            return max(valid, key=rank_index)
    
    # If points on table, play lowest point card to minimize loss
    if high_points:
        return min(high_points, key=rank_index)
    else:
        return min(non_points, key=rank_index) if non_points else min(valid, key=rank_index)


def _choose_opening_lead(sim_env, valid, my_team, tricks_remaining):
    """Strategic opening lead selection."""
    # Analyze hand structure
    hand = sim_env.hands[sim_env.turn]
    trump_suit = sim_env.trump_suit
    
    # Count cards by suit
    suit_counts = {}
    for c in hand:
        suit = card_suit(c)
        suit_counts[suit] = suit_counts.get(suit, 0) + 1
    
    # Prefer non-trump suits for opening leads
    non_trump_suits = {s: count for s, count in suit_counts.items() if s != trump_suit}
    
    if non_trump_suits:
        # Lead from longest non-trump suit
        best_suit = max(non_trump_suits.keys(), key=lambda s: non_trump_suits[s])
        suit_cards = [c for c in valid if card_suit(c) == best_suit]
        
        # Lead with highest card from longest suit
        return max(suit_cards, key=rank_index)
    else:
        # Only trumps in hand - lead with highest trump
        trump_cards = [c for c in valid if card_suit(c) == trump_suit]
        return max(trump_cards, key=rank_index)


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
    """Enhanced prior computation with better strategic awareness."""
    current_player = env.turn
    base = rank_index(action) / (len(RANKS) - 1)
    points = card_value(action) / 3.0
    is_trump = (card_suit(action) == env.trump_suit)
    
    if env.current_trick:
        win_prob = estimate_trick_win_prob(env, current_player, action, samples=4)
        
        # Enhanced prior with strategic considerations
        prior = 0.4 * win_prob + 0.2 * base + 0.2 * points + 0.2 * _strategic_bonus(env, action)
        
        # Trump management in concealed phase
        if env.phase == "concealed" and is_trump:
            lead_suit = card_suit(env.current_trick[0][1])
            has_lead = any(card_suit(c) == lead_suit for c in env.hands[current_player])
            pts_on_table = sum(card_value(c) for _, c in env.current_trick)
            
            # Don't trump unless points justify it
            if not has_lead and pts_on_table < 2:
                prior *= 0.3
            elif pts_on_table >= 3:
                prior *= 1.2  # Bonus for trumping high-value tricks
    else:
        # Opening lead - consider suit development
        hand = env.hands[current_player]
        suit_len = sum(1 for c in hand if card_suit(c) == card_suit(action))
        suit_development = suit_len / 8.0
        
        prior = 0.3 * base + 0.2 * points + 0.3 * suit_development + 0.2 * _strategic_bonus(env, action)
        
        # Prefer non-trump suits for opening leads
        if env.phase == "concealed" and is_trump:
            prior *= 0.4
    
    return max(0.01, min(1.0, prior))


def _strategic_bonus(env, action):
    """Compute strategic bonus for an action based on game state."""
    bonus = 0.0
    is_trump = (card_suit(action) == env.trump_suit)
    
    # Trump control bonus
    if is_trump:
        bonus += 0.1
    
    # High card bonus
    if card_value(action) > 0:
        bonus += 0.1
    
    # Suit length bonus for non-trump suits
    if not is_trump:
        hand = env.hands[env.turn]
        suit_len = sum(1 for c in hand if card_suit(c) == card_suit(action))
        if suit_len >= 3:
            bonus += 0.1
    
    return bonus


def _compute_action_prior_longbias(env, action):
    """Enhanced long-game bias prior with better strategic considerations."""
    current_player = env.turn
    base = rank_index(action) / (len(RANKS) - 1)
    points = card_value(action) / 3.0
    is_trump = (card_suit(action) == env.trump_suit)
    
    def long_game_bias() -> float:
        bias = 0.0
        if not env.current_trick:
            # Opening lead - develop long suits, preserve trumps
            suit_len = sum(1 for c in env.hands[current_player] if card_suit(c) == card_suit(action))
            if not is_trump:
                bias += 0.3 * (suit_len / 8.0)  # Stronger bias for suit development
            if env.phase == "concealed" and is_trump:
                bias -= 0.3  # Stronger penalty for trump leads in concealed phase
        else:
            # During trick - strategic trump management
            pts_on_table = sum(card_value(c) for _, c in env.current_trick)
            if is_trump and env.phase == "concealed":
                if pts_on_table < 2:
                    bias -= 0.25  # Don't trump low-value tricks
                elif pts_on_table >= 3:
                    bias += 0.2   # Trump high-value tricks
            else:
                bias += 0.1  # General bonus for non-trump plays
        return max(-0.4, min(0.4, bias))
    
    if env.current_trick:
        win_prob = estimate_trick_win_prob(env, current_player, action, samples=4)
        prior = 0.25 * win_prob + 0.2 * base + 0.2 * points + 0.2 * _strategic_bonus(env, action) + 0.15 * (long_game_bias() + 0.3)
        
        # Enhanced trump management
        if env.phase == "concealed" and is_trump:
            lead_suit = card_suit(env.current_trick[0][1])
            has_lead = any(card_suit(c) == lead_suit for c in env.hands[current_player])
            pts_on_table = sum(card_value(c) for _, c in env.current_trick)
            
            if not has_lead and pts_on_table < 2:
                prior *= 0.2  # Stronger penalty for unnecessary trumping
            elif pts_on_table >= 3:
                prior *= 1.3  # Stronger bonus for trumping high-value tricks
    else:
        # Opening lead with long-game bias
        hand = env.hands[current_player]
        suit_len = sum(1 for c in hand if card_suit(c) == card_suit(action))
        suit_development = suit_len / 8.0
        
        prior = 0.25 * base + 0.2 * points + 0.25 * suit_development + 0.15 * _strategic_bonus(env, action) + 0.15 * (long_game_bias() + 0.3)
        
        # Strong preference for non-trump opening leads
        if env.phase == "concealed" and is_trump:
            prior *= 0.3  # Even stronger penalty for trump leads
    
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
    # propagate belief fields if available
    if "void_suits_by_player" in node.state:
        temp_env.void_suits_by_player = [set(s) for s in node.state["void_suits_by_player"]]
    if "lead_suit_counts" in node.state:
        temp_env.lead_suit_counts = copy.deepcopy(node.state["lead_suit_counts"])
    temp_env.debug = False
    valid = temp_env.valid_moves(hand)
    if not valid:
        # If no valid moves and hand is empty, game is finished
        if not hand:
            return None
        # If hand has cards but no valid moves, something is wrong
        raise ValueError(f"No valid moves for hand {hand} at turn {node.state['turn']}")
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
            if "void_suits_by_player" in node.state:
                sim_env.void_suits_by_player = [set(s) for s in node.state["void_suits_by_player"]]
            if "lead_suit_counts" in node.state:
                sim_env.lead_suit_counts = copy.deepcopy(node.state["lead_suit_counts"])
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
    # Check if game is already finished
    if state.get("done", False) or all(len(hand) == 0 for hand in state.get("hands", [])):
        raise ValueError("Game is already finished")
    
    valid = env.valid_moves(state["hands"][state["turn"]])
    
    # Handle case where there are no valid moves
    if not valid:
        # Check if game is finished
        if state.get("done", False) or all(len(hand) == 0 for hand in state.get("hands", [])):
            raise ValueError("Game is already finished")
        
        # Return a default move if possible, otherwise raise an error
        if state["hands"][state["turn"]]:
            default_move = state["hands"][state["turn"]][0]
            return default_move, {default_move: 1.0}
        else:
            raise ValueError("No valid moves available and no cards in hand")
    
    root = MCTSNode(state)
    for _ in range(iterations):
        node = select(root, (config.c_puct if config else 1.0))
        child = expand(node, env, config)
        if child is None:
            child = node
        reward = simulate_from_state(env, child.state)
        backpropagate(child, reward)
    
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
    
    # Handle case where pi is empty
    if not pi:
        # Fallback to uniform distribution over valid moves
        for a in valid:
            pi[a] = 1.0 / len(valid)
    
    best_action = max(pi.items(), key=lambda kv: kv[1])[0]
    return best_action, pi


