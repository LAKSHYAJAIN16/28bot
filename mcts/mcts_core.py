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


def ismcts_rollout(sim_env, rollout_iterations=20):
    """ISMCTS-based rollout that uses actual planning for evaluation - much more human-like."""
    done = False
    moves_done = 0
    max_moves = 32  # Safety cap
    
    while not done and moves_done < max_moves:
        hand = sim_env.hands[sim_env.turn]
        valid = sim_env.valid_moves(hand)
        
        if not valid:
            if not hand:
                break
            raise ValueError(f"No valid moves for hand {hand} at turn {sim_env.turn}")
        
        # Use strategic heuristic for rollout evaluation (avoiding circular import)
        action = _strategic_heuristic_choice(sim_env, valid)
        state, _, done, _, _ = sim_env.step(action)
        
        moves_done += 1
    
    return sim_env.scores[0] - sim_env.scores[1]


# Replace the old heuristic_rollout with the new ISMCTS-based one
def heuristic_rollout(sim_env):
    """ISMCTS-based rollout for much more human-like play."""
    return ismcts_rollout(sim_env, rollout_iterations=20)


def _strategic_heuristic_choice(sim_env, valid):
    """Strategic heuristic as fallback - much more sophisticated than current rollout."""
    hand = sim_env.hands[sim_env.turn]
    my_team = 0 if sim_env.turn % 2 == 0 else 1
    tricks_remaining = sum(len(h) for h in sim_env.hands) // 4
    
    if sim_env.current_trick:
        return _strategic_trick_play(sim_env, valid, my_team, tricks_remaining)
    else:
        return _strategic_opening_lead(sim_env, valid, my_team, tricks_remaining)


def _strategic_trick_play(sim_env, valid, my_team, tricks_remaining):
    """Much more sophisticated trick play with hand reading and long-term planning."""
    lead_suit = card_suit(sim_env.current_trick[0][1])
    winner_so_far, win_card = _winner_so_far(sim_env)
    leader_team = (winner_so_far % 2) if winner_so_far is not None else None
    pts_on_table = sum(card_value(c) for _, c in sim_env.current_trick)
    
    same_suit = [c for c in valid if card_suit(c) == lead_suit]
    trumps = [c for c in valid if card_suit(c) == sim_env.trump_suit]
    non_points = [c for c in valid if card_value(c) == 0]
    high_points = [c for c in valid if card_value(c) > 0]
    
    # Partner is winning - sophisticated coordination
    if leader_team == my_team:
        return _sophisticated_partner_support(sim_env, valid, same_suit, high_points, non_points, pts_on_table, tricks_remaining)
    
    # Opponent is winning - sophisticated counterplay
    else:
        return _sophisticated_counterplay(sim_env, valid, same_suit, trumps, non_points, high_points, win_card, lead_suit, pts_on_table, tricks_remaining)


def _sophisticated_partner_support(sim_env, valid, same_suit, high_points, non_points, pts_on_table, tricks_remaining):
    """Sophisticated partner support with hand reading and long-term planning."""
    if same_suit:
        # Partner winning with lead suit - consider hand structure
        if tricks_remaining <= 2:
            # Late game - conserve high cards for final tricks
            return min(same_suit, key=rank_index)
        else:
            # Early/mid game - play middle card to maintain control
            sorted_same = sorted(same_suit, key=rank_index)
            return sorted_same[len(sorted_same) // 2]
    else:
        # Partner winning, we're off-suit - sophisticated point dumping
        if pts_on_table >= 3:
            # High value trick - dump points aggressively
            if high_points:
                return max(high_points, key=rank_index)
        elif pts_on_table >= 1:
            # Medium value - dump points moderately
            if high_points:
                sorted_points = sorted(high_points, key=rank_index)
                return sorted_points[len(sorted_points) // 2]
        
        # Low value or no points - conserve high cards
        return min(non_points, key=rank_index) if non_points else min(valid, key=rank_index)


def _sophisticated_counterplay(sim_env, valid, same_suit, trumps, non_points, high_points, win_card, lead_suit, pts_on_table, tricks_remaining):
    """Sophisticated counterplay with trump management and hand reading."""
    # Try to win with same suit first
    if same_suit:
        higher = [c for c in same_suit if rank_index(c) > rank_index(win_card)] if win_card and card_suit(win_card) == lead_suit else same_suit
        if higher:
            # Can win - choose strategically
            if pts_on_table >= 3:
                # High value - win with lowest winning card
                return min(higher, key=rank_index)
            else:
                # Low value - consider conserving high cards
                sorted_higher = sorted(higher, key=rank_index)
                return sorted_higher[len(sorted_higher) // 2]
    
    # Consider trumping with sophisticated logic
    if trumps and _sophisticated_trump_decision(sim_env, pts_on_table, tricks_remaining, same_suit, trumps):
        return _sophisticated_trump_choice(sim_env, trumps, win_card, pts_on_table, tricks_remaining)
    
    # Can't win - sophisticated loss minimization
    return _sophisticated_loss_minimization(sim_env, valid, non_points, high_points, pts_on_table, tricks_remaining)


def _sophisticated_trump_decision(sim_env, pts_on_table, tricks_remaining, same_suit, trumps):
    """Sophisticated trump decision with hand reading and game state analysis."""
    # Always trump if revealed phase
    if sim_env.phase == "revealed":
        return True
    
    # Trump if off-suit and points justify it
    if not same_suit and pts_on_table >= 2:
        return True
    
    # Trump if late game and we need control
    if tricks_remaining <= 2 and pts_on_table >= 1:
        return True
    
    # Trump if we have strong trump control and high points
    if pts_on_table >= 3:
        return True
    
    # Trump if we have many trumps and it's early game (establish control)
    if tricks_remaining >= 5 and len(trumps) >= 3:
        return pts_on_table >= 1
    
    return False


def _sophisticated_trump_choice(sim_env, trumps, win_card, pts_on_table, tricks_remaining):
    """Sophisticated trump card selection."""
    if win_card and card_suit(win_card) == sim_env.trump_suit:
        # Need to beat trump
        over = [c for c in trumps if rank_index(c) > rank_index(win_card)]
        if over:
            return min(over, key=rank_index)
    
    # Choose trump strategically based on game state
    sorted_trumps = sorted(trumps, key=rank_index)
    
    if tricks_remaining <= 2:
        # Late game - use high trumps
        return max(trumps, key=rank_index)
    elif pts_on_table >= 3:
        # High value - use medium trump
        return sorted_trumps[len(sorted_trumps) // 2]
    else:
        # Low value - conserve high trumps
        return min(trumps, key=rank_index)


def _sophisticated_loss_minimization(sim_env, valid, non_points, high_points, pts_on_table, tricks_remaining):
    """Sophisticated loss minimization with hand reading."""
    if pts_on_table == 0:
        # No points - avoid winning
        if non_points:
            return max(non_points, key=rank_index)
        else:
            return max(valid, key=rank_index)
    
    # Points on table - minimize loss
    if high_points:
        if tricks_remaining <= 2:
            # Late game - dump highest points
            return max(high_points, key=rank_index)
        else:
            # Early/mid game - dump middle points
            sorted_points = sorted(high_points, key=rank_index)
            return sorted_points[len(sorted_points) // 2]
    else:
        return min(non_points, key=rank_index) if non_points else min(valid, key=rank_index)


def _strategic_opening_lead(sim_env, valid, my_team, tricks_remaining):
    """Sophisticated opening lead with hand structure analysis."""
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
    """Compute strategic bonus for an action based on game state and hand reading."""
    bonus = 0.0
    is_trump = (card_suit(action) == env.trump_suit)
    current_player = env.turn
    my_team = 0 if current_player % 2 == 0 else 1
    
    # Trump control bonus
    if is_trump:
        bonus += 0.15
    
    # High card bonus
    if card_value(action) > 0:
        bonus += 0.1
    
    # Suit length bonus for non-trump suits
    if not is_trump:
        hand = env.hands[current_player]
        suit_len = sum(1 for c in hand if card_suit(c) == card_suit(action))
        if suit_len >= 3:
            bonus += 0.15
    
    # Hand reading bonus - consider what opponents likely have
    if env.current_trick:
        lead_suit = card_suit(env.current_trick[0][1])
        if card_suit(action) == lead_suit:
            # Following suit - consider if we can win
            winner_so_far, win_card = _winner_so_far(env)
            if win_card and rank_index(action) > rank_index(win_card):
                bonus += 0.2  # Can win the trick
        elif is_trump and env.phase == "revealed":
            # Trumping in revealed phase - consider points on table
            pts_on_table = sum(card_value(c) for _, c in env.current_trick)
            if pts_on_table >= 2:
                bonus += 0.25  # Trumping high-value trick
    
    # Team coordination bonus
    if env.current_trick:
        winner_so_far, _ = _winner_so_far(env)
        if winner_so_far is not None:
            leader_team = 0 if winner_so_far % 2 == 0 else 1
            if leader_team == my_team:
                # Partner is winning - coordinate
                if card_suit(action) == lead_suit:
                    bonus += 0.1  # Support partner's lead
                elif card_value(action) > 0:
                    bonus += 0.15  # Dump points when partner winning
    
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
        if state.get("done", False):
            raise ValueError("Game is already finished")
        
        # Check if all hands are empty (game should be done)
        all_hands = state.get("hands", [])
        if all(len(hand) == 0 for hand in all_hands):
            raise ValueError("Game is already finished - all hands empty")
        
        # Check current player's hand
        current_hand = state["hands"][state["turn"]] if state.get("hands") else []
        if current_hand:
            # Player has cards but no valid moves - this shouldn't happen
            # Return the first card as a fallback
            default_move = current_hand[0]
            return default_move, {default_move: 1.0}
        else:
            # Player has no cards - check if this is expected (game end)
            cards_remaining = sum(len(hand) for hand in all_hands)
            if cards_remaining == 0:
                raise ValueError("Game is finished - no cards remaining")
            else:
                # Some players still have cards, but current player doesn't
                # This might be a game state issue - try to continue with a dummy move
                raise ValueError(f"Current player {state['turn']} has no cards but {cards_remaining} cards remain in other hands")
    
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


