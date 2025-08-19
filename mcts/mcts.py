import random, math, copy

# Optional NN support for hybrid MCTS
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

# Ensure global is defined before any function references
NN_EVAL = None

# =========================
# Card Game 28 Environment
# =========================
SUITS = ["H", "D", "C", "S"]
RANKS = ["7", "8", "Q", "K", "10", "A", "9", "J"]  # low -> high
CARD_POINTS = {"J": 3, "9": 2, "10": 1, "A": 1, "7": 0, "8": 0, "Q": 0, "K": 0}

# Helpers
def card_rank(card):
    return card[:-1]

def card_suit(card):
    return card[-1]

def card_value(card):
    return CARD_POINTS[card_rank(card)]

def rank_index(card):
    return RANKS.index(card_rank(card))

# Suit strength heuristic for choosing trump
def suit_trump_strength(hand, suit):
    suit_cards = [c for c in hand if card_suit(c) == suit]
    if not suit_cards:
        return 0
    count = len(suit_cards)
    rank_power_sum = sum(rank_index(c) for c in suit_cards)
    point_sum = sum(card_value(c) for c in suit_cards)
    has_jack = any(card_rank(c) == "J" for c in suit_cards)
    has_nine = any(card_rank(c) == "9" for c in suit_cards)
    # Weighted combination: count, card strength, and point value; bonuses for J and 9
    return (
        3 * count
        + rank_power_sum
        + 2 * point_sum
        + (5 if has_jack else 0)
        + (3 if has_nine else 0)
    )

class HeuristicBiddingAgent:
    def propose_bid(self, first_four_cards, current_high_bid):
        # Evaluate best suit present in the first four cards
        present_suits = {s for s in SUITS if any(card_suit(c) == s for c in first_four_cards)}
        strengths = {s: suit_trump_strength(first_four_cards, s) for s in present_suits}
        if not strengths:
            return None
        best_suit = max(strengths, key=lambda s: (strengths[s], -SUITS.index(s)))
        count = sum(1 for c in first_four_cards if card_suit(c) == best_suit)
        pts = sum(card_value(c) for c in first_four_cards if card_suit(c) == best_suit)
        has_j = any(card_rank(c) == "J" for c in first_four_cards if card_suit(c) == best_suit)
        has_9 = any(card_rank(c) == "9" for c in first_four_cards if card_suit(c) == best_suit)
        # Base bid heuristic (min 16)
        bid = 16 + 2*count + 2*pts + (2 if has_j else 0) + (1 if has_9 else 0)
        bid = min(28, bid)
        if bid <= max(current_high_bid, 0):
            return None
        # Raise by at least 1 over current high
        return max(current_high_bid + 1, bid)

    def choose_trump(self, first_four_cards, final_bid):
        # Choose the strongest suit among those present in the four cards
        strengths = {s: suit_trump_strength(first_four_cards, s) for s in SUITS}
        # Prefer suits actually present
        def suit_key(s):
            present = any(card_suit(c) == s for c in first_four_cards)
            return (present, strengths[s], -SUITS.index(s))
        return max(SUITS, key=suit_key)

# =========================
# Environment
# =========================
class TwentyEightEnv:
    def __init__(self):
        self.num_players = 4
        self.debug = False

    def reset(self, initial_trump=None, first_player=0):
        deck = [r+s for r in ["7","8","9","10","J","Q","K","A"] for s in SUITS]
        random.shuffle(deck)
        # Two-stage deal: first 4 cards to each player
        self.hands = [[] for _ in range(4)]
        for _ in range(4):
            for p in range(4):
                self.hands[p].append(deck.pop())
        # Persist the exact auction hands used for bidding/trump decisions
        self.first_four_hands = [hand[:] for hand in self.hands]
        self.scores = [0,0]
        self.game_score = [0,0]
        self.current_trick = []
        self.turn = first_player
        # Bidding phase (official-style simplified). If initial_trump is provided, skip bidding
        if initial_trump is not None:
            self.bidder = first_player
            self.trump_suit = initial_trump
            self.bid_value = 16
        else:
            # Run official-style auction: start at 16, players may pass or raise until three consecutive passes after a bid
            agents = [MonteCarloBiddingAgent(num_samples=4, mcts_iterations=50) for _ in range(4)]
            order = [(first_player + i) % 4 for i in range(4)]
            current_high = 0
            current_bidder = None
            consecutive_passes = 0
            bidding_started = False
            idx = 0
            first_cycle_passes = 0
            # Safety cap to avoid infinite loops
            while idx < 100:
                p = order[idx % 4]
                # Use the exact 4-card auction hands
                proposal = agents[p].propose_bid(self.first_four_hands[p], current_high, my_seat=p, first_player=first_player)
                min_allowed = 16 if current_high < 16 else current_high + 1
                dbg = getattr(agents[p], 'last_debug', None)
                if proposal is not None and proposal >= min_allowed and proposal <= 28:
                    current_high = proposal
                    current_bidder = p
                    consecutive_passes = 0
                    bidding_started = True
                    print(f"Bid: Player {p} proposes {proposal} (min_allowed={min_allowed}) – suit_scores={dbg['suit_to_points'] if dbg else {}} chosen={dbg['chosen_suit'] if dbg else None} p30={dbg['p30_points'] if dbg else None} avg={dbg['avg_points'] if dbg else None}")
                else:
                    consecutive_passes += 1
                    if idx < 4:
                        first_cycle_passes += 1
                    print(f"Pass: Player {p} (min_allowed={min_allowed}) – analysis={dbg}")
                # If a bid has started, end when three subsequent passes occur
                if bidding_started and consecutive_passes == 3:
                    break
                # If no one bid in the first full cycle, force first player to bid 16
                if idx >= 3 and not bidding_started and first_cycle_passes >= 4:
                    break
                idx += 1
            # Fallback if no one bid: force first player at 16
            if current_bidder is None:
                current_bidder = order[0]
                current_high = 16
                print(f"Forced bid: All players passed. Player {current_bidder} takes minimum bid {current_high}.")
            self.bidder = current_bidder
            self.bid_value = current_high
            # Bidder chooses trump based on their exact 4-card auction hand
            self.trump_suit = agents[self.bidder].choose_trump(self.first_four_hands[self.bidder], self.bid_value)
            choose_dbg = getattr(agents[self.bidder], 'last_choose_debug', None)
            print(f"Auction winner: Player {self.bidder} with bid {self.bid_value}; chooses trump {self.trump_suit} – details={choose_dbg}")
        # Concealed phase with a face-down trump card from bidder
        self.phase = "concealed"
        trump_cards_in_bidder = [c for c in self.hands[self.bidder] if card_suit(c) == self.trump_suit]
        self.face_down_trump_card = max(trump_cards_in_bidder, key=rank_index) if trump_cards_in_bidder else None
        if self.face_down_trump_card is not None:
            self.hands[self.bidder].remove(self.face_down_trump_card)
        # Deal remaining 4 cards to each player
        for _ in range(4):
            for p in range(4):
                self.hands[p].append(deck.pop())
        self.last_exposer = None
        self.exposure_trick_index = None
        self.tricks_played = 0
        self.invalid_round = False
        self.last_trick_winner = None
        return self.get_state()

    def get_state(self):
        return {
            "hands": copy.deepcopy(self.hands),
            "turn": self.turn,
            "scores": self.scores[:],
            "game_score": self.game_score[:],
            "current_trick": copy.deepcopy(self.current_trick),
            "trump": self.trump_suit,
            "phase": self.phase,
            "bidder": self.bidder,
            "bid_value": getattr(self, "bid_value", 16),
            "face_down_trump": self.face_down_trump_card,
            "last_exposer": self.last_exposer,
            "exposure_trick_index": self.exposure_trick_index
        }

    def valid_moves(self, hand):
        if not self.current_trick:
            # Before exposure, bidder cannot lead trump unless only trump cards remain
            if self.phase == "concealed" and self.turn == self.bidder and self.trump_suit is not None:
                non_trump = [c for c in hand if card_suit(c) != self.trump_suit]
                if non_trump:
                    return non_trump
            return hand
        lead_suit = card_suit(self.current_trick[0][1])
        same_suit = [c for c in hand if card_suit(c) == lead_suit]
        return same_suit if same_suit else hand

    def step(self, card):
        # Enforce follow suit when not leading
        if self.current_trick:
            lead_suit = card_suit(self.current_trick[0][1])
            if any(card_suit(c) == lead_suit for c in self.hands[self.turn]) and card_suit(card) != lead_suit:
                raise ValueError("Illegal move: must follow suit")
        else:
            # Enforce bidder restriction on leading trump before exposure
            if self.phase == "concealed" and self.turn == self.bidder and self.trump_suit is not None:
                if any(card_suit(c) != self.trump_suit for c in self.hands[self.turn]) and card_suit(card) == self.trump_suit:
                    raise ValueError("Illegal lead: bidder cannot lead trump before exposure unless only trump remains")

        # Exposure logic: if cannot follow suit and plays trump during concealed phase, expose now
        if self.phase == "concealed" and self.current_trick:
            lead_suit = card_suit(self.current_trick[0][1])
            has_lead = any(card_suit(c) == lead_suit for c in self.hands[self.turn])
            if not has_lead and self.trump_suit is not None and card_suit(card) == self.trump_suit:
                self.phase = "revealed"
                self.last_exposer = self.turn
                self.exposure_trick_index = self.tricks_played + 1  # exposure happens during this trick
                if self.face_down_trump_card is not None:
                    self.hands[self.bidder].append(self.face_down_trump_card)
                    self.face_down_trump_card = None
                if self.debug:
                    print(f"-- Phase 2 begins: Trump revealed as {self.trump_suit} by Player {self.last_exposer} on trick {self.exposure_trick_index} --")

        self.hands[self.turn].remove(card)
        self.current_trick.append((self.turn, card))
        self.turn = (self.turn + 1) % 4
        done = all(len(hand)==0 for hand in self.hands)
        trick_winner = None
        trick_points = 0
        if len(self.current_trick) == 4:
            trick_winner, trick_points = self.resolve_trick()
            team = 0 if trick_winner%2==0 else 1
            self.scores[team] += trick_points
            self.current_trick = []
            self.turn = trick_winner
            self.last_trick_winner = trick_winner
            self.tricks_played += 1
            # If after seven tricks trump never exposed, round is invalid per rules
            if self.tricks_played >= 7 and self.phase == "concealed":
                self.invalid_round = True
                done = True
                # Round invalid; no change to cumulative game_score
                # Reset points to 0 for transparency
                self.scores = [0, 0]
        reward = self.scores[0] - self.scores[1] if done else 0
        # At round end, update cumulative game score (+1/-1) based on bid success
        if done and not self.invalid_round:
            bidder_team = 0 if self.bidder % 2 == 0 else 1
            if self.scores[bidder_team] >= getattr(self, 'bid_value', 16):
                self.game_score[bidder_team] += 1
            else:
                self.game_score[bidder_team] -= 1
            # Optionally, non-bidder team gets opposite
            other_team = 1 - bidder_team
            if self.scores[bidder_team] >= getattr(self, 'bid_value', 16):
                self.game_score[other_team] -= 0  # no change for defenders per your spec
            else:
                self.game_score[other_team] += 0
        return self.get_state(), reward, done, trick_winner, trick_points

    def resolve_trick(self):
        lead_suit = card_suit(self.current_trick[0][1])
        winning_card, winner = self.current_trick[0][1], self.current_trick[0][0]
        if self.phase == "concealed":
            for player, card in self.current_trick[1:]:
                if card_suit(card) == lead_suit and rank_index(card) > rank_index(winning_card):
                    winning_card, winner = card, player
        else:
            for player, card in self.current_trick[1:]:
                win_suit = card_suit(winning_card)
                if card_suit(card) == win_suit and rank_index(card) > rank_index(winning_card):
                    winning_card, winner = card, player
                elif self.trump_suit is not None and card_suit(card) == self.trump_suit and win_suit != self.trump_suit:
                    winning_card, winner = card, player
        points = sum(card_value(c) for _,c in self.current_trick)
        return winner, points

# =========================
# MCTS Node
# =========================
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
        return max(self.children, key=lambda c:c.visits)

# =========================
# MCTS Functions
# =========================
def select(node):
    while node.children:
        node = max(node.children, key=lambda c: c.uct_score())
    return node

def expand(node, env):
    hand = node.state["hands"][node.state["turn"]]
    tried = [c.action for c in node.children]
    # Compute valid moves in the context of the node's state
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
    # NN priors if available, else computed priors
    if NN_EVAL is not None:
        nn_priors, _ = NN_EVAL.predict(node.state, valid)
        priors = nn_priors
    else:
        priors = {a: compute_action_prior(temp_env, a) for a in valid}
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
            state, reward, done, _, _ = sim_env.step(action)
            child = MCTSNode(state, parent=node, action=action)
            child.prior = (priors.get(action, 0.01) / max_prior) if max_prior > 0 else 0.01
            node.children.append(child)
            return child
    return None

def estimate_trick_win_prob(env, acting_player, action, samples=6):
    wins = 0
    for _ in range(samples):
        sim = copy.deepcopy(env)
        sim.debug = False
        # Apply the candidate action
        _ = sim.step(action)
        # Finish the trick randomly
        # Keep stepping until current_trick empties (trick resolved)
        while sim.current_trick:
            hand = sim.hands[sim.turn]
            valid = sim.valid_moves(hand)
            if not valid:
                break
            rand_action = random.choice(valid)
            _, _, _, winner, _ = sim.step(rand_action)
        # After trick resolves, winner is set in last step; use sim.last_trick_winner
        winner = sim.last_trick_winner
        if winner is None:
            continue
        if winner % 2 == acting_player % 2:
            wins += 1
    return wins / max(1, samples)

# Override prior with a version that uses win-prob estimates and reveal penalties
def compute_action_prior(env, action):
    current_player = env.turn
    base = rank_index(action) / (len(RANKS) - 1)
    points = card_value(action) / 3.0
    is_trump = (card_suit(action) == env.trump_suit)
    if env.current_trick:
        # Estimate immediate trick win probability
        win_prob = estimate_trick_win_prob(env, current_player, action, samples=4)
        prior = 0.5 * win_prob + 0.25 * base + 0.25 * points
        # Penalize revealing trump on low-probability wins during concealed phase
        if env.phase == "concealed":
            lead_suit = card_suit(env.current_trick[0][1])
            has_lead = any(card_suit(c) == lead_suit for c in env.hands[current_player])
            if not has_lead and is_trump and win_prob < 0.45:
                prior *= 0.4
    else:
        # Lead prior: prefer long suit and strong card; avoid leading trump while concealed
        hand = env.hands[current_player]
        suit_len = sum(1 for c in hand if card_suit(c) == card_suit(action))
        prior = 0.45 * base + 0.25 * points + 0.3 * (suit_len / 8.0)
        if env.phase == "concealed" and is_trump:
            prior *= 0.5
    return max(0.01, min(1.0, prior))

def _winner_so_far(sim_env):
    if not sim_env.current_trick:
        return None, None
    lead_suit = card_suit(sim_env.current_trick[0][1])
    winning_card, winner = sim_env.current_trick[0][1], sim_env.current_trick[0][0]
    if sim_env.phase == "concealed":
        for player, card in sim_env.current_trick[1:]:
            if card_suit(card) == lead_suit and rank_index(card) > rank_index(winning_card):
                winning_card, winner = card, player
    else:
        for player, card in sim_env.current_trick[1:]:
            win_suit = card_suit(winning_card)
            if card_suit(card) == win_suit and rank_index(card) > rank_index(winning_card):
                winning_card, winner = card, player
            elif sim_env.trump_suit is not None and card_suit(card) == sim_env.trump_suit and win_suit != sim_env.trump_suit:
                winning_card, winner = card, player
    return winner, winning_card

def heuristic_rollout(sim_env):
    done = False
    while not done:
        hand = sim_env.hands[sim_env.turn]
        valid = sim_env.valid_moves(hand)
        if not valid: break
        if sim_env.current_trick:
            lead_suit = card_suit(sim_env.current_trick[0][1])
            winner_so_far, win_card = _winner_so_far(sim_env)
            my_team = 0 if sim_env.turn % 2 == 0 else 1
            leader_team = (winner_so_far % 2) if winner_so_far is not None else None
            same_suit = [c for c in valid if card_suit(c) == lead_suit]
            # Helper lists
            trumps = [c for c in valid if card_suit(c) == sim_env.trump_suit]
            non_points = [c for c in valid if card_value(c) == 0]
            # If teammate winning, play low to conserve strength
            if leader_team is not None and leader_team == my_team:
                if same_suit:
                    action = min(same_suit, key=rank_index)
                else:
                    action = min(non_points, key=rank_index) if non_points else min(valid, key=rank_index)
            else:
                # Try to win trick
                chosen = None
                if same_suit:
                    higher = [c for c in same_suit if rank_index(c) > rank_index(win_card)] if win_card and card_suit(win_card)==lead_suit else same_suit
                    if higher:
                        chosen = min(higher, key=rank_index)
                if chosen is None:
                    # Use trump if revealed or if playing a trump would expose (allowed)
                    if trumps and (sim_env.phase == "revealed" or not same_suit):
                        # choose minimal trump; try to overtrump if trump currently winning
                        if win_card and card_suit(win_card) == sim_env.trump_suit:
                            over = [c for c in trumps if rank_index(c) > rank_index(win_card)]
                            chosen = min(over, key=rank_index) if over else min(trumps, key=rank_index)
                        else:
                            chosen = min(trumps, key=rank_index)
                if chosen is None:
                    # Dump lowest non-point if cannot win
                    chosen = min(non_points, key=rank_index) if non_points else min(valid, key=rank_index)
                action = chosen
        else:
            # Lead: avoid trump in concealed phase (valid_moves already enforces for bidder). Lead strongest non-trump card.
            non_trump = [c for c in valid if card_suit(c) != sim_env.trump_suit]
            lead_pool = non_trump if non_trump else valid
            # Prefer high rank in longer suits
            suits = {}
            for c in lead_pool:
                suits.setdefault(card_suit(c), []).append(c)
            best_suit = max(suits.keys(), key=lambda s: (len(suits[s]), max(rank_index(c) for c in suits[s])))
            action = max(suits[best_suit], key=rank_index)
        _, _, done, _, _ = sim_env.step(action)
    return sim_env.scores[0]-sim_env.scores[1]

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
    # If NN is available, use value head as terminal evaluator when not done
    if NN_EVAL is not None:
        valid = sim_env.valid_moves(sim_env.hands[sim_env.turn])
        priors, value = NN_EVAL.predict(state, valid)
        # Heuristic rollout as backup blended with NN value
        rollout_value = heuristic_rollout(sim_env)
        return 0.5 * value * 28 + 0.5 * rollout_value
    return heuristic_rollout(sim_env)

def backpropagate(node, reward):
    while node:
        node.visits +=1
        node.value += reward
        node = node.parent

def mcts_search(env, state, iterations=50):
    root = MCTSNode(state)
    for _ in range(iterations):
        node = select(root)
        child = expand(node, env)
        if child is None:
            child = node
        reward = simulate_from_state(env, child.state)
        backpropagate(child, reward)
    return root.best_child().action

def mcts_plan(env, state, iterations=50):
    root = MCTSNode(state)
    for _ in range(iterations):
        node = select(root)
        child = expand(node, env)
        if child is None:
            child = node
        reward = simulate_from_state(env, child.state)
        backpropagate(child, reward)
    # Extract visit distribution over valid actions
    valid = env.valid_moves(state["hands"][state["turn"]])
    visits = {a: 0 for a in valid}
    for ch in root.children:
        if ch.action in visits:
            visits[ch.action] = ch.visits
    total = sum(visits.values())
    if total == 0:
        # fallback to equal or prior-based distribution
        for a in visits:
            visits[a] = 1
        total = len(visits)
    pi = {a: v/total for a, v in visits.items()}
    # Choose action as argmax visits
    best_action = max(pi.items(), key=lambda kv: kv[1])[0]
    return best_action, pi

def _percentile(values, q):
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(q * (len(s) - 1))))
    return s[k]

class MonteCarloBiddingAgent:
    def __init__(self, num_samples=6, mcts_iterations=60):
        self.num_samples = num_samples
        self.mcts_iterations = mcts_iterations
        self.last_debug = None
        self.last_choose_debug = None

    def _simulate_points_for_suit(self, my_first_four, suit, my_seat, first_player):
        all_ranks = ["7","8","9","10","J","Q","K","A"]
        full_deck = [r+s for r in all_ranks for s in SUITS]
        # Remove my known cards
        remaining = [c for c in full_deck if c not in my_first_four]
        team_points = []
        for _ in range(self.num_samples):
            pool = remaining[:]
            random.shuffle(pool)
            hands = [[] for _ in range(4)]
            # Other players get 8 each, bidder gets 4 more
            for p in range(4):
                if p == my_seat:
                    hands[p] = my_first_four + [pool.pop() for _ in range(4)]
                else:
                    hands[p] = [pool.pop() for _ in range(8)]
            # Build env from this completed deal
            env = TwentyEightEnv()
            env.hands = copy.deepcopy(hands)
            env.scores = [0, 0]
            env.game_score = [0, 0]
            env.current_trick = []
            env.turn = first_player
            env.bidder = my_seat
            env.trump_suit = suit
            env.phase = "concealed"
            env.debug = False
            trump_cards = [c for c in env.hands[env.bidder] if card_suit(c) == env.trump_suit]
            env.face_down_trump_card = max(trump_cards, key=rank_index) if trump_cards else None
            if env.face_down_trump_card:
                env.hands[env.bidder].remove(env.face_down_trump_card)
            env.last_exposer = None
            env.exposure_trick_index = None
            env.tricks_played = 0
            env.invalid_round = False
            env.last_trick_winner = None
            # Play the game with MCTS policy
            state = env.get_state()
            done = False
            # Cap total moves to avoid runaway
            safety_moves_cap = 8 * 4 + 2
            moves_done = 0
            while not done and moves_done < safety_moves_cap:
                move = mcts_search(env, state, self.mcts_iterations)
                state, _, done, _, _ = env.step(move)
                moves_done += 1
            bidder_team = 0 if my_seat % 2 == 0 else 1
            team_points.append(state["scores"][bidder_team])
        return team_points

    def propose_bid(self, my_first_four, current_high_bid, my_seat=0, first_player=0):
        # Consider only suits present in my first four
        present_suits = [s for s in SUITS if any(card_suit(c) == s for c in my_first_four)]
        if not present_suits:
            self.last_debug = {
                "present_suits": [],
                "suit_to_points": {},
                "chosen_suit": None,
                "avg_points": 0.0,
                "p30_points": 0.0,
                "proposed": None,
                "current_high": current_high_bid,
            }
            return None
            
        # Evaluate each present suit via Monte Carlo
        suit_to_points = {}
        for s in present_suits:
            pts = self._simulate_points_for_suit(my_first_four, s, my_seat, first_player)
            suit_to_points[s] = pts
        # Choose suit with highest average expected points
        def avg_pts(s):
            p = suit_to_points[s]
            return sum(p)/max(1,len(p))
        best_suit = max(present_suits, key=avg_pts)
        pts = suit_to_points[best_suit]
        avg_points = sum(pts)/max(1,len(pts))
        # Min allowed per auction state
        min_allowed = 16 if current_high_bid < 16 else current_high_bid + 1
        if avg_points > min_allowed:
            final_prop = min(28, int(math.ceil(avg_points)))
        else:
            final_prop = None
        self.last_debug = {
            "present_suits": present_suits,
            "suit_to_points": suit_to_points,
            "chosen_suit": best_suit,
            "avg_points": avg_points,
            "p30_points": _percentile(pts, 0.3),
            "proposed": final_prop,
            "current_high": current_high_bid,
        }
        return final_prop

    def choose_trump(self, my_first_four, final_bid):
        # Choose suit with highest expected points
        present_suits = [s for s in SUITS if any(card_suit(c) == s for c in my_first_four)]
        if not present_suits:
            chosen = max(SUITS, key=lambda s: -SUITS.index(s))
            self.last_choose_debug = {"chosen": chosen, "estimates": {}}
            return chosen
        best_s, best_e = None, -1
        estimates = {}
        for s in present_suits:
            # lightweight estimate without full sampling
            count = sum(1 for c in my_first_four if card_suit(c) == s)
            pts = sum(card_value(c) for c in my_first_four if card_suit(c) == s)
            est = 3*count + 2*pts
            estimates[s] = est
            if est > best_e:
                best_e = est
                best_s = s
        self.last_choose_debug = {"chosen": best_s, "estimates": estimates}
        return best_s

# =========================
# Play multiple games with concise debug
# =========================
def play_games(num_games=3, iterations=50):
    results_all=[]
    series_game_score = [0, 0]
    for g in range(1,num_games+1):
        print(f"\n===== GAME {g} =====")
        first_player = 0
        env = TwentyEightEnv()
        env.debug = True
        state = env.reset(initial_trump=None, first_player=first_player)
        done=False
        for i,hand in enumerate(env.hands):
            print(f"Player {i} hand : ",hand)
        # Show the exact first four cards used during the auction
        for i in range(4):
            print(f"Player {i} (auction 4 cards): {env.first_four_hands[i]}")
        print(f"Auction winner (bidder): Player {env.bidder} with bid {getattr(env,'bid_value',16)}")
        print(f"Bidder sets concealed trump suit: {state['trump']}")
        print(f"Phase: {state['phase']}, bidder concealed card: {env.face_down_trump_card}")
        print("")
        while not done:
            current_player = state['turn']  # store actual player
            move = policy_move(env, iterations)
            state, _, done, winner, trick_points = env.step(move)
            print(f"Player {current_player} plays {move}")
            if winner is not None:
                print(f"Player {winner} won the hand: {trick_points} points\n")  # blank line after hand
            if getattr(env, 'invalid_round', False):
                print("Round declared invalid: trump never exposed by end of 7th trick.")
                break
        print(f"Game {g} final points: Team A={state['scores'][0]}, Team B={state['scores'][1]}")
        # print(f"Game {g} cumulative game score (+1/-1 on bid): Team A={state['game_score'][0]}, Team B={state['game_score'][1]}")
        # Update series cumulative score across games
        series_game_score[0] += state['game_score'][0]
        series_game_score[1] += state['game_score'][1]
        print(f"Series cumulative game score so far: Team A={series_game_score[0]}, Team B={series_game_score[1]}")
        results_all.append(state['scores'])
    print("\nAll game results:", results_all)
    print(f"Final series cumulative game score: Team A={series_game_score[0]}, Team B={series_game_score[1]}")

def policy_move(env, iterations):
    state = env.get_state()
    # Run a modest MCTS to get candidates
    move = mcts_search(env, state, iterations)
    # Use the heuristic rollout policy to score all valid moves quickly
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
    # Blend: if heuristic best differs from MCTS by a large margin, pick heuristic; else stick with MCTS
    mcts_score = next((s for s,a2 in scored if a2 == move), None)
    best_score = max(s for s,_ in scored)
    if mcts_score is None or (best_score - mcts_score) > 2:
        return best_heur
    return move

if __name__=="__main__":
    play_games(num_games=10, iterations=5000)

DECK_RANKS_FULL = ["7","8","9","10","J","Q","K","A"]
FULL_DECK = [r+s for r in DECK_RANKS_FULL for s in SUITS]
CARD_TO_INDEX = {c:i for i,c in enumerate(FULL_DECK)}

def encode_state_to_tensor(state):
    # Flattened binary features for each player's hand (4 x 32), current_trick (4 x 32), trump (4), phase(2), turn(4), bidder(4)
    hands = state["hands"]
    features = []
    for p in range(4):
        vec = [0]*len(FULL_DECK)
        for c in hands[p]:
            vec[CARD_TO_INDEX[c]] = 1
        features.extend(vec)
    # Current trick vector
    trick_vec = [0]*len(FULL_DECK)
    for _, c in state["current_trick"]:
        trick_vec[CARD_TO_INDEX[c]] = 1
    features.extend(trick_vec)
    # Trump one-hot
    trump_vec = [0]*len(SUITS)
    if state["trump"] in SUITS:
        trump_vec[SUITS.index(state["trump"])]=1
    features.extend(trump_vec)
    # Phase one-hot
    phase_vec = [1,0] if state.get("phase","concealed")=="concealed" else [0,1]
    features.extend(phase_vec)
    # Turn one-hot
    turn_vec = [0,0,0,0]
    turn_vec[state["turn"]]=1
    features.extend(turn_vec)
    # Bidder one-hot
    bidder_vec = [0,0,0,0]
    bidder_vec[state.get("bidder",0)] = 1
    features.extend(bidder_vec)
    # Scores (points) normalized
    scores = state["scores"]
    features.extend([scores[0]/28.0, scores[1]/28.0])
    return features

class PolicyValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = 4*len(FULL_DECK) + len(FULL_DECK) + len(SUITS) + 2 + 4 + 4 + 2
        hidden = 512
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.policy_head = nn.Linear(hidden, len(FULL_DECK))
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x))  # [-1,1]
        return policy_logits, value

class NNEvaluator:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = PolicyValueNet().to(self.device)
        self.model.eval()

    def predict(self, state, valid_actions):
        with torch.no_grad():
            x = torch.tensor(encode_state_to_tensor(state), dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, value = self.model(x)
            logits = logits.squeeze(0)
            # Mask logits to valid actions
            mask = torch.full((len(FULL_DECK),), float('-inf'), device=self.device)
            for a in valid_actions:
                mask[CARD_TO_INDEX[a]] = 0.0
            masked_logits = logits + mask
            policy = F.softmax(masked_logits, dim=-1)
            # Return dict of action->prior and scalar value
            priors = {a: float(policy[CARD_TO_INDEX[a]].item()) for a in valid_actions}
            return priors, float(value.item())

    def load(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

if HAS_TORCH:
    try:
        NN_EVAL = NNEvaluator()
    except Exception:
        NN_EVAL = None

# ===== Self-play and Training Support =====
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.data = []
        self.idx = 0

    def add(self, state_vec, policy_vec, value):
        if len(self.data) < self.capacity:
            self.data.append((state_vec, policy_vec, value))
        else:
            self.data[self.idx] = (state_vec, policy_vec, value)
            self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.data, min(batch_size, len(self.data)))
        xs, ps, vs = zip(*batch)
        return list(xs), list(ps), list(vs)

# Map action distribution dict to full-deck policy vector
def policy_dict_to_vector(policy_dict):
    vec = [0.0]*len(FULL_DECK)
    for a, p in policy_dict.items():
        vec[CARD_TO_INDEX[a]] = p
    return vec


def self_play_episode(iterations=200):
    env = TwentyEightEnv()
    env.debug = False
    state = env.reset(initial_trump=None, first_player=0)
    episode = []
    done = False
    while not done:
        action, pi = mcts_plan(env, state, iterations)
        episode.append((encode_state_to_tensor(state), policy_dict_to_vector(pi)))
        state, _, done, _, _ = env.step(action)
        if getattr(env, 'invalid_round', False):
            break
    # Value target from bidder team perspective: +1 if bid met, else -1, normalized
    bidder_team = 0 if env.bidder % 2 == 0 else 1
    value = 1.0 if state["scores"][bidder_team] >= getattr(env, 'bid_value', 16) else -1.0
    return episode, value


def train_policy_value(episodes=10, iterations=200, batch_size=64, epochs=2, lr=1e-3):
    if NN_EVAL is None:
        print("Torch not available; skipping NN training.")
        return
    buffer = ReplayBuffer()
    # Collect self-play data
    for _ in range(episodes):
        ep, value = self_play_episode(iterations)
        for state_vec, pi_vec in ep:
            buffer.add(state_vec, pi_vec, value)
    # Train
    model = NN_EVAL.model
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        xs, ps, vs = buffer.sample(batch_size)
        if not xs:
            break
        x = torch.tensor(xs, dtype=torch.float32)
        p_target = torch.tensor(ps, dtype=torch.float32)
        v_target = torch.tensor(vs, dtype=torch.float32).unsqueeze(1)
        logits, v_pred = model(x)
        policy_loss = F.cross_entropy(logits, torch.argmax(p_target, dim=1))
        value_loss = F.mse_loss(v_pred, v_target)
        loss = policy_loss + value_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    model.eval()
    print(f"Training done. Avg loss: {sum(losses)/len(losses) if losses else 0:.4f}")
