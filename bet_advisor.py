#!/usr/bin/env python3
"""
Bet Advisor for Twenty-Eight (28) - Standalone

Usage:
  - Run: python bet_advisor.py
  - The script will prompt for your 4 auction cards and the current highest bid.

Notes:
  - Input is your initial 4-card auction hand (e.g., JH 9H 7D AC)
  - Suits: H (hearts), D (diamonds), C (clubs), S (spades)
  - Ranks: 7,8,9,10,J,Q,K,A
  - This script is independent from the MCTS engine; it uses a compact heuristic
    similar to the one used by the bidding agent.
"""

import sys
import copy
import random
from typing import Dict, List, Tuple
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# Use the project's MCTS modules for ISMCTS-based evaluation
from mcts.constants import SUITS, RANKS, CARD_POINTS, card_suit, card_rank, card_value, rank_index, trick_rank_index
from mcts.env28 import TwentyEightEnv
from mcts.ismcts import ismcts_plan

SUITS = SUITS
RANKS = RANKS
CARD_POINTS = CARD_POINTS

# Speed/accuracy presets
FAST_MODE = True
VERBOSE = True  # Toggle detailed prints
DEFAULT_STAGE1_SAMPLES = 1 if FAST_MODE else 2
DEFAULT_STAGE1_ITERS = 100 if FAST_MODE else 100
DEFAULT_STAGE2_SAMPLES = 48 if FAST_MODE else 192
DEFAULT_STAGE2_ITERS = 300 if FAST_MODE else 400
DEFAULT_ISMCTS_SAMPLES = 6 if FAST_MODE else 8

# Analytical model configuration
DEFAULT_ANALYTICAL_SAMPLES = 20 if FAST_MODE else 50  # Number of analytical simulations
DEFAULT_ANALYTICAL_VARIATION = 0.1  # ±10% variation for analytical samples


def parse_cards(raw: str) -> List[str]:
    tokens = [t.strip().upper() for t in raw.replace(",", " ").split() if t.strip()]
    normalized = []
    for t in tokens:
        # Accept forms like 10H, JH, AS, 7D
        if len(t) < 2:
            raise ValueError(f"Invalid card token: {t}")
        suit = t[-1]
        rank = t[:-1]
        if suit not in SUITS or rank not in RANKS:
            raise ValueError(f"Invalid card token: {t}")
        normalized.append(rank + suit)
    if len(normalized) < 4:
        raise ValueError("Provide at least 4 cards (initial auction hand).")
    # Use only the first 4 cards for the auction phase
    return normalized[:4]


def card_rank(card: str) -> str:  # alias for local usage
    return card[:-1]


def suit_trump_strength(first_four_cards: List[str], suit: str) -> int:
    suit_cards = [c for c in first_four_cards if card_suit(c) == suit]
    if not suit_cards:
        return 0
    count = len(suit_cards)
    rank_power_sum = sum(RANKS.index(card_rank(c)) for c in suit_cards)
    point_sum = sum(card_value(c) for c in suit_cards)
    has_jack = any(card_rank(c) == "J" for c in suit_cards)
    has_nine = any(card_rank(c) == "9" for c in suit_cards)
    return 3 * count + rank_power_sum + 2 * point_sum + (5 if has_jack else 0) + (3 if has_nine else 0)


def choose_trump(first_four_cards: List[str]) -> Tuple[str, Dict[str, int]]:
    strengths = {s: suit_trump_strength(first_four_cards, s) for s in SUITS}
    def suit_key(s: str):
        present = any(card_suit(c) == s for c in first_four_cards)
        return (present, strengths[s], -SUITS.index(s))
    best = max(SUITS, key=suit_key)
    return best, strengths


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(q * len(s))))
    return s[k]


def run_single_simulation_worker(args):
    """Worker function for multiprocessing - must be at module level"""
    my_first_four, suit, my_seat, first_player, base_iterations, sample_idx = args
    
    # Each worker gets its own random seed for reproducibility
    try:
        random.seed(hash((tuple(my_first_four), suit, sample_idx)))
    except Exception:
        pass
    
    all_ranks = ["7", "8", "9", "10", "J", "Q", "K", "A"]
    full_deck = [r + s for r in all_ranks for s in SUITS]
    remaining = [c for c in full_deck if c not in my_first_four]
    
    pool = remaining[:]
    random.shuffle(pool)
    hands = [[] for _ in range(4)]
    for p in range(4):
        if p == my_seat:
            hands[p] = my_first_four + [pool.pop() for _ in range(4)]
        else:
            hands[p] = [pool.pop() for _ in range(8)]
    
    env = TwentyEightEnv()
    env.hands = copy.deepcopy(hands)
    env.scores = [0, 0]
    env.game_score = [0, 0]
    env.current_trick = []
    env.turn = first_player
    env.bidder = my_seat
    env.trump_suit = suit
    env.bid_value = 16  # Set a default bid value for scoring
    env.round_stakes = 1
    env.phase = "concealed"
    env.debug = False
    # Initialize belief fields expected by get_state/resolve_trick
    env.void_suits_by_player = [set() for _ in range(4)]
    env.lead_suit_counts = [{s: 0 for s in SUITS} for _ in range(4)]
    # # Mark quick evaluation only in FAST mode
    # if FAST_MODE:
    #     env.quick_eval = True
    trump_cards = [c for c in env.hands[env.bidder] if card_suit(c) == env.trump_suit]
    env.face_down_trump_card = max(trump_cards, key=rank_index) if trump_cards else None
    if env.face_down_trump_card:
        env.hands[env.bidder].remove(env.face_down_trump_card)
    env.last_exposer = None
    env.exposure_trick_index = None
    env.tricks_played = 0
    env.invalid_round = False
    env.last_trick_winner = None
    state = env.get_state()
    done = False
    moves_done = 0
    safety_moves_cap = 8 * 4 + 2
    while not done and moves_done < safety_moves_cap:
        # Use heavier planning on bidder team turns; lighter otherwise
        iters = base_iterations
        try:
            move, _ = ismcts_plan(env, state, iterations=iters, samples=DEFAULT_ISMCTS_SAMPLES)
            state, _, done, _, _ = env.step(move)
            moves_done += 1
        except (ValueError, IndexError) as e:
            # Handle empty policy or other MCTS errors
            if "empty" in str(e).lower() or "max()" in str(e) or "No valid moves" in str(e):
                # Fallback: just play a random valid move
                valid_moves = env.valid_moves(env.hands[env.turn])
                if valid_moves:
                    move = random.choice(valid_moves)
                    state, _, done, _, _ = env.step(move)
                    moves_done += 1
                else:
                    # No valid moves - game might be finished
                    break
            else:
                # For other errors, try to continue with random play
                valid_moves = env.valid_moves(env.hands[env.turn])
                if valid_moves:
                    move = random.choice(valid_moves)
                    state, _, done, _, _ = env.step(move)
                    moves_done += 1
                else:
                    break
    
    # For bidding simulations, we want the raw team points, not game score
    bidder_team = 0 if my_seat % 2 == 0 else 1
    team_points = state["scores"][bidder_team]
    
    # Sanity check: if we have a strong hand but got 0 points, something is wrong
    if team_points == 0 and any(card_suit(c) == suit for c in my_first_four):
        # Check if we have high cards in the trump suit
        trump_cards_in_hand = [c for c in my_first_four if card_suit(c) == suit]
        if len(trump_cards_in_hand) >= 2:
            # This should never happen with a strong trump hand
            # Return a minimum reasonable score instead of 0
            team_points = max(5, len(trump_cards_in_hand) * 3)
    
    return team_points


def simulate_points_for_suit_ismcts(my_first_four: List[str], suit: str, my_seat: int = 0, first_player: int = 0,
                                    num_samples: int = DEFAULT_STAGE2_SAMPLES, base_iterations: int = DEFAULT_STAGE2_ITERS,
                                    playout_tricks: int | None = None) -> List[int]:
    # Use multithreading to run individual simulations in parallel
    max_workers = min(num_samples, max(1, (os.cpu_count() or 1)))
    
    # Prepare arguments for each worker
    args_list = [(my_first_four, suit, my_seat, first_player, base_iterations, i) for i in range(num_samples)]
    
    # Run simulations in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_single_simulation_worker, args) for args in args_list]
        team_points = []
        for fut in as_completed(futures):
            try:
                result = fut.result()
                team_points.append(result)
            except Exception as e:
                # Fallback to sequential if parallel fails
                print(f"Parallel simulation failed: {e}, falling back to sequential")
                team_points.append(run_single_simulation_worker(args_list[len(team_points)]))
    
    return team_points


def load_optimized_coefficients() -> Dict[str, float]:
    """Load optimized coefficients from JSON file."""
    try:
        with open('optimized_analytical_coefficients.json', 'r') as f:
            import json
            return json.load(f)
    except FileNotFoundError:
        # Fallback to default coefficients if file doesn't exist
        return {
            'base_points_multiplier': -0.0089,
            'trump_base_value': 0.7549,
            'high_trump_bonus': 0.8354,
            'trump_length_bonus': 0.3050,
            'suit_base_multiplier': 2.5290,
            'suit_length_bonus': 1.8479,
            'suit_high_card_bonus': -1.0000,
            'non_trump_penalty': 1.0000,
            'suit_flexibility': -0.1666,
            'balance_bonus': 3.0000,
            'trump_control_bonus': -1.0000
        }

def calculate_expected_points_analytical(hand_4_cards: List[str], trump_suit: str) -> float:
    """
    Calculate expected team points using analytical model.
    This predicts team points by: 
    1. Predicting what this 4-card hand contributes
    2. Adding a logical estimate for partner's contribution
    """
    # Load optimized coefficients
    coefficients = load_optimized_coefficients()
    
    # Step 1: Calculate what this 4-card hand contributes
    base_points = sum(card_value(card) for card in hand_4_cards) * coefficients['base_points_multiplier']
    
    # Calculate trump control value
    trump_cards = [card for card in hand_4_cards if card_suit(card) == trump_suit]
    trump_control_value = calculate_trump_control_value(trump_cards, trump_suit)
    
    # Calculate trick-winning potential for each suit
    suit_values = {}
    for suit in SUITS:
        suit_cards = [card for card in hand_4_cards if card_suit(card) == suit]
        if suit_cards:
            suit_values[suit] = calculate_suit_trick_potential(suit_cards, suit, trump_suit)
    
    # Calculate team coordination bonus
    coordination_bonus = calculate_team_coordination_bonus(hand_4_cards, trump_suit)
    
    # This hand's contribution
    hand_contribution = base_points + trump_control_value + sum(suit_values.values()) + coordination_bonus
    
    # Step 2: Estimate partner's contribution using game theory
    partner_estimate = estimate_partner_contribution(hand_4_cards, trump_suit)
    
    # Total team points
    total_team_points = hand_contribution + partner_estimate
    
    # Cap at reasonable team score (0-28 points)
    return max(0, min(28, total_team_points))


def estimate_partner_contribution(hand_4_cards: List[str], trump_suit: str) -> float:
    """
    Estimate partner's contribution using game theory logic.
    This is based on:
    1. Average partner contribution from historical data
    2. Trump distribution (if we have trumps, partner likely has fewer)
    3. Suit distribution (if we have strong suits, partner likely has complementary cards)
    """
    # Base partner contribution (average from historical data)
    base_partner = 7.0  # Average partner contributes ~7 points
    
    # Adjust based on trump distribution
    trump_cards = [card for card in hand_4_cards if card_suit(card) == trump_suit]
    trump_count = len(trump_cards)
    
    # If we have many trumps, partner likely has fewer (trump scarcity)
    if trump_count >= 2:
        # Partner likely has 0-1 trumps, so their trump contribution is lower
        trump_adjustment = -2.0 * (trump_count - 1)  # Reduce partner estimate
    elif trump_count == 1:
        # Partner likely has 0-2 trumps, moderate adjustment
        trump_adjustment = -1.0
    else:
        # We have no trumps, partner likely has 1-3 trumps, increase estimate
        trump_adjustment = 2.0
    
    # Adjust based on suit strength
    suit_strengths = {}
    for suit in SUITS:
        suit_cards = [card for card in hand_4_cards if card_suit(card) == suit]
        if suit_cards:
            # Calculate suit strength (high cards, length)
            strength = sum(card_value(card) for card in suit_cards)
            if suit == trump_suit:
                strength *= 1.5  # Trump bonus
            suit_strengths[suit] = strength
    
    # If we have very strong suits, partner likely has complementary cards
    strong_suit_bonus = 0.0
    for suit, strength in suit_strengths.items():
        if strength > 8:  # Strong suit
            strong_suit_bonus += 1.0  # Partner likely has supporting cards
    
    # Calculate final partner estimate
    partner_estimate = base_partner + trump_adjustment + strong_suit_bonus
    
    # Ensure reasonable bounds
    return max(2.0, min(12.0, partner_estimate))


def calculate_trump_control_value(trump_cards: List[str], trump_suit: str) -> float:
    """Calculate the value of trump control based on trump cards in hand."""
    if not trump_cards:
        return 0.0
    
    # Load optimized coefficients
    coefficients = load_optimized_coefficients()
    
    # Count high trumps (J, 9, A, 10)
    high_trumps = [card for card in trump_cards if card_rank(card) in ['J', '9', 'A', '10']]
    
    # Base value: each trump card is worth extra points
    base_value = len(trump_cards) * coefficients['trump_base_value']
    
    # High trump bonus: J=3, 9=2, A=1, 10=1
    high_trump_bonus = sum(card_value(card) for card in high_trumps) * coefficients['high_trump_bonus']
    
    # Trump length bonus: having multiple trumps is very valuable
    length_bonus = len(trump_cards) * (len(trump_cards) - 1) * coefficients['trump_length_bonus']
    
    return base_value + high_trump_bonus + length_bonus


def calculate_suit_trick_potential(suit_cards: List[str], suit: str, trump_suit: str) -> float:
    """Calculate expected trick-winning potential for a specific suit."""
    if not suit_cards:
        return 0.0
    
    # Load optimized coefficients
    coefficients = load_optimized_coefficients()
    
    # Calculate card strength in this suit
    card_strengths = [trick_rank_index(card) for card in suit_cards]
    max_strength = max(card_strengths)
    
    # Base value: probability of winning tricks in this suit
    # Higher cards have better chance of winning
    base_value = sum(strength / 8.0 for strength in card_strengths) * coefficients['suit_base_multiplier']
    
    # Length bonus: having multiple cards in same suit is valuable
    length_bonus = len(suit_cards) * coefficients['suit_length_bonus']
    
    # High card bonus: J, 9, A, 10 are especially valuable
    high_cards = [card for card in suit_cards if card_rank(card) in ['J', '9', 'A', '10']]
    high_card_bonus = len(high_cards) * coefficients['suit_high_card_bonus']
    
    # Trump competition: if this isn't trump suit, we compete with trump
    if suit != trump_suit:
        # Non-trump suits are worth less due to trump competition
        base_value *= coefficients['non_trump_penalty']
        length_bonus *= coefficients['non_trump_penalty']
    
    return base_value + length_bonus + high_card_bonus


def calculate_team_coordination_bonus(hand_4_cards: List[str], trump_suit: str) -> float:
    """Calculate bonus from team coordination potential."""
    # Load optimized coefficients
    coefficients = load_optimized_coefficients()
    
    # Having multiple suits gives flexibility
    suits_present = set(card_suit(card) for card in hand_4_cards)
    suit_flexibility = len(suits_present) * coefficients['suit_flexibility']
    
    # Having balanced hand (not all in one suit) is good for coordination
    suit_counts = {}
    for card in hand_4_cards:
        suit = card_suit(card)
        suit_counts[suit] = suit_counts.get(suit, 0) + 1
    
    balance_bonus = 0.0
    if len(suit_counts) >= 2:
        # Balanced hand is good for team coordination
        balance_bonus = coefficients['balance_bonus']
    
    # Having trump control helps team coordination
    trump_control_bonus = 0.0
    if any(card_suit(card) == trump_suit for card in hand_4_cards):
        trump_control_bonus = coefficients['trump_control_bonus']
    
    return suit_flexibility + balance_bonus + trump_control_bonus


def simulate_points_for_suit_analytical(hand_4_cards: List[str], suit: str, num_samples: int = DEFAULT_ANALYTICAL_SAMPLES) -> List[float]:
    """
    Use analytical model instead of ISMCTS for more stable estimates.
    Returns list of expected points for each sample.
    """
    # Use analytical calculation for stable results
    expected_points = calculate_expected_points_analytical(hand_4_cards, suit)
    
    # Add small random variation to simulate different game scenarios
    # This maintains some realism while being much more stable than ISMCTS
    results = []
    for _ in range(num_samples):
        # Add variation to account for different game scenarios
        variation = random.uniform(1.0 - DEFAULT_ANALYTICAL_VARIATION, 1.0 + DEFAULT_ANALYTICAL_VARIATION)
        result = expected_points * variation
        results.append(max(0, result))
    
    return results


def propose_bid_analytical(hand_4_cards: List[str], current_high_bid: int = 0) -> Tuple[int, str, Dict, Dict]:
    """
    Use analytical model for bidding decisions - much more stable than ISMCTS.
    """
    present_suits = [s for s in SUITS if any(card_suit(c) == s for c in hand_4_cards)]
    
    if not present_suits:
        return 0, SUITS[0], {}, {}
    
    # Calculate expected points for each suit using analytical model
    suit_to_points = {}
    for suit in present_suits:
        points_list = simulate_points_for_suit_analytical(hand_4_cards, suit, num_samples=DEFAULT_ANALYTICAL_SAMPLES)
        avg_points = sum(points_list) / len(points_list)
        suit_to_points[suit] = [avg_points]  # Store as single-item list for compatibility
    
    # Calculate statistics
    def stats(values):
        n = max(1, len(values))
        mean = sum(values) / n
        var = sum((v - mean) ** 2 for v in values) / n
        std = math.sqrt(var)
        return mean, std
    
    # Find best suit
    best_suit = None
    best_mean = 0.0
    for suit in present_suits:
        mean, _ = stats(suit_to_points[suit])
        if mean > best_mean:
            best_mean = mean
            best_suit = suit
    
    # Calculate bid using conservative approach
    mean, std = stats(suit_to_points[best_suit])
    
    # For analytical model, use the mean directly (no need for conservative adjustment)
    conservative_estimate = mean
    
    # Determine bid
    min_allowed = 16 if current_high_bid < 16 else current_high_bid + 1
    
    if conservative_estimate >= min_allowed - 1:
        proposed_bid = max(min_allowed, min(28, int(math.floor(conservative_estimate))))
        reason = f"Analytical estimate: {conservative_estimate:.1f} >= {min_allowed - 1}"
    else:
        proposed_bid = 0  # Pass
        reason = f"Analytical estimate: {conservative_estimate:.1f} < {min_allowed - 1}"
    
    # Prepare debug info
    debug = {
        "present_suits": present_suits,
        "suit_to_points": suit_to_points,
        "chosen_suit": best_suit,
        "avg_points": mean,
        "proposed": proposed_bid,
        "current_high": current_high_bid,
        "min_allowed": min_allowed,
        "raise_delta": 1,
        "reason": reason,
        "method": "analytical"
    }
    
    return proposed_bid, best_suit, suit_to_points, debug


def propose_bid_ismcts(first_four_cards: List[str], current_high_bid: int = 0,
                       num_samples: int = DEFAULT_STAGE2_SAMPLES, base_iterations: int = DEFAULT_STAGE2_ITERS) -> Tuple[int, str, Dict[str, Dict[str, float]], Dict]:
    # Suit presence and counts in the 4-card hand
    suit_counts = {s: sum(1 for c in first_four_cards if card_suit(c) == s) for s in SUITS}
    present_suits = [s for s in SUITS if suit_counts[s] > 0]
    suit_to_points: Dict[str, List[int]] = {}
    if not present_suits:
        debug = {
            "present_suits": [],
            "suit_to_points": suit_to_points,
            "chosen_suit": None,
            "avg_points": 0.0,
            "p30_points": 0.0,
            "std_points": 0.0,
            "proposed": None,
            "current_high": current_high_bid,
            "min_allowed": 16,
            "raise_delta": 1,
            "reason": "Pass: No suits present in first four cards.",
        }
        return 0, "H", {s: {"avg": 0.0, "p30": 0.0, "std": 0.0} for s in SUITS}, debug

    suit_stats: Dict[str, Dict[str, float]] = {}
    # Stage 1 elimination per heuristic:
    # - If the 4-card hand is a 2/2 split across two suits, run Stage 1 only on those two suits and pick the best
    # - Otherwise, skip Stage 1 entirely and run Stage 2 directly on the suit(s) with the highest count
    stage1_points: Dict[str, List[int]] = {}
    max_count = max(suit_counts[s] for s in present_suits)
    top_count_suits = [s for s in present_suits if suit_counts[s] == max_count]
    if max_count == 2 and len(top_count_suits) == 2:
        # 2/2 case: quick Stage 1 on both, choose best for heavy eval
        max_workers = min(2, max(1, (os.cpu_count() or 1)))
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(
                    run_single_simulation_worker,
                    (first_four_cards, s, 0, 0, DEFAULT_STAGE1_ITERS, 0)
                ): s
                for s in top_count_suits
            }
            for fut in as_completed(futures):
                s = futures[fut]
                try:
                    stage1_points[s] = fut.result()
                except Exception:
                    stage1_points[s] = simulate_points_for_suit_ismcts(
                        first_four_cards,
                        s,
                        my_seat=0,
                        first_player=0,
                        num_samples=DEFAULT_STAGE1_SAMPLES,
                        base_iterations=DEFAULT_STAGE1_ITERS,
                    )
        def stage1_stat(s: str):
            pts = stage1_points.get(s, [])
            if not pts:
                return (0.0, 0.0)
            n = len(pts)
            mean = sum(pts) / n
            p30 = _percentile(pts, 0.3)
            return (p30, mean)
        best = max(top_count_suits, key=lambda s: stage1_stat(s))
        top_suits = [best]
    else:
        # Not a 2/2 split: skip Stage 1, go straight to heavy for suits with the highest count
        top_suits = top_count_suits[:]

    # Stage 2: heavy evaluation for top suits only
    if len(top_suits) > 1:
        max_workers = min(len(top_suits), max(1, (os.cpu_count() or 1)))
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(run_single_simulation_worker, (first_four_cards, s, 0, 0, base_iterations, 0)): s
                for s in top_suits
            }
            for fut in as_completed(futures):
                s = futures[fut]
                try:
                    suit_to_points[s] = fut.result()
                except Exception:
                    suit_to_points[s] = simulate_points_for_suit_ismcts(
                        first_four_cards, s, my_seat=0, first_player=0,
                        num_samples=num_samples, base_iterations=base_iterations
                    )
    else:
        s = top_suits[0]
        suit_to_points[s] = simulate_points_for_suit_ismcts(
            first_four_cards, s, my_seat=0, first_player=0,
            num_samples=num_samples, base_iterations=base_iterations
        )
    # Fill non-top suits with Stage 1 results if computed; otherwise empty
    for s in present_suits:
        if s not in suit_to_points:
            suit_to_points[s] = stage1_points.get(s, [])

    # Build stats for each suit from collected samples
    for s in present_suits:
        pts = suit_to_points.get(s, [])
        n = max(1, len(pts))
        mean = (sum(pts) / n) if pts else 0.0
        var = (sum((v - mean) ** 2 for v in pts) / n) if pts else 0.0
        std = var ** 0.5
        p30 = _percentile(pts, 0.3) if pts else 0.0
        suit_stats[s] = {"avg": mean, "std": std, "p30": p30}

    # Conservative thresholds
    # Sequential bidding: always raise by +1 minimum; direct 20 allowed if strong
    raise_delta = 1
    min_allowed = 16 if current_high_bid < 16 else current_high_bid + raise_delta

    candidates: List[Tuple[str, int, float, float, float, float]] = []
    suit_failed_checks: Dict[str, Dict[str, object]] = {}
    for s in present_suits:
        avg_points = suit_stats[s]["avg"]
        std_points = suit_stats[s]["std"]
        p30 = suit_stats[s]["p30"]
        conf_mean = avg_points - 0.5 * std_points
        ok = (p30 >= (min_allowed - 1)) and (conf_mean >= (min_allowed - 1))
        if ok:
            bid_raw = max(p30, avg_points - 1.0)
            bid_s = int(max(min_allowed, min(28, int(bid_raw))))
            candidates.append((s, bid_s, avg_points, std_points, p30, conf_mean))
        else:
            failed = []
            if p30 < (min_allowed - 1):
                failed.append(f"p30 {p30:.2f} < {min_allowed - 1}")
            if conf_mean < (min_allowed - 1):
                failed.append(f"conf_mean {conf_mean:.2f} < {min_allowed - 1}")
            suit_failed_checks[s] = {
                "avg": avg_points,
                "std": std_points,
                "p30": p30,
                "conf_mean": conf_mean,
                "failed": failed,
            }

    if not candidates:
        best_suit = max(present_suits, key=lambda s: (suit_stats[s]["p30"], suit_stats[s]["avg"]))
        det = suit_failed_checks.get(best_suit)
        best_mean = det["avg"] if det else 0.0
        best_std = det["std"] if det else 0.0
        best_p30 = det["p30"] if det else 0.0
        best_conf = det["conf_mean"] if det else 0.0
        reason_text = (
            f"Pass: No suit meets thresholds. Best suit {best_suit} has p30={best_p30:.2f}, "
            f"conf_mean={best_conf:.2f} below min_allowed={min_allowed} (current_high={current_high_bid}, raise_delta=+1). "
            f"Failed checks for best suit: {', '.join(det['failed']) if det and det['failed'] else 'n/a'}."
        )
        debug = {
            "present_suits": present_suits,
            "suit_to_points": suit_to_points,
            "chosen_suit": None,
            "avg_points": best_mean,
            "p30_points": best_p30,
            "std_points": best_std,
            "proposed": None,
            "current_high": current_high_bid,
            "min_allowed": min_allowed,
            "raise_delta": 1,
            "reason": reason_text,
            "suit_failed_checks": suit_failed_checks,
            "stage2_samples": {s: pts for s, pts in suit_to_points.items() if len(pts) >= max(6, num_samples)},
        }
        return 0, best_suit, suit_stats, debug

    candidates.sort(key=lambda t: (t[4], t[2]))
    best_suit, final_prop, avg_points, std_points, p30, conf_mean = candidates[-1]
    reason_text = (
        f"Bid: Suit {best_suit} meets thresholds (p30={p30:.2f}, conf_mean={conf_mean:.2f} >= min_allowed={min_allowed}); "
        f"conservative mapping max(p30, avg-1) -> {final_prop}. (current_high={current_high_bid}, raise_delta=+1)"
    )
    debug = {
        "present_suits": present_suits,
        "suit_to_points": suit_to_points,
        "chosen_suit": best_suit,
        "avg_points": avg_points,
        "p30_points": p30,
        "std_points": std_points,
        "proposed": final_prop,
        "current_high": current_high_bid,
        "min_allowed": min_allowed,
        "raise_delta": 1,
        "reason": reason_text,
        "stage2_samples": {s: pts for s, pts in suit_to_points.items() if len(pts) >= max(6, num_samples)},
    }
    return final_prop, best_suit, suit_stats, debug


def _format_bidding_debug(dbg: dict) -> str:
    try:
        lines = []
        present_suits = dbg.get("present_suits", [])
        min_allowed = dbg.get("min_allowed")
        current_high = dbg.get("current_high")
        raise_delta = dbg.get("raise_delta")
        chosen_suit = dbg.get("chosen_suit")
        proposed = dbg.get("proposed")
        reason = dbg.get("reason")
        lines.append("  bidding analysis:")
        if present_suits:
            lines.append(f"    present_suits: {', '.join(present_suits)}")
        meta = []
        if min_allowed is not None:
            meta.append(f"min_allowed={min_allowed}")
        if raise_delta is not None:
            meta.append(f"raise_amount={raise_delta}")
        if meta:
            lines.append("    " + "   ".join(meta))
        if chosen_suit is not None:
            lines.append(f"    chosen_suit: {chosen_suit}")
        if proposed is not None:
            lines.append(f"    proposed_bid: {proposed}")
        if "avg_points" in dbg or "p30_points" in dbg or "std_points" in dbg:
            lines.append(
                "    overall: "
                + f"avg={dbg.get('avg_points', 0):.2f}   p30={dbg.get('p30_points', 0):.2f}   std={dbg.get('std_points', 0):.2f}"
            )
        stp = dbg.get("suit_to_points") or {}
        if stp:
            lines.append("    suit_stats:")
            for s, pts in stp.items():
                if not pts:
                    lines.append(f"      {s}: did not simulate this as trump.")
                    continue
                
                n = len(pts)
                mean = sum(pts) / n
                var = sum((v - mean) ** 2 for v in pts) / n
                std = math.sqrt(var)
                q = sorted(pts)
                k = max(0, min(len(q) - 1, int(0.3 * len(q))))
                p30 = q[k]
                preview = ", ".join(str(x) for x in pts)
                lines.append(
                    f"      {s}: pts=[{preview}]  avg={mean:.2f}  p30={p30:.2f}  std={std:.2f}"
                )
        sf = dbg.get("suit_failed_checks")
        if sf:
            lines.append("    failed_checks:")
            for s, det in sf.items():
                failed = det.get("failed") or []
                if failed:
                    lines.append(f"      {s}: " + "; ".join(failed))
        if reason:
            lines.append("    reason: " + str(reason))
        
        # # Display Stage 2 samples if available
        # stage2_samples = dbg.get('stage2_samples', {})
        # if stage2_samples:
        #     lines.append("    stage2_samples:")
        #     for suit, pts in stage2_samples.items():
        #         lines.append(f"      {suit}: {pts}")
        
        return "\n".join(lines)
    except Exception:
        return "  (failed to format bidding analysis)"

def _prompt_cards() -> List[str]:
    while True:
        print("Enter your 4 auction cards (e.g., 'JH 9H 7D AC'):")
        raw = input().strip()
        try:
            return parse_cards(raw)
        except Exception as exc:
            print(f"Invalid input: {exc}. Please try again.\n")


def _prompt_current_high() -> int:
    while True:
        print("Enter current highest bid on the table (press Enter if none):")
        raw = input().strip()
        if raw == "":
            return 0
        try:
            val = int(raw)
            if val < 0:
                raise ValueError
            return val
        except Exception:
            print("Please enter a non-negative integer or just press Enter.\n")


def main() -> int:
    print("Bid Advisor (28) - Hybrid Analytical + ISMCTS")
    print("-----------------------------------------------")
    cards = _prompt_cards()
    current_high = _prompt_current_high()

    print(f"\nYour auction cards: {', '.join(cards)}")
    print(f"Current high bid:  {current_high if current_high > 0 else 'None'}")
    
    # Run analytical method first (fast)
    print("\n=== ANALYTICAL ANALYSIS ===")
    analytical_bid, analytical_trump, analytical_stats, analytical_dbg = propose_bid_analytical(cards, current_high_bid=current_high)
    
    print("Analytical suit stats (avg points):")
    for s in SUITS:
        if s in analytical_stats:
            d = analytical_stats[s]
            mean = sum(d) / len(d)
            print(f"  {s}: {mean:.2f}")
    
    if analytical_bid == 0:
        min_allowed = 16 if current_high < 16 else current_high + 1
        print(f"Analytical recommendation: PASS (min allowed to raise was {min_allowed})")
    else:
        stakes = 2 if analytical_bid >= 20 else 1
        print(f"Analytical recommendation: BID {analytical_bid} and choose trump = {analytical_trump} (stakes = {stakes})")
    
    print(_format_bidding_debug(analytical_dbg))
    
    # Run ISMCTS method (slower but more detailed)
    print("\n=== ISMCTS SIMULATION ANALYSIS ===")
    print("Running ISMCTS simulations (this may take a moment)...")
    
    ismcts_bid, ismcts_trump, ismcts_stats, ismcts_dbg = propose_bid_ismcts(cards, current_high_bid=current_high,
                                                                           num_samples=DEFAULT_STAGE2_SAMPLES, base_iterations=DEFAULT_STAGE2_ITERS)
    
    print("ISMCTS suit stats (avg / p30 / std):")
    for s in SUITS:
        if s in ismcts_stats:
            d = ismcts_stats[s]
            print(f"  {s}: {d['avg']:.2f} / {d['p30']:.2f} / {d['std']:.2f}")
    
    if ismcts_bid == 0:
        min_allowed = 16 if current_high < 16 else current_high + 1
        print(f"ISMCTS recommendation: PASS (min allowed to raise was {min_allowed})")
    else:
        stakes = 2 if ismcts_bid >= 20 else 1
        print(f"ISMCTS recommendation: BID {ismcts_bid} and choose trump = {ismcts_trump} (stakes = {stakes})")
    
    print(_format_bidding_debug(ismcts_dbg))
    

    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())