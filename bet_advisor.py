#!/usr/bin/env python3
"""
Bet Advisor for Twenty-Eight (28) – Standalone

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

# Use the project's MCTS modules for ISMCTS-based evaluation
from mcts.constants import SUITS, RANKS, CARD_POINTS, card_suit, card_rank, card_value, rank_index
from mcts.env28 import TwentyEightEnv
from mcts.ismcts import ismcts_plan


SUITS = SUITS
RANKS = RANKS
CARD_POINTS = CARD_POINTS


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
    k = max(0, min(len(s) - 1, int(q * (len(s) - 1))))
    return s[k]


def simulate_points_for_suit_ismcts(my_first_four: List[str], suit: str, my_seat: int = 0, first_player: int = 0,
                                    num_samples: int = 6, base_iterations: int = 200) -> List[int]:
    all_ranks = ["7", "8", "9", "10", "J", "Q", "K", "A"]
    full_deck = [r + s for r in all_ranks for s in SUITS]
    remaining = [c for c in full_deck if c not in my_first_four]
    team_points: List[int] = []
    for _ in range(num_samples):
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
        state = env.get_state()
        done = False
        moves_done = 0
        safety_moves_cap = 8 * 4 + 2
        while not done and moves_done < safety_moves_cap:
            iters = max(10, base_iterations // 5)
            move, _ = ismcts_plan(env, state, iterations=iters, samples=6)
            state, _, done, _, _ = env.step(move)
            moves_done += 1
        bidder_team = 0 if my_seat % 2 == 0 else 1
        team_points.append(state["scores"][bidder_team])
    return team_points


def propose_bid_ismcts(first_four_cards: List[str], current_high_bid: int = 0,
                       num_samples: int = 6, base_iterations: int = 200) -> Tuple[int, str, Dict[str, Dict[str, float]]]:
    present_suits = [s for s in SUITS if any(card_suit(c) == s for c in first_four_cards)]
    if not present_suits:
        return 0, "H", {s: {"avg": 0.0, "p30": 0.0, "std": 0.0} for s in SUITS}

    suit_stats: Dict[str, Dict[str, float]] = {}
    for s in present_suits:
        pts = simulate_points_for_suit_ismcts(first_four_cards, s, my_seat=0, first_player=0,
                                              num_samples=num_samples, base_iterations=base_iterations)
        n = max(1, len(pts))
        mean = sum(pts) / n
        var = sum((v - mean) ** 2 for v in pts) / n
        std = var ** 0.5
        p30 = _percentile(pts, 0.3)
        suit_stats[s] = {"avg": mean, "std": std, "p30": p30}

    # Conservative thresholds
    raise_delta = 2 if current_high_bid < 20 else 1
    min_allowed = 16 if current_high_bid < 16 else current_high_bid + raise_delta

    candidates: List[Tuple[str, int, float, float, float]] = []
    for s in present_suits:
        avg_points = suit_stats[s]["avg"]
        std_points = suit_stats[s]["std"]
        p30 = suit_stats[s]["p30"]
        conf_mean = avg_points - 0.75 * std_points
        # Allow direct jump to 20 anytime current_high < 20 if strength supports it
        ok = (p30 >= min_allowed and conf_mean >= min_allowed) or (
            current_high_bid < 20 and p30 >= 20 and conf_mean >= 20
        )
        if ok:
            bid_raw = max(p30, avg_points - 1.0)
            bid_s = int(max(min_allowed, min(28, int(bid_raw))))
            # Normalize to 20 if we crossed 20 and current_high < 20 to reflect direct jump rule
            if current_high_bid < 20 and bid_s < 20 and (p30 >= 20 or conf_mean >= 20):
                bid_s = 20
            candidates.append((s, bid_s, avg_points, std_points, p30))

    if not candidates:
        # No safe raise → pass with the best suit suggestion for info
        best_suit = max(present_suits, key=lambda s: (suit_stats[s]["p30"], suit_stats[s]["avg"]))
        return 0, best_suit, suit_stats

    candidates.sort(key=lambda t: (t[4], t[2]))
    best_suit, final_prop, avg_points, std_points, p30 = candidates[-1]
    return final_prop, best_suit, suit_stats


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
    print("Bid Advisor (28)")
    print("----------------")
    cards = _prompt_cards()
    current_high = _prompt_current_high()

    suggested, trump, stats = propose_bid_ismcts(cards, current_high_bid=current_high,
                                                 num_samples=6, base_iterations=200)

    print(f"\nYour auction cards: {', '.join(cards)}")
    print(f"Current high bid:  {current_high if current_high > 0 else 'None'}")
    print("Estimated suit stats (avg / p30 / std):")
    for s in SUITS:
        if s in stats:
            d = stats[s]
            print(f"  {s}: {d['avg']:.2f} / {d['p30']:.2f} / {d['std']:.2f}")
    if suggested == 0:
        raise_delta = 2 if current_high < 20 else 1
        min_allowed = 16 if current_high < 16 else current_high + raise_delta
        print(f"\nRecommendation: PASS (min allowed to raise was {min_allowed}; direct 20 allowed if strong)")
    else:
        stakes = 2 if suggested >= 20 else 1
        print(f"\nRecommendation: BID {suggested} and choose trump = {trump} (stakes = {stakes})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())