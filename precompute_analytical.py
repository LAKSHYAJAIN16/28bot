#!/usr/bin/env python3
"""
Precompute analytical bid recommendations for all 4-card combinations.
This uses the analytical model (game theory + coefficients) from bet_advisor.py.

Run:
  python precompute_analytical.py
"""

import itertools
import json
import math
import os
import time
from typing import List, Dict, Tuple

from mcts.constants import FULL_DECK, card_suit, RANKS, SUITS as SUITS_CONST, card_value, card_rank
from bet_advisor import (
    calculate_expected_points_analytical, 
    load_optimized_coefficients,
    calculate_trump_control_value,
    calculate_suit_trick_potential,
    calculate_team_coordination_bonus,
    estimate_partner_contribution
)

# Predefined output configuration
OUT_PATH = "precomputed_analytical.jsonl"  # Separate file for analytical precomputations
LIMIT = 0  # 0 = all combinations
SUITS = SUITS_CONST


def normalize_four(cards: List[str]) -> List[str]:
    # Sort by suit then rank text to canonicalize
    return sorted(cards, key=lambda c: (card_suit(c), c))

def _rank_key(card: str) -> int:
    rank = card[:-1]
    return RANKS.index(rank)

def canonical_key(cards: List[str]) -> tuple:
    # Generate a suit-permutation-invariant key by applying all suit permutations
    # and picking the lexicographically smallest normalized tuple
    best = None
    for perm in itertools.permutations(SUITS_CONST):
        # mapping original suit -> permuted suit by position
        suit_map = {s: perm[i] for i, s in enumerate(SUITS_CONST)}
        mapped = [card[:-1] + suit_map[card[-1]] for card in cards]
        mapped_sorted = sorted(mapped, key=lambda c: (_rank_key(c), SUITS.index(card_suit(c))))
        tup = tuple(mapped_sorted)
        if best is None or tup < best:
            best = tup
    return best


def calculate_analytical_suit_stats(first_four_cards: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Calculate analytical statistics for each suit using the analytical model.
    Returns suit_stats with avg, std, p30 for each suit.
    """
    suit_stats = {}
    present_suits = [s for s in SUITS if any(card_suit(c) == s for c in first_four_cards)]
    
    for suit in present_suits:
        # Run multiple analytical simulations with variation
        points_list = []
        for _ in range(20):  # 20 analytical samples
            points = calculate_expected_points_analytical(first_four_cards, suit)
            points_list.append(points)
        
        # Calculate statistics
        n = len(points_list)
        mean = sum(points_list) / n
        var = sum((p - mean) ** 2 for p in points_list) / n
        std = var ** 0.5
        p30 = sorted(points_list)[max(0, min(n-1, int(0.3 * n)))]
        
        suit_stats[suit] = {
            "avg": mean,
            "std": std,
            "p30": p30,
            "samples": points_list
        }
    
    return suit_stats


def propose_bid_analytical(first_four_cards: List[str], current_high_bid: int = 0) -> Tuple[int, str, Dict]:
    """
    Propose a bid using analytical model - same logic as bet_advisor.py
    """
    present_suits = [s for s in SUITS if any(card_suit(c) == s for c in first_four_cards)]
    if not present_suits:
        return None, None, {}
    
    # Get analytical suit statistics
    suit_stats = calculate_analytical_suit_stats(first_four_cards)
    
    # Conservative thresholds
    raise_delta = 1
    min_allowed = 16 if current_high_bid < 16 else current_high_bid + raise_delta
    
    candidates = []
    suit_failed_checks = {}
    
    for s in present_suits:
        stats = suit_stats[s]
        avg_points = stats["avg"]
        std_points = stats["std"]
        p30 = stats["p30"]
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
        # Report best suit stats even when passing
        def suit_key(s):
            return (suit_stats[s]["p30"], suit_stats[s]["avg"])
        best_s = max(present_suits, key=suit_key)
        det = suit_failed_checks.get(best_s)
        best_mean = det["avg"] if det else 0.0
        best_std = det["std"] if det else 0.0
        best_p30 = det["p30"] if det else 0.0
        best_conf = det["conf_mean"] if det else 0.0
        
        return None, best_s, {
            "present_suits": present_suits,
            "suit_stats": suit_stats,
            "chosen_suit": None,
            "avg_points": best_mean,
            "p30_points": best_p30,
            "std_points": best_std,
            "proposed": None,
            "current_high": current_high_bid,
            "min_allowed": min_allowed,
            "raise_delta": raise_delta,
            "suit_failed_checks": suit_failed_checks,
        }
    
    candidates.sort(key=lambda t: (t[4], t[2]))
    best_suit, final_prop, avg_points, std_points, p30, conf_mean = candidates[-1]
    
    return final_prop, best_suit, {
        "present_suits": present_suits,
        "suit_stats": suit_stats,
        "chosen_suit": best_suit,
        "avg_points": avg_points,
        "p30_points": p30,
        "std_points": std_points,
        "proposed": final_prop,
        "current_high": current_high_bid,
        "min_allowed": min_allowed,
        "raise_delta": raise_delta,
    }


def main() -> int:
    all_cards = FULL_DECK
    total_raw = math.comb(len(all_cards), 4)
    combos = itertools.combinations(all_cards, 4)
    count_unique = 0  # number of NEW unique entries written in this run
    seen = set()
    
    # Load existing precomputed entries and seed the seen-set to avoid recomputing
    pre_existing = 0
    try:
        with open(OUT_PATH, "r", encoding="utf-8") as f_in:
            for line in f_in:
                try:
                    obj = json.loads(line)
                    four = obj.get("four_cards") or obj.get("four")
                    if isinstance(four, list):
                        seen.add(canonical_key(four))
                except Exception:
                    continue
        pre_existing = len(seen)
    except FileNotFoundError:
        pre_existing = 0
    
    visited_raw = 0
    t0 = time.perf_counter()
    print(f"Starting analytical precompute (append mode): raw_combos={total_raw}, existing_unique={pre_existing}, OUT_PATH={os.path.abspath(OUT_PATH)}")
    
    # Append new results to the existing JSONL
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        for comb in combos:
            visited_raw += 1
            if LIMIT and count_unique >= LIMIT:
                break
                
            first_four = normalize_four(list(comb))
            key = canonical_key(first_four)
            if key in seen:
                if visited_raw % 5000 == 0:
                    elapsed = time.perf_counter() - t0
                    rate_v = visited_raw / max(1e-6, elapsed)
                    total_unique = pre_existing + count_unique
                    dedup_ratio = total_unique / max(1, visited_raw)
                    print(f"visited={visited_raw}/{total_raw} unique_total={total_unique} (new={count_unique}) dedup={dedup_ratio:.3f} elapsed={elapsed:.1f}s visited_rate={rate_v:.1f}/s")
                continue
                
            seen.add(key)
            count_unique += 1
            
            # Calculate analytical bid recommendation
            bid, trump, analysis = propose_bid_analytical(first_four, current_high_bid=0)
            
            # Prepare output
            output = {
                "four_cards": first_four,
                "canonical_key": list(key),
                "analytical_suit_stats": analysis.get("suit_stats", {}),
                "analytical_bid": bid,
                "analytical_trump": trump,
                "analytical_analysis": analysis,
                "timestamp": time.time()
            }
            
            f.write(json.dumps(output) + "\n")
            f.flush()  # Ensure immediate write
            
            if count_unique % 100 == 0:
                elapsed = time.perf_counter() - t0
                rate_v = visited_raw / max(1e-6, elapsed)
                rate_u = count_unique / max(1e-6, elapsed)
                total_unique = pre_existing + count_unique
                dedup_ratio = total_unique / max(1, visited_raw)
                print(f"visited={visited_raw}/{total_raw} unique_total={total_unique} (new={count_unique}) dedup={dedup_ratio:.3f} elapsed={elapsed:.1f}s visited_rate={rate_v:.1f}/s unique_rate={rate_u:.1f}/s")
    
    elapsed = time.perf_counter() - t0
    total_unique = pre_existing + count_unique
    print(f"Completed analytical precompute: visited={visited_raw}/{total_raw} unique_total={total_unique} (new={count_unique}) elapsed={elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    exit(main())
