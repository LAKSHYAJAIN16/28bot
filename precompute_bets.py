#!/usr/bin/env python3
"""
Precompute bid recommendations for all 4-card combinations (orderless) at current_high=0 (min_allowed=16).
Writes JSON lines with: four_cards, suit_stats, recommendation (bid, trump), and analysis.

Run:
  python precompute_bets.py
"""

import itertools
import json
import math
import os
import time
from typing import List

from mcts.constants import FULL_DECK, card_suit, RANKS, SUITS as SUITS_CONST
from bet_advisor import propose_bid_ismcts, simulate_points_for_suit_ismcts, SUITS, DEFAULT_STAGE2_SAMPLES, DEFAULT_STAGE2_ITERS, FAST_MODE

# Predefined output configuration
OUT_PATH = "precomputed_bets.jsonl"
LIMIT = 0  # 0 = all combinations
HEAVY_TOP_K = 2  # heavy-evaluate top K suits
HEAVY_SAMPLES = max(DEFAULT_STAGE2_SAMPLES, 6)
HEAVY_ITERS = max(DEFAULT_STAGE2_ITERS, 220)


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
    print(f"Starting precompute (append mode): raw_combos={total_raw}, existing_unique={pre_existing}, OUT_PATH={os.path.abspath(OUT_PATH)}")
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
            # Accuracy-first: two-stage evaluation
            present_suits = [s for s in SUITS if any(card_suit(c) == s for c in first_four)]
            stage1 = {}
            for s in present_suits:
                pts = simulate_points_for_suit_ismcts(first_four, s, my_seat=0, first_player=0,
                                                      num_samples=2, base_iterations=80, playout_tricks=4)
                n = max(1, len(pts))
                stage1[s] = (sum(pts)/n, sorted(pts)[max(0, min(len(pts)-1, int(0.3*len(pts))))])
            top = sorted(present_suits, key=lambda s: (stage1[s][1], stage1[s][0]))[-max(1, HEAVY_TOP_K):]
            suit_stats = {}
            suit_to_points = {}
            for s in present_suits:
                if s in top:
                    pts = simulate_points_for_suit_ismcts(first_four, s, my_seat=0, first_player=0,
                                                          num_samples=HEAVY_SAMPLES, base_iterations=HEAVY_ITERS)
                else:
                    pts = simulate_points_for_suit_ismcts(first_four, s, my_seat=0, first_player=0,
                                                          num_samples=2, base_iterations=80, playout_tricks=4)
                suit_to_points[s] = pts
                n = max(1, len(pts))
                mean = sum(pts)/n
                p30 = sorted(pts)[max(0, min(len(pts)-1, int(0.3*len(pts))))] if pts else 0.0
                var = sum((v-mean)**2 for v in pts)/n if pts else 0.0
                suit_stats[s] = {"avg": mean, "p30": p30, "std": var**0.5}

            # Final recommendation using the same thresholds as the advisor (current_high=0)
            from bet_advisor import propose_bid_ismcts as advisor_bid
            bid, trump, _, dbg = advisor_bid(first_four, current_high_bid=0,
                                             num_samples=HEAVY_SAMPLES, base_iterations=HEAVY_ITERS)
            rec = {
                "four_cards": first_four,
                "suit_stats": suit_stats,
                "recommendation": {"bid": bid, "trump": trump},
                "analysis": dbg,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count_unique += 1
            if count_unique % 100 == 0:
                elapsed = time.perf_counter() - t0
                rate_u = count_unique / max(1e-6, elapsed)
                rate_v = visited_raw / max(1e-6, elapsed)
                total_unique = pre_existing + count_unique
                dedup_ratio = total_unique / max(1, visited_raw)
                print(f"unique_total={total_unique} (new={count_unique}) visited={visited_raw}/{total_raw} dedup={dedup_ratio:.3f} elapsed={elapsed:.1f}s unique_rate={rate_u:.2f}/s visited_rate={rate_v:.1f}/s")
    elapsed = time.perf_counter() - t0
    total_unique = pre_existing + count_unique
    print(f"Done. Appended {count_unique} new unique combos to {OUT_PATH} (unique_total={total_unique}). Total visited raw={visited_raw}/{total_raw}. Elapsed={elapsed:.1f}s, new_unique_rate={(count_unique/max(1e-6,elapsed)):.2f}/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


