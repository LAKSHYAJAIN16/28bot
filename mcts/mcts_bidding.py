import copy
import math
import random
import os
import itertools
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from .constants import SUITS, card_suit, card_rank, card_value, rank_index, suit_trump_strength
from .ismcts import ismcts_plan


def _percentile(values, q):
    if not values:
        return 0.0
    s = sorted(values)
    # Use index based on q * n to avoid defaulting to min for small n
    k = max(0, min(len(s) - 1, int(q * len(s))))
    return s[k]


class MonteCarloBiddingAgent:
    def __init__(self, num_samples=6, mcts_iterations=60):
        self.num_samples = num_samples
        self.mcts_iterations = mcts_iterations
        self.last_debug = None
        self.last_choose_debug = None
        # Lazy-loaded precomputed map: canonical_key -> suit_stats
        self._precomp_map: dict[tuple[str, ...], dict[str, dict[str, float]]] | None = None

    @staticmethod
    def _rank_key(card: str) -> int:
        r = card[:-1]
        try:
            from .constants import RANKS
            return RANKS.index(r)
        except Exception:
            order = ["7","8","9","10","J","Q","K","A"]
            return order.index(r)

    @staticmethod
    def _canonical_key(cards: list[str]) -> tuple[str, ...]:
        from .constants import SUITS as SUITS_CONST
        best: tuple[str, ...] | None = None
        for perm in itertools.permutations(SUITS_CONST):
            suit_map = {s: perm[i] for i, s in enumerate(SUITS_CONST)}
            mapped = [c[:-1] + suit_map[c[-1]] for c in cards]
            mapped_sorted = sorted(mapped, key=lambda c: (MonteCarloBiddingAgent._rank_key(c), SUITS.index(card_suit(c))))
            tup = tuple(mapped_sorted)
            if best is None or tup < best:
                best = tup
        return best if best is not None else tuple(sorted(cards))

    def _load_precomputed(self) -> None:
        if self._precomp_map is not None:
            return
        self._precomp_map = {}
        path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "precomputed_bets.jsonl"))
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        four = obj.get("four_cards") or obj.get("four")
                        ss = obj.get("suit_stats")
                        if isinstance(four, list) and isinstance(ss, dict):
                            key = self._canonical_key(four)
                            self._precomp_map[key] = ss
                    except Exception:
                        continue
        except FileNotFoundError:
            pass

    def _lookup_precomputed_stats(self, my_first_four: list[str]) -> dict[str, dict[str, float]] | None:
        self._load_precomputed()
        assert self._precomp_map is not None
        key = self._canonical_key(my_first_four)
        stats = self._precomp_map.get(key)
        if not stats:
            return None
        # Filter to only suits present in hand
        present_suits = {card_suit(c) for c in my_first_four}
        return {s: v for s, v in stats.items() if s in present_suits}

    def _simulate_points_for_suit(self, my_first_four, suit, my_seat, first_player, playout_tricks: int | None = None):
        # Lazy import to avoid circular import during env initialization
        from .env28 import TwentyEightEnv
        all_ranks = ["7", "8", "9", "10", "J", "Q", "K", "A"]
        full_deck = [r + s for r in all_ranks for s in SUITS]
        remaining = [c for c in full_deck if c not in my_first_four]
        team_points = []
        for _ in range(self.num_samples):
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
            # Initialize belief fields expected by get_state
            env.void_suits_by_player = [set() for _ in range(4)]
            env.lead_suit_counts = [{s: 0 for s in SUITS} for _ in range(4)]
            env.quick_eval = True
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
                # Imperfect-information planning for bidding simulation
                iters = max(8, self.mcts_iterations // 5) if env.turn % 2 == my_seat % 2 else max(6, self.mcts_iterations // 12)
                try:
                    move, _ = ismcts_plan(env, state, iterations=iters, samples=6)
                    state, _, done, _, _ = env.step(move)
                    moves_done += 1
                except Exception as e:
                    # If ISMCTS fails, fall back to random play
                    valid_moves = env.valid_moves(state["hands"][state["turn"]])
                    if valid_moves:
                        move = random.choice(valid_moves)
                        state, _, done, _, _ = env.step(move)
                        moves_done += 1
                    else:
                        # If no valid moves, break out of the loop
                        break
                if playout_tricks is not None and (moves_done // 4) >= playout_tricks:
                    break
            bidder_team = 0 if my_seat % 2 == 0 else 1
            team_points.append(state["scores"][bidder_team])
        return team_points

    def propose_bid(self, my_first_four, current_high_bid, my_seat=0, first_player=0):
        present_suits = [s for s in SUITS if any(card_suit(c) == s for c in my_first_four)]
        if not present_suits:
            self.last_debug = {
                "present_suits": [],
                "suit_to_points": {},
                "chosen_suit": None,
                "avg_points": 0.0,
                "p30_points": 0.0,
                "std_points": 0.0,
                "proposed": None,
                "current_high": current_high_bid,
            }
            return None

        # Fast path: use precomputed suit stats if available
        pre_ss = self._lookup_precomputed_stats(my_first_four)
        if pre_ss:
            raise_delta = 1
            min_allowed = 16 if current_high_bid < 16 else current_high_bid + raise_delta
            candidates = []
            suit_failed_checks = {}
            for s in present_suits:
                det = pre_ss.get(s)
                if not det:
                    continue
                avg_points = float(det.get("avg", 0.0))
                std_points = float(det.get("std", 0.0))
                p30 = float(det.get("p30", 0.0))
                conf_mean = avg_points - 0.5 * std_points
                ok = (p30 >= (min_allowed - 1)) and (conf_mean >= (min_allowed - 1))
                if ok:
                    bid_raw = max(p30, avg_points - 1.0)
                    bid_s = int(math.floor(bid_raw))
                    bid_s = max(min_allowed, min(28, bid_s))
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
                def suit_key(s):
                    det = pre_ss.get(s) or {}
                    return (float(det.get("p30", 0.0)), float(det.get("avg", 0.0)))
                best_s = max(present_suits, key=suit_key)
                det = pre_ss.get(best_s) or {}
                best_mean = float(det.get("avg", 0.0))
                best_std = float(det.get("std", 0.0))
                best_p30 = float(det.get("p30", 0.0))
                best_conf = best_mean - 0.5 * best_std
                reason_text = (
                    f"Pass: No suit meets thresholds. Best suit {best_s} has p30={best_p30:.2f}, "
                    f"conf_mean={best_conf:.2f} below min_allowed={min_allowed} (current_high={current_high_bid}, raise_delta=+1). "
                    f"Failed checks for best suit: {', '.join((suit_failed_checks.get(best_s) or {}).get('failed', [])) or 'n/a'}."
                )
                self.last_debug = {
                    "present_suits": present_suits,
                    "suit_to_points": {},
                    "chosen_suit": None,
                    "avg_points": best_mean,
                    "p30_points": best_p30,
                    "std_points": best_std,
                    "proposed": None,
                    "current_high": current_high_bid,
                    "min_allowed": min_allowed,
                    "raise_delta": raise_delta,
                    "reason": reason_text,
                    "suit_failed_checks": suit_failed_checks,
                }
                return None
            candidates.sort(key=lambda t: (t[4], t[2]))
            best_suit, final_prop, avg_points, std_points, p30, conf_mean = candidates[-1]
            reason_text = (
                f"Bid (precomputed): Suit {best_suit} meets thresholds (p30={p30:.2f}, conf_mean={conf_mean:.2f} >= min_allowed={min_allowed}); "
                f"conservative mapping max(p30, avg-1) -> {final_prop}. (current_high={current_high_bid}, raise_delta=+1)"
            )
            self.last_debug = {
                "present_suits": present_suits,
                "suit_to_points": {},
                "chosen_suit": best_suit,
                "avg_points": avg_points,
                "p30_points": p30,
                "std_points": std_points,
                "proposed": final_prop,
                "current_high": current_high_bid,
                "min_allowed": min_allowed,
                "raise_delta": raise_delta,
                "reason": reason_text,
            }
            return final_prop

        # Stage 1 elimination heuristic:
        # - If the 4-card hand is 2/2 across two suits, run Stage 1 only on those two suits and pick the best
        # - Otherwise, skip Stage 1 and go straight to Stage 2 for suits with the highest count
        suit_counts = {s: sum(1 for c in my_first_four if card_suit(c) == s) for s in present_suits}
        max_count = max(suit_counts.values()) if suit_counts else 0
        top_count_suits = [s for s in present_suits if suit_counts[s] == max_count]
        stage1_points: dict[str, list[int]] = {}
        max_workers = min(len(top_count_suits), max(1, (os.cpu_count() or 1)))
        if max_count == 2 and len(top_count_suits) == 2:
            # 2/2 case: quick screen both, then pick best by (p30, mean)
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = {
                    ex.submit(_simulate_points_for_suit_worker, my_first_four, s, my_seat, first_player, 2, max(60, self.mcts_iterations)): s
                    for s in top_count_suits
                }
                for fut in as_completed(futures):
                    s = futures[fut]
                    try:
                        stage1_points[s] = fut.result()
                    except Exception:
                        stage1_points[s] = self._simulate_points_for_suit(my_first_four, s, my_seat, first_player, playout_tricks=4)
            def stage1_stat(suit: str):
                pts = stage1_points.get(suit, [])
                if not pts:
                    return (0.0, 0.0)
                n = len(pts)
                return (_percentile(pts, 0.3), sum(pts)/n)
            best = max(top_count_suits, key=lambda su: stage1_stat(su))
            top_suits = [best]
        else:
            # Not a 2/2 split: heavy only on suits with the highest count
            top_suits = top_count_suits[:]
        # Stage 2: heavy for top suits
        suit_to_points = {}
        if len(top_suits) > 1:
            with ProcessPoolExecutor(max_workers=min(len(top_suits), max_workers)) as ex:
                futures = {
                    ex.submit(_simulate_points_for_suit_worker, my_first_four, s, my_seat, first_player, max(6, self.num_samples), max(220, self.mcts_iterations)): s
                    for s in top_suits
                }
                for fut in as_completed(futures):
                    s = futures[fut]
                    try:
                        suit_to_points[s] = fut.result()
                    except Exception:
                        suit_to_points[s] = self._simulate_points_for_suit(my_first_four, s, my_seat, first_player)
        else:
            s = top_suits[0]
            suit_to_points[s] = self._simulate_points_for_suit(my_first_four, s, my_seat, first_player)
        # Non-top suits keep stage1 if computed (only in 2/2), else remain empty
        for s in present_suits:
            if s not in suit_to_points:
                suit_to_points[s] = stage1_points.get(s, [])

        def stats(values):
            n = max(1, len(values))
            mean = sum(values) / n
            var = sum((v - mean) ** 2 for v in values) / n
            std = math.sqrt(var)
            return mean, std

        # Sequential bidding: always raise by +1 minimum; direct 20 still allowed by env check
        raise_delta = 1
        min_allowed = 16 if current_high_bid < 16 else current_high_bid + raise_delta
        candidates = []
        suit_failed_checks = {}
        for s in present_suits:
            pts = suit_to_points[s]
            avg_points, std_points = stats(pts)
            p30 = _percentile(pts, 0.3)
            conf_mean = avg_points - 0.5 * std_points
            ok = (p30 >= (min_allowed - 1)) and (conf_mean >= (min_allowed - 1))
            if ok:
                bid_raw = max(p30, avg_points - 1.0)
                bid_s = int(math.floor(bid_raw))
                bid_s = max(min_allowed, min(28, bid_s))
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
            # Report best suit stats even when passing for better visibility
            def suit_key(s):
                pts = suit_to_points[s]
                mean, _std = stats(pts)
                return (_percentile(pts, 0.3), mean)
            best_s = max(present_suits, key=suit_key)
            det = suit_failed_checks.get(best_s)
            best_mean = det["avg"] if det else 0.0
            best_std = det["std"] if det else 0.0
            best_p30 = det["p30"] if det else 0.0
            best_conf = det["conf_mean"] if det else 0.0
            reason_text = (
                f"Pass: No suit meets thresholds. Best suit {best_s} has p30={best_p30:.2f}, "
                f"conf_mean={best_conf:.2f} below min_allowed={min_allowed} (current_high={current_high_bid}, raise_delta=+1). "
                f"Failed checks for best suit: {', '.join(det['failed']) if det and det['failed'] else 'n/a'}."
            )
            self.last_debug = {
                "present_suits": present_suits,
                "suit_to_points": suit_to_points,
                "chosen_suit": None,
                "avg_points": best_mean,
                "p30_points": best_p30,
                "std_points": best_std,
                "proposed": None,
                "current_high": current_high_bid,
                "min_allowed": min_allowed,
                "raise_delta": raise_delta,
                "reason": reason_text,
                "suit_failed_checks": suit_failed_checks,
            }
            return None
        candidates.sort(key=lambda t: (t[4], t[2]))
        best_suit, final_prop, avg_points, std_points, p30, conf_mean = candidates[-1]
        reason_text = (
            f"Bid: Suit {best_suit} meets thresholds (p30={p30:.2f}, conf_mean={conf_mean:.2f} >= min_allowed={min_allowed}); "
            f"conservative mapping max(p30, avg-1) -> {final_prop}. (current_high={current_high_bid}, raise_delta=+1)"
        )
        self.last_debug = {
            "present_suits": present_suits,
            "suit_to_points": suit_to_points,
            "chosen_suit": best_suit,
            "avg_points": avg_points,
            "p30_points": p30,
            "std_points": std_points,
            "proposed": final_prop,
            "current_high": current_high_bid,
            "min_allowed": min_allowed,
            "raise_delta": raise_delta,
            "reason": reason_text,
            "stage2_samples": {s: pts for s, pts in suit_to_points.items() if len(pts) >= max(6, self.num_samples)},
        }
        return final_prop

    def choose_trump(self, my_first_four, final_bid):
        present_suits = [s for s in SUITS if any(card_suit(c) == s for c in my_first_four)]
        if not present_suits:
            chosen = max(SUITS, key=lambda s: -SUITS.index(s))
            self.last_choose_debug = {"chosen": chosen, "estimates": {}}
            return chosen

        # Prefer precomputed averages if available
        estimates: dict[str, float] = {}
        pre_ss = self._lookup_precomputed_stats(my_first_four)
        missing: list[str] = []
        for s in present_suits:
            if pre_ss and s in pre_ss:
                try:
                    estimates[s] = float(pre_ss[s].get("avg", 0.0))
                except Exception:
                    estimates[s] = 0.0
            else:
                missing.append(s)

        # For missing suits, run heavier ISMCTS evaluation and use average points
        if missing:
            max_workers = min(len(missing), max(1, (os.cpu_count() or 1)))
            if len(missing) > 1:
                with ProcessPoolExecutor(max_workers=max_workers) as ex:
                    futures = {
                        ex.submit(
                            _simulate_points_for_suit_worker,
                            my_first_four,
                            s,
                            0,
                            0,
                            max(6, self.num_samples),
                            max(220, self.mcts_iterations),
                        ): s
                        for s in missing
                    }
                    for fut in as_completed(futures):
                        s = futures[fut]
                        try:
                            pts = fut.result()
                        except Exception:
                            pts = self._simulate_points_for_suit(my_first_four, s, 0, 0)
                        n = max(1, len(pts))
                        estimates[s] = sum(pts) / n
            else:
                s = missing[0]
                pts = self._simulate_points_for_suit(my_first_four, s, 0, 0)
                n = max(1, len(pts))
                estimates[s] = sum(pts) / n

        best_suit = max(present_suits, key=lambda s: estimates.get(s, 0.0))
        self.last_choose_debug = {"chosen": best_suit, "estimates": estimates}
        return best_suit


def _simulate_points_for_suit_worker(my_first_four, suit, my_seat, first_player, num_samples, mcts_iterations):
    # Worker function to run in a separate process
    # Recreate a minimal agent context
    agent = MonteCarloBiddingAgent(num_samples=num_samples, mcts_iterations=mcts_iterations)
    return agent._simulate_points_for_suit(my_first_four, suit, my_seat, first_player)


