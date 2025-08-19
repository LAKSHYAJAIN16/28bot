import copy
import math
import random
from .constants import SUITS, card_suit, card_rank, card_value, rank_index, suit_trump_strength
from .env28 import TwentyEightEnv
from .ismcts import ismcts_plan


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
                iters = max(10, self.mcts_iterations // 5)
                move, _ = ismcts_plan(env, state, iterations=iters, samples=6)
                state, _, done, _, _ = env.step(move)
                moves_done += 1
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

        suit_to_points = {}
        for s in present_suits:
            pts = self._simulate_points_for_suit(my_first_four, s, my_seat, first_player)
            suit_to_points[s] = pts

        def stats(values):
            n = max(1, len(values))
            mean = sum(values) / n
            var = sum((v - mean) ** 2 for v in values) / n
            std = math.sqrt(var)
            return mean, std

        raise_delta = 2 if current_high_bid < 20 else 1
        min_allowed = 16 if current_high_bid < 16 else current_high_bid + raise_delta
        candidates = []
        for s in present_suits:
            pts = suit_to_points[s]
            avg_points, std_points = stats(pts)
            p30 = _percentile(pts, 0.3)
            conf_mean = avg_points - 0.75 * std_points
            ok = (p30 >= min_allowed) and (conf_mean >= min_allowed)
            if ok:
                bid_raw = max(p30, avg_points - 1.0)
                bid_s = int(math.floor(bid_raw))
                bid_s = max(min_allowed, min(28, bid_s))
                candidates.append((s, bid_s, avg_points, std_points, p30))
        if not candidates:
            self.last_debug = {
                "present_suits": present_suits,
                "suit_to_points": suit_to_points,
                "chosen_suit": None,
                "avg_points": 0.0,
                "p30_points": 0.0,
                "std_points": 0.0,
                "proposed": None,
                "current_high": current_high_bid,
                "min_allowed": min_allowed,
                "reason": "no_suit_meets_thresholds",
            }
            return None
        candidates.sort(key=lambda t: (t[4], t[2]))
        best_suit, final_prop, avg_points, std_points, p30 = candidates[-1]
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
        }
        return final_prop

    def choose_trump(self, my_first_four, final_bid):
        present_suits = [s for s in SUITS if any(card_suit(c) == s for c in my_first_four)]
        if not present_suits:
            chosen = max(SUITS, key=lambda s: -SUITS.index(s))
            self.last_choose_debug = {"chosen": chosen, "estimates": {}}
            return chosen
        best_s, best_e = None, -1
        estimates = {}
        for s in present_suits:
            count = sum(1 for c in my_first_four if card_suit(c) == s)
            pts = sum(card_value(c) for c in my_first_four if card_suit(c) == s)
            est = 3 * count + 2 * pts
            estimates[s] = est
            if est > best_e:
                best_e = est
                best_s = s
        self.last_choose_debug = {"chosen": best_s, "estimates": estimates}
        return best_s


