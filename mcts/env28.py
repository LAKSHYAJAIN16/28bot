import copy
import random
import math
from .constants import SUITS, RANKS, card_suit, card_rank, card_value, rank_index, trick_rank_index, suit_trump_strength


class TwentyEightEnv:
    def __init__(self):
        self.num_players = 4
        self.debug = False

    def reset(self, initial_trump=None, first_player=0):
        deck = [r + s for r in ["7", "8", "9", "10", "J", "Q", "K", "A"] for s in SUITS]
        random.shuffle(deck)
        self.hands = [[] for _ in range(4)]
        for _ in range(4):
            for p in range(4):
                self.hands[p].append(deck.pop())
        self.first_four_hands = [hand[:] for hand in self.hands]
        self.scores = [0, 0]
        self.game_score = [0, 0]
        self.current_trick = []
        self.turn = first_player
        # Belief tracking
        self.void_suits_by_player = [set() for _ in range(4)]
        self.lead_suit_counts = [
            {s: 0 for s in SUITS} for _ in range(4)
        ]
        if initial_trump is not None:
            self.bidder = first_player
            self.trump_suit = initial_trump
            self.bid_value = 16
            self.round_stakes = 1
        else:
            from .mcts_bidding import MonteCarloBiddingAgent

            agents = [MonteCarloBiddingAgent(num_samples=4, mcts_iterations=50) for _ in range(4)]
            order = [(first_player + i) % 4 for i in range(4)]
            current_high = 0
            current_bidder = None
            bidding_started = False
            passed = {p: False for p in order}
            idx = 0
            safety = 0
            if self.debug:
                print("\n===== AUCTION =====")
            while safety < 200:
                safety += 1
                # End if only one active remains and at least one bid was made
                active_players = [p for p in order if not passed[p]]
                if bidding_started and len(active_players) == 1:
                    break
                # If no one bid in first cycle and all passed, force first player to 16
                if not bidding_started and all(passed[p] for p in order):
                    break
                p = order[idx % 4]
                idx += 1
                if passed[p]:
                    continue
                proposal = agents[p].propose_bid(self.first_four_hands[p], current_high, my_seat=p, first_player=first_player)
                min_allowed = 16 if current_high < 16 else current_high + 1
                dbg = getattr(agents[p], "last_debug", None)
                # Enrich precomputed bets incrementally for each player's first-four
                try:
                    from .runner import _maybe_append_precomputed
                    _maybe_append_precomputed(self.first_four_hands[p], bidder_bid=proposal, trump=None, dbg=dbg)
                except Exception:
                    pass
                if proposal is not None and 16 <= proposal <= 28 and proposal >= min_allowed:
                    current_high = proposal
                    current_bidder = p
                    bidding_started = True
                    if self.debug:
                        print(f"Bid: Player {p} proposes {proposal} (min_allowed={min_allowed})")
                        if dbg:
                            print(_format_bidding_debug(dbg))
                else:
                    passed[p] = True
                    if self.debug:
                        print(f"Pass (locked): Player {p} (min_allowed={min_allowed})")
                        if dbg:
                            print(_format_bidding_debug(dbg))
            # Fallbacks when loop exits
            if not bidding_started:
                # No one bid: force first player (order[0]) to 16
                current_bidder = order[0]
                current_high = 16
                if self.debug:
                    print(f"Forced bid: All players passed. Player {current_bidder} takes minimum bid {current_high}.")
            elif len([p for p in order if not passed[p]]) == 1 and current_bidder is None:
                # Only one active remained but never raised; assign bidder to that player at min 16
                current_bidder = [p for p in order if not passed[p]][0]
                current_high = max(16, current_high)
            self.bidder = current_bidder
            self.bid_value = current_high
            # Stakes: bid >= 20 doubles the round value
            self.round_stakes = 2 if self.bid_value >= 20 else 1
            self.trump_suit = agents[self.bidder].choose_trump(self.first_four_hands[self.bidder], self.bid_value)
            choose_dbg = getattr(agents[self.bidder], "last_choose_debug", None)
            # Opportunistically persist this first-four to precomputed JSONL if new
            try:
                from .runner import _maybe_append_precomputed
                bidder_dbg = getattr(agents[self.bidder], "last_debug", None)
                _maybe_append_precomputed(self.first_four_hands[self.bidder], bidder_bid=self.bid_value, trump=self.trump_suit, dbg=bidder_dbg)
            except Exception:
                pass
            if self.debug:
                print(
                    f"\nAUCTION WINNER: Player {self.bidder} with bid {self.bid_value}; chooses trump {self.trump_suit} "
                )
                print(f"Details={choose_dbg}")
        self.phase = "concealed"
        trump_cards_in_bidder = [c for c in self.hands[self.bidder] if card_suit(c) == self.trump_suit]
        self.face_down_trump_card = max(trump_cards_in_bidder, key=rank_index) if trump_cards_in_bidder else None
        if self.face_down_trump_card is not None:
            self.hands[self.bidder].remove(self.face_down_trump_card)
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
            "stakes": getattr(self, "round_stakes", 1),
            "face_down_trump": self.face_down_trump_card,
            "last_exposer": self.last_exposer,
            "exposure_trick_index": self.exposure_trick_index,
            "void_suits_by_player": [set(s) for s in getattr(self, "void_suits_by_player", [set() for _ in range(4)])],
            "lead_suit_counts": copy.deepcopy(getattr(self, "lead_suit_counts", [{s: 0 for s in SUITS} for _ in range(4)])),
        }

    def valid_moves(self, hand):
        if not self.current_trick:
            if self.phase == "concealed" and self.turn == self.bidder and self.trump_suit is not None:
                non_trump = [c for c in hand if card_suit(c) != self.trump_suit]
                if non_trump:
                    return non_trump
            return hand
        lead_suit = card_suit(self.current_trick[0][1])
        same_suit = [c for c in hand if card_suit(c) == lead_suit]
        return same_suit if same_suit else hand

    def step(self, card):
        if self.current_trick:
            lead_suit = card_suit(self.current_trick[0][1])
            if any(card_suit(c) == lead_suit for c in self.hands[self.turn]) and card_suit(card) != lead_suit:
                raise ValueError("Illegal move: must follow suit")
        else:
            if self.phase == "concealed" and self.turn == self.bidder and self.trump_suit is not None:
                if any(card_suit(c) != self.trump_suit for c in self.hands[self.turn]) and card_suit(card) == self.trump_suit:
                    raise ValueError(
                        "Illegal lead: bidder cannot lead trump before exposure unless only trump remains"
                    )

        if self.phase == "concealed" and self.current_trick:
            lead_suit = card_suit(self.current_trick[0][1])
            has_lead = any(card_suit(c) == lead_suit for c in self.hands[self.turn])
            if not has_lead and self.trump_suit is not None and card_suit(card) == self.trump_suit:
                self.phase = "revealed"
                self.last_exposer = self.turn
                self.exposure_trick_index = self.tricks_played + 1
                if self.face_down_trump_card is not None:
                    self.hands[self.bidder].append(self.face_down_trump_card)
                    self.face_down_trump_card = None
                if self.debug:
                    print(
                        f"-- Phase 2 begins: Trump revealed as {self.trump_suit} by Player {self.last_exposer} on trick {self.exposure_trick_index} --"
                    )

        # Track suit leads for coordination
        try:
            if not self.current_trick:
                self.lead_suit_counts[self.turn][card_suit(card)] += 1
        except Exception:
            pass
        self.hands[self.turn].remove(card)
        self.current_trick.append((self.turn, card))
        self.turn = (self.turn + 1) % 4
        done = all(len(hand) == 0 for hand in self.hands)
        trick_winner = None
        trick_points = 0
        if len(self.current_trick) == 4:
            trick_winner, trick_points = self.resolve_trick()
            team = 0 if trick_winner % 2 == 0 else 1
            self.scores[team] += trick_points
            self.current_trick = []
            self.turn = trick_winner
            self.last_trick_winner = trick_winner
            self.tricks_played += 1
            if self.tricks_played >= 7 and self.phase == "concealed":
                self.invalid_round = True
                done = True
                self.scores = [0, 0]
        reward = self.scores[0] - self.scores[1] if done else 0
        if done and not self.invalid_round:
            bidder_team = 0 if self.bidder % 2 == 0 else 1
            if self.scores[bidder_team] >= getattr(self, "bid_value", 16):
                self.game_score[bidder_team] += getattr(self, "round_stakes", 1)
            else:
                self.game_score[bidder_team] -= getattr(self, "round_stakes", 1)
        return self.get_state(), reward, done, trick_winner, trick_points

    def resolve_trick(self):
        lead_suit = card_suit(self.current_trick[0][1])
        winning_card, winner = self.current_trick[0][1], self.current_trick[0][0]
        # Belief update: record voids for players who could not follow suit
        for p_idx, card in self.current_trick[1:]:
            if card_suit(card) != lead_suit:
                self.void_suits_by_player[p_idx].add(lead_suit)
        if self.phase == "concealed":
            for player, card in self.current_trick[1:]:
                if card_suit(card) == lead_suit and trick_rank_index(card) > trick_rank_index(winning_card):
                    winning_card, winner = card, player
        else:
            for player, card in self.current_trick[1:]:
                win_suit = card_suit(winning_card)
                if card_suit(card) == win_suit and trick_rank_index(card) > trick_rank_index(winning_card):
                    winning_card, winner = card, player
                elif self.trump_suit is not None and card_suit(card) == self.trump_suit and win_suit != self.trump_suit:
                    winning_card, winner = card, player
        points = sum(card_value(c) for _, c in self.current_trick)
        return winner, points


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
        if current_high is not None:
            meta.append(f"current_high={current_high}")
        if min_allowed is not None:
            meta.append(f"min_allowed={min_allowed}")
        if raise_delta is not None:
            meta.append(f"raise_delta={raise_delta}")
        if meta:
            lines.append("    " + "  ".join(meta))
        if chosen_suit is not None:
            lines.append(f"    chosen_suit: {chosen_suit}")
        if proposed is not None:
            lines.append(f"    proposed_bid: {proposed}")
        # Overall point estimates on best/selected suit
        if "avg_points" in dbg or "p30_points" in dbg or "std_points" in dbg:
            lines.append(
                "    overall: "
                + f"avg={dbg.get('avg_points', 0):.2f}  p30={dbg.get('p30_points', 0):.2f}  std={dbg.get('std_points', 0):.2f}"
            )
        # Per-suit stats
        stp = dbg.get("suit_to_points") or {}
        if stp:
            lines.append("    suit_stats:")
            for s, pts in stp.items():
                if not pts:
                    lines.append(f"      {s}: pts=[]")
                    continue
                n = len(pts)
                mean = sum(pts) / n
                var = sum((v - mean) ** 2 for v in pts) / n
                std = math.sqrt(var)
                q = sorted(pts)
                k = max(0, min(len(q) - 1, int(0.3 * len(q))))
                p30 = q[k]
                preview = ", ".join(str(x) for x in pts[:6]) + (" ..." if len(pts) > 6 else "")
                lines.append(
                    f"      {s}: pts=[{preview}]  avg={mean:.2f}  p30={p30:.2f}  std={std:.2f}"
                )
        # Failed checks detail (when present)
        sf = dbg.get("suit_failed_checks")
        if sf:
            lines.append("    failed_checks:")
            for s, det in sf.items():
                failed = det.get("failed") or []
                if failed:
                    lines.append(f"      {s}: " + "; ".join(failed))
        if reason:
            lines.append("    reason: " + str(reason))
        return "\n".join(lines)
    except Exception:
        return "  (failed to format bidding analysis)"

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
            "stakes": getattr(self, "round_stakes", 1),
            "face_down_trump": self.face_down_trump_card,
            "last_exposer": self.last_exposer,
            "exposure_trick_index": self.exposure_trick_index,
        }

    def valid_moves(self, hand):
        if not self.current_trick:
            if self.phase == "concealed" and self.turn == self.bidder and self.trump_suit is not None:
                non_trump = [c for c in hand if card_suit(c) != self.trump_suit]
                if non_trump:
                    return non_trump
            return hand
        lead_suit = card_suit(self.current_trick[0][1])
        same_suit = [c for c in hand if card_suit(c) == lead_suit]
        return same_suit if same_suit else hand

    def step(self, card):
        if self.current_trick:
            lead_suit = card_suit(self.current_trick[0][1])
            if any(card_suit(c) == lead_suit for c in self.hands[self.turn]) and card_suit(card) != lead_suit:
                raise ValueError("Illegal move: must follow suit")
        else:
            if self.phase == "concealed" and self.turn == self.bidder and self.trump_suit is not None:
                if any(card_suit(c) != self.trump_suit for c in self.hands[self.turn]) and card_suit(card) == self.trump_suit:
                    raise ValueError(
                        "Illegal lead: bidder cannot lead trump before exposure unless only trump remains"
                    )

        if self.phase == "concealed" and self.current_trick:
            lead_suit = card_suit(self.current_trick[0][1])
            has_lead = any(card_suit(c) == lead_suit for c in self.hands[self.turn])
            if not has_lead and self.trump_suit is not None and card_suit(card) == self.trump_suit:
                self.phase = "revealed"
                self.last_exposer = self.turn
                self.exposure_trick_index = self.tricks_played + 1
                if self.face_down_trump_card is not None:
                    self.hands[self.bidder].append(self.face_down_trump_card)
                    self.face_down_trump_card = None
                if self.debug:
                    print(
                        f"-- Phase 2 begins: Trump revealed as {self.trump_suit} by Player {self.last_exposer} on trick {self.exposure_trick_index} --"
                    )

        self.hands[self.turn].remove(card)
        self.current_trick.append((self.turn, card))
        self.turn = (self.turn + 1) % 4
        done = all(len(hand) == 0 for hand in self.hands)
        trick_winner = None
        trick_points = 0
        if len(self.current_trick) == 4:
            trick_winner, trick_points = self.resolve_trick()
            team = 0 if trick_winner % 2 == 0 else 1
            self.scores[team] += trick_points
            self.current_trick = []
            self.turn = trick_winner
            self.last_trick_winner = trick_winner
            self.tricks_played += 1
            if self.tricks_played >= 7 and self.phase == "concealed":
                self.invalid_round = True
                done = True
                self.scores = [0, 0]
        reward = self.scores[0] - self.scores[1] if done else 0
        if done and not self.invalid_round:
            bidder_team = 0 if self.bidder % 2 == 0 else 1
            if self.scores[bidder_team] >= getattr(self, "bid_value", 16):
                self.game_score[bidder_team] += getattr(self, "round_stakes", 1)
            else:
                self.game_score[bidder_team] -= getattr(self, "round_stakes", 1)
        return self.get_state(), reward, done, trick_winner, trick_points

    def resolve_trick(self):
        lead_suit = card_suit(self.current_trick[0][1])
        winning_card, winner = self.current_trick[0][1], self.current_trick[0][0]
        if self.phase == "concealed":
            for player, card in self.current_trick[1:]:
                if card_suit(card) == lead_suit and trick_rank_index(card) > trick_rank_index(winning_card):
                    winning_card, winner = card, player
        else:
            for player, card in self.current_trick[1:]:
                win_suit = card_suit(winning_card)
                if card_suit(card) == win_suit and trick_rank_index(card) > trick_rank_index(winning_card):
                    winning_card, winner = card, player
                elif self.trump_suit is not None and card_suit(card) == self.trump_suit and win_suit != self.trump_suit:
                    winning_card, winner = card, player
        points = sum(card_value(c) for _, c in self.current_trick)
        return winner, points


