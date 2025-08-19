# file: game28_simulator.py
import numpy as np

NUM_PLAYERS = 4
NUM_CARDS = 28  # 4 suits * 7 ranks
HAND_SIZE = 7

# Suit/Rank mapping for simplicity
SUITS = ["hearts", "diamonds", "clubs", "spades"]
RANKS = [7, 8, 9, 10, 11, 12, 13]  # arbitrary ordering for 28 cards

class Game28:
    """
    Core 28 game simulator. Handles card dealing, trick taking, scoring.
    Bidding is left out initially.
    """

    def __init__(self):
        self.hands = [np.zeros(NUM_CARDS, dtype=np.int8) for _ in range(NUM_PLAYERS)]
        self.played_mask = np.zeros(NUM_CARDS, dtype=np.int8)
        self.current_trick = [None] * NUM_PLAYERS  # list of (player, card_index)
        self.current_player = 0
        self.tricks_won = [0, 0]  # team 0 (0+2) and team 1 (1+3)
        self.trump = np.random.randint(0, 4)
        self.deck = np.arange(NUM_CARDS)
        np.random.shuffle(self.deck)
        self.deal_cards()
        
    def deal_cards(self):
        for p in range(NUM_PLAYERS):
            cards = self.deck[p*HAND_SIZE:(p+1)*HAND_SIZE]
            self.hands[p][cards] = 1

    def play_card(self, player, card):
        """
        Player plays a card (0-27)
        Returns: legal move (bool)
        """
        if self.hands[player][card] == 0:
            return False  # illegal
        self.hands[player][card] = 0
        self.played_mask[card] = 1
        for i in range(NUM_PLAYERS):
            if self.current_trick[i] is None:
                self.current_trick[i] = (player, card)
                break
        self.current_player = (player + 1) % NUM_PLAYERS
        return True

    def is_trick_complete(self):
        return all([c is not None for c in self.current_trick])

    def resolve_trick(self):
        """
        Determine winner of trick based on lead suit and trump
        Returns: winning player index
        """
        lead_card = self.current_trick[0][1]
        lead_suit = lead_card // 7
        best_card = lead_card
        winner = self.current_trick[0][0]
        for p, card in self.current_trick[1:]:
            suit = card // 7
            if suit == self.trump and best_card // 7 != self.trump:
                best_card = card
                winner = p
            elif suit == best_card // 7:
                if card % 7 > best_card % 7:
                    best_card = card
                    winner = p
        team = winner % 2
        self.tricks_won[team] += 1
        self.current_trick = [None] * NUM_PLAYERS
        self.current_player = winner
        return winner

    def is_hand_over(self):
        return sum([h.sum() for h in self.hands]) == 0
