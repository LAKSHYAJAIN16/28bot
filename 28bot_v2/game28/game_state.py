"""
Game state management for Game 28
"""

import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .constants import *


@dataclass
class Card:
    """Represents a playing card"""
    suit: str
    rank: str
    
    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"
    
    def __hash__(self) -> int:
        return hash((self.suit, self.rank))
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Card):
            return False
        return self.suit == other.suit and self.rank == other.rank


@dataclass
class Trick:
    """Represents a trick in the game"""
    cards: List[Tuple[int, Card]] = field(default_factory=list)  # (player, card)
    lead_suit: Optional[str] = None
    winner: Optional[int] = None
    points: int = 0
    
    def add_card(self, player: int, card: Card) -> None:
        """Add a card to the trick"""
        if not self.cards:
            self.lead_suit = card.suit
        self.cards.append((player, card))
        self.points += CARD_VALUES[card.rank]
    
    def get_winner(self, trump_suit: Optional[str] = None, trump_revealed: bool = False) -> int:
        """Determine the winner of the trick"""
        if not self.cards:
            return -1
        
        winning_card = self.cards[0][1]
        winning_player = self.cards[0][0]
        
        for player, card in self.cards[1:]:
            if self._beats(card, winning_card, trump_suit, trump_revealed):
                winning_card = card
                winning_player = player
        
        self.winner = winning_player
        return winning_player
    
    def _beats(self, card1: Card, card2: Card, trump_suit: Optional[str], trump_revealed: bool) -> bool:
        """Check if card1 beats card2"""
        # If trump is not revealed, only same suit can win
        if not trump_revealed:
            if card1.suit == self.lead_suit and card2.suit != self.lead_suit:
                return True
            if card1.suit != self.lead_suit and card2.suit == self.lead_suit:
                return False
            if card1.suit != self.lead_suit and card2.suit != self.lead_suit:
                return False
        
        # Trump revealed or same suit comparison
        if trump_suit and card1.suit == trump_suit and card2.suit != trump_suit:
            return True
        if trump_suit and card1.suit != trump_suit and card2.suit == trump_suit:
            return False
        
        # Same suit comparison
        if card1.suit == card2.suit:
            return TRICK_RANKINGS[card1.rank] > TRICK_RANKINGS[card2.rank]
        
        return False


@dataclass
class Game28State:
    """Represents the complete state of a Game 28 game"""
    
    # Player hands
    hands: List[List[Card]] = field(default_factory=lambda: [[] for _ in range(4)])
    
    # Game state
    phase: GamePhase = GamePhase.BIDDING
    current_player: int = 0
    bidder: Optional[int] = None
    winning_bid: Optional[int] = None
    trump_suit: Optional[str] = None
    trump_revealed: bool = False
    face_down_trump: Optional[Card] = None
    
    # Bidding state
    current_bid: int = MIN_BID
    bid_history: List[Tuple[int, int]] = field(default_factory=list)  # (player, bid)
    passed_players: List[int] = field(default_factory=list)
    
    # Trick state
    current_trick: Trick = field(default_factory=Trick)
    tricks: List[Trick] = field(default_factory=list)
    trick_leader: int = 0
    
    # Scoring
    team_scores: Dict[str, int] = field(default_factory=lambda: {'A': 0, 'B': 0})
    game_points: Dict[str, int] = field(default_factory=lambda: {'A': 0, 'B': 0})
    
    # Game tracking
    round_number: int = 0
    game_over: bool = False
    
    def __post_init__(self):
        """Initialize the game state"""
        self.deck = self._create_deck()
        self.deal_initial_cards()
    
    def _create_deck(self) -> List[Card]:
        """Create a deck of 32 cards"""
        deck = []
        for suit in SUITS:
            for rank in RANKS:
                deck.append(Card(suit, rank))
        return deck
    
    def deal_initial_cards(self) -> None:
        """Deal 4 cards to each player"""
        random.shuffle(self.deck)
        for i in range(4):
            for player in range(4):
                self.hands[player].append(self.deck.pop())
    
    def deal_remaining_cards(self) -> None:
        """Deal the remaining 4 cards to each player after bidding"""
        for i in range(4):
            for player in range(4):
                self.hands[player].append(self.deck.pop())
    
    def get_legal_bids(self, player: int) -> List[int]:
        """Get legal bids for a player"""
        if player in self.passed_players:
            return []
        
        legal_bids = []
        for bid in BID_RANGE:
            if bid > self.current_bid:
                legal_bids.append(bid)
        legal_bids.append(-1)  # Pass
        return legal_bids
    
    def make_bid(self, player: int, bid: int) -> bool:
        """Make a bid (returns True if bidding continues)"""
        if bid == -1:  # Pass
            self.passed_players.append(player)
            if len(self.passed_players) == 3:
                # Bidding ends, determine winner
                active_players = [p for p in range(4) if p not in self.passed_players]
                if active_players:
                    self.bidder = active_players[0]
                    self.winning_bid = self.current_bid
                    return False
                else:
                    # All passed, invalid round
                    self.game_over = True
                    return False
        else:
            self.current_bid = bid
            self.bid_history.append((player, bid))
            self.passed_players = []
        
        # Move to next player
        self.current_player = (self.current_player + 1) % 4
        while self.current_player in self.passed_players:
            self.current_player = (self.current_player + 1) % 4
        
        return True
    
    def set_trump(self, trump_suit: str) -> None:
        """Set the trump suit and place face-down trump"""
        self.trump_suit = trump_suit
        self.phase = GamePhase.CONCEALED
        
        # Find highest trump in bidder's hand
        trump_cards = [card for card in self.hands[self.bidder] if card.suit == trump_suit]
        if trump_cards:
            self.face_down_trump = max(trump_cards, key=lambda c: TRICK_RANKINGS[c.rank])
            self.hands[self.bidder].remove(self.face_down_trump)
        
        # Deal remaining cards
        self.deal_remaining_cards()
        self.current_player = self.bidder
    
    def get_legal_plays(self, player: int) -> List[Card]:
        """Get legal cards to play for a player"""
        if not self.current_trick.cards:
            # Leading
            if self.phase == GamePhase.CONCEALED and self.trump_suit:
                # Can't lead trump unless only trump remains
                non_trump = [card for card in self.hands[player] if card.suit != self.trump_suit]
                if non_trump:
                    return non_trump
            return self.hands[player]
        
        # Following
        lead_suit = self.current_trick.lead_suit
        following_suit = [card for card in self.hands[player] if card.suit == lead_suit]
        
        if following_suit:
            return following_suit
        else:
            return self.hands[player]
    
    def play_card(self, player: int, card: Card) -> None:
        """Play a card"""
        if card not in self.hands[player]:
            raise ValueError(f"Card {card} not in player {player}'s hand")
        
        # Remove card from hand
        self.hands[player].remove(card)
        
        # Add to current trick
        self.current_trick.add_card(player, card)
        
        # Check if trump was revealed
        if not self.trump_revealed and self.trump_suit and card.suit == self.trump_suit:
            self.trump_revealed = True
            self.phase = GamePhase.REVEALED
            # Return face-down trump to bidder's hand
            if self.face_down_trump:
                self.hands[self.bidder].append(self.face_down_trump)
                self.face_down_trump = None
        
        # Move to next player
        self.current_player = (self.current_player + 1) % 4
        
        # Check if trick is complete
        if len(self.current_trick.cards) == 4:
            winner = self.current_trick.get_winner(self.trump_suit, self.trump_revealed)
            self.tricks.append(self.current_trick)
            self.trick_leader = winner
            self.current_player = winner
            
            # Check if game is over
            if len(self.tricks) == 8:
                self._end_round()
            else:
                self.current_trick = Trick()
    
    def _end_round(self) -> None:
        """End the round and calculate scores"""
        # Calculate team scores
        team_a_score = sum(trick.points for trick in self.tricks 
                          if any(player in TEAM_A for player, _ in trick.cards))
        team_b_score = sum(trick.points for trick in self.tricks 
                          if any(player in TEAM_B for player, _ in trick.cards))
        
        self.team_scores['A'] = team_a_score
        self.team_scores['B'] = team_b_score
        
        # Determine bid success/failure
        if self.bidder is not None and self.winning_bid is not None:
            bidder_team = 'A' if self.bidder in TEAM_A else 'B'
            team_score = team_a_score if bidder_team == 'A' else team_b_score
            
            if team_score >= self.winning_bid:
                self.game_points[bidder_team] += 1
            else:
                self.game_points[bidder_team] -= 1
        
        self.round_number += 1
        self.game_over = True
    
    def get_observation(self, player: int) -> Dict:
        """Get the observation for a specific player"""
        return {
            'hand': self.hands[player],
            'phase': self.phase,
            'current_player': self.current_player,
            'bidder': self.bidder,
            'winning_bid': self.winning_bid,
            'trump_suit': self.trump_suit,
            'trump_revealed': self.trump_revealed,
            'current_bid': self.current_bid,
            'bid_history': self.bid_history,
            'passed_players': self.passed_players,
            'current_trick': self.current_trick,
            'tricks': self.tricks,
            'trick_leader': self.trick_leader,
            'team_scores': self.team_scores,
            'game_points': self.game_points,
            'round_number': self.round_number,
            'game_over': self.game_over
        }
    
    def copy(self) -> 'Game28State':
        """Create a deep copy of the game state"""
        import copy
        return copy.deepcopy(self)
