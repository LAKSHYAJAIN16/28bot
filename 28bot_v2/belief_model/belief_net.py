"""
Belief network for opponent hand inference in Game 28
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ..game28.game_state import Game28State, Card
from ..game28.constants import *


@dataclass
class BeliefState:
    """Represents the belief state for opponent hands"""
    player_id: int
    opponent_hands: Dict[int, List[float]]  # player_id -> card probabilities
    known_cards: List[Card]  # cards we know the location of
    played_cards: List[Card]  # cards that have been played
    trump_suit: Optional[str] = None
    trump_revealed: bool = False


class BeliefNetwork(nn.Module):
    """
    Neural network for predicting opponent hand distributions
    """
    
    def __init__(self, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        
        # Input features: hand (32) + bidding history (4) + played cards (32) + game state (10)
        input_dim = 32 + 4 + 32 + 10
        
        # Feature extractor
        layers = []
        prev_dim = input_dim
        for i in range(num_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output heads for each opponent
        self.opponent_heads = nn.ModuleDict({
            str(i): nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 32),  # 32 cards
                nn.Sigmoid()
            ) for i in range(4)
        })
        
        # Trump prediction head
        self.trump_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4),  # 4 suits
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor, player_id: int) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        features = self.feature_extractor(x)
        
        # Get opponent predictions
        opponent_predictions = {}
        for opp_id in range(4):
            if opp_id != player_id:
                opponent_predictions[str(opp_id)] = self.opponent_heads[str(opp_id)](features)
        
        # Get trump prediction
        trump_prediction = self.trump_head(features)
        
        return {
            'opponent_hands': opponent_predictions,
            'trump_suit': trump_prediction
        }
    
    def predict_beliefs(self, game_state: Game28State, player_id: int) -> BeliefState:
        """Predict belief state for the current game state"""
        # Encode input
        x = self._encode_game_state(game_state, player_id)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.forward(x, player_id)
        
        # Convert to belief state
        opponent_hands = {}
        for opp_id in range(4):
            if opp_id != player_id:
                card_probs = predictions['opponent_hands'][str(opp_id)].cpu().numpy()
                opponent_hands[opp_id] = card_probs.tolist()
        
        # Get known and played cards
        known_cards = game_state.hands[player_id].copy()
        if game_state.face_down_trump:
            known_cards.append(game_state.face_down_trump)
        
        played_cards = []
        for trick in game_state.tricks:
            for _, card in trick.cards:
                played_cards.append(card)
        for _, card in game_state.current_trick.cards:
            played_cards.append(card)
        
        return BeliefState(
            player_id=player_id,
            opponent_hands=opponent_hands,
            known_cards=known_cards,
            played_cards=played_cards,
            trump_suit=game_state.trump_suit,
            trump_revealed=game_state.trump_revealed
        )
    
    def _encode_game_state(self, game_state: Game28State, player_id: int) -> torch.Tensor:
        """Encode game state into tensor"""
        # Hand encoding
        hand_encoding = np.zeros(32)
        for card in game_state.hands[player_id]:
            card_idx = SUITS.index(card.suit) * 8 + RANKS.index(card.rank)
            hand_encoding[card_idx] = 1
        
        # Bidding history encoding
        bidding_history = np.zeros(4)
        for i, (player, bid) in enumerate(game_state.bid_history[-4:]):
            bidding_history[i] = bid if bid != -1 else 0
        
        # Played cards encoding
        played_encoding = np.zeros(32)
        for trick in game_state.tricks:
            for _, card in trick.cards:
                card_idx = SUITS.index(card.suit) * 8 + RANKS.index(card.rank)
                played_encoding[card_idx] = 1
        for _, card in game_state.current_trick.cards:
            card_idx = SUITS.index(card.suit) * 8 + RANKS.index(card.rank)
            played_encoding[card_idx] = 1
        
        # Game state encoding
        game_state_encoding = np.array([
            game_state.current_bid,
            player_id,
            list(GamePhase).index(game_state.phase),
            4 if game_state.trump_suit is None else SUITS.index(game_state.trump_suit),
            int(game_state.trump_revealed),
            4 if game_state.bidder is None else game_state.bidder,
            game_state.winning_bid if game_state.winning_bid else 0,
            game_state.team_scores['A'],
            game_state.team_scores['B'],
            len(game_state.tricks)
        ])
        
        # Concatenate all features
        features = np.concatenate([
            hand_encoding,
            bidding_history,
            played_encoding,
            game_state_encoding
        ])
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)


class BeliefUpdater:
    """
    Updates belief state based on new observations
    """
    
    def __init__(self):
        pass
    
    def update_beliefs(self, belief_state: BeliefState, action: str, player: int, card: Optional[Card] = None) -> BeliefState:
        """Update belief state based on new action"""
        # Create new belief state
        new_belief_state = BeliefState(
            player_id=belief_state.player_id,
            opponent_hands=belief_state.opponent_hands.copy(),
            known_cards=belief_state.known_cards.copy(),
            played_cards=belief_state.played_cards.copy(),
            trump_suit=belief_state.trump_suit,
            trump_revealed=belief_state.trump_revealed
        )
        
        if action == "play_card" and card:
            # Card was played - update probabilities
            card_idx = SUITS.index(card.suit) * 8 + RANKS.index(card.rank)
            new_belief_state.played_cards.append(card)
            
            # Set probability to 0 for all opponents
            for opp_id in new_belief_state.opponent_hands:
                new_belief_state.opponent_hands[opp_id][card_idx] = 0.0
            
            # Renormalize probabilities
            self._renormalize_beliefs(new_belief_state)
        
        elif action == "reveal_trump":
            new_belief_state.trump_revealed = True
        
        elif action == "set_trump" and card:
            new_belief_state.trump_suit = card.suit
            # Update face-down trump
            new_belief_state.known_cards.append(card)
            
            # Update probabilities for bidder
            card_idx = SUITS.index(card.suit) * 8 + RANKS.index(card.rank)
            bidder = None  # Need to get from game state
            if bidder in new_belief_state.opponent_hands:
                new_belief_state.opponent_hands[bidder][card_idx] = 0.0
                self._renormalize_beliefs(new_belief_state)
        
        return new_belief_state
    
    def _renormalize_beliefs(self, belief_state: BeliefState):
        """Renormalize belief probabilities"""
        for opp_id in belief_state.opponent_hands:
            probs = np.array(belief_state.opponent_hands[opp_id])
            # Set known cards to 0
            for card in belief_state.known_cards:
                card_idx = SUITS.index(card.suit) * 8 + RANKS.index(card.rank)
                probs[card_idx] = 0.0
            for card in belief_state.played_cards:
                card_idx = SUITS.index(card.suit) * 8 + RANKS.index(card.rank)
                probs[card_idx] = 0.0
            
            # Renormalize
            total = np.sum(probs)
            if total > 0:
                probs = probs / total
            else:
                # If all cards are known, set remaining to uniform
                remaining_cards = 8 - len(belief_state.known_cards) - len(belief_state.played_cards)
                if remaining_cards > 0:
                    probs = np.ones(32) / remaining_cards
                    for card in belief_state.known_cards + belief_state.played_cards:
                        card_idx = SUITS.index(card.suit) * 8 + RANKS.index(card.rank)
                        probs[card_idx] = 0.0
            
            belief_state.opponent_hands[opp_id] = probs.tolist()
    
    def sample_opponent_hands(self, belief_state: BeliefState, num_samples: int = 100) -> List[Dict[int, List[Card]]]:
        """Sample opponent hands from belief distribution"""
        samples = []
        
        for _ in range(num_samples):
            sample = {}
            
            for opp_id in belief_state.opponent_hands:
                probs = np.array(belief_state.opponent_hands[opp_id])
                
                # Sample 8 cards for this opponent
                opponent_hand = []
                available_cards = list(range(32))
                
                # Remove known and played cards
                for card in belief_state.known_cards + belief_state.played_cards:
                    card_idx = SUITS.index(card.suit) * 8 + RANKS.index(card.rank)
                    if card_idx in available_cards:
                        available_cards.remove(card_idx)
                
                # Sample cards
                for _ in range(8):
                    if not available_cards:
                        break
                    
                    # Get probabilities for available cards
                    available_probs = [probs[i] for i in available_cards]
                    total_prob = sum(available_probs)
                    
                    if total_prob > 0:
                        # Sample based on probabilities
                        available_probs = [p / total_prob for p in available_probs]
                        chosen_idx = np.random.choice(len(available_cards), p=available_probs)
                    else:
                        # Uniform sampling
                        chosen_idx = np.random.randint(len(available_cards))
                    
                    card_idx = available_cards.pop(chosen_idx)
                    suit_idx = card_idx // 8
                    rank_idx = card_idx % 8
                    card = Card(SUITS[suit_idx], RANKS[rank_idx])
                    opponent_hand.append(card)
                
                sample[opp_id] = opponent_hand
            
            samples.append(sample)
        
        return samples
