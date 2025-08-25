"""
Simplified Advanced Belief Network
A working version that focuses on core functionality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from game28.game_state import Game28State, Card
from game28.constants import *
from .advanced_parser import GameState, Move


@dataclass
class BeliefPrediction:
    """Comprehensive belief predictions"""
    opponent_hands: Dict[int, torch.Tensor]  # player_id -> card probabilities (32)
    trump_suit: torch.Tensor  # suit probabilities (4)
    void_suits: Dict[int, torch.Tensor]  # player_id -> void probabilities (4)
    uncertainty: torch.Tensor  # prediction confidence (1)


class SimpleAdvancedBeliefNetwork(nn.Module):
    """
    Simplified Advanced Belief Network
    Focuses on core functionality with proper dimensions
    """
    
    def __init__(self, 
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Input encoders
        self.card_encoder = nn.Sequential(
            nn.Linear(32 * 6, hidden_dim),  # 32 cards * 6 features each
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.player_encoder = nn.Sequential(
            nn.Linear(4 * 15, hidden_dim),  # 4 players * 15 features each
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.game_encoder = nn.Sequential(
            nn.Linear(20, hidden_dim),  # 20 game features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.temporal_encoder = nn.Sequential(
            nn.Linear(20 * 10, hidden_dim),  # 20 moves * 10 features each
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Feature fusion
        total_dim = hidden_dim * 4  # card + player + game + temporal
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Multi-layer processing
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_layers)
        ])
        
        # Prediction heads
        self.hand_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 32),  # 32 cards
            nn.Sigmoid()
        )
        
        self.trump_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 4),  # 4 suits
            nn.Softmax(dim=-1)
        )
        
        self.void_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 4),  # 4 suits
            nn.Sigmoid()
        )
        
        self.uncertainty_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, 
                card_features: torch.Tensor,
                player_features: torch.Tensor,
                game_features: torch.Tensor,
                temporal_features: torch.Tensor,
                player_id: int) -> BeliefPrediction:
        """
        Forward pass with comprehensive feature processing
        
        Args:
            card_features: [batch_size, 32, 6] - Card-level features
            player_features: [batch_size, 4, 15] - Player-level features
            game_features: [batch_size, 20] - Game-level features
            temporal_features: [batch_size, 20, 10] - Temporal features
            player_id: Current player ID
        """
        batch_size = card_features.size(0)
        
        # Flatten and encode inputs
        card_flat = card_features.view(batch_size, -1)  # [batch, 32*6]
        player_flat = player_features.view(batch_size, -1)  # [batch, 4*15]
        temporal_flat = temporal_features.view(batch_size, -1)  # [batch, 20*10]
        
        card_emb = self.card_encoder(card_flat)
        player_emb = self.player_encoder(player_flat)
        game_emb = self.game_encoder(game_features)
        temporal_emb = self.temporal_encoder(temporal_flat)
        
        # Concatenate all features
        combined_features = torch.cat([
            card_emb,
            player_emb,
            game_emb,
            temporal_emb
        ], dim=-1)
        
        # Feature fusion
        features = self.feature_fusion(combined_features)
        
        # Multi-layer processing
        for layer in self.layers:
            features = features + layer(features)  # Residual connection
        
        # Generate predictions for each opponent
        opponent_hands = {}
        void_suits = {}
        
        for opp_id in range(4):
            if opp_id != player_id:
                # Hand prediction
                hand_pred = self.hand_predictor(features)
                opponent_hands[opp_id] = hand_pred
                
                # Void prediction
                void_pred = self.void_predictor(features)
                void_suits[opp_id] = void_pred
        
        # Trump prediction
        trump_pred = self.trump_predictor(features)
        
        # Overall uncertainty
        uncertainty = self.uncertainty_predictor(features)
        
        return BeliefPrediction(
            opponent_hands=opponent_hands,
            trump_suit=trump_pred,
            void_suits=void_suits,
            uncertainty=uncertainty
        )
    
    def predict_beliefs(self, game_state: GameState, player_id: int) -> BeliefPrediction:
        """Predict beliefs for a given game state"""
        # Encode game state into features
        card_features = self._encode_card_features(game_state, player_id)
        player_features = self._encode_player_features(game_state, player_id)
        game_features = self._encode_game_features(game_state)
        temporal_features = self._encode_temporal_features(game_state)
        
        # Forward pass
        with torch.no_grad():
            predictions = self.forward(
                card_features.unsqueeze(0),
                player_features.unsqueeze(0),
                game_features.unsqueeze(0),
                temporal_features.unsqueeze(0),
                player_id
            )
        
        return predictions
    
    def _encode_card_features(self, game_state: GameState, player_id: int) -> torch.Tensor:
        """Encode card features for all 32 cards"""
        features = torch.zeros(32, 6)
        
        for card_idx in range(32):
            suit_idx = card_idx // 8
            rank_idx = card_idx % 8
            
            # Basic card features
            features[card_idx, 0] = suit_idx  # Suit
            features[card_idx, 1] = rank_idx  # Rank
            features[card_idx, 2] = CARD_VALUES[RANKS[rank_idx]]  # Point value
            features[card_idx, 3] = TRICK_RANKINGS[RANKS[rank_idx]]  # Trick ranking
            
            # Trump ranking
            if game_state.trump_suit:
                trump_rank = TRICK_RANKINGS[RANKS[rank_idx]] if SUITS[suit_idx] == game_state.trump_suit else 0
                features[card_idx, 4] = trump_rank
            else:
                features[card_idx, 4] = 0
            
            # Playability
            features[card_idx, 5] = 1.0
        
        return features
    
    def _encode_player_features(self, game_state: GameState, player_id: int) -> torch.Tensor:
        """Encode player features for all 4 players"""
        features = torch.zeros(4, 15)
        
        for pid in range(4):
            if pid in game_state.hands:
                hand = game_state.hands[pid]
                
                # Basic hand features
                features[pid, 0] = len(hand)  # Hand size
                
                # Suit counts
                for suit_idx, suit in enumerate(SUITS):
                    suit_count = sum(1 for card in hand if card.suit == suit)
                    features[pid, 1 + suit_idx] = suit_count
                
                # Void suits
                for suit_idx, suit in enumerate(SUITS):
                    has_suit = any(card.suit == suit for card in hand)
                    features[pid, 5 + suit_idx] = 0.0 if has_suit else 1.0
                
                # High cards
                high_cards = sum(1 for card in hand if card.rank in ['A', 'K', 'Q', 'J'])
                features[pid, 9] = high_cards
                
                # Trump cards
                if game_state.trump_suit:
                    trump_cards = sum(1 for card in hand if card.suit == game_state.trump_suit)
                    features[pid, 10] = trump_cards
                else:
                    features[pid, 10] = 0
                
                # Bidding features
                if game_state.bidding_history:
                    player_bids = [bid for player, bid in game_state.bidding_history if player == pid]
                    features[pid, 11] = max(player_bids) if player_bids else 0
                else:
                    features[pid, 11] = 0
                
                # Role encoding
                if pid == game_state.bidder:
                    features[pid, 12] = 0  # Bidder
                elif pid in [0, 2] and player_id in [0, 2] or pid in [1, 3] and player_id in [1, 3]:
                    features[pid, 12] = 1  # Partner
                elif pid == player_id:
                    features[pid, 12] = 2  # Self
                else:
                    features[pid, 12] = 3  # Opponent
                
                # Score features
                if pid in [0, 2]:
                    features[pid, 13] = game_state.team_a_score
                else:
                    features[pid, 13] = game_state.team_b_score
                
                # Tricks won
                tricks_won = sum(1 for play in game_state.played_cards if play.player_id == pid and play.won_trick)
                features[pid, 14] = tricks_won
        
        return features
    
    def _encode_game_features(self, game_state: GameState) -> torch.Tensor:
        """Encode game-level features"""
        features = torch.zeros(20)
        
        # Phase encoding
        phase_map = {'bidding': 0, 'concealed': 1, 'revealed': 2}
        features[0] = phase_map.get(game_state.phase, 0)
        
        # Trump encoding
        if game_state.trump_suit:
            trump_idx = SUITS.index(game_state.trump_suit)
            features[1] = trump_idx
        else:
            features[1] = 4  # Unknown
        
        # Game progress
        features[2] = game_state.game_progress
        
        # Scores
        features[3] = game_state.team_a_score
        features[4] = game_state.team_b_score
        
        # Bidding information
        features[5] = game_state.current_bid
        features[6] = game_state.bidder if game_state.bidder is not None else -1
        
        # Trick information
        features[7] = game_state.tricks_played
        features[8] = game_state.cards_played
        
        # Trump revelation
        features[9] = 1.0 if game_state.trump_revealed else 0.0
        features[10] = game_state.trump_revealer if game_state.trump_revealer is not None else -1
        
        # Game pressure
        features[11] = game_state.game_progress * 10
        
        # Risk level
        features[12] = abs(game_state.team_a_score - game_state.team_b_score) / 28.0
        
        # Expected value
        features[13] = (game_state.team_a_score + game_state.team_b_score) / 28.0
        
        # Variance
        features[14] = 0.5
        
        # Information entropy
        features[15] = 1.0 - game_state.game_progress
        
        # Time remaining
        features[16] = 1.0 - game_state.game_progress
        
        # Lead suit
        if game_state.current_trick and game_state.current_trick.cards:
            lead_suit = game_state.current_trick.cards[0][1].suit
            features[17] = SUITS.index(lead_suit)
        else:
            features[17] = -1
        
        # Current player
        features[18] = game_state.current_player
        
        # Game completion
        features[19] = game_state.game_progress
        
        return features
    
    def _encode_temporal_features(self, game_state: GameState) -> torch.Tensor:
        """Encode temporal features from move history"""
        max_moves = 20  # Fixed sequence length
        
        if not game_state.played_cards:
            return torch.zeros(max_moves, 10)  # Empty sequence padded
        
        # Take last 20 moves
        recent_moves = game_state.played_cards[-max_moves:]
        
        features = torch.zeros(max_moves, 10)
        
        for i, move in enumerate(recent_moves):
            # Move features
            features[i, 0] = move.player_id
            features[i, 1] = SUITS.index(move.card.suit)
            features[i, 2] = RANKS.index(move.card.rank)
            features[i, 3] = move.trick_number
            features[i, 4] = move.position_in_trick
            features[i, 5] = 1.0 if move.won_trick else 0.0
            features[i, 6] = move.points_earned
            features[i, 7] = 1.0 if move.trump_played else 0.0
            features[i, 8] = 1.0 if move.high_card_played else 0.0
            features[i, 9] = 1.0 if move.forced_play else 0.0
        
        return features
