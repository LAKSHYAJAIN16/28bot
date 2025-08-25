"""
Advanced Belief Network with Multi-Head Attention and Game-Specific Components
State-of-the-art neural architecture for opponent modeling in imperfect information games
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math

from game28.game_state import Game28State, Card
from game28.constants import *
from .advanced_parser import GameState, Move


@dataclass
class BeliefPrediction:
    """Comprehensive belief predictions"""
    opponent_hands: Dict[int, torch.Tensor]  # player_id -> card probabilities (32)
    trump_suit: torch.Tensor  # suit probabilities (4)
    void_suits: Dict[int, torch.Tensor]  # player_id -> void probabilities (4)
    card_counts: Dict[int, torch.Tensor]  # player_id -> suit counts (4)
    play_style: Dict[int, torch.Tensor]  # player_id -> style features (16)
    uncertainty: torch.Tensor  # prediction confidence (1)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with game-specific enhancements"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(context)
        return output


class GameSpecificAttention(nn.Module):
    """Game-specific attention mechanisms for card relationships"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Suit-aware attention
        self.suit_attention = nn.ModuleDict({
            suit: MultiHeadAttention(d_model // 4, 4) for suit in ['H', 'D', 'C', 'S']
        })
        
        # Rank-aware attention
        self.rank_attention = nn.ModuleDict({
            rank: MultiHeadAttention(d_model // 8, 2) for rank in ['7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        })
        
        # Trick-aware attention
        self.trick_attention = MultiHeadAttention(d_model, 4)
        
        # Player-role attention
        self.role_attention = nn.ModuleDict({
            'bidder': MultiHeadAttention(d_model, 4),
            'partner': MultiHeadAttention(d_model, 4),
            'opponent': MultiHeadAttention(d_model, 4)
        })
        
        # Cross-modal attention
        self.cross_attention = MultiHeadAttention(d_model, 4)
    
    def forward(self, card_features: torch.Tensor, player_features: torch.Tensor, 
                game_features: torch.Tensor) -> torch.Tensor:
        # Suit-based attention
        suit_attended = []
        for i, suit in enumerate(['H', 'D', 'C', 'S']):
            suit_mask = torch.arange(32) // 8 == i
            suit_cards = card_features[:, suit_mask, :]
            if suit_cards.size(1) > 0:
                attended = self.suit_attention[suit](suit_cards, suit_cards, suit_cards)
                suit_attended.append(attended)
        
        # Combine suit attention
        card_attended = torch.cat(suit_attended, dim=1) if suit_attended else card_features
        
        # Player-role attention
        role_attended = []
        for role in ['bidder', 'partner', 'opponent']:
            attended = self.role_attention[role](player_features, player_features, player_features)
            role_attended.append(attended)
        
        player_attended = torch.stack(role_attended, dim=1).mean(dim=1)
        
        # Cross-modal attention between cards and players
        cross_attended = self.cross_attention(card_attended, player_attended, player_attended)
        
        return cross_attended


class CardEncoder(nn.Module):
    """Advanced card encoding with rich features"""
    
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model
        
        # Card-level features
        self.suit_embedding = nn.Embedding(4, d_model // 4)  # H, D, C, S
        self.rank_embedding = nn.Embedding(8, d_model // 4)  # 7, 8, 9, 10, J, Q, K, A
        self.point_embedding = nn.Embedding(4, d_model // 4)  # 0, 1, 2, 3 points
        self.trick_rank_embedding = nn.Embedding(8, d_model // 4)  # 0-7 trick ranking
        
        # Positional encoding for card order
        self.positional_encoding = nn.Parameter(torch.randn(32, d_model))
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, card_features: torch.Tensor) -> torch.Tensor:
        # card_features: [batch_size, 32, 6] where 6 = [suit, rank, points, trick_rank, trump_rank, playability]
        batch_size = card_features.size(0)
        
        # Extract features
        suits = card_features[:, :, 0].long()
        ranks = card_features[:, :, 1].long()
        points = card_features[:, :, 2].long()
        trick_ranks = card_features[:, :, 3].long()
        
        # Embeddings
        suit_emb = self.suit_embedding(suits)  # [batch, 32, d_model//4]
        rank_emb = self.rank_embedding(ranks)  # [batch, 32, d_model//4]
        point_emb = self.point_embedding(points)  # [batch, 32, d_model//8]
        trick_emb = self.trick_rank_embedding(trick_ranks)  # [batch, 32, d_model//8]
        
        # Concatenate embeddings
        combined = torch.cat([suit_emb, rank_emb, point_emb, trick_emb], dim=-1)
        
        # Add positional encoding
        combined = combined + self.positional_encoding.unsqueeze(0)
        
        # Feature fusion
        output = self.feature_fusion(combined)
        
        return output


class PlayerEncoder(nn.Module):
    """Advanced player encoding with behavioral features"""
    
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        
        # Player-specific features
        self.hand_size_embedding = nn.Embedding(9, d_model // 8)  # 0-8 cards
        self.suit_count_embedding = nn.Embedding(9, d_model // 8)  # 0-8 cards per suit
        self.bid_embedding = nn.Embedding(29, d_model // 8)  # 0-28 bids
        self.role_embedding = nn.Embedding(4, d_model // 8)  # bidder, partner, opponent, self
        
        # Behavioral features
        self.behavior_encoder = nn.Sequential(
            nn.Linear(8, d_model // 2),  # 8 behavioral features (remaining after other features)
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model // 2)
        )
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, player_features: torch.Tensor) -> torch.Tensor:
        # player_features: [batch_size, 4, 15] where 15 = player features
        batch_size = player_features.size(0)
        
        # Extract features
        hand_sizes = player_features[:, :, 0].long()
        suit_counts = player_features[:, :, 1:5].long()
        bids = player_features[:, :, 5].long()
        roles = player_features[:, :, 6].long()
        behavior = player_features[:, :, 7:15]  # 8 behavioral features
        
        # Embeddings
        hand_emb = self.hand_size_embedding(hand_sizes)
        suit_emb = self.suit_count_embedding(suit_counts).mean(dim=2)
        bid_emb = self.bid_embedding(bids)
        role_emb = self.role_embedding(roles)
        
        # Behavioral encoding
        behavior_emb = self.behavior_encoder(behavior)
        
        # Concatenate
        combined = torch.cat([hand_emb, suit_emb, bid_emb, role_emb, behavior_emb], dim=-1)
        
        # Feature fusion
        output = self.feature_fusion(combined)
        
        return output


class GameEncoder(nn.Module):
    """Advanced game state encoding"""
    
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.d_model = d_model
        
        # Game phase embedding
        self.phase_embedding = nn.Embedding(3, d_model // 4)  # bidding, concealed, revealed
        
        # Trump suit embedding
        self.trump_embedding = nn.Embedding(5, d_model // 4)  # H, D, C, S, unknown
        
        # Game progress encoding
        self.progress_encoder = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model // 4)
        )
        
        # Score encoding
        self.score_encoder = nn.Sequential(
            nn.Linear(2, d_model // 4),  # team_a_score, team_b_score
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model // 4)
        )
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, game_features: torch.Tensor) -> torch.Tensor:
        # game_features: [batch_size, 20] where 20 = game features
        batch_size = game_features.size(0)
        
        # Extract features
        phases = game_features[:, 0].long()
        trumps = game_features[:, 1].long()
        progress = game_features[:, 2:3]  # game progress
        scores = game_features[:, 3:5]  # team scores
        
        # Embeddings
        phase_emb = self.phase_embedding(phases)
        trump_emb = self.trump_embedding(trumps)
        progress_emb = self.progress_encoder(progress)
        score_emb = self.score_encoder(scores)
        
        # Concatenate
        combined = torch.cat([phase_emb, trump_emb, progress_emb, score_emb], dim=-1)
        
        # Feature fusion
        output = self.feature_fusion(combined)
        
        return output


class TemporalEncoder(nn.Module):
    """Temporal sequence encoding for move history"""
    
    def __init__(self, d_model: int = 128, max_sequence_length: int = 20):
        super().__init__()
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        
        # Move encoding
        self.move_encoder = nn.Sequential(
            nn.Linear(10, d_model),  # 10 move features to full d_model
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
        
        # Temporal attention
        self.temporal_attention = MultiHeadAttention(d_model, 8)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(max_sequence_length, d_model))
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, move_sequences: torch.Tensor) -> torch.Tensor:
        # move_sequences: [batch_size, seq_len, 10] where 10 = move features
        batch_size, seq_len = move_sequences.size(0), move_sequences.size(1)
        
        # Encode moves
        move_emb = self.move_encoder(move_sequences)
        
        # Add positional encoding
        pos_emb = self.positional_encoding[:seq_len].unsqueeze(0)
        move_emb = move_emb + pos_emb
        
        # Temporal attention
        attended = self.temporal_attention(move_emb, move_emb, move_emb)
        
        # Global pooling
        pooled = attended.mean(dim=1)  # [batch_size, d_model]
        
        # Feature fusion
        output = self.feature_fusion(pooled)
        
        return output


class HandPredictionHead(nn.Module):
    """Specialized head for hand prediction"""
    
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model
        
        # Multi-layer prediction
        self.prediction_layers = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 4, 32),  # 32 cards
            nn.Sigmoid()
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        predictions = self.prediction_layers(features)
        uncertainty = self.uncertainty_head(features)
        
        return {
            'predictions': predictions,
            'uncertainty': uncertainty
        }


class VoidPredictionHead(nn.Module):
    """Specialized head for void suit detection"""
    
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model
        
        self.void_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 4),  # 4 suits
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.void_predictor(features)


class TrumpPredictionHead(nn.Module):
    """Specialized head for trump prediction"""
    
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model
        
        self.trump_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 4),  # 4 suits
            nn.Softmax(dim=-1)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.trump_predictor(features)


class CountPredictionHead(nn.Module):
    """Specialized head for card counting"""
    
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model
        
        self.count_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 4),  # 4 suits
            nn.Softmax(dim=-1)  # Probabilities that sum to hand size
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.count_predictor(features)


class BehaviorPredictor(nn.Module):
    """Specialized head for play style prediction"""
    
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model
        
        self.behavior_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 16),  # 16 behavioral features
            nn.Tanh()  # Normalized behavioral features
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.behavior_predictor(features)


class AdvancedBeliefNetwork(nn.Module):
    """
    Advanced Belief Network with Multi-Head Attention and Game-Specific Components
    State-of-the-art neural architecture for opponent modeling
    """
    
    def __init__(self, 
                 card_dim: int = 256,
                 player_dim: int = 128,
                 game_dim: int = 64,
                 temporal_dim: int = 128,
                 hidden_dim: int = 512,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        # Input encoders
        self.card_encoder = CardEncoder(card_dim)
        self.player_encoder = PlayerEncoder(player_dim)
        self.game_encoder = GameEncoder(game_dim)
        self.temporal_encoder = TemporalEncoder(temporal_dim)
        
        # Game-specific attention
        self.game_attention = GameSpecificAttention(card_dim)
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Feature fusion
        total_dim = card_dim + player_dim + game_dim + temporal_dim
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Prediction heads
        self.hand_predictor = HandPredictionHead(hidden_dim)
        self.void_predictor = VoidPredictionHead(hidden_dim)
        self.trump_predictor = TrumpPredictionHead(hidden_dim)
        self.count_predictor = CountPredictionHead(hidden_dim)
        self.behavior_predictor = BehaviorPredictor(hidden_dim)
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
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
            temporal_features: [batch_size, seq_len, 10] - Temporal features
            player_id: Current player ID
        """
        batch_size = card_features.size(0)
        
        # Encode inputs
        card_emb = self.card_encoder(card_features)  # [batch, 32, card_dim]
        player_emb = self.player_encoder(player_features)  # [batch, 4, player_dim]
        game_emb = self.game_encoder(game_features)  # [batch, game_dim]
        temporal_emb = self.temporal_encoder(temporal_features)  # [batch, temporal_dim]
        
        # Game-specific attention
        attended_cards = self.game_attention(card_emb, player_emb, game_emb.unsqueeze(1))
        
        # Global card representation
        card_global = attended_cards.mean(dim=1)  # [batch, card_dim]
        
        # Concatenate all features
        combined_features = torch.cat([
            card_global,
            player_emb.mean(dim=1),  # Global player representation
            game_emb,
            temporal_emb
        ], dim=-1)
        
        # Feature fusion
        features = self.feature_fusion(combined_features)
        
        # Multi-layer attention processing
        for i, (attention, norm) in enumerate(zip(self.attention_layers, self.layer_norms)):
            # Self-attention on features
            attended = attention(features.unsqueeze(1), features.unsqueeze(1), features.unsqueeze(1))
            attended = attended.squeeze(1)
            
            # Residual connection and normalization
            features = norm(features + self.dropout(attended))
        
        # Generate predictions for each opponent
        opponent_hands = {}
        void_suits = {}
        card_counts = {}
        play_styles = {}
        
        for opp_id in range(4):
            if opp_id != player_id:
                # Opponent-specific features (could be enhanced with player-specific processing)
                opp_features = features  # For now, use same features for all opponents
                
                # Hand prediction
                hand_pred = self.hand_predictor(opp_features)
                opponent_hands[opp_id] = hand_pred['predictions']
                
                # Void prediction
                void_pred = self.void_predictor(opp_features)
                void_suits[opp_id] = void_pred
                
                # Count prediction
                count_pred = self.count_predictor(opp_features)
                card_counts[opp_id] = count_pred
                
                # Behavior prediction
                behavior_pred = self.behavior_predictor(opp_features)
                play_styles[opp_id] = behavior_pred
        
        # Trump prediction
        trump_pred = self.trump_predictor(features)
        
        # Overall uncertainty
        uncertainty = self.hand_predictor(features)['uncertainty']
        
        return BeliefPrediction(
            opponent_hands=opponent_hands,
            trump_suit=trump_pred,
            void_suits=void_suits,
            card_counts=card_counts,
            play_style=play_styles,
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
            
            # Playability (simplified)
            features[card_idx, 5] = 1.0  # All cards are playable initially
        
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
        
        # Game pressure (simplified)
        features[11] = game_state.game_progress * 10  # Pressure increases as game progresses
        
        # Risk level (simplified)
        features[12] = abs(game_state.team_a_score - game_state.team_b_score) / 28.0
        
        # Expected value (simplified)
        features[13] = (game_state.team_a_score + game_state.team_b_score) / 28.0
        
        # Variance (simplified)
        features[14] = 0.5  # Placeholder
        
        # Information entropy (simplified)
        features[15] = 1.0 - game_state.game_progress  # More entropy early in game
        
        # Time remaining (simplified)
        features[16] = 1.0 - game_state.game_progress
        
        # Lead suit (if current trick exists)
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
        if not game_state.played_cards:
            return torch.zeros(1, 10)  # Empty sequence
        
        # Take last 20 moves (or pad if fewer)
        recent_moves = game_state.played_cards[-20:]
        
        features = torch.zeros(len(recent_moves), 10)
        
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
