"""
Improved Belief Network
A much better implementation that properly encodes game state and makes realistic predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random

from game28.game_state import Game28State, Card, Trick
from game28.constants import *


@dataclass
class BeliefPrediction:
    """Comprehensive belief predictions"""
    opponent_hands: Dict[int, torch.Tensor]  # player_id -> card probabilities (32)
    trump_suit: torch.Tensor  # suit probabilities (4)
    void_suits: Dict[int, torch.Tensor]  # player_id -> void probabilities (4)
    uncertainty: torch.Tensor  # prediction confidence (1)


class ImprovedBeliefNetwork(nn.Module):
    """
    Improved Belief Network
    Properly encodes game state and makes realistic predictions
    """
    
    def __init__(self, 
                 hidden_dim: int = 512,
                 num_layers: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Card encoder - encodes each card's features
        self.card_encoder = nn.Sequential(
            nn.Linear(8, hidden_dim // 4),  # 8 features per card
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        # Hand encoder - processes all cards in a hand
        self.hand_encoder = nn.Sequential(
            nn.Linear(8 * 8, hidden_dim // 2),  # 8 cards * 8 features each
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Game state encoder
        self.game_encoder = nn.Sequential(
            nn.Linear(50, hidden_dim),  # 50 game state features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Trick history encoder
        self.trick_encoder = nn.Sequential(
            nn.Linear(8 * 16, hidden_dim // 2),  # 8 tricks * 16 features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Feature fusion
        total_dim = hidden_dim + hidden_dim // 2 + hidden_dim // 2  # game + hand + trick
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
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, game_state: Game28State, player_id: int) -> BeliefPrediction:
        """Forward pass with improved encoding"""
        
        # Encode our hand
        our_hand = game_state.hands[player_id]
        hand_features = self._encode_hand(our_hand, game_state)
        
        # Encode game state
        game_features = self._encode_game_state(game_state, player_id)
        
        # Encode trick history
        trick_features = self._encode_trick_history(game_state)
        
        # Concatenate features
        combined_features = torch.cat([game_features, hand_features, trick_features], dim=-1)
        
        # Feature fusion
        fused_features = self.feature_fusion(combined_features)
        
        # Multi-layer processing
        processed_features = fused_features
        for layer in self.layers:
            processed_features = layer(processed_features)
        
        # Make predictions
        hand_probs = self.hand_predictor(processed_features)
        trump_probs = self.trump_predictor(processed_features)
        void_probs = self.void_predictor(processed_features)
        uncertainty = self.uncertainty_predictor(processed_features)
        
        # Create opponent hand predictions
        opponent_hands = {}
        for opp_id in range(4):
            if opp_id != player_id:
                # Adjust probabilities based on what we know
                opp_hand_probs = hand_probs.clone()
                
                # Remove cards we know are in our hand
                for card in our_hand:
                    card_idx = self._card_to_index(card)
                    opp_hand_probs[card_idx] = 0.0
                
                # Remove cards we've seen played
                for trick in game_state.tricks:
                    for _, card in trick.cards:
                        card_idx = self._card_to_index(card)
                        opp_hand_probs[card_idx] = 0.0
                
                # Normalize probabilities
                if opp_hand_probs.sum() > 0:
                    opp_hand_probs = opp_hand_probs / opp_hand_probs.sum()
                
                opponent_hands[opp_id] = opp_hand_probs
        
        return BeliefPrediction(
            opponent_hands=opponent_hands,
            trump_suit=trump_probs,
            void_suits={opp_id: void_probs for opp_id in range(4) if opp_id != player_id},
            uncertainty=uncertainty
        )
    
    def predict_beliefs(self, game_state: Game28State, player_id: int) -> BeliefPrediction:
        """Predict beliefs with proper game state encoding"""
        with torch.no_grad():
            return self.forward(game_state, player_id)
    
    def _encode_hand(self, hand: List[Card], game_state: Game28State) -> torch.Tensor:
        """Encode our hand with rich features"""
        # Create a tensor for all 8 possible cards
        hand_tensor = torch.zeros(8, 8)  # 8 cards, 8 features each
        
        for i, card in enumerate(hand):
            if i >= 8:  # Safety check
                break
            
            # Card features
            hand_tensor[i, 0] = SUITS.index(card.suit) / 3.0  # Suit (normalized)
            hand_tensor[i, 1] = RANKS.index(card.rank) / 7.0  # Rank (normalized)
            hand_tensor[i, 2] = CARD_VALUES[card.rank] / 10.0  # Point value (normalized)
            hand_tensor[i, 3] = TRICK_RANKINGS[card.rank] / 7.0  # Trick ranking (normalized)
            
            # Trump features
            if game_state.trump_suit:
                is_trump = card.suit == game_state.trump_suit
                hand_tensor[i, 4] = 1.0 if is_trump else 0.0
                if is_trump:
                    hand_tensor[i, 5] = TRICK_RANKINGS[card.rank] / 7.0  # Trump ranking
                else:
                    hand_tensor[i, 5] = 0.0
            else:
                hand_tensor[i, 4] = 0.0
                hand_tensor[i, 5] = 0.0
            
            # High card features
            hand_tensor[i, 6] = 1.0 if card.rank in ['A', 'K', 'Q', 'J'] else 0.0
            
            # Suit strength (how many cards we have in this suit)
            suit_count = sum(1 for c in hand if c.suit == card.suit)
            hand_tensor[i, 7] = suit_count / 8.0
        
        # Flatten and encode
        hand_flat = hand_tensor.view(-1)
        hand_encoded = self.hand_encoder(hand_flat)
        
        return hand_encoded
    
    def _encode_game_state(self, game_state: Game28State, player_id: int) -> torch.Tensor:
        """Encode comprehensive game state features"""
        features = torch.zeros(50)
        
        # Phase encoding
        if game_state.phase == GamePhase.BIDDING:
            features[0] = 0.0
        elif game_state.phase == GamePhase.CONCEALED:
            features[0] = 0.5
        else:  # REVEALED
            features[0] = 1.0
        
        # Trump information
        if game_state.trump_suit:
            trump_idx = SUITS.index(game_state.trump_suit)
            features[1] = trump_idx / 3.0
            features[2] = 1.0  # Trump is set
        else:
            features[1] = 0.0
            features[2] = 0.0  # No trump set
        
        # Game progress
        game_progress = len(game_state.tricks) / 8.0
        features[3] = game_progress
        
        # Team scores
        features[4] = game_state.game_points.get('A', 0) / 100.0
        features[5] = game_state.game_points.get('B', 0) / 100.0
        
        # Bidding information
        features[6] = game_state.winning_bid / 28.0 if game_state.winning_bid else 0.0
        features[7] = game_state.bidder / 3.0 if game_state.bidder is not None else 0.0
        
        # Player role encoding
        features[8] = 1.0 if player_id == game_state.bidder else 0.0  # Are we the bidder?
        features[9] = 1.0 if player_id in [0, 2] else 0.0  # Are we on team A?
        
        # Current trick information
        if game_state.current_trick and game_state.current_trick.cards:
            features[10] = len(game_state.current_trick.cards) / 4.0  # Cards played in current trick
            if game_state.current_trick.lead_suit:
                features[11] = SUITS.index(game_state.current_trick.lead_suit) / 3.0
            else:
                features[11] = 0.0
            
            # Current high card in trick
            high_card_value = 0.0
            for _, card in game_state.current_trick.cards:
                card_value = CARD_VALUES[card.rank]
                if card.suit == game_state.current_trick.lead_suit:
                    card_value += 100  # Lead suit bonus
                elif game_state.trump_suit and card.suit == game_state.trump_suit:
                    card_value += 200  # Trump bonus
                high_card_value = max(high_card_value, card_value)
            features[12] = high_card_value / 300.0
        else:
            features[10] = 0.0
            features[11] = 0.0
            features[12] = 0.0
        
        # Our hand strength
        our_hand = game_state.hands[player_id]
        features[13] = sum(CARD_VALUES[card.rank] for card in our_hand) / 100.0  # Total points
        features[14] = len(our_hand) / 8.0  # Hand size
        
        # Suit distribution in our hand
        for suit_idx, suit in enumerate(SUITS):
            suit_count = sum(1 for card in our_hand if card.suit == suit)
            features[15 + suit_idx] = suit_count / 8.0
        
        # Void suits in our hand
        for suit_idx, suit in enumerate(SUITS):
            has_suit = any(card.suit == suit for card in our_hand)
            features[19 + suit_idx] = 0.0 if has_suit else 1.0
        
        # Trump cards in our hand
        if game_state.trump_suit:
            trump_count = sum(1 for card in our_hand if card.suit == game_state.trump_suit)
            features[23] = trump_count / 8.0
            trump_points = sum(CARD_VALUES[card.rank] for card in our_hand if card.suit == game_state.trump_suit)
            features[24] = trump_points / 100.0
        else:
            features[23] = 0.0
            features[24] = 0.0
        
        # High cards in our hand
        high_cards = sum(1 for card in our_hand if card.rank in ['A', 'K', 'Q', 'J'])
        features[25] = high_cards / 8.0
        
        # Cards played so far
        cards_played = 0
        for trick in game_state.tricks:
            cards_played += len(trick.cards)
        features[26] = cards_played / 32.0
        
        # Information about other players' hands
        for opp_id in range(4):
            if opp_id != player_id:
                opp_hand = game_state.hands[opp_id]
                opp_hand_size = len(opp_hand)
                features[27 + opp_id] = opp_hand_size / 8.0
        
        # Trick winners
        team_a_tricks = 0
        team_b_tricks = 0
        for trick in game_state.tricks:
            if trick.winner in [0, 2]:
                team_a_tricks += 1
            else:
                team_b_tricks += 1
        
        features[31] = team_a_tricks / 8.0
        features[32] = team_b_tricks / 8.0
        
        # Game pressure (how close to end)
        features[33] = game_progress
        
        # Score difference
        score_diff = abs(game_state.game_points.get('A', 0) - game_state.game_points.get('B', 0))
        features[34] = score_diff / 100.0
        
        # Risk level (based on bidding)
        if game_state.winning_bid:
            risk = game_state.winning_bid / 28.0
            features[35] = risk
        else:
            features[35] = 0.0
        
        # Position in current trick
        if game_state.current_trick and game_state.current_trick.cards:
            position = len(game_state.current_trick.cards)
            features[36] = position / 4.0
        else:
            features[36] = 0.0
        
        # Can we follow suit?
        if game_state.current_trick and game_state.current_trick.cards:
            lead_suit = game_state.current_trick.lead_suit
            can_follow = any(card.suit == lead_suit for card in our_hand)
            features[37] = 1.0 if can_follow else 0.0
        else:
            features[37] = 1.0  # Leading
        
        # Uncertainty based on game progress
        features[38] = 1.0 - game_progress  # More uncertainty early in game
        
        # Team advantage
        team_a_score = game_state.game_points.get('A', 0)
        team_b_score = game_state.game_points.get('B', 0)
        if team_a_score > team_b_score:
            features[39] = (team_a_score - team_b_score) / 100.0
        else:
            features[39] = -(team_b_score - team_a_score) / 100.0
        
        # Remaining features for future expansion
        features[40:50] = 0.0
        
        return self.game_encoder(features)
    
    def _encode_trick_history(self, game_state: Game28State) -> torch.Tensor:
        """Encode trick history with rich features"""
        # Create tensor for 8 tricks, 16 features each
        trick_tensor = torch.zeros(8, 16)
        
        for trick_idx, trick in enumerate(game_state.tricks):
            if trick_idx >= 8:  # Safety check
                break
            
            # Trick features
            trick_tensor[trick_idx, 0] = trick_idx / 8.0  # Trick number (normalized)
            trick_tensor[trick_idx, 1] = len(trick.cards) / 4.0  # Cards played
            
            if trick.cards:
                # Lead suit
                lead_suit = trick.lead_suit
                if lead_suit:
                    trick_tensor[trick_idx, 2] = SUITS.index(lead_suit) / 3.0
                
                # Winner
                if trick.winner is not None:
                    trick_tensor[trick_idx, 3] = trick.winner / 3.0
                    trick_tensor[trick_idx, 4] = 1.0 if trick.winner in [0, 2] else 0.0  # Team A won
                
                # Points won
                trick_tensor[trick_idx, 5] = trick.points / 10.0
                
                # High card value in trick
                high_value = max(CARD_VALUES[card.rank] for _, card in trick.cards)
                trick_tensor[trick_idx, 6] = high_value / 10.0
                
                # Trump played
                trump_played = any(card.suit == game_state.trump_suit for _, card in trick.cards) if game_state.trump_suit else False
                trick_tensor[trick_idx, 7] = 1.0 if trump_played else 0.0
                
                # Cards by suit
                for suit_idx, suit in enumerate(SUITS):
                    suit_count = sum(1 for _, card in trick.cards if card.suit == suit)
                    trick_tensor[trick_idx, 8 + suit_idx] = suit_count / 4.0
                
                # High cards played
                high_cards = sum(1 for _, card in trick.cards if card.rank in ['A', 'K', 'Q', 'J'])
                trick_tensor[trick_idx, 12] = high_cards / 4.0
                
                # Position of winner
                if trick.winner is not None:
                    winner_pos = 0
                    for pos, (player, _) in enumerate(trick.cards):
                        if player == trick.winner:
                            winner_pos = pos
                            break
                    trick_tensor[trick_idx, 13] = winner_pos / 4.0
                
                # Trick complexity (how many different suits)
                suits_played = len(set(card.suit for _, card in trick.cards))
                trick_tensor[trick_idx, 14] = suits_played / 4.0
                
                # Trick value (total points)
                total_points = sum(CARD_VALUES[card.rank] for _, card in trick.cards)
                trick_tensor[trick_idx, 15] = total_points / 40.0
        
        # Flatten and encode
        trick_flat = trick_tensor.view(-1)
        trick_encoded = self.trick_encoder(trick_flat)
        
        return trick_encoded
    
    def _card_to_index(self, card: Card) -> int:
        """Convert card to index in 32-card deck"""
        suit_idx = SUITS.index(card.suit)
        rank_idx = RANKS.index(card.rank)
        return suit_idx * 8 + rank_idx


def create_improved_belief_model() -> ImprovedBeliefNetwork:
    """Create and initialize an improved belief model"""
    model = ImprovedBeliefNetwork()
    
    # Initialize weights properly
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    return model


def train_improved_belief_model(model: ImprovedBeliefNetwork, 
                               training_data: List[Tuple[Game28State, int, Dict]], 
                               epochs: int = 100,
                               learning_rate: float = 0.001) -> ImprovedBeliefNetwork:
    """Train the improved belief model on realistic data"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for game_state, player_id, target_beliefs in training_data:
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(game_state, player_id)
            
            # Calculate loss for each prediction type
            loss = 0.0
            
            # Hand prediction loss
            for opp_id, target_hand in target_beliefs.get('hands', {}).items():
                if opp_id in predictions.opponent_hands:
                    pred_hand = predictions.opponent_hands[opp_id]
                    target_tensor = torch.tensor(target_hand, dtype=torch.float32)
                    loss += criterion(pred_hand, target_tensor)
            
            # Trump prediction loss
            if 'trump' in target_beliefs:
                target_trump = torch.tensor(target_beliefs['trump'], dtype=torch.float32)
                loss += criterion(predictions.trump_suit, target_trump)
            
            # Void prediction loss
            for opp_id, target_void in target_beliefs.get('voids', {}).items():
                if opp_id in predictions.void_suits:
                    pred_void = predictions.void_suits[opp_id]
                    target_tensor = torch.tensor(target_void, dtype=torch.float32)
                    loss += criterion(pred_void, target_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(training_data):.4f}")
    
    return model


def generate_realistic_training_data(num_games: int = 1000) -> List[Tuple[Game28State, int, Dict]]:
    """Generate realistic training data for the belief model"""
    training_data = []
    
    for game_idx in range(num_games):
        # Create a realistic game state
        game_state = Game28State()
        
        # Simulate some game progress
        num_tricks = random.randint(0, 8)
        for _ in range(num_tricks):
            # Simulate a trick
            trick = Trick()
            for player in range(4):
                # Random card
                suit = random.choice(SUITS)
                rank = random.choice(RANKS)
                card = Card(suit, rank)
                trick.add_card(player, card)
            
            # Determine winner (simplified)
            winner = random.randint(0, 3)
            trick.winner = winner
            trick.points = sum(CARD_VALUES[card.rank] for _, card in trick.cards)
            
            game_state.tricks.append(trick)
        
        # Set random trump
        if random.random() > 0.3:  # 70% chance of trump being set
            game_state.trump_suit = random.choice(SUITS)
            game_state.phase = GamePhase.CONCEALED if random.random() > 0.5 else GamePhase.REVEALED
        
        # Set random bidder
        if random.random() > 0.2:  # 80% chance of having a bidder
            game_state.bidder = random.randint(0, 3)
            game_state.winning_bid = random.randint(16, 28)
        
        # Generate realistic target beliefs
        for player_id in range(4):
            target_beliefs = {}
            
            # Hand beliefs (simplified - assume uniform distribution for unknown cards)
            for opp_id in range(4):
                if opp_id != player_id:
                    # Create realistic hand probabilities
                    hand_probs = [0.1] * 32  # Base probability
                    
                    # Remove cards we know about
                    for card in game_state.hands[player_id]:
                        card_idx = SUITS.index(card.suit) * 8 + RANKS.index(card.rank)
                        hand_probs[card_idx] = 0.0
                    
                    # Remove cards we've seen played
                    for trick in game_state.tricks:
                        for _, card in trick.cards:
                            card_idx = SUITS.index(card.suit) * 8 + RANKS.index(card.rank)
                            hand_probs[card_idx] = 0.0
                    
                    # Normalize
                    total_prob = sum(hand_probs)
                    if total_prob > 0:
                        hand_probs = [p / total_prob for p in hand_probs]
                    
                    target_beliefs.setdefault('hands', {})[opp_id] = hand_probs
            
            # Trump beliefs
            if game_state.trump_suit:
                trump_probs = [0.0, 0.0, 0.0, 0.0]
                trump_idx = SUITS.index(game_state.trump_suit)
                trump_probs[trump_idx] = 1.0
            else:
                trump_probs = [0.25, 0.25, 0.25, 0.25]  # Uniform
            
            target_beliefs['trump'] = trump_probs
            
            # Void beliefs
            for opp_id in range(4):
                if opp_id != player_id:
                    void_probs = [0.1, 0.1, 0.1, 0.1]  # Base void probability
                    
                    # Adjust based on what we've seen
                    for trick in game_state.tricks:
                        for _, card in trick.cards:
                            if card.suit in SUITS:
                                suit_idx = SUITS.index(card.suit)
                                void_probs[suit_idx] = 0.0  # Can't be void if we've seen them play it
                    
                    target_beliefs.setdefault('voids', {})[opp_id] = void_probs
            
            training_data.append((game_state, player_id, target_beliefs))
    
    return training_data
