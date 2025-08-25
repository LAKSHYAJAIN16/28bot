#!/usr/bin/env python3
"""
Point prediction model for Game 28 bidding
Predicts expected points from 4-card hands using canonicalization
"""

import sys
import os
import json
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

# Add the parent directory to Python path to import game28 module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game28.game_state import Card
from game28.constants import CARD_VALUES, TOTAL_POINTS

class HandCanonicalizer:
    """Canonicalize hands to treat equivalent hands as the same"""
    
    def __init__(self):
        # Game 28 ranking: J < 9 < A < 10 < K < Q < 8 < 7
        self.rank_order = ['J', '9', 'A', '10', 'K', 'Q', '8', '7']
        self.suit_order = ['C', 'D', 'H', 'S']
    
    def canonicalize_hand(self, hand: List[str]) -> Tuple[str, Dict[str, str]]:
        """
        Canonicalize a hand to a standard form
        Returns: (canonical_hand, suit_mapping)
        
        Example:
        Input: ['JC', '9C', 'AC', '10C'] 
        Output: ('7C', '8C', 'QC', 'KC', {'C': 'C'})
        
        Input: ['JH', '9H', 'AH', '10H']
        Output: ('7C', '8C', 'QC', 'KC', {'H': 'C'})
        
        Note: Game 28 ranking is J < 9 < A < 10 < K < Q < 8 < 7
        """
        # Parse cards
        cards = []
        for card_str in hand:
            if len(card_str) == 2:
                rank, suit = card_str[0], card_str[1]
            else:  # 10
                rank, suit = card_str[:2], card_str[2]
            cards.append((rank, suit))
        
        # Sort by rank (7 > 8 > Q > K > 10 > A > 9 > J)
        cards.sort(key=lambda x: self.rank_order.index(x[0]), reverse=True)
        
        # Create canonical form by mapping suits to standard order
        suit_mapping = {}
        canonical_cards = []
        
        for i, (rank, suit) in enumerate(cards):
            if suit not in suit_mapping:
                # Map to next available canonical suit
                canonical_suit = self.suit_order[len(suit_mapping)]
                suit_mapping[suit] = canonical_suit
            
            canonical_suit = suit_mapping[suit]
            canonical_cards.append(f"{rank}{canonical_suit}")
        
        canonical_hand = tuple(canonical_cards)
        return canonical_hand, suit_mapping
    
    def get_hand_features(self, hand: List[str]) -> Dict[str, Any]:
        """Extract features from a hand"""
        # Parse cards
        ranks = []
        suits = []
        total_points = 0
        
        for card_str in hand:
            if len(card_str) == 2:
                rank, suit = card_str[0], card_str[1]
            else:  # 10
                rank, suit = card_str[:2], card_str[2]
            
            ranks.append(rank)
            suits.append(suit)
            total_points += CARD_VALUES.get(rank, 0)
        
        # Count suits
        suit_counts = Counter(suits)
        
        # Find longest suit
        longest_suit = max(suit_counts, key=suit_counts.get) if suit_counts else None
        longest_suit_count = suit_counts[longest_suit] if longest_suit else 0
        
        # Calculate suit strength (points in longest suit)
        suit_strength = 0
        if longest_suit:
            for i, suit in enumerate(suits):
                if suit == longest_suit:
                    suit_strength += CARD_VALUES.get(ranks[i], 0)
        
        return {
            'total_points': total_points,
            'longest_suit': longest_suit,
            'longest_suit_count': longest_suit_count,
            'suit_strength': suit_strength,
            'suit_distribution': dict(suit_counts),
            'ranks': ranks,
            'suits': suits
        }

class PointPredictionDataset(Dataset):
    """Dataset for point prediction training"""
    
    def __init__(self, mcts_data_file: str = "mcts_bidding_analysis.json"):
        self.canonicalizer = HandCanonicalizer()
        self.data = []
        self.load_mcts_data(mcts_data_file)
    
    def load_mcts_data(self, mcts_data_file: str):
        """Load and process MCTS data for point prediction"""
        try:
            with open(mcts_data_file, 'r') as f:
                mcts_data = json.load(f)
            
            print(f"Loading {len(mcts_data['training_data'])} training examples for point prediction...")
            
            for example in mcts_data['training_data']:
                hand = example['initial_hand']  # Use full 7-card hand
                
                if len(hand) == 7:  # Ensure we have full hands
                    # Canonicalize the hand
                    canonical_hand, suit_mapping = self.canonicalizer.canonicalize_hand(hand)
                    
                    # Get hand features
                    features = self.canonicalizer.get_hand_features(hand)
                    
                    # Use the actual points directly
                    actual_points = example.get('actual_points', 0.0)
                    
                    # Create training example
                    training_example = {
                        'canonical_hand': canonical_hand,
                        'original_hand': hand,
                        'features': features,
                        'actual_points': actual_points,
                        'player_id': example.get('player_id', 0),
                        'game_id': example.get('game_id', 'unknown'),
                        'trump_suit': example.get('trump_suit'),
                        'auction_winner': example.get('auction_winner'),
                        'winning_bid': example.get('winning_bid'),
                        'bid_success': example.get('bid_success')
                    }
                    
                    self.data.append(training_example)
            
            print(f"Created {len(self.data)} training examples")
            
        except Exception as e:
            print(f"Error loading MCTS data: {e}")
            import traceback
            traceback.print_exc()
            self.data = []
    

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        # Convert hand to tensor representation
        hand_tensor = self.hand_to_tensor(example['canonical_hand'])
        
        return {
            'hand_tensor': hand_tensor,
            'features': torch.tensor([
                example['features']['total_points'] / TOTAL_POINTS,
                example['features']['longest_suit_count'] / 7.0,  # Normalize by 7 cards
                example['features']['suit_strength'] / TOTAL_POINTS
            ], dtype=torch.float32),
            'actual_points': torch.tensor(example['actual_points'], dtype=torch.float32)
        }
    
    def hand_to_tensor(self, canonical_hand: Tuple[str, ...]) -> torch.Tensor:
        """Convert canonical hand to tensor representation"""
        # Create a 7x8 tensor (7 cards, 8 possible ranks in Game 28)
        tensor = torch.zeros(7, 8)
        
        rank_to_idx = {rank: idx for idx, rank in enumerate(['J', '9', 'A', '10', 'K', 'Q', '8', '7'])}
        
        for i, card in enumerate(canonical_hand):
            if len(card) == 2:
                rank = card[0]
            else:  # 10
                rank = card[:2]
            
            if rank in rank_to_idx:
                tensor[i, rank_to_idx[rank]] = 1.0
        
        return tensor

class PointPredictionModel(nn.Module):
    """Neural network for predicting expected points from 4-card hands"""
    
    def __init__(self):
        super().__init__()
        
        # Hand representation layers
        self.hand_conv = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=3, padding=1),  # 4 cards, not 7 ranks
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Feature processing
        self.feature_fc = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        # Combined prediction
        self.prediction_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, hand_tensor, features):
        # Process hand
        hand_flat = hand_tensor.view(hand_tensor.size(0), 4, -1)  # (batch, 4, 13) - 4 cards, 13 ranks
        hand_features = self.hand_conv(hand_flat).squeeze(-1)  # (batch, 64)
        
        # Process additional features
        feature_embedding = self.feature_fc(features)  # (batch, 64)
        
        # Combine
        combined = torch.cat([hand_features, feature_embedding], dim=1)
        
        # Predict points
        predicted_points = self.prediction_head(combined)
        
        return predicted_points.squeeze(-1)

class PointPredictionTrainer:
    """Trainer for the point prediction model"""
    
    def __init__(self, mcts_data_file: str = "mcts_bidding_analysis.json"):
        self.dataset = PointPredictionDataset(mcts_data_file)
        self.model = PointPredictionModel()
        self.canonicalizer = HandCanonicalizer()
        
        # Split data
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=32, shuffle=False)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
    
    def train(self, epochs: int = 100):
        """Train the model"""
        print(f"Training point prediction model for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                
                hand_tensor = batch['hand_tensor']
                features = batch['features']
                actual_points = batch['actual_points']
                
                predicted_points = self.model(hand_tensor, features)
                loss = self.criterion(predicted_points, actual_points)
                
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for batch in self.val_loader:
                    hand_tensor = batch['hand_tensor']
                    features = batch['features']
                    actual_points = batch['actual_points']
                    
                    predicted_points = self.model(hand_tensor, features)
                    loss = self.criterion(predicted_points, actual_points)
                    
                    val_loss += loss.item()
                    predictions.extend(predicted_points.numpy())
                    actuals.extend(actual_points.numpy())
            
            # Calculate correlation
            correlation = np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss: {train_loss/len(self.train_loader):.4f}, "
                      f"Val Loss: {val_loss/len(self.val_loader):.4f}, "
                      f"Correlation: {correlation:.4f}")
        
        print("Training completed!")
    
    def predict_points(self, hand: List[str]) -> float:
        """Predict expected points for a given 7-card hand"""
        self.model.eval()
        
        # Canonicalize hand
        canonical_hand, _ = self.canonicalizer.canonicalize_hand(hand)
        features = self.canonicalizer.get_hand_features(hand)
        
        # Convert to tensor
        hand_tensor = self.dataset.hand_to_tensor(canonical_hand).unsqueeze(0)
        feature_tensor = torch.tensor([
            features['total_points'] / TOTAL_POINTS,
            features['longest_suit_count'] / 7.0,  # Normalize by 7 cards
            features['suit_strength'] / TOTAL_POINTS
        ], dtype=torch.float32).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            predicted_points = self.model(hand_tensor, feature_tensor)
        
        return predicted_points.item()
    
    def save_model(self, path: str = "models/point_prediction_model.pth"):
        """Save the trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str = "models/point_prediction_model.pth"):
        """Load a trained model"""
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}")

def main():
    """Main training function"""
    print("Training Point Prediction Model")
    print("="*50)
    
    # Create trainer
    trainer = PointPredictionTrainer()
    
    if len(trainer.dataset) == 0:
        print("No training data available. Please run analyze_mcts_data.py first.")
        return
    
    # Train the model
    trainer.train(epochs=1000)
    
    # Save the model
    trainer.save_model()
    
    # Test predictions
    print("\nTesting predictions:")
    test_hands = [
        ['JC', '9C', 'AC', '10C', 'KC', 'QC', '8C'],  # Strong clubs
        ['7H', '8H', '9H', '10H', 'JH', 'QH', 'KH'],  # Strong hearts
        ['AS', 'KS', 'QS', 'JS', '10S', '9S', '8S'],  # Very strong spades
        ['7D', '8C', '9H', '9S', '10D', 'JD', 'QD'],  # Mixed
    ]
    
    for hand in test_hands:
        predicted_points = trainer.predict_points(hand)
        print(f"Hand {hand}: Predicted {predicted_points:.2f} points")
    
    print("\nModel training completed!")

if __name__ == "__main__":
    main()
