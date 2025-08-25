#!/usr/bin/env python3
"""
Fix the point prediction model by retraining it with proper 4-card hand data
"""

import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game28.constants import RANKS, CARD_VALUES, TOTAL_POINTS

class SimplePointPredictionModel(nn.Module):
    """Simple point prediction model for 4-card hands"""
    
    def __init__(self):
        super().__init__()
        
        # Hand representation layers (4 cards, 13 ranks)
        self.hand_conv = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=3, padding=1),
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
        # Process hand (batch, 4, 13) -> (batch, 64)
        hand_features = self.hand_conv(hand_tensor).squeeze(-1)
        
        # Process additional features
        feature_embedding = self.feature_fc(features)
        
        # Combine
        combined = torch.cat([hand_features, feature_embedding], dim=1)
        
        # Predict points
        predicted_points = self.prediction_head(combined)
        
        return predicted_points.squeeze(-1)

class PointPredictionDataset(Dataset):
    """Dataset for point prediction training with 4-card hands"""
    
    def __init__(self, data_file: str = "data/mcts_bidding_analysis.json"):
        self.data = []
        self.load_data(data_file)
    
    def load_data(self, data_file: str):
        """Load and process data for point prediction"""
        try:
            with open(data_file, 'r') as f:
                mcts_data = json.load(f)
            
            print(f"Loading {len(mcts_data['training_data'])} training examples...")
            
            for example in mcts_data['training_data']:
                if 'bidding_hand' in example and len(example['bidding_hand']) == 4:
                    bidding_hand = example['bidding_hand']
                    
                    # Calculate actual points from the game result
                    # For now, use a simple heuristic: successful bids get positive points
                    actual_points = 0.0
                    if example.get('success', False):
                        actual_points = example.get('bid', 16)  # Use bid value as points
                    else:
                        actual_points = 0.0  # Failed bids get 0 points
                    
                    # Create training example
                    training_example = {
                        'bidding_hand': bidding_hand,
                        'actual_points': actual_points,
                        'hand_strength': example.get('hand_strength', 0.0),
                        'bid': example.get('bid', 16),
                        'success': example.get('success', False)
                    }
                    
                    self.data.append(training_example)
            
            print(f"Created {len(self.data)} training examples")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            self.data = []
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        # Convert hand to tensor representation
        hand_tensor = torch.zeros(4, 13)  # 4 cards, 13 ranks
        
        for i, card in enumerate(example['bidding_hand']):
            rank = card[0] if len(card) == 2 else card[:2]
            rank_idx = RANKS.index(rank)
            hand_tensor[i, rank_idx] = 1.0
        
        # Create features
        features = torch.tensor([
            len(example['bidding_hand']) / 4.0,  # Normalized hand size
            example['hand_strength'],  # Hand strength
            1.0  # Bias term
        ], dtype=torch.float32)
        
        return {
            'hand_tensor': hand_tensor,
            'features': features,
            'actual_points': torch.tensor(example['actual_points'], dtype=torch.float32)
        }

def train_point_prediction_model():
    """Train a new point prediction model"""
    print("Training Simple Point Prediction Model")
    print("="*50)
    
    # Create dataset
    dataset = PointPredictionDataset()
    
    if len(dataset) == 0:
        print("No training data available!")
        return None
    
    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = SimplePointPredictionModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print(f"Training on {len(train_dataset)} examples, validating on {len(val_dataset)} examples")
    
    # Train
    best_val_loss = float('inf')
    for epoch in range(100):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            hand_tensor = batch['hand_tensor']
            features = batch['features']
            actual_points = batch['actual_points']
            
            predicted_points = model(hand_tensor, features)
            loss = criterion(predicted_points, actual_points)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch in val_loader:
                hand_tensor = batch['hand_tensor']
                features = batch['features']
                actual_points = batch['actual_points']
                
                predicted_points = model(hand_tensor, features)
                loss = criterion(predicted_points, actual_points)
                
                val_loss += loss.item()
                predictions.extend(predicted_points.numpy())
                actuals.extend(actual_points.numpy())
        
        # Calculate correlation
        correlation = np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Val Loss: {val_loss/len(val_loader):.4f}, "
                  f"Correlation: {correlation:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, "models/point_prediction_model_fixed.pth")
    
    print("Training completed!")
    return model

def test_fixed_model():
    """Test the fixed model"""
    print("\nTesting Fixed Point Prediction Model")
    print("="*50)
    
    # Load the fixed model
    model = SimplePointPredictionModel()
    model_path = "models/point_prediction_model_fixed.pth"
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"✓ Loaded fixed model from {model_path}")
    else:
        print(f"✗ Fixed model not found: {model_path}")
        return
    
    # Test with different hands
    test_hands = [
        ['AS', 'QH', '7H', '9H'],  # Original
        ['AH', 'KH', 'QH', 'JH'],  # Strong hearts
        ['7S', '8S', '9S', '10S'], # Weak spades
        ['AD', 'KD', 'QD', 'JD'],  # Strong diamonds
    ]
    
    for hand in test_hands:
        # Create hand tensor
        hand_tensor = torch.zeros(1, 4, 13)
        for i, card in enumerate(hand):
            rank_idx = RANKS.index(card[0] if len(card) == 2 else card[:2])
            hand_tensor[0, i, rank_idx] = 1.0
        
        # Create features
        hand_strength = sum(CARD_VALUES[card[0] if len(card) == 2 else card[:2]] for card in hand) / TOTAL_POINTS
        features = torch.tensor([
            len(hand) / 4.0,
            hand_strength,
            1.0
        ], dtype=torch.float32).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            prediction = model(hand_tensor, features)
            print(f"Hand {hand}: strength={hand_strength:.3f}, prediction={prediction.item():.6f}")

def main():
    """Main function"""
    print("Fixing Point Prediction Model")
    print("="*50)
    
    # Train new model
    model = train_point_prediction_model()
    
    if model is not None:
        # Test the model
        test_fixed_model()
        
        print("\n" + "="*50)
        print("FIX COMPLETED!")
        print("The point prediction model has been retrained and should now produce")
        print("reasonable positive predictions for 4-card hands.")
        print("\nTo use the fixed model, update the model path in main_game_simulation.py:")
        print("Change: models/point_prediction_model.pth")
        print("To: models/point_prediction_model_fixed.pth")

if __name__ == "__main__":
    main()
