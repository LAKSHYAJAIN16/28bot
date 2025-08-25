#!/usr/bin/env python3
"""
Test script to verify point prediction model functionality
"""

import sys
import os
import torch
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.point_prediction_model import PointPredictionModel
from game28.constants import RANKS, CARD_VALUES, TOTAL_POINTS

def test_point_prediction_model():
    """Test the point prediction model with different inputs"""
    
    print("Testing Point Prediction Model")
    print("="*50)
    
    # Create model
    model = PointPredictionModel()
    
    # Test 1: 4-card hand (what main simulation is doing)
    print("\nTest 1: 4-card hand (bidding hand)")
    hand_tensor_4 = torch.zeros(1, 4, 13)  # (batch, 4 cards, 13 ranks)
    
    # Example: AS, QH, 7H, 9H
    test_cards_4 = ['AS', 'QH', '7H', '9H']
    for i, card in enumerate(test_cards_4):
        rank = card[0] if len(card) == 2 else card[:2]
        rank_idx = RANKS.index(rank)
        hand_tensor_4[0, i, rank_idx] = 1.0
    
    features_4 = torch.tensor([
        4.0 / 4.0,  # Normalized hand size
        sum(CARD_VALUES[card[0] if len(card) == 2 else card[:2]] for card in test_cards_4) / TOTAL_POINTS,  # Hand strength
        1.0  # Bias term
    ], dtype=torch.float32).unsqueeze(0)
    
    print(f"4-card hand: {test_cards_4}")
    print(f"Hand tensor shape: {hand_tensor_4.shape}")
    print(f"Features shape: {features_4.shape}")
    
    with torch.no_grad():
        prediction_4 = model(hand_tensor_4, features_4)
        print(f"Prediction: {prediction_4.item():.4f}")
    
    # Test 2: 7-card hand (what the model was designed for)
    print("\nTest 2: 7-card hand (full hand)")
    hand_tensor_7 = torch.zeros(1, 7, 8)  # (batch, 7 cards, 8 ranks for Game 28)
    
    # Example: AS, QH, 7H, 9H, JC, 8S, 10D
    test_cards_7 = ['AS', 'QH', '7H', '9H', 'JC', '8S', '10D']
    rank_to_idx = {rank: idx for idx, rank in enumerate(['J', '9', 'A', '10', 'K', 'Q', '8', '7'])}
    
    for i, card in enumerate(test_cards_7):
        rank = card[0] if len(card) == 2 else card[:2]
        if rank in rank_to_idx:
            hand_tensor_7[0, i, rank_to_idx[rank]] = 1.0
    
    features_7 = torch.tensor([
        sum(CARD_VALUES[card[0] if len(card) == 2 else card[:2]] for card in test_cards_7) / TOTAL_POINTS,  # Total points
        2.0 / 7.0,  # Longest suit count (assuming 2 cards in one suit)
        sum(CARD_VALUES[card[0] if len(card) == 2 else card[:2]] for card in test_cards_7[:2]) / TOTAL_POINTS  # Suit strength
    ], dtype=torch.float32).unsqueeze(0)
    
    print(f"7-card hand: {test_cards_7}")
    print(f"Hand tensor shape: {hand_tensor_7.shape}")
    print(f"Features shape: {features_7.shape}")
    
    with torch.no_grad():
        prediction_7 = model(hand_tensor_7, features_7)
        print(f"Prediction: {prediction_7.item():.4f}")
    
    # Test 3: Check if model file exists
    print("\nTest 3: Check model file")
    model_path = "models/point_prediction_model.pth"
    if os.path.exists(model_path):
        print(f"Model file exists: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            print("Model file can be loaded")
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Model file does not exist: {model_path}")
    
    print("\n" + "="*50)
    print("CONCLUSION:")
    print("The point prediction model expects 7-card hands with 8 ranks (Game 28 ranking)")
    print("But main simulation is passing 4-card hands with 13 ranks (standard ranking)")
    print("This mismatch is causing the negative predictions!")

if __name__ == "__main__":
    test_point_prediction_model()
