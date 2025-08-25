#!/usr/bin/env python3
"""
Quick test script for the current belief model
"""

import torch
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from belief_model.simple_advanced_belief_net import SimpleAdvancedBeliefNetwork
from belief_model.advanced_parser import extract_all_game_states
from game28.constants import *

def test_current_model():
    """Test the current belief model"""
    print("🧪 Testing Current Belief Model")
    print("=" * 40)
    
    # Check if model exists
    model_path = "models/belief_model/advanced_belief_model_best.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleAdvancedBeliefNetwork().to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"✅ Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load some test data
    print("\n📊 Loading test data...")
    game_states = extract_all_game_states(["../logs/game28/mcts_games"], max_games=5)
    
    if not game_states:
        print("❌ No game states found for testing")
        return
    
    print(f"✅ Loaded {len(game_states)} game states for testing")
    
    # Test predictions
    print("\n🔮 Making predictions...")
    for i, game_state in enumerate(game_states[:3]):  # Test first 3 states
        print(f"\nGame State {i+1}:")
        print(f"  Phase: {game_state.phase}")
        print(f"  Tricks played: {game_state.tricks_played}")
        print(f"  Trump suit: {game_state.trump_suit}")
        
        # Test for player 0
        if 0 in game_state.hands:
            try:
                predictions = model.predict_beliefs(game_state, 0)
                
                # Trump prediction
                trump_probs = predictions.trump_suit.cpu().numpy()
                trump_suit = SUITS[trump_probs.argmax()]
                trump_confidence = trump_probs.max()
                print(f"  Trump prediction: {trump_suit} (confidence: {trump_confidence:.3f})")
                
                # Void predictions for opponents
                for opp_id in [1, 3]:  # Opponents of player 0
                    if opp_id in predictions.void_suits:
                        void_probs = predictions.void_suits[opp_id].cpu().numpy()
                        void_suits = [SUITS[j] for j in range(4) if void_probs[j] > 0.5]
                        print(f"  Opponent {opp_id} void suits: {void_suits}")
                
                # Uncertainty
                uncertainty = predictions.uncertainty.cpu().numpy()[0]
                print(f"  Prediction uncertainty: {uncertainty:.3f}")
                
            except Exception as e:
                print(f"  ❌ Error predicting: {e}")
    
    print("\n✅ Testing completed!")

if __name__ == "__main__":
    test_current_model()
