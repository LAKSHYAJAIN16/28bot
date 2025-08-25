#!/usr/bin/env python3
"""
Quick test to verify belief model can make predictions with Game28State
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game28.game_state import Game28State
from belief_model.simple_advanced_belief_net import SimpleAdvancedBeliefNetwork
import torch

def test_belief_prediction():
    """Test that the belief model can make predictions"""
    print("🧪 Testing Belief Model Prediction")
    print("=" * 50)
    
    try:
        # Create a game state
        print("Creating Game28State...")
        game_state = Game28State()
        
        # Load the belief model
        print("Loading belief model...")
        model = SimpleAdvancedBeliefNetwork()
        model_path = "models/belief_model/advanced_belief_model_best.pt"
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            print("✓ Belief model loaded successfully")
        else:
            print("✗ Belief model not found, using untrained model")
        
        # Test prediction
        print("Making prediction...")
        with torch.no_grad():
            predictions = model.predict_beliefs(game_state, player_id=0)
        
        print("✓ Prediction successful!")
        print(f"  Trump probabilities: {predictions.trump_suit.cpu().numpy().flatten()}")
        print(f"  Uncertainty: {predictions.uncertainty.cpu().numpy().item():.3f}")
        print(f"  Opponent hands: {len(predictions.opponent_hands)} players")
        print(f"  Void suits: {len(predictions.void_suits)} players")
        
        return True
        
    except Exception as e:
        print(f"❌ Belief model prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_belief_prediction()
