#!/usr/bin/env python3
"""
Debug script to see what the belief model is actually returning
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game28.game_state import Game28State
from belief_model.simple_advanced_belief_net import SimpleAdvancedBeliefNetwork
import torch

def debug_belief_output():
    """Debug what the belief model is returning"""
    print("🔍 Debugging Belief Model Output")
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
            print("✓ Model loaded successfully")
        else:
            print("✗ Model not found, using untrained model")
        
        # Make a prediction
        print("Making prediction...")
        with torch.no_grad():
            predictions = model.predict_beliefs(game_state, 0)
        
        print("Prediction structure:")
        print(f"  Type: {type(predictions)}")
        print(f"  Trump suit shape: {predictions.trump_suit.shape}")
        print(f"  Uncertainty shape: {predictions.uncertainty.shape}")
        print(f"  Opponent hands keys: {list(predictions.opponent_hands.keys())}")
        print(f"  Void suits keys: {list(predictions.void_suits.keys())}")
        
        # Check void suits specifically
        print("\nVoid suits details:")
        for opp_id, void_tensor in predictions.void_suits.items():
            print(f"  Opponent {opp_id}:")
            print(f"    Type: {type(void_tensor)}")
            print(f"    Shape: {void_tensor.shape}")
            print(f"    Values: {void_tensor}")
            
            # Convert to numpy
            void_numpy = void_tensor.cpu().numpy()
            print(f"    Numpy type: {type(void_numpy)}")
            print(f"    Numpy shape: {void_numpy.shape}")
            print(f"    Numpy values: {void_numpy}")
            
            # Try to flatten
            void_flat = void_numpy.flatten()
            print(f"    Flattened shape: {void_flat.shape}")
            print(f"    Flattened values: {void_flat}")
            
            # Check truth value
            print(f"    Truth value test: {bool(void_flat.any())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_belief_output()
