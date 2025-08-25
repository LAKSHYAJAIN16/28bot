#!/usr/bin/env python3
"""
Debug script to check the shape of opponent hand predictions
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game28.game_state import Game28State
from belief_model.simple_advanced_belief_net import SimpleAdvancedBeliefNetwork
import torch

def debug_hand_probs():
    """Debug the shape of opponent hand predictions"""
    print("🔍 Debugging Opponent Hand Predictions")
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
        
        print("Opponent hand predictions details:")
        for opp_id, hand_tensor in predictions.opponent_hands.items():
            print(f"  Opponent {opp_id}:")
            print(f"    Type: {type(hand_tensor)}")
            print(f"    Shape: {hand_tensor.shape}")
            print(f"    Values (first 10): {hand_tensor.flatten()[:10]}")
            
            # Convert to numpy
            hand_numpy = hand_tensor.cpu().numpy()
            print(f"    Numpy type: {type(hand_numpy)}")
            print(f"    Numpy shape: {hand_numpy.shape}")
            print(f"    Numpy values (first 10): {hand_numpy.flatten()[:10]}")
            
            # Try to access individual elements
            print(f"    First element: {hand_numpy.flatten()[0]}")
            print(f"    First element type: {type(hand_numpy.flatten()[0])}")
            
            # Test the calculation
            expected_points = 0.0
            for i, prob in enumerate(hand_numpy.flatten()):
                if i < 32:  # Only first 32 elements
                    suit_idx = i // 8
                    rank_idx = i % 8
                    if rank_idx < 8:  # Valid rank
                        try:
                            expected_points += float(prob) * 1.0  # Simplified calculation
                        except Exception as e:
                            print(f"    Error at index {i}: {e}")
                            break
            print(f"    Expected points calculation: {expected_points}")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_hand_probs()
