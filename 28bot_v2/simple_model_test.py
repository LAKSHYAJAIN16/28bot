#!/usr/bin/env python3
"""
Simple test script for the belief model - focusing on working features
"""

import torch
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from belief_model.simple_advanced_belief_net import SimpleAdvancedBeliefNetwork
from belief_model.advanced_parser import extract_all_game_states
from game28.constants import *

def simple_model_test():
    """Simple test focusing on working features"""
    print("🧪 Simple Belief Model Test")
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
        print(f"📱 Using device: {device}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load test data
    print("\n📊 Loading test data...")
    game_states = extract_all_game_states(["../logs/game28/mcts_games"], max_games=5)
    
    if not game_states:
        print("❌ No game states found for testing")
        return
    
    print(f"✅ Loaded {len(game_states)} game states for testing")
    
    # Test predictions
    print("\n🔮 Making predictions...")
    
    trump_correct = 0
    trump_total = 0
    
    for i, game_state in enumerate(game_states[:3]):  # Test first 3 states
        print(f"\n🎮 Game State {i+1}:")
        print(f"  📋 Phase: {game_state.phase}")
        print(f"  🃏 Tricks played: {game_state.tricks_played}")
        print(f"  🎯 Trump suit: {game_state.trump_suit}")
        print(f"  📈 Game progress: {game_state.game_progress:.2f}")
        
        # Test for player 0 only
        if 0 in game_state.hands:
            print(f"\n  👤 Player 0 (hand size: {len(game_state.hands[0])}):")
            
            try:
                predictions = model.predict_beliefs(game_state, 0)
                
                # Trump prediction
                trump_probs = predictions.trump_suit.cpu().numpy().flatten()  # Flatten to 1D
                trump_suit = SUITS[trump_probs.argmax()]
                trump_confidence = trump_probs.max()
                
                # Check if trump prediction is correct
                if game_state.trump_suit:
                    trump_total += 1
                    if trump_suit == game_state.trump_suit:
                        trump_correct += 1
                        print(f"    ✅ Trump prediction: {trump_suit} (confidence: {trump_confidence:.3f}) - CORRECT!")
                    else:
                        print(f"    ❌ Trump prediction: {trump_suit} (confidence: {trump_confidence:.3f}) - WRONG! Actual: {game_state.trump_suit}")
                else:
                    print(f"    🎯 Trump prediction: {trump_suit} (confidence: {trump_confidence:.3f})")
                
                # Uncertainty
                uncertainty = predictions.uncertainty.cpu().numpy().item()
                print(f"    🤔 Prediction uncertainty: {uncertainty:.3f}")
                
                # Show trump probabilities for all suits
                print(f"    🎯 Trump probabilities:")
                for suit_idx, suit in enumerate(SUITS):
                    prob = trump_probs[suit_idx]
                    print(f"      {suit}: {prob:.3f}")
                
            except Exception as e:
                print(f"    ❌ Error predicting: {e}")
    
    # Print summary statistics
    print(f"\n📊 Summary Statistics:")
    if trump_total > 0:
        trump_accuracy = trump_correct / trump_total
        print(f"  🎯 Trump prediction accuracy: {trump_accuracy:.3f} ({trump_correct}/{trump_total})")
    
    print(f"  🎮 Game states tested: {min(3, len(game_states))}")
    
    print("\n✅ Simple testing completed!")

if __name__ == "__main__":
    simple_model_test()
