#!/usr/bin/env python3
"""
Comprehensive test script for the belief model
"""

import torch
import sys
import os
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from belief_model.simple_advanced_belief_net import SimpleAdvancedBeliefNetwork
from belief_model.advanced_parser import extract_all_game_states
from game28.constants import *

def test_model_comprehensive():
    """Comprehensive test of the belief model"""
    print("🧪 Comprehensive Belief Model Test")
    print("=" * 50)
    
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
    game_states = extract_all_game_states(["../logs/game28/mcts_games"], max_games=10)
    
    if not game_states:
        print("❌ No game states found for testing")
        return
    
    print(f"✅ Loaded {len(game_states)} game states for testing")
    
    # Test predictions
    print("\n🔮 Making predictions...")
    
    trump_correct = 0
    trump_total = 0
    
    for i, game_state in enumerate(game_states[:5]):  # Test first 5 states
        print(f"\n🎮 Game State {i+1}:")
        print(f"  📋 Phase: {game_state.phase}")
        print(f"  🃏 Tricks played: {game_state.tricks_played}")
        print(f"  🎯 Trump suit: {game_state.trump_suit}")
        print(f"  📈 Game progress: {game_state.game_progress:.2f}")
        
        # Test for each player
        for player_id in range(4):
            if player_id in game_state.hands:
                print(f"\n  👤 Player {player_id} (hand size: {len(game_state.hands[player_id])}):")
                
                try:
                    predictions = model.predict_beliefs(game_state, player_id)
                    
                    # Trump prediction
                    trump_probs = predictions.trump_suit.cpu().numpy()
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
                    
                    # Void predictions for opponents
                    for opp_id in range(4):
                        if opp_id != player_id and opp_id in predictions.void_suits:
                            void_probs = predictions.void_suits[opp_id].cpu().numpy()
                            void_suits = []
                            for j in range(4):
                                if void_probs[j] > 0.5:
                                    void_suits.append(SUITS[j])
                            
                            if len(void_suits) > 0:
                                print(f"    🚫 Opponent {opp_id} predicted void suits: {void_suits}")
                            else:
                                print(f"    ✅ Opponent {opp_id} predicted to have all suits")
                    
                    # Uncertainty
                    uncertainty = predictions.uncertainty.cpu().numpy()[0]
                    print(f"    🤔 Prediction uncertainty: {uncertainty:.3f}")
                    
                    # Show some hand predictions for opponents
                    for opp_id in range(4):
                        if opp_id != player_id and opp_id in predictions.opponent_hands:
                            hand_probs = predictions.opponent_hands[opp_id].cpu().numpy()
                            # Show top 5 most likely cards
                            top_indices = np.argsort(hand_probs)[-5:][::-1]
                            print(f"    🃏 Opponent {opp_id} most likely cards:")
                            for idx in top_indices:
                                suit_idx = idx // 8
                                rank_idx = idx % 8
                                card = f"{RANKS[rank_idx]}{SUITS[suit_idx]}"
                                prob = hand_probs[idx]
                                print(f"      {card}: {prob:.3f}")
                    
                except Exception as e:
                    print(f"    ❌ Error predicting for player {player_id}: {e}")
    
    # Print summary statistics
    print(f"\n📊 Summary Statistics:")
    if trump_total > 0:
        trump_accuracy = trump_correct / trump_total
        print(f"  🎯 Trump prediction accuracy: {trump_accuracy:.3f} ({trump_correct}/{trump_total})")
    
    print(f"  🎮 Game states tested: {min(5, len(game_states))}")
    print(f"  👥 Players tested per state: 4")
    
    print("\n✅ Comprehensive testing completed!")

def compare_models():
    """Compare different trained models"""
    print("\n🔄 Model Comparison")
    print("=" * 30)
    
    models = [
        ("Advanced Best", "models/belief_model/advanced_belief_model_best.pt"),
        ("Epoch 10", "models/belief_model/belief_model_epoch_10.pt"),
        ("Final", "models/belief_model/belief_model_final.pt")
    ]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for model_name, model_path in models:
        if os.path.exists(model_path):
            print(f"\n📁 Testing {model_name} model...")
            
            try:
                model = SimpleAdvancedBeliefNetwork().to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                
                # Get model size
                param_count = sum(p.numel() for p in model.parameters())
                print(f"  📏 Parameters: {param_count:,}")
                print(f"  💾 Size: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")
                
                # Quick test on one game state
                game_states = extract_all_game_states(["../logs/game28/mcts_games"], max_games=1)
                if game_states:
                    game_state = game_states[0]
                    predictions = model.predict_beliefs(game_state, 0)
                    trump_probs = predictions.trump_suit.cpu().numpy()
                    trump_suit = SUITS[trump_probs.argmax()]
                    trump_confidence = trump_probs.max()
                    print(f"  🎯 Sample trump prediction: {trump_suit} (confidence: {trump_confidence:.3f})")
                
            except Exception as e:
                print(f"  ❌ Error loading {model_name}: {e}")
        else:
            print(f"  ❌ {model_name} model not found")

if __name__ == "__main__":
    test_model_comprehensive()
    compare_models()
