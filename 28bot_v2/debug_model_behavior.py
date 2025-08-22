#!/usr/bin/env python3
"""
Debug script to understand what the bidding model is learning
"""

import sys
import os
import numpy as np
from collections import defaultdict

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from rl_bidding.env_adapter import Game28Env
from game28.game_state import Card
from game28.constants import BID_RANGE, CARD_VALUES, TOTAL_POINTS

def analyze_model_behavior():
    """Analyze what the model is actually learning"""
    
    # Load the model
    try:
        model = PPO.load("models/bidding_policy/best_model/best_model.zip")
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Create environment
    env = Game28Env(player_id=0)
    
    print("\n" + "="*60)
    print("ANALYZING MODEL BEHAVIOR")
    print("="*60)
    
    # Test different scenarios
    scenarios = [
        ("Strong hand", [Card('H', 'A'), Card('H', 'K'), Card('D', 'J'), Card('C', '9')]),
        ("Weak hand", [Card('S', '7'), Card('D', '8'), Card('C', '7'), Card('H', '8')]),
        ("Medium hand", [Card('H', '10'), Card('D', 'Q'), Card('C', 'K'), Card('S', '9')]),
        ("Mixed hand", [Card('H', 'A'), Card('S', '7'), Card('D', 'J'), Card('C', '8')])
    ]
    
    for scenario_name, hand in scenarios:
        print(f"\n--- {scenario_name} ---")
        print(f"Hand: {[str(card) for card in hand]}")
        
        # Calculate hand strength
        hand_strength = sum(CARD_VALUES[card.rank] for card in hand) / TOTAL_POINTS
        print(f"Hand strength: {hand_strength:.3f}")
        
        # Test different bidding situations
        for current_bid in [16, 18, 20, 22]:
            # Set up environment
            obs, _ = env.reset()
            env.game_state.hands[0] = hand
            env.game_state.current_bid = current_bid
            env.game_state.current_player = 0
            
            # Get model's decision
            obs = env._get_observation()
            action, _ = model.predict(obs, deterministic=True)
            
            # Convert action to bid
            if action == len(BID_RANGE):
                model_bid = "PASS"
            else:
                model_bid = BID_RANGE[action]
            
            print(f"  Current bid: {current_bid} → Model: {model_bid}")
    
    print("\n" + "="*60)
    print("ANALYSIS OF TRAINING DATA")
    print("="*60)
    
    # Analyze what the model is training against
    print("\n1. OPPONENT BEHAVIOR:")
    print("   - Other players use simple heuristics:")
    print("     * Hand strength > 0.6 → Bid aggressively")
    print("     * Hand strength > 0.4 → Bid moderately") 
    print("     * Hand strength ≤ 0.4 → Pass")
    print("   - No strategic thinking or game theory")
    
    print("\n2. REWARD STRUCTURE:")
    print("   - Only get reward at game end (sparse rewards)")
    print("   - Binary outcome: +1 for win, -1 for loss, 0 for tie")
    print("   - No intermediate feedback for good bidding decisions")
    print("   - No consideration of bid efficiency")
    
    print("\n3. TRAINING DATA QUALITY:")
    print("   - Self-play against simple heuristics")
    print("   - No real human gameplay data")
    print("   - No expert demonstrations")
    print("   - Limited strategic scenarios")
    
    print("\n4. WHAT THE MODEL IS LEARNING:")
    print("   - To play against predictable, simple opponents")
    print("   - Basic hand strength correlation with bidding")
    print("   - No complex strategic patterns")
    print("   - No adaptation to different opponent types")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR IMPROVEMENT")
    print("="*60)
    
    print("\n1. COLLECT REAL DATA:")
    print("   - Record games from skilled human players")
    print("   - Use expert demonstrations for imitation learning")
    print("   - Analyze successful bidding patterns")
    
    print("\n2. IMPROVE OPPONENT MODELS:")
    print("   - Create more sophisticated opponent strategies")
    print("   - Implement different playing styles")
    print("   - Add game theory considerations")
    
    print("\n3. BETTER REWARD SHAPING:")
    print("   - Add intermediate rewards for good decisions")
    print("   - Consider bid efficiency and risk")
    print("   - Reward strategic thinking")
    
    print("\n4. INCREASE TRAINING DIVERSITY:")
    print("   - More training episodes")
    print("   - Curriculum learning")
    print("   - Multi-agent training")
    
    print("\n5. ADD STRATEGIC ELEMENTS:")
    print("   - Position-based bidding")
    print("   - Partner signaling")
    print("   - Risk assessment")
    print("   - Game theory principles")

def test_model_consistency():
    """Test if the model is consistent in its decisions"""
    
    print("\n" + "="*60)
    print("TESTING MODEL CONSISTENCY")
    print("="*60)
    
    try:
        model = PPO.load("models/bidding_policy/best_model/best_model.zip")
    except:
        print("Could not load model for consistency test")
        return
    
    env = Game28Env(player_id=0)
    
    # Test same situation multiple times
    test_hand = [Card('H', 'A'), Card('H', 'K'), Card('D', 'J'), Card('C', '9')]
    current_bid = 16
    
    print(f"\nTesting consistency with hand: {[str(card) for card in test_hand]}")
    print(f"Current bid: {current_bid}")
    
    decisions = []
    for i in range(10):
        obs, _ = env.reset()
        env.game_state.hands[0] = test_hand
        env.game_state.current_bid = current_bid
        env.game_state.current_player = 0
        
        obs = env._get_observation()
        action, _ = model.predict(obs, deterministic=True)
        
        if action == len(BID_RANGE):
            decision = "PASS"
        else:
            decision = BID_RANGE[action]
        
        decisions.append(decision)
        print(f"  Trial {i+1}: {decision}")
    
    # Check consistency
    unique_decisions = set(decisions)
    print(f"\nUnique decisions: {unique_decisions}")
    print(f"Consistency: {len(unique_decisions) == 1}")

if __name__ == "__main__":
    analyze_model_behavior()
    test_model_consistency()
