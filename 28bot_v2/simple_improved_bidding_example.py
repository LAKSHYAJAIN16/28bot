#!/usr/bin/env python3
"""
Simple example of using the improved bidding model
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from improved_bidding_trainer import ImprovedBiddingTrainer
from game28.constants import BID_RANGE

def simple_bidding_example():
    """Simple example of using the improved bidding model"""
    
    print("Loading improved bidding model...")
    
    # Load the trained model
    try:
        model = PPO.load("models/improved_bidding_model.zip")
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Make sure you've run: python improved_bidding_trainer.py")
        return
    
    # Create the improved environment
    trainer = ImprovedBiddingTrainer()
    env = trainer.create_improved_environment()
    
    print("\nTesting the model with a sample game...")
    
    # Play a few games to see the model in action
    for game_num in range(3):
        print(f"\n--- Game {game_num + 1} ---")
        
        obs, _ = env.reset()
        done = False
        step_count = 0
        
        while not done and step_count < 10:  # Limit steps to avoid infinite loops
            # Get model's decision
            action, _ = model.predict(obs, deterministic=True)
            
            # Convert action to bid
            if action == len(BID_RANGE):  # Pass
                bid = "PASS"
            else:
                bid = BID_RANGE[action]
            
            print(f"Step {step_count + 1}: Model chose action {action} -> Bid: {bid}")
            
            # Take the action
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1
            
            if done:
                print(f"Game ended. Final reward: {reward:.3f}")
                break
    
    print("\n" + "="*50)
    print("HOW TO USE IN YOUR CODE")
    print("="*50)
    
    print("\n1. Load the model:")
    print("   model = PPO.load('models/improved_bidding_model')")
    
    print("\n2. Create environment:")
    print("   trainer = ImprovedBiddingTrainer()")
    print("   env = trainer.create_improved_environment()")
    
    print("\n3. Get predictions:")
    print("   obs, _ = env.reset()")
    print("   action, _ = model.predict(obs, deterministic=True)")
    print("   obs, reward, done, truncated, info = env.step(action)")
    
    print("\n4. Convert action to bid:")
    print("   if action == len(BID_RANGE):")
    print("       bid = -1  # Pass")
    print("   else:")
    print("       bid = BID_RANGE[action]")
    
    print("\nThe improved model features:")
    print("• MCTS-based opponent strategies")
    print("• Enhanced reward functions")
    print("• Better training data from 279 MCTS games")
    print("• Improved network architecture")

if __name__ == "__main__":
    simple_bidding_example()
