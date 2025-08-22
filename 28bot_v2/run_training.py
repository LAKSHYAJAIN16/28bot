#!/usr/bin/env python3
"""
Script to run the RL bidding training from the correct directory
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the training
from rl_bidding.train_policy import train_bidding_policy, evaluate_policy

if __name__ == "__main__":
    print("Starting Game 28 RL Bidding Training...")
    
    # Train the bidding policy
    model = train_bidding_policy(
        num_episodes=50000,
        learning_rate=3e-4,
        batch_size=64
    )
    
    # Evaluate the trained model
    results = evaluate_policy("models/bidding_policy/final_model", num_games=100)
    
    print("Training completed!")
