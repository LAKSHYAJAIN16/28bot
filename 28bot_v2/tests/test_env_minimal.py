#!/usr/bin/env python3
"""
Minimal test to isolate observation space issue
"""

import sys
import os
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl_bidding.env_adapter import Game28Env
from stable_baselines3 import PPO

def test_minimal():
    """Test minimal environment creation and training"""
    print("Creating environment...")
    env = Game28Env(player_id=0)
    
    print("Environment created successfully")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test reset
    print("\nTesting reset...")
    obs, info = env.reset()
    print(f"Reset successful, observation keys: {list(obs.keys())}")
    
    # Test step
    print("\nTesting step...")
    action = 0  # First action
    obs, reward, done, truncated, info = env.step(action)
    print(f"Step successful, reward: {reward}, done: {done}")
    
    # Test PPO creation
    print("\nTesting PPO creation...")
    try:
        model = PPO("MultiInputPolicy", env, verbose=1)
        print("PPO model created successfully")
        
        # Test training for a few steps
        print("\nTesting training...")
        model.learn(total_timesteps=100, progress_bar=False)
        print("Training completed successfully")
        
    except Exception as e:
        print(f"Error during PPO creation/training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_minimal()
