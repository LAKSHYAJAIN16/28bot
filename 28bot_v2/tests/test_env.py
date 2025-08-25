#!/usr/bin/env python3
"""
Test script to understand environment behavior
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl_bidding.env_adapter import Game28Env
import random

def test_environment():
    """Test the environment to understand episode lengths"""
    env = Game28Env(player_id=0)
    
    print("Testing environment behavior...")
    
    total_steps = 0
    num_episodes = 10
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        steps = 0
        done = False
        
        print(f"\nEpisode {episode + 1}:")
        
        while not done:
            # Random action
            action = random.randint(0, env.action_space.n - 1)
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            
            print(f"  Step {steps}: action={action}, done={done}, bidding_continues={info.get('bidding_continues', 'N/A')}")
            
            if steps > 20:  # Prevent infinite loops
                print("  Stopping episode due to too many steps")
                break
        
        total_steps += steps
        print(f"  Episode ended after {steps} steps")
    
    avg_steps = total_steps / num_episodes
    print(f"\nAverage steps per episode: {avg_steps}")
    print(f"Expected timesteps for 1000 episodes: {1000 * avg_steps}")

if __name__ == "__main__":
    test_environment()
