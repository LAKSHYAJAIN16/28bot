#!/usr/bin/env python3
"""
Debug script to test observation space values
"""

import sys
import os
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl_bidding.env_adapter import Game28Env
from game28.constants import MAX_BID, MIN_BID

def test_observation_space():
    """Test the observation space to find the issue"""
    env = Game28Env(player_id=0)
    
    print("Observation space definition:")
    for key, space in env.observation_space.spaces.items():
        print(f"  {key}: {space}")
    
    print("\nTesting observation values:")
    obs, _ = env.reset()
    
    print(f"Keys in observation: {list(obs.keys())}")
    print(f"Keys in observation space: {list(env.observation_space.spaces.keys())}")
    
    # Check each key individually
    for key in sorted(env.observation_space.spaces.keys()):
        if key in obs:
            value = obs[key]
            space = env.observation_space.spaces[key]
            print(f"  {key}: {value} (type: {type(value)})")
            
            # Check if discrete space values are within bounds
            if hasattr(space, 'n'):
                if isinstance(value, (int, np.integer)):
                    if value >= space.n:
                        print(f"    ERROR: Value {value} >= space.n {space.n}")
                    elif value < 0:
                        print(f"    ERROR: Value {value} < 0")
                elif isinstance(value, np.ndarray):
                    if np.any(value >= space.n):
                        print(f"    ERROR: Some values >= space.n {space.n}")
                    if np.any(value < 0):
                        print(f"    ERROR: Some values < 0")
        else:
            print(f"  {key}: MISSING")
    
    # Check for missing keys
    missing_keys = set(env.observation_space.spaces.keys()) - set(obs.keys())
    if missing_keys:
        print(f"\nMISSING KEYS: {missing_keys}")

if __name__ == "__main__":
    test_observation_space()
