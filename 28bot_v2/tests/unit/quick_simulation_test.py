#!/usr/bin/env python3
"""
Quick test to run a single game simulation with belief models
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main_game_simulation import create_agents, GameSimulator

def quick_simulation_test():
    """Run a quick single game simulation"""
    print("🎮 Quick Belief Model Simulation Test")
    print("=" * 60)
    
    try:
        # Create agents
        print("Creating 4 belief model agents...")
        agents = create_agents()
        
        # Create simulator
        print("Creating game simulator...")
        simulator = GameSimulator(agents, game_id=1)
        
        # Run simulation
        print("Running simulation...")
        results = simulator.simulate_game()
        
        print("\n✅ Simulation completed successfully!")
        print(f"Winner: {results['winner']}")
        print(f"Team A Score: {results['team_a_score']}")
        print(f"Team B Score: {results['team_b_score']}")
        if results['bidder'] is not None:
            print(f"Bidder: Player {results['bidder']}")
            print(f"Winning Bid: {results['winning_bid']}")
            print(f"Trump Suit: {results['trump_suit']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simulation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    quick_simulation_test()
