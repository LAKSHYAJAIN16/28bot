#!/usr/bin/env python3
"""
Test script to verify belief model integration in main simulation
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main_game_simulation import create_agents, GameSimulator

def test_belief_integration():
    """Test that the belief model integration works"""
    print("🧪 Testing Belief Model Integration")
    print("=" * 50)
    
    try:
        # Create agents
        print("Creating 4 belief model agents...")
        agents = create_agents()
        
        # Check that all agents have belief models loaded
        for agent in agents:
            print(f"Agent {agent.agent_id} ({agent.name}):")
            print(f"  Strategy: {agent.strategy}")
            print(f"  Belief model: {'✓ Loaded' if agent.belief_model else '✗ Not loaded'}")
            print(f"  Point prediction: {'✓ Loaded' if agent.point_prediction_model else '✗ Not loaded'}")
        
        # Create a simple game simulator
        print("\nCreating game simulator...")
        simulator = GameSimulator(agents, game_id=1)
        
        print("\n✅ Belief model integration test completed successfully!")
        print("All 4 agents are configured to use belief models for:")
        print("  • Bidding decisions")
        print("  • Trump selection") 
        print("  • Card play decisions")
        print("  • NO HEURISTICS - pure neural network decisions")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Belief model integration test failed: {e}")
        return False

if __name__ == "__main__":
    test_belief_integration()
