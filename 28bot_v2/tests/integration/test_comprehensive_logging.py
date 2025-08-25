#!/usr/bin/env python3
"""
Test script to run the updated simulation with comprehensive logging
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main_game_simulation import create_agents, GameSimulator

def test_comprehensive_logging():
    """Test the comprehensive logging functionality"""
    print("🧪 Testing Comprehensive Logging")
    print("=" * 50)
    
    try:
        # Create agents
        print("Creating agents...")
        agents = create_agents()
        
        # Create simulator
        print("Creating simulator...")
        simulator = GameSimulator(agents, game_id=1)
        
        # Run a single game
        print("Running game...")
        results = simulator.simulate_game()
        
        print("\n✅ Test completed successfully!")
        print(f"Condensed log: {results.get('condensed_log_file', 'N/A')}")
        print(f"Comprehensive log: {results.get('comprehensive_log_file', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_comprehensive_logging()
