#!/usr/bin/env python3
"""
Test script to verify all major improvements work together:
1. Trick winner leads next trick
2. Hybrid agent (belief model + ISMCTS)
3. File restructuring
4. Timestamp-based logging
"""

import sys
import os
import time
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_trick_order():
    """Test that trick winner leads the next trick"""
    print("Testing trick order functionality...")
    
    from game28.game_state import Game28State, Card
    from game28.constants import GamePhase
    
    # Create a game state
    game_state = Game28State()
    
    # Deal some cards to players
    game_state.hands[0] = [Card('H', 'A'), Card('D', 'K'), Card('C', 'Q')]
    game_state.hands[1] = [Card('H', 'K'), Card('S', 'A'), Card('D', 'Q')]
    game_state.hands[2] = [Card('H', 'Q'), Card('S', 'K'), Card('C', 'A')]
    game_state.hands[3] = [Card('H', 'J'), Card('D', 'A'), Card('S', 'Q')]
    
    # Set up game state for card play
    game_state.phase = GamePhase.CONCEALED
    game_state.current_player = 0  # Player 0 starts
    game_state.trump_suit = 'H'
    game_state.trump_revealed = False
    
    print(f"Initial current player: {game_state.current_player}")
    
    # Simulate first trick
    print("\n=== First Trick ===")
    print(f"Player {game_state.current_player} leads")
    
    # Play 4 cards
    for i in range(4):
        current_player = game_state.current_player
        card = game_state.hands[current_player][0]  # Take first card
        game_state.play_card(current_player, card)
        print(f"Player {current_player} plays {card}")
        print(f"Current player after play: {game_state.current_player}")
    
    # Check trick winner
    if game_state.tricks:
        last_trick = game_state.tricks[-1]
        winner = last_trick.winner
        print(f"\nTrick winner: Player {winner}")
        print(f"Current player after trick completion: {game_state.current_player}")
        
        # Verify that the winner is now the current player
        if game_state.current_player == winner:
            print("✅ SUCCESS: Trick winner is now the current player!")
            return True
        else:
            print("❌ FAILURE: Trick winner is NOT the current player!")
            print(f"Expected: {winner}, Got: {game_state.current_player}")
            return False
    else:
        print("❌ FAILURE: No tricks were created!")
        return False

def test_file_structure():
    """Test that files have been properly restructured"""
    print("\nTesting file structure...")
    
    expected_dirs = [
        "agents",
        "tests/debug", 
        "tests/integration",
        "tests/unit",
        "utils",
        "docs/technical"
    ]
    
    expected_files = [
        "agents/hybrid_agent.py",
        "tests/integration/test_trick_order.py",
        "utils/use_belief_model.py",
        "docs/technical/BELIEF_MODEL_SCORING_EXPLAINED.md"
    ]
    
    all_good = True
    
    # Check directories
    for dir_path in expected_dirs:
        if os.path.exists(dir_path):
            print(f"✅ Directory exists: {dir_path}")
        else:
            print(f"❌ Directory missing: {dir_path}")
            all_good = False
    
    # Check files
    for file_path in expected_files:
        if os.path.exists(file_path):
            print(f"✅ File exists: {file_path}")
        else:
            print(f"❌ File missing: {file_path}")
            all_good = False
    
    return all_good

def test_logging_format():
    """Test that logging uses timestamp format"""
    print("\nTesting logging format...")
    
    from main_game_simulation import GameLogger
    
    # Create a test logger
    test_logger = GameLogger(game_id=999, log_dir="test_logs", log_type="test")
    
    # Check filename format
    expected_format = f"{test_logger.timestamp}_game_999_test.log"
    if test_logger.log_filename == expected_format:
        print(f"✅ Log filename format correct: {test_logger.log_filename}")
        
        # Clean up test file
        if os.path.exists(test_logger.log_path):
            os.remove(test_logger.log_path)
        if os.path.exists("test_logs"):
            os.rmdir("test_logs")
        
        return True
    else:
        print(f"❌ Log filename format incorrect:")
        print(f"  Expected: {expected_format}")
        print(f"  Got: {test_logger.log_filename}")
        return False

def test_hybrid_agent_import():
    """Test that hybrid agent can be imported"""
    print("\nTesting hybrid agent import...")
    
    try:
        from agents.hybrid_agent import HybridAgent, HybridDecision
        print("✅ Hybrid agent imports successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import hybrid agent: {e}")
        return False

def test_main_simulation_imports():
    """Test that main simulation can import from new structure"""
    print("\nTesting main simulation imports...")
    
    try:
        # Test that main simulation can still run
        from main_game_simulation import create_agents, GameSimulator
        print("✅ Main simulation imports successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import main simulation components: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing All Major Improvements")
    print("=" * 50)
    
    tests = [
        ("Trick Order", test_trick_order),
        ("File Structure", test_file_structure),
        ("Logging Format", test_logging_format),
        ("Hybrid Agent Import", test_hybrid_agent_import),
        ("Main Simulation Imports", test_main_simulation_imports)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All improvements are working correctly!")
    else:
        print("⚠️  Some improvements need attention")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
