#!/usr/bin/env python3
"""
Test script to verify that the hybrid agent is properly integrated into the main game simulation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main_game_simulation import create_agents, GameSimulator
from game28.game_state import Game28State

def test_hybrid_agent_creation():
    """Test that hybrid agents are created correctly"""
    print("Testing hybrid agent creation...")
    
    try:
        agents = create_agents()
        
        if len(agents) != 4:
            print(f"❌ Expected 4 agents, got {len(agents)}")
            return False
        
        for i, agent in enumerate(agents):
            print(f"  Agent {i}: {agent.name} (strategy: {agent.strategy})")
            
            # Check if hybrid agent is loaded
            if agent.hybrid_agent is None:
                print(f"    ❌ Hybrid agent not loaded for {agent.name}")
                return False
            else:
                print(f"    ✅ Hybrid agent loaded for {agent.name}")
            
            # Check if belief model is loaded
            if agent.belief_model is None:
                print(f"    ❌ Belief model not loaded for {agent.name}")
                return False
            else:
                print(f"    ✅ Belief model loaded for {agent.name}")
        
        print("✅ All hybrid agents created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create agents: {e}")
        return False

def test_hybrid_agent_bidding():
    """Test that hybrid agents can make bidding decisions"""
    print("\nTesting hybrid agent bidding...")
    
    try:
        agents = create_agents()
        game_state = Game28State()
        
        # Test bidding for each agent
        for agent in agents:
            print(f"  Testing {agent.name}...")
            
            # Get legal bids
            legal_bids = game_state.get_legal_bids(agent.agent_id)
            print(f"    Legal bids: {legal_bids}")
            
            # Make bidding decision
            bid = agent.decide_bid(game_state, game_state.current_bid)
            print(f"    Chosen bid: {bid}")
            
            if bid not in legal_bids:
                print(f"    ❌ Invalid bid {bid} for {agent.name}")
                return False
            else:
                print(f"    ✅ Valid bid {bid} for {agent.name}")
        
        print("✅ All hybrid agents can make bidding decisions!")
        return True
        
    except Exception as e:
        print(f"❌ Bidding test failed: {e}")
        return False

def test_hybrid_agent_trump_selection():
    """Test that hybrid agents can select trump"""
    print("\nTesting hybrid agent trump selection...")
    
    try:
        agents = create_agents()
        game_state = Game28State()
        
        # Set up a bidding scenario
        game_state.bidder = 0
        game_state.winning_bid = 20
        
        # Test trump selection for the bidder
        agent = agents[0]  # Agent 0 is the bidder
        print(f"  Testing {agent.name} trump selection...")
        
        trump_suit = agent.choose_trump(game_state)
        print(f"    Selected trump: {trump_suit}")
        
        if trump_suit not in ['H', 'D', 'C', 'S']:
            print(f"    ❌ Invalid trump suit {trump_suit}")
            return False
        else:
            print(f"    ✅ Valid trump suit {trump_suit}")
        
        print("✅ Hybrid agent can select trump!")
        return True
        
    except Exception as e:
        print(f"❌ Trump selection test failed: {e}")
        return False

def test_hybrid_agent_card_selection():
    """Test that hybrid agents can select cards"""
    print("\nTesting hybrid agent card selection...")
    
    try:
        agents = create_agents()
        game_state = Game28State()
        
        # Set up a card play scenario
        game_state.phase = game_state.phase.CONCEALED
        game_state.trump_suit = 'H'
        game_state.current_player = 0
        
        # Test card selection for agent 0
        agent = agents[0]
        print(f"  Testing {agent.name} card selection...")
        
        # Get legal cards
        legal_cards = game_state.get_legal_plays(agent.agent_id)
        print(f"    Legal cards: {[str(c) for c in legal_cards]}")
        
        if not legal_cards:
            print("    ❌ No legal cards available")
            return False
        
        # Choose a card
        chosen_card = agent.choose_card(game_state, legal_cards)
        print(f"    Chosen card: {chosen_card}")
        
        if chosen_card not in legal_cards:
            print(f"    ❌ Invalid card choice {chosen_card}")
            return False
        else:
            print(f"    ✅ Valid card choice {chosen_card}")
        
        print("✅ Hybrid agent can select cards!")
        return True
        
    except Exception as e:
        print(f"❌ Card selection test failed: {e}")
        return False

def test_hybrid_performance_stats():
    """Test that hybrid agents track performance statistics"""
    print("\nTesting hybrid agent performance tracking...")
    
    try:
        agents = create_agents()
        
        for agent in agents:
            print(f"  Testing {agent.name} performance stats...")
            
            if agent.hybrid_agent is None:
                print(f"    ❌ No hybrid agent for {agent.name}")
                return False
            
            # Get performance stats
            stats = agent.hybrid_agent.get_performance_stats()
            print(f"    Performance stats: {stats}")
            
            # Check that stats are properly initialized
            if stats['total_decisions'] != 0:
                print(f"    ❌ Expected 0 total decisions, got {stats['total_decisions']}")
                return False
            
            print(f"    ✅ Performance stats properly initialized for {agent.name}")
        
        print("✅ All hybrid agents track performance correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Performance tracking test failed: {e}")
        return False

def test_single_game_simulation():
    """Test a single game simulation with hybrid agents"""
    print("\nTesting single game simulation with hybrid agents...")
    
    try:
        # Create agents
        agents = create_agents()
        
        # Create simulator
        simulator = GameSimulator(agents, game_id=999, log_dir="test_logs")
        
        # Run a single game
        print("  Running single game simulation...")
        results = simulator.simulate_game()
        
        print(f"  Game results: {results}")
        
        # Check that the game completed
        if 'winning_team' not in results:
            print("    ❌ Game did not complete properly")
            return False
        
        print("    ✅ Game simulation completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Game simulation test failed: {e}")
        return False

def main():
    """Run all hybrid agent integration tests"""
    print("Testing Hybrid Agent Integration")
    print("=" * 60)
    
    tests = [
        ("Agent Creation", test_hybrid_agent_creation),
        ("Bidding Decisions", test_hybrid_agent_bidding),
        ("Trump Selection", test_hybrid_agent_trump_selection),
        ("Card Selection", test_hybrid_agent_card_selection),
        ("Performance Tracking", test_hybrid_performance_stats),
        ("Single Game Simulation", test_single_game_simulation)
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
    print("\n" + "=" * 60)
    print("HYBRID AGENT INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Hybrid agent is fully integrated and working!")
        print("\nYou can now run the main simulation with:")
        print("python main_game_simulation.py")
    else:
        print("⚠️  Some integration issues need attention")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
