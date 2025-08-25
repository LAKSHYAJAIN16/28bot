#!/usr/bin/env python3
"""
Test script to verify that ISMCTS is actually working
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game28.game_state import Game28State, Card
from game28.constants import *
from belief_model.improved_belief_net import ImprovedBeliefNetwork
from ismcts.ismcts_bidding import ISMCTSBidding, BeliefAwareISMCTS
from agents.hybrid_agent import HybridAgent, HybridDecision

def test_ismcts_basic():
    """Test basic ISMCTS functionality"""
    print("Testing basic ISMCTS functionality...")
    
    # Create a game state
    game_state = Game28State()
    
    # Create ISMCTS instance
    ismcts = ISMCTSBidding(num_simulations=100)  # Reduced for testing
    
    # Test action selection
    try:
        action = ismcts.select_action(game_state, player_id=0)
        print(f"✅ ISMCTS selected action: {action}")
        return True
    except Exception as e:
        print(f"❌ ISMCTS failed: {e}")
        return False

def test_belief_aware_ismcts():
    """Test belief-aware ISMCTS"""
    print("\nTesting belief-aware ISMCTS...")
    
    # Create a game state
    game_state = Game28State()
    
    # Create belief network (dummy for now)
    belief_network = None  # We'll use uniform sampling
    
    # Create belief-aware ISMCTS
    ismcts = BeliefAwareISMCTS(belief_network=belief_network, num_simulations=100)
    
    try:
        action, confidence = ismcts.select_action_with_confidence(game_state, player_id=0)
        print(f"✅ Belief-aware ISMCTS selected action: {action} with confidence: {confidence:.3f}")
        return True
    except Exception as e:
        print(f"❌ Belief-aware ISMCTS failed: {e}")
        return False

def test_hybrid_agent_ismcts():
    """Test that hybrid agent actually uses ISMCTS"""
    print("\nTesting hybrid agent ISMCTS integration...")
    
    # Create a game state
    game_state = Game28State()
    
    # Create belief network (dummy)
    belief_network = None
    
    # Create ISMCTS
    ismcts = BeliefAwareISMCTS(belief_network=belief_network, num_simulations=50)
    
    # Create hybrid agent
    hybrid_agent = HybridAgent(
        agent_id=0,
        belief_model=belief_network,
        ismcts=ismcts,
        use_hybrid=True
    )
    
    # Test bidding decision
    legal_bids = game_state.get_legal_bids(0)
    print(f"Legal bids: {legal_bids}")
    
    try:
        decision = hybrid_agent._ismcts_based_bid(game_state, legal_bids)
        print(f"✅ Hybrid agent ISMCTS decision: {decision.action} (method: {decision.method})")
        print(f"   Confidence: {decision.confidence:.3f}")
        print(f"   Reasoning: {decision.reasoning}")
        
        # Check if it actually used ISMCTS
        if decision.method == "ismcts":
            print("✅ Hybrid agent successfully used ISMCTS!")
            return True
        else:
            print("❌ Hybrid agent fell back to belief model")
            return False
            
    except Exception as e:
        print(f"❌ Hybrid agent ISMCTS failed: {e}")
        return False

def test_ismcts_vs_belief():
    """Compare ISMCTS vs belief model decisions"""
    print("\nComparing ISMCTS vs belief model decisions...")
    
    # Create a game state
    game_state = Game28State()
    
    # Create belief network (dummy)
    belief_network = None
    
    # Create ISMCTS
    ismcts = BeliefAwareISMCTS(belief_network=belief_network, num_simulations=100)
    
    # Create hybrid agent
    hybrid_agent = HybridAgent(
        agent_id=0,
        belief_model=belief_network,
        ismcts=ismcts,
        use_hybrid=True
    )
    
    legal_bids = game_state.get_legal_bids(0)
    
    try:
        # Get belief-based decision
        belief_decision = hybrid_agent._belief_based_bid(game_state, legal_bids)
        print(f"Belief model decision: {belief_decision.action} (confidence: {belief_decision.confidence:.3f})")
        
        # Get ISMCTS decision
        ismcts_decision = hybrid_agent._ismcts_based_bid(game_state, legal_bids)
        print(f"ISMCTS decision: {ismcts_decision.action} (confidence: {ismcts_decision.confidence:.3f})")
        
        # Get hybrid decision
        hybrid_decision = hybrid_agent._hybrid_bid(game_state, legal_bids)
        print(f"Hybrid decision: {hybrid_decision.action} (method: {hybrid_decision.method})")
        
        print("✅ All decision methods working!")
        return True
        
    except Exception as e:
        print(f"❌ Decision comparison failed: {e}")
        return False

def main():
    """Run all ISMCTS tests"""
    print("Testing ISMCTS Functionality")
    print("=" * 50)
    
    tests = [
        ("Basic ISMCTS", test_ismcts_basic),
        ("Belief-Aware ISMCTS", test_belief_aware_ismcts),
        ("Hybrid Agent ISMCTS", test_hybrid_agent_ismcts),
        ("ISMCTS vs Belief Comparison", test_ismcts_vs_belief)
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
    print("ISMCTS TEST SUMMARY")
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
        print("🎉 ISMCTS is working correctly!")
    else:
        print("⚠️  Some ISMCTS functionality needs attention")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
