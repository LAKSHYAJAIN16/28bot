"""
Comprehensive example demonstrating 28bot v2 capabilities
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, List

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game28.game_state import Game28State
from game28.constants import *
from belief_model.belief_net import BeliefNetwork, BeliefState
from ismcts.ismcts_bidding import ISMCTSBidding, BeliefAwareISMCTS
from viz.render import BidExplanation, BeliefVisualization, create_game_state_visualization
from experiments.exploitability import evaluate_exploitability


def example_basic_game():
    """Example 1: Basic game state and mechanics"""
    print("=== Example 1: Basic Game State ===")
    
    # Create a new game
    game_state = Game28State()
    
    print(f"Game phase: {game_state.phase}")
    print(f"Current player: {game_state.current_player}")
    print(f"Current bid: {game_state.current_bid}")
    
    # Show player hands
    for i, hand in enumerate(game_state.hands):
        print(f"Player {i} hand: {[str(card) for card in hand]}")
        hand_points = sum(CARD_VALUES[card.rank] for card in hand)
        print(f"  Points: {hand_points}")
    
    # Show legal bids for current player
    legal_bids = game_state.get_legal_bids(game_state.current_player)
    print(f"Legal bids for player {game_state.current_player}: {legal_bids}")
    
    print()


def example_bidding_phase():
    """Example 2: Complete bidding phase"""
    print("=== Example 2: Bidding Phase ===")
    
    game_state = Game28State()
    
    # Simulate bidding
    while not game_state.game_over and game_state.phase == GamePhase.BIDDING:
        current_player = game_state.current_player
        legal_bids = game_state.get_legal_bids(current_player)
        
        if not legal_bids:
            break
        
        # Simple heuristic bidding
        hand = game_state.hands[current_player]
        hand_strength = sum(CARD_VALUES[card.rank] for card in hand) / TOTAL_POINTS
        
        if hand_strength > 0.5:
            bid = min(game_state.current_bid + 1, MAX_BID)
        else:
            bid = -1  # Pass
        
        print(f"Player {current_player} bids: {bid} (hand strength: {hand_strength:.2f})")
        
        # Apply bid
        game_state.make_bid(current_player, bid)
    
    print(f"Bidding ended. Bidder: {game_state.bidder}")
    print(f"Winning bid: {game_state.winning_bid}")
    
    if game_state.bidder is not None:
        # Set trump
        trump_suit = 'H'  # Simple choice
        game_state.set_trump(trump_suit)
        print(f"Trump suit set to: {trump_suit}")
    
    print()


def example_belief_network():
    """Example 3: Belief network for opponent modeling"""
    print("=== Example 3: Belief Network ===")
    
    # Create belief network
    belief_network = BeliefNetwork()
    
    # Create a game state
    game_state = Game28State()
    
    # Get belief predictions
    belief_state = belief_network.predict_beliefs(game_state, player_id=0)
    
    print(f"Known cards: {[str(card) for card in belief_state.known_cards]}")
    print(f"Played cards: {[str(card) for card in belief_state.played_cards]}")
    
    # Show opponent hand probabilities
    for opp_id in belief_state.opponent_hands:
        probs = belief_state.opponent_hands[opp_id]
        high_prob_cards = [i for i, p in enumerate(probs) if p > 0.1]
        print(f"Opponent {opp_id} high probability cards: {len(high_prob_cards)}")
    
    print()


def example_ismcts():
    """Example 4: ISMCTS for bidding decisions"""
    print("=== Example 4: ISMCTS Bidding ===")
    
    # Create ISMCTS with belief network
    belief_network = BeliefNetwork()
    ismcts = BeliefAwareISMCTS(belief_network=belief_network, num_simulations=100)
    
    # Create game state
    game_state = Game28State()
    
    # Get ISMCTS decision
    action, confidence = ismcts.select_action_with_confidence(game_state, player_id=0)
    
    print(f"ISMCTS decision: {action}")
    print(f"Confidence: {confidence:.3f}")
    
    # Show legal actions
    legal_bids = game_state.get_legal_bids(0)
    print(f"Legal bids: {legal_bids}")
    
    print()


def example_bid_explanation():
    """Example 5: Bid explanation and visualization"""
    print("=== Example 5: Bid Explanation ===")
    
    # Create bid explainer
    bid_explainer = BidExplanation()
    
    # Create game state
    game_state = Game28State()
    
    # Generate explanation for a bid
    bid = 20
    confidence = 0.75
    explanation = bid_explainer.explain_bid(game_state, player_id=0, bid=bid, confidence=confidence)
    
    print(f"Bid: {explanation['bid']}")
    print(f"Confidence: {explanation['confidence']}")
    print(f"Hand strength: {explanation['hand_strength']['strength_category']}")
    print(f"Expected points: {explanation['expected_points']:.1f}")
    print("Reasoning:")
    for reason in explanation['reasoning']:
        print(f"  - {reason}")
    
    print()


def example_visualization():
    """Example 6: Game state visualization"""
    print("=== Example 6: Visualization ===")
    
    # Create game state
    game_state = Game28State()
    
    # Simulate some bidding
    for _ in range(3):
        if game_state.game_over or game_state.phase != GamePhase.BIDDING:
            break
        
        current_player = game_state.current_player
        legal_bids = game_state.get_legal_bids(current_player)
        
        if legal_bids:
            bid = legal_bids[0] if legal_bids[0] != -1 else -1
            game_state.make_bid(current_player, bid)
    
    # Create visualization
    fig = create_game_state_visualization(game_state, player_id=0)
    
    # Save visualization
    fig.write_html("game_state_visualization.html")
    print("Game state visualization saved to game_state_visualization.html")
    
    print()


def example_exploitability():
    """Example 7: Exploitability evaluation"""
    print("=== Example 7: Exploitability Evaluation ===")
    
    # Define some simple strategies
    def random_strategy(game_state, player_id):
        legal_bids = game_state.get_legal_bids(player_id)
        if legal_bids:
            return np.random.choice(legal_bids)
        return -1
    
    def conservative_strategy(game_state, player_id):
        hand = game_state.hands[player_id]
        hand_strength = sum(CARD_VALUES[card.rank] for card in hand) / TOTAL_POINTS
        
        if hand_strength > 0.5:
            return min(game_state.current_bid + 1, MAX_BID)
        else:
            return -1
    
    def aggressive_strategy(game_state, player_id):
        hand = game_state.hands[player_id]
        hand_strength = sum(CARD_VALUES[card.rank] for card in hand) / TOTAL_POINTS
        
        if hand_strength > 0.3:
            return min(game_state.current_bid + 2, MAX_BID)
        else:
            return -1
    
    strategies = [random_strategy, conservative_strategy, aggressive_strategy]
    
    # Run exploitability evaluation (with fewer games for demo)
    results = evaluate_exploitability(strategies, num_games=50, save_results=False)
    
    print("Exploitability Results:")
    for strategy_name, result in results.items():
        overall = result['overall_metrics']
        print(f"{strategy_name}:")
        print(f"  Average win rate: {overall['avg_win_rate']:.3f}")
        print(f"  Exploitability: {overall['exploitability']:.3f}")
    
    print()


def example_complete_game():
    """Example 8: Complete game with AI components"""
    print("=== Example 8: Complete Game ===")
    
    from run_game import Game28Runner
    
    # Create game runner with AI components
    runner = Game28Runner(
        use_belief_network=True,
        use_ismcts=True,
        num_simulations=100,  # Reduced for demo
        visualize=True
    )
    
    # Run a single round
    results = runner.run_game(num_rounds=1, log_game=True)
    
    print("Game completed!")
    print(f"Final scores: {results['total_game_points']}")
    
    print()


def main():
    """Run all examples"""
    print("28bot v2 - Comprehensive Example\n")
    print("This example demonstrates all major components of the 28bot v2 system.\n")
    
    try:
        example_basic_game()
        example_bidding_phase()
        example_belief_network()
        example_ismcts()
        example_bid_explanation()
        example_visualization()
        example_exploitability()
        example_complete_game()
        
        print("All examples completed successfully!")
        print("\nKey features demonstrated:")
        print("- Game state management and mechanics")
        print("- Bidding phase simulation")
        print("- Belief network for opponent modeling")
        print("- ISMCTS for decision making")
        print("- Bid explanation and reasoning")
        print("- Game state visualization")
        print("- Exploitability evaluation")
        print("- Complete game with AI components")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure all dependencies are installed and models are available.")


if __name__ == "__main__":
    main()
