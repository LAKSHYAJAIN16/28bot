#!/usr/bin/env python3
"""
Example of using the trained belief model
"""

import sys
import os
import torch
import numpy as np

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from belief_model.belief_net import BeliefNetwork
from game28.game_state import Game28State
from game28.constants import SUITS, RANKS

def load_belief_model(model_path: str = "models/belief_model/belief_model_final.pt"):
    """Load the trained belief model"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BeliefNetwork().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"✓ Belief model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"✗ Error loading belief model: {e}")
        print("Make sure you've trained the model first using:")
        print("python scripts/train_belief_model.py")
        return None

def print_belief_predictions(belief_state, player_id: int):
    """Print belief predictions in a readable format"""
    print(f"\nBelief Predictions for Player {player_id}:")
    print("="*50)
    
    # Print opponent hand predictions
    for opp_id, hand_probs in belief_state.opponent_hands.items():
        if int(opp_id) != player_id:
            print(f"\nOpponent {opp_id} Hand Probabilities:")
            
            # Group by suit
            for suit_idx, suit in enumerate(SUITS):
                print(f"  {suit}: ", end="")
                for rank_idx, rank in enumerate(RANKS):
                    card_idx = suit_idx * 8 + rank_idx
                    prob = hand_probs[card_idx]
                    if prob > 0.3:  # Only show high probability cards
                        print(f"{rank}({prob:.2f}) ", end="")
                print()
    
    # Print trump prediction
    if hasattr(belief_state, 'trump_suit') and belief_state.trump_suit is not None:
        trump_probs = belief_state.trump_suit
        print(f"\nTrump Suit Probabilities:")
        for suit_idx, suit in enumerate(SUITS):
            prob = trump_probs[suit_idx]
            print(f"  {suit}: {prob:.3f}")

def demonstrate_belief_model():
    """Demonstrate the belief model in action"""
    print("="*60)
    print("BELIEF MODEL DEMONSTRATION")
    print("="*60)
    
    # Load the model
    model = load_belief_model()
    if model is None:
        return
    
    # Create a game state
    game_state = Game28State()
    
    print(f"\nGame State:")
    print(f"  Phase: {game_state.phase}")
    print(f"  Current bid: {game_state.current_bid}")
    print(f"  Your hand: {[str(card) for card in game_state.hands[0]]}")
    
    # Get belief predictions for player 0
    player_id = 0
    belief_state = model.predict_beliefs(game_state, player_id)
    
    # Print predictions
    print_belief_predictions(belief_state, player_id)
    
    # Simulate some bidding
    print(f"\n" + "="*50)
    print("SIMULATING BIDDING")
    print("="*50)
    
    # Player 1 bids 18
    game_state.make_bid(1, 18)
    print(f"Player 1 bids 18")
    
    # Player 2 passes
    game_state.make_bid(2, -1)
    print(f"Player 2 passes")
    
    # Player 3 bids 19
    game_state.make_bid(3, 19)
    print(f"Player 3 bids 19")
    
    # Player 0 passes
    game_state.make_bid(0, -1)
    print(f"Player 0 passes")
    
    # Player 1 passes
    game_state.make_bid(1, -1)
    print(f"Player 1 passes")
    
    print(f"\nBidding completed!")
    print(f"Winner: Player {game_state.bidder} with bid {game_state.winning_bid}")
    
    # Get updated beliefs after bidding
    belief_state_after_bidding = model.predict_beliefs(game_state, player_id)
    
    print(f"\nBeliefs after bidding:")
    print_belief_predictions(belief_state_after_bidding, player_id)
    
    # Simulate some card play
    print(f"\n" + "="*50)
    print("SIMULATING CARD PLAY")
    print("="*50)
    
    # Set trump (let's say Hearts)
    game_state.set_trump('H')
    print(f"Trump set to: {game_state.trump_suit}")
    
    # Play some cards
    legal_cards = game_state.get_legal_plays(0)
    if legal_cards:
        card_played = legal_cards[0]
        game_state.play_card(0, card_played)
        print(f"Player 0 plays: {card_played}")
    
    # Get updated beliefs after card play
    belief_state_after_play = model.predict_beliefs(game_state, player_id)
    
    print(f"\nBeliefs after card play:")
    print_belief_predictions(belief_state_after_play, player_id)

def compare_beliefs_with_reality():
    """Compare belief predictions with actual opponent hands"""
    print("="*60)
    print("BELIEF vs REALITY COMPARISON")
    print("="*60)
    
    # Load the model
    model = load_belief_model()
    if model is None:
        return
    
    # Create a game state
    game_state = Game28State()
    
    print(f"\nActual opponent hands:")
    for player_id in range(1, 4):
        hand = [str(card) for card in game_state.hands[player_id]]
        print(f"  Player {player_id}: {hand}")
    
    # Get belief predictions
    belief_state = model.predict_beliefs(game_state, 0)
    
    print(f"\nBelief predictions for high-probability cards:")
    for opp_id, hand_probs in belief_state.opponent_hands.items():
        print(f"\nOpponent {opp_id}:")
        
        # Find cards with high probability (>0.5)
        high_prob_cards = []
        for card_idx, prob in enumerate(hand_probs):
            if prob > 0.5:
                suit_idx = card_idx // 8
                rank_idx = card_idx % 8
                suit = SUITS[suit_idx]
                rank = RANKS[rank_idx]
                high_prob_cards.append(f"{rank}{suit}({prob:.2f})")
        
        if high_prob_cards:
            print(f"  High probability cards: {', '.join(high_prob_cards)}")
        else:
            print(f"  No high probability cards predicted")

def main():
    """Main function"""
    print("Belief Model Usage Examples")
    print("="*60)
    
    # Check if model exists
    model_path = "models/belief_model/belief_model_final.pt"
    if not os.path.exists(model_path):
        print("❌ Belief model not found!")
        print(f"Expected path: {model_path}")
        print("\nTo train the belief model, run:")
        print("python scripts/train_belief_model.py")
        return
    
    # Run demonstrations
    demonstrate_belief_model()
    
    print(f"\n" + "="*60)
    compare_beliefs_with_reality()
    
    print(f"\n" + "="*60)
    print("USAGE SUMMARY")
    print("="*60)
    
    print("\nHow to use the belief model in your code:")
    print("""
# Load the model
from belief_model.belief_net import BeliefNetwork
import torch

model = BeliefNetwork()
model.load_state_dict(torch.load("models/belief_model/belief_model_final.pt"))
model.eval()

# Get predictions
from game28.game_state import Game28State

game_state = Game28State()
belief_state = model.predict_beliefs(game_state, player_id=0)

# Access predictions
opponent_hands = belief_state.opponent_hands  # Dict of opponent hand probabilities
trump_probability = belief_state.trump_suit   # Trump suit probabilities
    """)

if __name__ == "__main__":
    main()
