#!/usr/bin/env python3
"""
Test script to verify that the trick winner leads the next trick
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game28.game_state import Game28State, Card
from game28.constants import SUITS, RANKS, GamePhase

def test_trick_order():
    """Test that trick winner leads the next trick"""
    print("Testing trick order functionality...")
    
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
    
    # Player 0 plays
    card = game_state.hands[0][0]  # Ace of Hearts
    game_state.play_card(0, card)
    print(f"Player 0 plays {card}")
    print(f"Current player after play: {game_state.current_player}")
    
    # Player 1 plays
    card = game_state.hands[1][0]  # King of Hearts
    game_state.play_card(1, card)
    print(f"Player 1 plays {card}")
    print(f"Current player after play: {game_state.current_player}")
    
    # Player 2 plays
    card = game_state.hands[2][0]  # Queen of Hearts
    game_state.play_card(2, card)
    print(f"Player 2 plays {card}")
    print(f"Current player after play: {game_state.current_player}")
    
    # Player 3 plays
    card = game_state.hands[3][0]  # Jack of Hearts
    game_state.play_card(3, card)
    print(f"Player 3 plays {card}")
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
        else:
            print("❌ FAILURE: Trick winner is NOT the current player!")
            print(f"Expected: {winner}, Got: {game_state.current_player}")
    else:
        print("❌ FAILURE: No tricks were created!")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_trick_order()
