#!/usr/bin/env python3
"""
Human vs AI Game for Twenty-Eight (28)

Usage:
  - Run: python run_human_game.py
  - Choose your player number (0-3)
  - Input your 8 cards
  - Input other players' moves as they happen
  - Get AI suggestions for your best move

This script allows you to play a game of 28 where you control one player
and input the moves of other players, with the AI suggesting the best moves for you.
"""

import sys
import copy
import random
from typing import Dict, List, Tuple, Optional
import os

# Use the project's MCTS modules
from mcts.constants import SUITS, RANKS, CARD_POINTS, card_suit, card_rank, card_value, rank_index, trick_rank_index
from mcts.env28 import TwentyEightEnv
from mcts.ismcts import ismcts_plan
from mcts.policy import policy_move

SUITS = SUITS
RANKS = RANKS
CARD_POINTS = CARD_POINTS

# Game configuration
AI_ITERATIONS = 1000  # MCTS iterations for AI moves
AI_SAMPLES = 16  # ISMCTS samples for AI moves
SEARCH_MODE = "regular"  # MCTS search mode

# MCTS config for AI
MCTS_CONFIG = {
    'mcts_samples': AI_SAMPLES,
    'mcts_iters_per_sample': max(20, AI_ITERATIONS // 4),
    'mcts_c_puct': 1.5,
    'bidding_samples': 8,
    'bidding_iterations': 100,
    'bidding_stage2_iterations': 200,
    'rollout_iterations': 100,
}


def parse_cards(raw: str) -> List[str]:
    """Parse card input from user."""
    tokens = [t.strip().upper() for t in raw.replace(",", " ").split() if t.strip()]
    normalized = []
    for t in tokens:
        # Accept forms like 10H, JH, AS, 7D
        if len(t) < 2:
            raise ValueError(f"Invalid card token: {t}")
        suit = t[-1]
        rank = t[:-1]
        if suit not in SUITS or rank not in RANKS:
            raise ValueError(f"Invalid card token: {t}")
        normalized.append(rank + suit)
    return normalized


def parse_single_card(raw: str) -> str:
    """Parse a single card input."""
    raw = raw.strip().upper()
    if len(raw) < 2:
        raise ValueError(f"Invalid card: {raw}")
    suit = raw[-1]
    rank = raw[:-1]
    if suit not in SUITS or rank not in RANKS:
        raise ValueError(f"Invalid card: {raw}")
    return rank + suit


def display_game_state(env: TwentyEightEnv, state: Dict, human_hand: List[str], human_player: int):
    """Display the current game state."""
    print("\n" + "="*50)
    print("CURRENT GAME STATE")
    print("="*50)
    
    # Show your hand
    print(f"Your hand (Player {human_player}): {', '.join(human_hand)}")
    
    # Show current trick
    if env.current_trick:
        print(f"Current trick: {', '.join(env.current_trick)}")
    else:
        print("Current trick: (empty)")
    
    # Show scores
    print(f"Scores - Team A: {state['scores'][0]}, Team B: {state['scores'][1]}")
    
    # Show game info
    if hasattr(env, 'bidder') and env.bidder is not None:
        print(f"Bidder: Player {env.bidder} (bid: {getattr(env, 'bid_value', 16)})")
        print(f"Trump suit: {env.trump_suit}")
        print(f"Phase: {state.get('phase', 'unknown')}")
        if hasattr(env, 'face_down_trump_card') and env.face_down_trump_card:
            print(f"Concealed trump card: {env.face_down_trump_card}")
    
    print(f"Current player: {state['turn']}")
    print("="*50)


def get_valid_moves(hand: List[str], current_trick: List[str]) -> List[str]:
    """Get valid moves for the current player."""
    if not current_trick:
        # First player can play any card
        return hand
    
    # Must follow suit if possible
    lead_suit = card_suit(current_trick[0])
    cards_in_lead_suit = [card for card in hand if card_suit(card) == lead_suit]
    
    if cards_in_lead_suit:
        return cards_in_lead_suit
    else:
        return hand  # Can play any card if void in lead suit


def evaluate_trick_winner(trick: List[str], trump_suit: str) -> int:
    """Determine who wins the trick."""
    if len(trick) != 4:
        return -1
    
    lead_suit = card_suit(trick[0])
    trump_cards = [i for i, card in enumerate(trick) if card_suit(card) == trump_suit]
    
    if trump_cards:
        # Trump wins
        best_trump = max(trump_cards, key=lambda i: trick_rank_index(trick[i]))
        return best_trump
    else:
        # Lead suit wins
        lead_cards = [i for i, card in enumerate(trick) if card_suit(card) == lead_suit]
        best_lead = max(lead_cards, key=lambda i: trick_rank_index(trick[i]))
        return best_lead


def get_ai_move(env: TwentyEightEnv, state: Dict, player: int) -> str:
    """Get AI move for the specified player."""
    # Temporarily set the environment to the current player's turn
    original_turn = env.turn
    env.turn = player
    
    try:
        # Get AI move using MCTS
        move = policy_move(env, AI_ITERATIONS, SEARCH_MODE, MCTS_CONFIG)
        return move
    except Exception as e:
        print(f"AI move calculation failed: {e}")
        # Fallback to random valid move
        valid_moves = get_valid_moves(env.hands[player], env.current_trick)
        return random.choice(valid_moves) if valid_moves else env.hands[player][0]
    finally:
        env.turn = original_turn


def get_other_player_move(env: TwentyEightEnv, state: Dict, player: int) -> str:
    """Get move for other players (input by user)."""
    current_hand = env.hands[player]
    valid_moves = get_valid_moves(current_hand, env.current_trick)
    
    print(f"Player {player} valid moves: {', '.join(valid_moves)}")
    
    while True:
        print(f"Enter Player {player}'s move:")
        try:
            move = parse_single_card(input().strip())
            if move in valid_moves:
                return move
            else:
                print(f"Invalid move. Valid moves are: {', '.join(valid_moves)}")
        except Exception as e:
            print(f"Invalid input: {e}")


def main() -> int:
    print("Human vs AI Game - Twenty-Eight (28)")
    print("="*50)
    print("You will input moves for all other players")
    print("AI will suggest the best move for you")
    print("="*50)
    
    # Get human player number
    while True:
        print("\nEnter your player number (0-3):")
        try:
            human_player = int(input().strip())
            if 0 <= human_player <= 3:
                break
            else:
                print("Player number must be 0, 1, 2, or 3.")
        except ValueError:
            print("Please enter a valid number.")
    
    print(f"You are Player {human_player}")
    print(f"You are on Team {'A' if human_player % 2 == 0 else 'B'}")
    
    # Get human player's cards
    while True:
        print("\nEnter your 8 cards (e.g., 'JH 9H 7D AC 8S KS QD 10C'):")
        try:
            human_cards = parse_cards(input().strip())
            if len(human_cards) != 8:
                print(f"Expected 8 cards, got {len(human_cards)}. Please try again.")
                continue
            break
        except Exception as e:
            print(f"Invalid input: {e}. Please try again.")
    
    print(f"\nYour cards: {', '.join(human_cards)}")
    
    # Get trump suit
    print("\nEnter trump suit (H/D/C/S):")
    while True:
        trump_input = input().strip().upper()
        if trump_input in SUITS:
            trump_suit = trump_input
            break
        else:
            print("Invalid suit. Please enter H, D, C, or S.")
    
    # Get starting player
    print("\nEnter starting player (0-3):")
    while True:
        try:
            starting_player = int(input().strip())
            if 0 <= starting_player <= 3:
                break
            else:
                print("Player number must be 0, 1, 2, or 3.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Initialize game environment
    env = TwentyEightEnv()
    env.debug = False
    
    # Set up the game with human cards
    all_ranks = ["7", "8", "9", "10", "J", "Q", "K", "A"]
    full_deck = [r + s for r in all_ranks for s in SUITS]
    remaining_cards = [c for c in full_deck if c not in human_cards]
    
    # Shuffle remaining cards and distribute to other players
    random.shuffle(remaining_cards)
    other_hands = [
        remaining_cards[:8],  # Player 0
        remaining_cards[8:16],  # Player 1
        remaining_cards[16:24],  # Player 2
        remaining_cards[24:32]   # Player 3
    ]
    
    # Set up hands - replace the human player's hand with their actual cards
    env.hands = other_hands.copy()
    env.hands[human_player] = human_cards
    
    # Initialize game state
    env.scores = [0, 0]
    env.game_score = [0, 0]
    env.current_trick = []
    env.turn = starting_player  # Start with specified player
    env.bidder = human_player  # You are the bidder for simplicity
    env.trump_suit = trump_suit
    env.bid_value = 16
    env.round_stakes = 1
    env.phase = "concealed"
    env.void_suits_by_player = [set() for _ in range(4)]
    env.lead_suit_counts = [{s: 0 for s in SUITS} for _ in range(4)]
    
    # Set up trump card for bidder
    trump_cards = [c for c in human_cards if card_suit(c) == env.trump_suit]
    if trump_cards:
        env.face_down_trump_card = max(trump_cards, key=rank_index)
        # Remove from your hand
        human_cards.remove(env.face_down_trump_card)
        env.hands[human_player] = human_cards
    
    env.last_exposer = None
    env.exposure_trick_index = None
    env.tricks_played = 0
    env.invalid_round = False
    env.last_trick_winner = None
    
    # Get initial state
    state = env.get_state()
    
    print(f"\nGame initialized!")
    print(f"Starting player: {starting_player}")
    print(f"Trump suit: {env.trump_suit}")
    print(f"Your concealed trump card: {env.face_down_trump_card}")
    
    # Main game loop
    while not state.get('done', False):
        display_game_state(env, state, env.hands[human_player], human_player)
        
        if state['turn'] == human_player:
            # Human player's turn
            current_hand = env.hands[human_player]
            valid_moves = get_valid_moves(current_hand, env.current_trick)
            
            print(f"\nYour turn! Valid moves: {', '.join(valid_moves)}")
            
            # Get AI suggestion
            ai_suggestion = get_ai_move(env, state, human_player)
            print(f"AI suggests: {ai_suggestion}")
            
            # Get human move
            while True:
                print("Enter your move (or 'ai' to use AI suggestion):")
                move_input = input().strip().upper()
                
                if move_input == 'AI':
                    move = ai_suggestion
                    break
                else:
                    try:
                        move = parse_single_card(move_input)
                        if move in valid_moves:
                            break
                        else:
                            print(f"Invalid move. Valid moves are: {', '.join(valid_moves)}")
                    except Exception as e:
                        print(f"Invalid input: {e}")
            
            # Make the move
            current_hand.remove(move)
            env.current_trick.append(move)
            print(f"You play: {move}")
            
            # Move to next player
            state['turn'] = (state['turn'] + 1) % 4
            
        else:
            # Other players' turns - get input from user
            current_player = state['turn']
            current_hand = env.hands[current_player]
            
            # Get move from user for this player
            other_move = get_other_player_move(env, state, current_player)
            
            # Remove the card from the player's hand
            if other_move in current_hand:
                current_hand.remove(other_move)
            
            # Add to current trick
            env.current_trick.append(other_move)
            print(f"Player {current_player} plays: {other_move}")
            
            # Move to next player
            state['turn'] = (state['turn'] + 1) % 4
        
        # Check if trick is complete
        if len(env.current_trick) == 4:
            # Determine winner
            winner = evaluate_trick_winner(env.current_trick, env.trump_suit)
            trick_points = sum(card_value(card) for card in env.current_trick)
            
            print(f"\nPlayer {winner} wins the trick with {trick_points} points")
            
            # Update scores
            winner_team = 0 if winner % 2 == 0 else 1
            state['scores'][winner_team] += trick_points
            
            # Clear trick and set next player (winner leads next trick)
            env.current_trick = []
            state['turn'] = winner
            
            # Check if game is done
            if all(len(hand) == 0 for hand in env.hands):
                state['done'] = True
    
    # Game over
    print("\n" + "="*50)
    print("GAME OVER!")
    print("="*50)
    print(f"Final scores - Team A: {state['scores'][0]}, Team B: {state['scores'][1]}")
    
    if state['scores'][0] > state['scores'][1]:
        print("Team A wins!")
    elif state['scores'][1] > state['scores'][0]:
        print("Team B wins!")
    else:
        print("It's a tie!")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
