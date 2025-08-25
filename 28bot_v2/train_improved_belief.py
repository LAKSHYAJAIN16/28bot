#!/usr/bin/env python3
"""
Train and test the improved belief model
"""

import sys
import os
import torch
import numpy as np
from typing import List, Tuple, Dict

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from belief_model.improved_belief_net import (
    create_improved_belief_model, 
    train_improved_belief_model
)
from belief_model.advanced_parser import extract_all_game_states
from game28.game_state import Game28State, Card, Trick
from game28.constants import SUITS, RANKS, CARD_VALUES, GamePhase


def _phase_from_str(phase_str: str) -> GamePhase:
    mapping = {
        'bidding': GamePhase.BIDDING,
        'concealed': GamePhase.CONCEALED,
        'revealed': GamePhase.REVEALED,
    }
    return mapping.get(phase_str, GamePhase.BIDDING)


def _one_hot_trump(suit: str) -> list:
    return [1.0 if s == suit else 0.0 for s in SUITS]


def _to_game28_state(parsed_state) -> Game28State:
    """Convert advanced_parser.GameState into a minimal Game28State for training."""
    g = Game28State()
    # Overwrite dealt hands with parsed hands if available
    if hasattr(parsed_state, 'hands') and isinstance(parsed_state.hands, dict):
        g.hands = [parsed_state.hands.get(i, []) for i in range(4)]
    # Phase
    if hasattr(parsed_state, 'phase'):
        g.phase = _phase_from_str(parsed_state.phase)
    # Trump
    if hasattr(parsed_state, 'trump_suit'):
        g.trump_suit = parsed_state.trump_suit
        g.trump_revealed = bool(getattr(parsed_state, 'trump_revealed', False))
    # Bidding
    if hasattr(parsed_state, 'bidding_history') and parsed_state.bidding_history:
        g.bidder = parsed_state.bidding_history[-1][0]
        g.winning_bid = parsed_state.bidding_history[-1][1]
        g.current_bid = g.winning_bid
    # Current player
    if hasattr(parsed_state, 'current_player'):
        g.current_player = parsed_state.current_player
    return g


def build_training_data_from_logs(parsed_states):
    """Turn parsed GameState objects into (Game28State, player_id, target_beliefs) tuples.
    We supervise at least trump prediction; hand/void targets left empty unless readily derivable.
    """
    training = []
    for ps in parsed_states:
        g = _to_game28_state(ps)
        # Targets
        targets = {}
        if getattr(ps, 'trump_suit', None) in SUITS:
            targets['trump'] = _one_hot_trump(ps.trump_suit)
        # Optionally add more targets later (hands/voids) when reliable
        for pid in range(4):
            training.append((g, pid, targets))
    return training

def test_improved_belief_model():
    """Test the improved belief model with realistic scenarios"""
    
    print("Creating improved belief model...")
    model = create_improved_belief_model()
    
    print("Loading training data from logs...")
    # Use both legacy and new improved game logs
    log_dirs = [
        os.path.join("..", "logs", "game28", "mcts_games"),
        os.path.join("..", "logs", "improved_games")
    ]
    parsed_states = extract_all_game_states(log_dirs)
    training_data = build_training_data_from_logs(parsed_states)
    
    print(f"Training data size: {len(training_data)} examples")
    
    print("Training model...")
    trained_model = train_improved_belief_model(
        model, 
        training_data, 
        epochs=50,  # Reduced for faster training
        learning_rate=0.001,
    )
    
    print("Testing model predictions...")
    
    # Test 1: Early game (bidding phase)
    print("\n=== Test 1: Early Game (Bidding Phase) ===")
    early_game = Game28State()
    early_game.phase = GamePhase.BIDDING
    
    predictions = trained_model.predict_beliefs(early_game, 0)
    
    print("Trump probabilities:")
    trump_probs = predictions.trump_suit.cpu().numpy()
    for i, (suit, prob) in enumerate(zip(SUITS, trump_probs)):
        print(f"  {suit}: {prob:.4f}")
    
    print(f"Uncertainty: {predictions.uncertainty.cpu().numpy().item():.4f}")
    
    # Test 2: Mid game (concealed phase)
    print("\n=== Test 2: Mid Game (Concealed Phase) ===")
    mid_game = Game28State()
    mid_game.phase = GamePhase.CONCEALED
    mid_game.trump_suit = "H"  # Hearts
    mid_game.bidder = 1
    mid_game.winning_bid = 20
    
    # Add some tricks
    for i in range(3):
        trick = Trick()
        for player in range(4):
            suit = random.choice(SUITS)
            rank = random.choice(RANKS)
            card = Card(suit, rank)
            trick.add_card(player, card)
        trick.winner = random.randint(0, 3)
        trick.points = sum(CARD_VALUES[card.rank] for _, card in trick.cards)
        mid_game.tricks.append(trick)
    
    predictions = trained_model.predict_beliefs(mid_game, 0)
    
    print("Trump probabilities (should be high for Hearts):")
    trump_probs = predictions.trump_suit.cpu().numpy()
    for i, (suit, prob) in enumerate(zip(SUITS, trump_probs)):
        print(f"  {suit}: {prob:.4f}")
    
    print(f"Uncertainty: {predictions.uncertainty.cpu().numpy().item():.4f}")
    
    # Test 3: Late game (revealed phase)
    print("\n=== Test 3: Late Game (Revealed Phase) ===")
    late_game = Game28State()
    late_game.phase = GamePhase.REVEALED
    late_game.trump_suit = "D"  # Diamonds
    late_game.bidder = 2
    late_game.winning_bid = 24
    
    # Add many tricks
    for i in range(6):
        trick = Trick()
        for player in range(4):
            suit = random.choice(SUITS)
            rank = random.choice(RANKS)
            card = Card(suit, rank)
            trick.add_card(player, card)
        trick.winner = random.randint(0, 3)
        trick.points = sum(CARD_VALUES[card.rank] for _, card in trick.cards)
        late_game.tricks.append(trick)
    
    predictions = trained_model.predict_beliefs(late_game, 0)
    
    print("Trump probabilities (should be high for Diamonds):")
    trump_probs = predictions.trump_suit.cpu().numpy()
    for i, (suit, prob) in enumerate(zip(SUITS, trump_probs)):
        print(f"  {suit}: {prob:.4f}")
    
    print(f"Uncertainty: {predictions.uncertainty.cpu().numpy().item():.4f}")
    
    # Test 4: Check opponent hand predictions
    print("\n=== Test 4: Opponent Hand Predictions ===")
    for opp_id in range(1, 4):
        if opp_id in predictions.opponent_hands:
            opp_probs = predictions.opponent_hands[opp_id].cpu().numpy()
            
            # Find highest probability cards
            top_indices = np.argsort(opp_probs)[-5:]  # Top 5
            print(f"Opponent {opp_id} - Top 5 most likely cards:")
            for idx in reversed(top_indices):
                suit_idx = idx // 8
                rank_idx = idx % 8
                suit = SUITS[suit_idx]
                rank = RANKS[rank_idx]
                prob = opp_probs[idx]
                print(f"  {rank}{suit}: {prob:.4f}")
    
    # Save the trained model
    print("\nSaving trained model...")
    torch.save(trained_model.state_dict(), "models/improved_belief_model.pt")
    print("Model saved to models/improved_belief_model.pt")
    
    return trained_model


def test_model_consistency():
    """Test that the model makes consistent predictions"""
    
    print("\n=== Testing Model Consistency ===")
    
    # Load the trained model
    model = create_improved_belief_model()
    model.load_state_dict(torch.load("models/improved_belief_model.pt"))
    model.eval()
    
    # Create a game state
    game_state = Game28State()
    game_state.phase = GamePhase.CONCEALED
    game_state.trump_suit = "S"  # Spades
    game_state.bidder = 0
    game_state.winning_bid = 18
    
    # Test multiple predictions on the same state
    predictions1 = model.predict_beliefs(game_state, 0)
    predictions2 = model.predict_beliefs(game_state, 0)
    
    # Check if predictions are consistent
    trump_diff = torch.abs(predictions1.trump_suit - predictions2.trump_suit).max().item()
    uncertainty_diff = abs(predictions1.uncertainty.item() - predictions2.uncertainty.item())
    
    print(f"Trump prediction consistency: {trump_diff:.6f} (should be ~0)")
    print(f"Uncertainty consistency: {uncertainty_diff:.6f} (should be ~0)")
    
    if trump_diff < 1e-5 and uncertainty_diff < 1e-5:
        print("✓ Model predictions are consistent")
    else:
        print("✗ Model predictions are inconsistent")


if __name__ == "__main__":
    import random
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Training and Testing Improved Belief Model")
    print("=" * 50)
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Train and test the model
    model = test_improved_belief_model()
    
    # Test consistency
    test_model_consistency()
    
    print("\n" + "=" * 50)
    print("Improved belief model training and testing completed!")
    print("The model should now make much more realistic predictions.")
