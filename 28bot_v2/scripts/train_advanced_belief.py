#!/usr/bin/env python3
"""
Advanced Belief Model Training Script
Trains the world's best belief model for opponent modeling in 28 card game
"""

import os
import sys
import torch
import numpy as np
from typing import List, Optional

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from belief_model.advanced_trainer import train_advanced_belief_model
from belief_model.advanced_parser import extract_all_game_states
from belief_model.advanced_belief_net import AdvancedBeliefNetwork
from game28.constants import *


def main():
    """Main training function"""
    print("🚀 Starting Advanced Belief Model Training")
    print("=" * 60)
    
    # Configuration
    log_dirs = [
        "../logs/game28/mcts_games",
        "../logs/improved_games"
    ]
    
    # Training parameters
    max_games = 1000  # Use more games for better training
    batch_size = 16   # Smaller batch size for better gradient estimates
    learning_rate = 1e-3
    num_epochs = 50
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model directory if it doesn't exist
    os.makedirs("models/belief_model", exist_ok=True)
    
    # Extract game states
    print("\n📊 Extracting game states from logs...")
    game_states = extract_all_game_states(log_dirs, max_games)
    
    if len(game_states) == 0:
        print("❌ No game states extracted. Check log directories and parsing.")
        return None
    
    print(f"✅ Extracted {len(game_states)} game states")
    
    # Print some statistics
    print("\n📈 Game State Statistics:")
    phases = [state.phase for state in game_states]
    phase_counts = {}
    for phase in phases:
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    
    for phase, count in phase_counts.items():
        print(f"  {phase}: {count} states ({count/len(game_states)*100:.1f}%)")
    
    # Check for trump information
    trump_states = [state for state in game_states if state.trump_suit]
    print(f"  States with trump: {len(trump_states)} ({len(trump_states)/len(game_states)*100:.1f}%)")
    
    # Check for played cards
    states_with_plays = [state for state in game_states if len(state.played_cards) > 0]
    print(f"  States with plays: {len(states_with_plays)} ({len(states_with_plays)/len(game_states)*100:.1f}%)")
    
    # Train the model
    print(f"\n🧠 Training Advanced Belief Model...")
    print(f"  Max games: {max_games}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {num_epochs}")
    
    model = train_advanced_belief_model(
        log_dirs=log_dirs,
        max_games=max_games,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        device=device
    )
    
    if model is None:
        print("❌ Training failed!")
        return None
    
    print("\n✅ Training completed successfully!")
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Test the model on a few examples
    print(f"\n🧪 Testing model on sample data...")
    test_model(model, game_states[:5])
    
    return model


def test_model(model: AdvancedBeliefNetwork, game_states: List):
    """Test the trained model on sample data"""
    model.eval()
    
    print(f"Testing on {len(game_states)} game states...")
    
    for i, game_state in enumerate(game_states):
        print(f"\nGame State {i+1}:")
        print(f"  Phase: {game_state.phase}")
        print(f"  Tricks played: {game_state.tricks_played}")
        print(f"  Cards played: {game_state.cards_played}")
        print(f"  Trump suit: {game_state.trump_suit}")
        print(f"  Game progress: {game_state.game_progress:.2f}")
        
        # Test for each player
        for player_id in range(4):
            if player_id in game_state.hands:
                print(f"\n  Player {player_id} (hand size: {len(game_state.hands[player_id])}):")
                
                try:
                    # Get predictions
                    predictions = model.predict_beliefs(game_state, player_id)
                    
                    # Print trump prediction
                    trump_probs = predictions.trump_suit.cpu().numpy()
                    trump_suit = SUITS[trump_probs.argmax()]
                    trump_confidence = trump_probs.max()
                    print(f"    Trump prediction: {trump_suit} (confidence: {trump_confidence:.3f})")
                    
                    # Print void predictions for opponents
                    for opp_id in range(4):
                        if opp_id != player_id and opp_id in predictions.void_suits:
                            void_probs = predictions.void_suits[opp_id].cpu().numpy()
                            void_suits = [SUITS[j] for j in range(4) if void_probs[j] > 0.5]
                            print(f"    Opponent {opp_id} void suits: {void_suits}")
                    
                    # Print uncertainty
                    uncertainty = predictions.uncertainty.cpu().numpy()[0]
                    print(f"    Prediction uncertainty: {uncertainty:.3f}")
                    
                except Exception as e:
                    print(f"    Error predicting for player {player_id}: {e}")


def evaluate_model_performance(model: AdvancedBeliefNetwork, game_states: List, num_samples: int = 100):
    """Evaluate model performance on test data"""
    print(f"\n📊 Evaluating model performance on {num_samples} samples...")
    
    model.eval()
    
    # Sample game states
    test_states = np.random.choice(game_states, min(num_samples, len(game_states)), replace=False)
    
    hand_accuracies = []
    trump_accuracies = []
    void_accuracies = []
    
    with torch.no_grad():
        for game_state in test_states:
            for player_id in range(4):
                if player_id in game_state.hands:
                    try:
                        predictions = model.predict_beliefs(game_state, player_id)
                        
                        # Hand accuracy (simplified)
                        for opp_id in range(4):
                            if opp_id != player_id and opp_id in game_state.hands:
                                actual_hand = game_state.hands[opp_id]
                                pred_hand = predictions.opponent_hands[opp_id].cpu().numpy()
                                
                                # Calculate accuracy for cards in hand
                                correct = 0
                                total = 0
                                
                                for card in actual_hand:
                                    card_idx = SUITS.index(card.suit) * 8 + RANKS.index(card.rank)
                                    if pred_hand[card_idx] > 0.5:
                                        correct += 1
                                    total += 1
                                
                                if total > 0:
                                    hand_accuracies.append(correct / total)
                        
                        # Trump accuracy
                        if game_state.trump_suit:
                            trump_probs = predictions.trump_suit.cpu().numpy()
                            pred_trump = SUITS[trump_probs.argmax()]
                            trump_accuracies.append(1.0 if pred_trump == game_state.trump_suit else 0.0)
                        
                        # Void accuracy
                        for opp_id in range(4):
                            if opp_id != player_id and opp_id in game_state.hands:
                                actual_hand = game_state.hands[opp_id]
                                pred_void = predictions.void_suits[opp_id].cpu().numpy()
                                
                                for suit_idx, suit in enumerate(SUITS):
                                    has_suit = any(card.suit == suit for card in actual_hand)
                                    pred_void_prob = pred_void[suit_idx]
                                    
                                    if has_suit and pred_void_prob <= 0.5:
                                        void_accuracies.append(1.0)
                                    elif not has_suit and pred_void_prob > 0.5:
                                        void_accuracies.append(1.0)
                                    else:
                                        void_accuracies.append(0.0)
                    
                    except Exception as e:
                        continue
    
    # Print results
    if hand_accuracies:
        print(f"  Hand prediction accuracy: {np.mean(hand_accuracies):.3f} ± {np.std(hand_accuracies):.3f}")
    
    if trump_accuracies:
        print(f"  Trump prediction accuracy: {np.mean(trump_accuracies):.3f} ± {np.std(trump_accuracies):.3f}")
    
    if void_accuracies:
        print(f"  Void suit prediction accuracy: {np.mean(void_accuracies):.3f} ± {np.std(void_accuracies):.3f}")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Train the model
    model = main()
    
    if model is not None:
        print("\n🎉 Advanced Belief Model training completed successfully!")
        print("Model saved to:")
        print("  - models/belief_model/advanced_belief_model_final.pt")
        print("  - models/belief_model/advanced_belief_model_best.pt")
        
        # Load some game states for evaluation
        print("\n🔍 Loading game states for evaluation...")
        game_states = extract_all_game_states(["../logs/game28/mcts_games"], max_games=200)
        
        if game_states:
            evaluate_model_performance(model, game_states)
    else:
        print("\n❌ Training failed!")
        sys.exit(1)
