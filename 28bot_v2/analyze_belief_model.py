#!/usr/bin/env python3
"""
Analyze the belief model performance in detail
"""

import torch
import numpy as np
from belief_model.belief_net import BeliefNetwork
from belief_model.train_beliefs import RealGameDataset
from game28.constants import SUITS, RANKS

def analyze_model_performance():
    """Analyze what the belief model is actually learning"""
    
    # Load the trained model
    device = torch.device("cpu")
    model = BeliefNetwork().to(device)
    model.load_state_dict(torch.load("models/belief_model/belief_model_final.pt", map_location=device))
    model.eval()
    
    # Load some test data
    dataset = RealGameDataset(["../logs/game28/mcts_games"], max_games=20)
    
    print(f"Analyzing {len(dataset)} examples...")
    print("="*60)
    
    # Analyze a few specific examples
    for i in range(min(5, len(dataset))):
        example = dataset[i]
        print(f"\n--- Example {i+1} ---")
        
        # Get model prediction
        input_tensor = torch.tensor(example['input'], dtype=torch.float32).unsqueeze(0)
        player_id = example['player_id']
        
        with torch.no_grad():
            predictions = model(input_tensor, player_id)
        
        # Analyze hand predictions
        print(f"Player {player_id}'s perspective:")
        
        # Show actual vs predicted for each opponent
        for opp_id_str in example['target_hands']:
            opp_id = int(opp_id_str)
            actual_hand = example['target_hands'][opp_id_str]
            predicted_hand = predictions['opponent_hands'][opp_id_str].cpu().numpy()[0]
            
            print(f"\nOpponent {opp_id}:")
            
            # Find cards that are actually in the hand
            actual_cards = []
            predicted_cards = []
            
            for card_idx in range(32):
                suit_idx = card_idx // 8
                rank_idx = card_idx % 8
                card_name = f"{RANKS[rank_idx]}{SUITS[suit_idx]}"
                
                if actual_hand[card_idx] == 1:
                    actual_cards.append((card_name, predicted_hand[card_idx]))
                
                if predicted_hand[card_idx] > 0.5:  # High confidence prediction
                    predicted_cards.append((card_name, predicted_hand[card_idx]))
            
            print(f"  Actual cards: {[card for card, _ in actual_cards]}")
            print(f"  Predicted cards (>0.5): {[f'{card}({prob:.2f})' for card, prob in predicted_cards]}")
            
            # Calculate accuracy for this opponent
            correct_predictions = sum(1 for card_idx in range(32) 
                                    if (actual_hand[card_idx] == 1 and predicted_hand[card_idx] > 0.5) or
                                       (actual_hand[card_idx] == 0 and predicted_hand[card_idx] <= 0.5))
            accuracy = correct_predictions / 32
            
            # Also calculate precision and recall for cards actually in hand
            true_positives = sum(1 for card_idx in range(32) 
                               if actual_hand[card_idx] == 1 and predicted_hand[card_idx] > 0.5)
            false_positives = sum(1 for card_idx in range(32) 
                                if actual_hand[card_idx] == 0 and predicted_hand[card_idx] > 0.5)
            false_negatives = sum(1 for card_idx in range(32) 
                                if actual_hand[card_idx] == 1 and predicted_hand[card_idx] <= 0.5)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            print(f"  Accuracy: {accuracy:.3f} ({correct_predictions}/32)")
            print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}")
            print(f"  True cards found: {true_positives}/{sum(actual_hand)} actual cards")
        
        # Analyze trump prediction
        trump_probs = predictions['trump_suit'].cpu().numpy()[0]
        actual_trump = np.argmax(example['target_trump'])
        predicted_trump = np.argmax(trump_probs)
        
        print(f"\nTrump prediction:")
        print(f"  Actual: {SUITS[actual_trump]}")
        print(f"  Predicted: {SUITS[predicted_trump]} (confidence: {trump_probs[predicted_trump]:.3f})")
        print(f"  All probabilities: {dict(zip(SUITS, trump_probs))}")
        
    print("\n" + "="*60)
    
    # Overall statistics
    total_hand_accuracy = 0
    total_trump_accuracy = 0
    total_examples = 0
    
    for example in dataset:
        input_tensor = torch.tensor(example['input'], dtype=torch.float32).unsqueeze(0)
        player_id = example['player_id']
        
        with torch.no_grad():
            predictions = model(input_tensor, player_id)
        
        # Hand accuracy - count examples, not opponents
        example_accuracy = 0
        num_opponents = 0
        for opp_id_str in example['target_hands']:
            actual_hand = example['target_hands'][opp_id_str]
            predicted_hand = predictions['opponent_hands'][opp_id_str].cpu().numpy()[0]
            
            correct_predictions = sum(1 for card_idx in range(32) 
                                    if (actual_hand[card_idx] == 1 and predicted_hand[card_idx] > 0.5) or
                                       (actual_hand[card_idx] == 0 and predicted_hand[card_idx] <= 0.5))
            example_accuracy += correct_predictions / 32
            num_opponents += 1
        
        total_hand_accuracy += example_accuracy / num_opponents  # Average across opponents for this example
        
        # Trump accuracy
        actual_trump = np.argmax(example['target_trump'])
        predicted_trump = np.argmax(predictions['trump_suit'].cpu().numpy()[0])
        total_trump_accuracy += (actual_trump == predicted_trump)
        
        total_examples += 1
    
    print(f"Overall Performance Summary:")
    print(f"  Average Hand Accuracy: {total_hand_accuracy / total_examples:.3f}")
    print(f"  Trump Accuracy: {total_trump_accuracy / total_examples:.3f}")
    
    # Analyze what makes a good vs bad prediction
    print(f"\nModel Analysis:")
    print(f"  - Hand accuracy of ~{total_hand_accuracy / total_examples:.3f} means the model correctly predicts")
    print(f"    whether each card is in an opponent's hand {100 * total_hand_accuracy / total_examples:.1f}% of the time")
    print(f"  - Trump accuracy of {total_trump_accuracy / total_examples:.3f} means it correctly identifies")
    print(f"    the trump suit {100 * total_trump_accuracy / total_examples:.1f}% of the time (random = 25%)")
    
    # Expected performance analysis
    baseline_hand_accuracy = 24/32  # 24 cards not in hand, 8 in hand - random guess would be ~75% correct
    print(f"  - Baseline hand accuracy (random): ~{baseline_hand_accuracy:.3f}")
    print(f"  - Model improvement: {total_hand_accuracy / total_examples - baseline_hand_accuracy:.3f}")
    
    if total_hand_accuracy / total_examples > baseline_hand_accuracy + 0.05:
        print("  ✅ Model is learning meaningful patterns!")
    else:
        print("  ❌ Model may not be learning much beyond random guessing")

if __name__ == "__main__":
    analyze_model_performance()
