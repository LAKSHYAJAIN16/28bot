#!/usr/bin/env python3
"""
Simple script to use the point prediction model for bidding decisions
"""

import sys
import os
import argparse

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from point_prediction_model import PointPredictionTrainer, HandCanonicalizer

def test_canonicalization():
    """Test the hand canonicalization"""
    print("Testing Hand Canonicalization")
    print("="*40)
    
    canonicalizer = HandCanonicalizer()
    
    # Test equivalent hands
    hand1 = ['JC', '9C', 'AC', '10C']
    hand2 = ['JH', '9H', 'AH', '10H']
    
    canonical1, mapping1 = canonicalizer.canonicalize_hand(hand1)
    canonical2, mapping2 = canonicalizer.canonicalize_hand(hand2)
    
    print(f"Hand 1: {hand1}")
    print(f"Canonical 1: {canonical1}")
    print(f"Hand 2: {hand2}")
    print(f"Canonical 2: {canonical2}")
    print(f"Equivalent: {canonical1 == canonical2}")
    print()

def test_point_predictions():
    """Test point predictions for various hands"""
    print("Testing Point Predictions")
    print("="*40)
    
    # Create trainer and load model
    trainer = PointPredictionTrainer()
    
    # Try to load existing model
    trainer.load_model()
    
    # Test hands
    test_hands = [
        (['JC', '9C', 'AC', '10C'], "Strong clubs"),
        (['JH', '9H', 'AH', '10H'], "Strong hearts (should be same as clubs)"),
        (['7H', '8H', '9H', '10H'], "Weak hearts"),
        (['AS', 'KS', 'QS', 'JS'], "Very strong spades"),
        (['7D', '8C', '9H', '10S'], "Mixed weak"),
        (['AD', 'KD', 'QD', 'JD'], "Very strong diamonds"),
        (['7C', '8C', '9C', '10C'], "Weak clubs"),
        (['AC', 'KC', 'QC', 'JC'], "Very strong clubs"),
    ]
    
    for hand, description in test_hands:
        try:
            predicted_points = trainer.predict_points(hand)
            print(f"{description:25} {hand}: {predicted_points:6.2f} points")
        except Exception as e:
            print(f"{description:25} {hand}: Error - {e}")
    
    print()

def interactive_mode():
    """Interactive mode for testing hands"""
    print("Interactive Point Prediction Mode")
    print("="*40)
    print("Enter 4-card hands to get point predictions")
    print("Format: JC 9C AC 10C (space-separated)")
    print("Type 'quit' to exit")
    print()
    
    trainer = PointPredictionTrainer()
    trainer.load_model()
    
    while True:
        try:
            hand_input = input("Enter hand: ").strip()
            
            if hand_input.lower() == 'quit':
                break
            
            # Parse hand
            cards = hand_input.split()
            if len(cards) != 4:
                print("Please enter exactly 4 cards")
                continue
            
            # Predict points
            predicted_points = trainer.predict_points(cards)
            print(f"Predicted points: {predicted_points:.2f}")
            
            # Show canonical form
            canonicalizer = HandCanonicalizer()
            canonical_hand, _ = canonicalizer.canonicalize_hand(cards)
            print(f"Canonical form: {canonical_hand}")
            
            # Show features
            features = canonicalizer.get_hand_features(cards)
            print(f"Features: {features['total_points']} total points, "
                  f"{features['longest_suit_count']} cards in longest suit ({features['longest_suit']})")
            print()
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            print()

def compare_hands():
    """Compare multiple hands"""
    print("Hand Comparison")
    print("="*40)
    
    trainer = PointPredictionTrainer()
    trainer.load_model()
    
    hands_to_compare = [
        (['JC', '9C', 'AC', '10C'], "Strong clubs"),
        (['AS', 'KS', 'QS', 'JS'], "Strong spades"),
        (['AD', 'KD', 'QD', 'JD'], "Strong diamonds"),
        (['AH', 'KH', 'QH', 'JH'], "Strong hearts"),
        (['7C', '8C', '9C', '10C'], "Weak clubs"),
        (['7H', '8H', '9H', '10H'], "Weak hearts"),
    ]
    
    # Get predictions
    predictions = []
    for hand, description in hands_to_compare:
        try:
            predicted_points = trainer.predict_points(hand)
            predictions.append((description, hand, predicted_points))
        except Exception as e:
            print(f"Error with {description}: {e}")
    
    # Sort by predicted points
    predictions.sort(key=lambda x: x[2], reverse=True)
    
    print("Hands ranked by predicted points:")
    for i, (description, hand, points) in enumerate(predictions, 1):
        print(f"{i:2d}. {description:15} {hand}: {points:6.2f} points")
    
    print()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Point Prediction Model Usage")
    parser.add_argument('--mode', choices=['test', 'interactive', 'compare', 'canonical'], 
                       default='test', help='Mode to run')
    parser.add_argument('--train', action='store_true', help='Train the model first')
    
    args = parser.parse_args()
    
    if args.train:
        print("Training point prediction model...")
        trainer = PointPredictionTrainer()
        trainer.train(epochs=100)  # Quick training
        trainer.save_model()
        print("Training completed!")
        print()
    
    if args.mode == 'canonical':
        test_canonicalization()
    elif args.mode == 'test':
        test_point_predictions()
    elif args.mode == 'interactive':
        interactive_mode()
    elif args.mode == 'compare':
        compare_hands()

if __name__ == "__main__":
    main()
