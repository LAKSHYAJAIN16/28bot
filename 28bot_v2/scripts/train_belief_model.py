#!/usr/bin/env python3
"""
Simple script to train the belief model
"""

import sys
import os
import argparse

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from belief_model.train_beliefs import train_belief_model, evaluate_belief_model

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train the belief model")
    parser.add_argument("--num-games", type=int, default=5000,
                       help="Number of games to simulate for training")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=30,
                       help="Number of training epochs")
    parser.add_argument("--hidden-dim", type=int, default=256,
                       help="Hidden layer dimension")
    parser.add_argument("--num-layers", type=int, default=3,
                       help="Number of hidden layers")
    parser.add_argument("--save-dir", type=str, default="models/belief_model",
                       help="Directory to save the model")
    parser.add_argument("--evaluate", action="store_true",
                       help="Evaluate the model after training")
    parser.add_argument("--test-games", type=int, default=500,
                       help="Number of test games for evaluation")
    
    args = parser.parse_args()
    
    print("="*60)
    print("BELIEF MODEL TRAINING")
    print("="*60)
    
    print(f"Training Parameters:")
    print(f"  Number of games: {args.num_games}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Number of epochs: {args.num_epochs}")
    print(f"  Hidden dimension: {args.hidden_dim}")
    print(f"  Number of layers: {args.num_layers}")
    print(f"  Save directory: {args.save_dir}")
    print()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Train the model
    print("Starting training...")
    model = train_belief_model(
        num_games=args.num_games,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        save_dir=args.save_dir
    )
    
    print(f"\nTraining completed! Model saved to {args.save_dir}")
    
    # Evaluate if requested
    if args.evaluate:
        print("\n" + "="*60)
        print("EVALUATING MODEL")
        print("="*60)
        
        model_path = os.path.join(args.save_dir, "belief_model_final.pt")
        
        if os.path.exists(model_path):
            results = evaluate_belief_model(
                model_path,
                num_test_games=args.test_games
            )
            
            print(f"\nEvaluation Results:")
            print(f"  Hand Prediction Accuracy: {results['hand_accuracy']:.3f}")
            print(f"  Trump Prediction Accuracy: {results['trump_accuracy']:.3f}")
        else:
            print(f"Model file not found: {model_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    
    print("\nNext steps:")
    print("1. Use the trained model in your Game 28 bot")
    print("2. Integrate with bidding advisor for better decisions")
    print("3. Combine with MCTS for improved search")
    print("4. Fine-tune parameters if needed")

if __name__ == "__main__":
    main()
