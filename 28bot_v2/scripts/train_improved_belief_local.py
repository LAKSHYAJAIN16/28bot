#!/usr/bin/env python3
"""
Local training script for the improved belief model
Works on both CPU and GPU with backwards compatibility
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Tuple, Dict
import argparse

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from belief_model.improved_belief_net import create_improved_belief_model, train_improved_belief_model
from belief_model.advanced_parser import extract_all_game_states
from train_improved_belief import build_training_data_from_logs


def setup_device():
    """Setup device with backwards compatibility"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def train_with_epoch_saving_local(model, training_data, epochs, learning_rate, use_amp=True, save_dir="models"):
    """Local training function with epoch saving and backwards compatibility"""
    device = next(model.parameters()).device
    print(f"Training on device: {device}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss()
    
    # Backwards compatible AMP setup
    if use_amp and device.type == 'cuda':
        try:
            # Try new API first (PyTorch 2.0+)
            scaler = torch.amp.GradScaler('cuda')
            print("Using PyTorch 2.0+ AMP API")
        except:
            # Fallback to old API (PyTorch 1.x)
            scaler = torch.cuda.amp.GradScaler()
            print("Using PyTorch 1.x AMP API")
    else:
        scaler = None
        print("AMP disabled (CPU training or not available)")
    
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        batch_count = 0
        
        for game_state, player_id, target_beliefs in training_data:
            try:
                optimizer.zero_grad()
                
                # Forward pass with backwards compatible autocast
                if use_amp and device.type == 'cuda':
                    try:
                        # Try new API first (PyTorch 2.0+)
                        with torch.amp.autocast('cuda'):
                            predictions = model(game_state, player_id)
                            loss = _calculate_loss(predictions, target_beliefs, criterion, device)
                    except:
                        # Fallback to old API (PyTorch 1.x)
                        with torch.cuda.amp.autocast():
                            predictions = model(game_state, player_id)
                            loss = _calculate_loss(predictions, target_beliefs, criterion, device)
                else:
                    # CPU training
                    predictions = model(game_state, player_id)
                    loss = _calculate_loss(predictions, target_beliefs, criterion, device)
                
                # Backward pass
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
            except Exception as e:
                print(f"Error in batch {batch_count}: {e}")
                continue
        
        if batch_count > 0:
            avg_loss = total_loss / batch_count
        else:
            avg_loss = 0.0
        
        # Save model after each epoch
        epoch_save_path = os.path.join(save_dir, f"improved_belief_model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), epoch_save_path)
        
        if epoch % 5 == 0 or epoch == epochs - 1:  # Print every 5 epochs and final epoch
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Saved: {epoch_save_path}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model


def _calculate_loss(predictions, target_beliefs, criterion, device):
    """Calculate loss with proper device handling"""
    loss = 0.0

    # Hand prediction loss
    for opp_id, target_hand in target_beliefs.get('hands', {}).items():
        if opp_id in predictions.opponent_hands:
            pred_hand = predictions.opponent_hands[opp_id]
            target_tensor = torch.tensor(target_hand, dtype=torch.float32, device=device)
            loss += criterion(pred_hand, target_tensor)

    # Trump prediction loss
    if 'trump' in target_beliefs:
        target_trump = torch.tensor(target_beliefs['trump'], dtype=torch.float32, device=device)
        loss += criterion(predictions.trump_suit, target_trump)

    # Void prediction loss
    for opp_id, target_void in target_beliefs.get('voids', {}).items():
        if opp_id in predictions.void_suits:
            pred_void = predictions.void_suits[opp_id]
            target_tensor = torch.tensor(target_void, dtype=torch.float32, device=device)
            loss += criterion(pred_void, target_tensor)

    return loss


def main():
    parser = argparse.ArgumentParser(description='Train improved belief model locally')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--no-amp', action='store_true', help='Disable AMP (mixed precision)')
    parser.add_argument('--save-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--logs-dir', type=str, default='../logs', help='Directory containing game logs')
    
    args = parser.parse_args()
    
    # Setup device
    device = setup_device()
    
    # Load training data
    print("Loading training data...")
    log_dirs = [
        os.path.join(args.logs_dir, "game28", "mcts_games"),
        os.path.join(args.logs_dir, "improved_games"),
    ]
    
    # Check if log directories exist
    existing_dirs = []
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            existing_dirs.append(log_dir)
        else:
            print(f"Warning: Log directory not found: {log_dir}")
    
    if not existing_dirs:
        print("Error: No log directories found!")
        return
    
    parsed = extract_all_game_states(existing_dirs)
    training = build_training_data_from_logs(parsed)
    print(f"Training data loaded: {len(training)} samples")
    
    # Create model
    print("Creating model...")
    model = create_improved_belief_model()
    model = model.to(device)
    print(f"Model created on device: {next(model.parameters()).device}")
    
    # Train model
    print(f"\n=== STARTING TRAINING ===")
    print(f"Training on {len(training)} samples for {args.epochs} epochs")
    print(f"Using device: {device}")
    print(f"AMP enabled: {not args.no_amp}")
    
    model = train_with_epoch_saving_local(
        model,
        training,
        epochs=args.epochs,
        learning_rate=args.lr,
        use_amp=not args.no_amp,
        save_dir=args.save_dir
    )
    
    # Save final model
    final_save_path = os.path.join(args.save_dir, "improved_belief_model_final.pt")
    torch.save(model.state_dict(), final_save_path)
    print(f"Final model saved: {final_save_path}")
    
    # Also save as the default name for compatibility
    default_save_path = os.path.join(args.save_dir, "improved_belief_model.pt")
    torch.save(model.state_dict(), default_save_path)
    print(f"Default model saved: {default_save_path}")
    
    # List all saved models
    print("\nAll saved models:")
    for file in sorted(os.listdir(args.save_dir)):
        if file.startswith("improved_belief_model"):
            file_path = os.path.join(args.save_dir, file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"  {file} ({file_size:.2f} MB)")
    
    print("\n✅ Training completed successfully!")


if __name__ == "__main__":
    main()
