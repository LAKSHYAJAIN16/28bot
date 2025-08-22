#!/usr/bin/env python3
"""
Script to demonstrate how to use the trained bidding model
"""

import sys
import os
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from rl_bidding.env_adapter import Game28Env
from game28.constants import BID_RANGE, MIN_BID, MAX_BID

def load_bidding_model(model_path: str = "models/bidding_policy/best_model/best_model.zip"):
    """Load the trained bidding model"""
    try:
        model = PPO.load(model_path)
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have trained the model first using run_training.py")
        return None

def get_bid_from_model(model, observation):
    """Get a bid decision from the model"""
    action, _ = model.predict(observation, deterministic=True)
    
    # Convert action to bid
    if action == len(BID_RANGE):  # Pass
        return -1
    else:
        return BID_RANGE[action]

def play_single_game_with_model(model_path: str = "models/bidding_policy/best_model/best_model.zip"):
    """Play a single game using the trained model"""
    # Load the model
    model = load_bidding_model(model_path)
    if model is None:
        return
    
    # Create environment
    env = Game28Env(player_id=0)
    
    print("Starting a game with the trained bidding model...")
    print("=" * 50)
    
    # Reset environment
    obs, _ = env.reset()
    step = 0
    total_reward = 0
    
    while True:
        step += 1
        
        # Get current game state info
        current_bid = env.game_state.current_bid
        passed_players = len(env.game_state.passed_players)
        
        print(f"\nStep {step}:")
        print(f"Current bid: {current_bid}")
        print(f"Players passed: {passed_players}")
        print(f"Your hand: {[str(card) for card in env.game_state.hands[0]]}")
        
        # Get model's decision
        model_action = model.predict(obs, deterministic=True)[0]
        model_bid = get_bid_from_model(model, obs)
        
        if model_bid == -1:
            print(f"Model decides to: PASS")
        else:
            print(f"Model decides to bid: {model_bid}")
        
        # Take the action
        obs, reward, done, truncated, info = env.step(model_action)
        total_reward += reward
        
        print(f"Reward: {reward}")
        print(f"Bidding continues: {info['bidding_continues']}")
        
        if done:
            print("\n" + "=" * 50)
            print("Game finished!")
            print(f"Final reward: {total_reward}")
            
            if env.game_state.bidder is not None:
                print(f"Winner: Player {env.game_state.bidder} with bid {env.game_state.winning_bid}")
                if env.game_state.trump_suit:
                    print(f"Trump suit: {env.game_state.trump_suit}")
            else:
                print("No winner - all players passed")
            
            break

def evaluate_model_performance(model_path: str = "models/bidding_policy/best_model/best_model.zip", num_games: int = 100):
    """Evaluate the model's performance over multiple games"""
    # Load the model
    model = load_bidding_model(model_path)
    if model is None:
        return
    
    # Create environment
    env = Game28Env(player_id=0)
    
    print(f"Evaluating model over {num_games} games...")
    
    wins = 0
    total_reward = 0
    successful_bids = 0
    total_bids = 0
    
    for game in range(num_games):
        obs, _ = env.reset()
        done = False
        game_reward = 0
        
        while not done:
            # Get model's decision
            model_action = model.predict(obs, deterministic=True)[0]
            model_bid = get_bid_from_model(model, obs)
            
            # Take the action
            obs, reward, done, truncated, info = env.step(model_action)
            game_reward += reward
            
            # Count bids
            if model_bid != -1:
                total_bids += 1
        
        # Record results
        if game_reward > 0:
            wins += 1
        total_reward += game_reward
        
        # Check if we won the bidding
        if env.game_state.bidder == 0 and env.game_state.winning_bid is not None:
            successful_bids += 1
    
    # Calculate statistics
    win_rate = wins / num_games
    avg_reward = total_reward / num_games
    bid_success_rate = successful_bids / max(total_bids, 1)
    
    print(f"\nEvaluation Results:")
    print(f"Win rate: {win_rate:.3f} ({wins}/{num_games})")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Bid success rate: {bid_success_rate:.3f}")

def interactive_bidding(model_path: str = "models/bidding_policy/best_model/best_model.zip"):
    """Interactive mode where you can see the model's suggestions"""
    # Load the model
    model = load_bidding_model(model_path)
    if model is None:
        return
    
    # Create environment
    env = Game28Env(player_id=0)
    
    print("Interactive bidding mode")
    print("The model will suggest bids, and you can choose to follow or override")
    print("Enter 'q' to quit, 's' to see model suggestion, 'b <bid>' to make a bid, 'p' to pass")
    
    while True:
        obs, _ = env.reset()
        done = False
        
        print(f"\nNew game started!")
        print(f"Your hand: {[str(card) for card in env.game_state.hands[0]]}")
        
        while not done:
            current_bid = env.game_state.current_bid
            print(f"\nCurrent bid: {current_bid}")
            
            # Get model suggestion
            model_bid = get_bid_from_model(model, obs)
            if model_bid == -1:
                print(f"Model suggests: PASS")
            else:
                print(f"Model suggests: bid {model_bid}")
            
            # Get user input
            user_input = input("Your action (s/b <bid>/p/q): ").strip().lower()
            
            if user_input == 'q':
                print("Quitting...")
                return
            elif user_input == 's':
                print(f"Model suggests: {model_bid if model_bid != -1 else 'PASS'}")
                continue
            elif user_input == 'p':
                action = len(BID_RANGE)  # Pass
            elif user_input.startswith('b '):
                try:
                    bid = int(user_input[2:])
                    if bid in BID_RANGE:
                        action = BID_RANGE.index(bid)
                    else:
                        print(f"Invalid bid. Must be between {MIN_BID} and {MAX_BID}")
                        continue
                except ValueError:
                    print("Invalid input. Use 'b <number>' to bid")
                    continue
            else:
                print("Invalid input. Use 's' for suggestion, 'b <bid>' to bid, 'p' to pass, 'q' to quit")
                continue
            
            # Take the action
            obs, reward, done, truncated, info = env.step(action)
            print(f"Action taken. Reward: {reward}")
            
            if done:
                print("Game finished!")
                if env.game_state.bidder is not None:
                    print(f"Winner: Player {env.game_state.bidder} with bid {env.game_state.winning_bid}")
                break

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Use the trained bidding model")
    parser.add_argument("--mode", choices=["play", "evaluate", "interactive"], 
                       default="play", help="Mode to run")
    parser.add_argument("--model", default="models/bidding_policy/best_model/best_model.zip",
                       help="Path to the model file")
    parser.add_argument("--games", type=int, default=100,
                       help="Number of games for evaluation")
    
    args = parser.parse_args()
    
    if args.mode == "play":
        play_single_game_with_model(args.model)
    elif args.mode == "evaluate":
        evaluate_model_performance(args.model, args.games)
    elif args.mode == "interactive":
        interactive_bidding(args.model)
