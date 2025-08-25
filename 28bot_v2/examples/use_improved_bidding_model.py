#!/usr/bin/env python3
"""
Script to use the improved bidding model
"""

import sys
import os
import numpy as np
from typing import List, Dict, Any

# Add the parent directory to Python path to access scripts
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from rl_bidding.env_adapter import Game28Env
from scripts.improved_bidding_trainer import ImprovedBiddingTrainer
from game28.game_state import Game28State
from game28.constants import BID_RANGE, MIN_BID, MAX_BID

class ImprovedBiddingModel:
    """Wrapper for the improved bidding model"""
    
    def __init__(self, model_path: str = "models/improved_bidding_model.zip"):
        self.model_path = model_path
        self.model = None
        self.trainer = ImprovedBiddingTrainer()
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = PPO.load(self.model_path)
            print(f"Successfully loaded improved bidding model from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Make sure the model has been trained first using improved_bidding_trainer.py")
            self.model = None
    
    def get_bid_suggestion(self, hand: List[str], current_bid: int, 
                          bid_history: List[tuple], position: int) -> int:
        """
        Get a bid suggestion using the improved model
        
        Args:
            hand: List of card strings (e.g., ['H7', 'D9', 'CJ', ...])
            current_bid: Current highest bid
            bid_history: List of (player, bid) tuples
            position: Player position (0-3)
        
        Returns:
            Suggested bid value or -1 for pass
        """
        if self.model is None:
            print("Model not loaded. Using fallback strategy.")
            return self._get_fallback_bid(hand, current_bid)
        
        # Create environment with current state
        env = self.trainer.create_improved_environment()
        
        # Set up the game state
        env.game_state.current_bid = current_bid
        env.game_state.bid_history = bid_history
        env.player_id = position
        
        # Convert hand strings to Card objects
        from game28.game_state import Card
        card_objects = []
        for card_str in hand:
            suit = card_str[0]  # First character is suit
            rank = card_str[1:]  # Rest is rank
            card_objects.append(Card(suit, rank))
        
        env.game_state.hands[position] = card_objects
        
        # Get observation
        obs, _ = env.reset()
        
        # Get model prediction
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Convert action to bid
        if action == len(BID_RANGE):  # Pass
            return -1
        else:
            return BID_RANGE[action]
    
    def _get_fallback_bid(self, hand: List[str], current_bid: int) -> int:
        print("Using fallback bidding strategy")
        """Fallback bidding strategy if model is not available"""
        # Simple heuristic based on hand strength
        from game28.constants import CARD_VALUES, TOTAL_POINTS
        
        # Only consider first 4 cards for bidding decisions
        bidding_cards = hand[:4] if len(hand) >= 4 else hand
        total_points = 0
        for card_str in bidding_cards:
            rank = card_str[0]  # Get rank (first character)
            total_points += CARD_VALUES[rank]
        
        hand_strength = total_points / TOTAL_POINTS
        
        if hand_strength > 0.6:
            return min(current_bid + 2, MAX_BID)
        elif hand_strength > 0.4:
            return min(current_bid + 1, MAX_BID)
        else:
            return -1  # Pass
    
    def evaluate_model(self, num_games: int = 50):
        """Evaluate the model performance"""
        if self.model is None:
            print("Model not loaded. Cannot evaluate.")
            return
        
        print(f"Evaluating improved bidding model over {num_games} games...")
        
        env = self.trainer.create_improved_environment()
        wins = 0
        total_reward = 0
        
        for game in range(num_games):
            obs, _ = env.reset()
            done = False
            game_reward = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                game_reward += reward
            
            if game_reward > 0:
                wins += 1
            total_reward += game_reward
        
        win_rate = wins / num_games
        avg_reward = total_reward / num_games
        
        print(f"Evaluation Results:")
        print(f"Win rate: {win_rate:.3f} ({wins}/{num_games})")
        print(f"Average reward: {avg_reward:.3f}")

def main():
    """Main function to demonstrate model usage"""
    print("="*60)
    print("IMPROVED BIDDING MODEL USAGE")
    print("="*60)
    
    # Create model wrapper
    bidding_model = ImprovedBiddingModel()
    
    if bidding_model.model is None:
        print("\nModel not available. Please train the model first:")
        print("python improved_bidding_trainer.py")
        return
    
    # Sample hand and game state
    hand = ['H7', 'S7', 'D7', 'C7']
    current_bid = 16
    bid_history = []  # Player 1 bid 16, Player 2 bid 17, Player 3 bid 18
    position = 0  # We are player 0
    
    print(f"Hand: {hand}")
    print(f"Current bid: {current_bid}")
    print(f"Bid history: {bid_history}")
    print(f"Position: {position}")
    
    suggestion = bidding_model.get_bid_suggestion(hand, current_bid, bid_history, position)
    
    if suggestion == -1:
        print(f"Model suggestion: PASS")
    else:
        print(f"Model suggestion: {suggestion}")
    
    #

if __name__ == "__main__":
    main()
