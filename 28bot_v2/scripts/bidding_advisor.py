#!/usr/bin/env python3
"""
Simple bidding advisor using the trained model
"""

import sys
import os
import numpy as np
from typing import List, Dict, Any, Optional

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from rl_bidding.env_adapter import Game28Env
from game28.game_state import Game28State, Card
from game28.constants import BID_RANGE, MIN_BID, MAX_BID, CARD_VALUES, TOTAL_POINTS

class BiddingAdvisor:
    """Simple advisor for bidding decisions using the trained model"""
    
    def __init__(self, model_path: str = "models/bidding_policy/best_model/best_model.zip"):
        """Initialize the bidding advisor"""
        self.model = None
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        """Load the trained model"""
        try:
            self.model = PPO.load(self.model_path)
            print(f"Successfully loaded bidding model from {self.model_path}")
        except Exception as e:
            print(f"Warning: Could not load model from {self.model_path}")
            print(f"Error: {e}")
            print("Will use fallback heuristic bidding")
            self.model = None
    
    def get_bid_suggestion(self, hand: List[Card], current_bid: int, 
                          bid_history: List[tuple], position: int) -> Dict[str, Any]:
        """
        Get a bidding suggestion for the current situation
        
        Args:
            hand: List of cards in hand
            current_bid: Current highest bid
            bid_history: List of (player, bid) tuples
            position: Player position (0-3)
        
        Returns:
            Dictionary with bid suggestion and confidence
        """
        if self.model is not None:
            return self._get_model_suggestion(hand, current_bid, bid_history, position)
        else:
            return self._get_heuristic_suggestion(hand, current_bid, bid_history, position)
    
    def _get_model_suggestion(self, hand: List[Card], current_bid: int, 
                             bid_history: List[tuple], position: int) -> Dict[str, Any]:
        """Get suggestion from the trained model"""
        # Create a temporary environment to get the observation
        env = Game28Env(player_id=position)
        env.game_state.hands[position] = hand
        env.game_state.current_bid = current_bid
        env.game_state.bid_history = bid_history
        env.game_state.current_player = position
        
        # Get observation
        obs = env._get_observation()
        
        # Get model prediction
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Convert action to bid
        if action == len(BID_RANGE):  # Pass
            suggested_bid = -1
        else:
            suggested_bid = BID_RANGE[action]
        
        # Calculate confidence based on hand strength
        hand_strength = self._calculate_hand_strength(hand)
        
        return {
            'suggested_bid': suggested_bid,
            'confidence': hand_strength,
            'method': 'trained_model',
            'hand_strength': hand_strength
        }
    
    def _get_heuristic_suggestion(self, hand: List[Card], current_bid: int, 
                                 bid_history: List[tuple], position: int) -> Dict[str, Any]:
        """Get suggestion using heuristic rules"""
        hand_strength = self._calculate_hand_strength(hand)
        
        # Simple heuristic bidding
        if hand_strength > 0.6:
            # Strong hand - bid aggressively
            if current_bid < MAX_BID - 2:
                suggested_bid = current_bid + 2
            else:
                suggested_bid = -1  # Pass if too high
        elif hand_strength > 0.4:
            # Medium hand - bid moderately
            if current_bid < MAX_BID - 1:
                suggested_bid = current_bid + 1
            else:
                suggested_bid = -1  # Pass
        else:
            # Weak hand - pass
            suggested_bid = -1
        
        return {
            'suggested_bid': suggested_bid,
            'confidence': hand_strength,
            'method': 'heuristic',
            'hand_strength': hand_strength
        }
    
    def _calculate_hand_strength(self, hand: List[Card]) -> float:
        """Calculate hand strength for bidding"""
        total_points = sum(CARD_VALUES[card.rank] for card in hand)
        return total_points / TOTAL_POINTS
    
    def format_suggestion(self, suggestion: Dict[str, Any]) -> str:
        """Format the suggestion for display"""
        if suggestion['suggested_bid'] == -1:
            bid_text = "PASS"
        else:
            bid_text = f"bid {suggestion['suggested_bid']}"
        
        confidence_text = f"{suggestion['confidence']:.2f}"
        method_text = suggestion['method']
        
        return f"Suggested: {bid_text} (confidence: {confidence_text}, method: {method_text})"

def main():
    """Example usage of the bidding advisor"""
    advisor = BiddingAdvisor()
    
    # Example hand
    from game28.game_state import Card
    example_hand = [
        Card('C', '7'), Card('D', '7'), Card('H', '7'), Card('S', '7')
    ]
    
    print("Bidding Advisor Example")
    print("=" * 30)
    print(f"Hand: {[str(card) for card in example_hand]}")
    print(f"Current bid: 16")
    print(f"Position: 0")
    
    suggestion = advisor.get_bid_suggestion(
        hand=example_hand,
        current_bid=18,
        bid_history=[],
        position=0
    )
    
    print(f"\n{advisor.format_suggestion(suggestion)}")

if __name__ == "__main__":
    main()
