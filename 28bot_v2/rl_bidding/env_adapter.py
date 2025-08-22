"""
RL Environment adapter for Game 28 bidding
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import torch

from game28.game_state import Game28State, Card
from game28.constants import *


class Game28Env(gym.Env):
    """
    Gymnasium environment for Game 28 bidding
    """
    
    def __init__(self, player_id: int = 0):
        super().__init__()
        self.player_id = player_id
        self.game_state = None
        self.reset()
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(BID_RANGE) + 1)  # Bids + pass
        
        # Observation space: hand (32 cards) + bidding history + game state
        self.observation_space = spaces.Dict({
            'hand': spaces.MultiBinary(32),  # One-hot encoding of cards in hand
            'bidding_history': spaces.Box(low=0, high=28, shape=(4,), dtype=np.int32),
            'current_bid': spaces.Discrete(29),
            'position': spaces.Discrete(4),
            'phase': spaces.Discrete(3),  # Bidding, concealed, revealed
            'trump_suit': spaces.Discrete(5),  # 4 suits + None
            'trump_revealed': spaces.Discrete(2),
            'bidder': spaces.Discrete(5),  # 4 players + None
            'winning_bid': spaces.Discrete(29),
            'team_scores': spaces.Box(low=0, high=28, shape=(2,), dtype=np.int32),
            'game_points': spaces.Box(low=-10, high=10, shape=(2,), dtype=np.int32),
            'legal_actions': spaces.MultiBinary(len(BID_RANGE) + 1)
        })
    
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment"""
        super().reset(seed=seed)
        self.game_state = Game28State()
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment"""
        # Convert action to bid
        if action == len(BID_RANGE):  # Pass
            bid = -1
        else:
            bid = BID_RANGE[action]
        
        # Make the bid
        bidding_continues = self.game_state.make_bid(self.player_id, bid)
        
        # Get reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self.game_state.game_over or not bidding_continues
        
        # Get observation
        observation = self._get_observation()
        
        # Add debug info
        info = {
            'bidding_continues': bidding_continues,
            'game_over': self.game_state.game_over,
            'current_bid': self.game_state.current_bid,
            'passed_players': len(self.game_state.passed_players)
        }
        
        return observation, reward, done, False, info
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get the current observation"""
        obs = self.game_state.get_observation(self.player_id)
        
        # Encode hand as one-hot vector
        hand_encoding = self._encode_hand(obs['hand'])
        
        # Encode bidding history
        bidding_history = np.zeros(4, dtype=np.int32)
        for i, (player, bid) in enumerate(obs['bid_history'][-4:]):
            bidding_history[i] = bid if bid != -1 else 0
        
        # Encode trump suit
        trump_suit_idx = 4  # None
        if obs['trump_suit']:
            trump_suit_idx = SUITS.index(obs['trump_suit'])
        
        # Encode bidder
        bidder_idx = 4  # None
        if obs['bidder'] is not None:
            bidder_idx = obs['bidder']
        
        # Get legal actions
        legal_actions = np.zeros(len(BID_RANGE) + 1, dtype=np.int32)
        legal_bids = self.game_state.get_legal_bids(self.player_id)
        for bid in legal_bids:
            if bid == -1:
                legal_actions[-1] = 1
            else:
                legal_actions[BID_RANGE.index(bid)] = 1
        
        return {
            'hand': hand_encoding,
            'bidding_history': bidding_history,
            'current_bid': obs['current_bid'],
            'position': self.player_id,
            'phase': list(GamePhase).index(obs['phase']),
            'trump_suit': trump_suit_idx,
            'trump_revealed': int(obs['trump_revealed']),
            'bidder': bidder_idx,
            'winning_bid': obs['winning_bid'] if obs['winning_bid'] else 0,
            'team_scores': np.array([obs['team_scores']['A'], obs['team_scores']['B']], dtype=np.int32),
            'game_points': np.array([obs['game_points']['A'], obs['game_points']['B']], dtype=np.int32),
            'legal_actions': legal_actions
        }
    
    def _encode_hand(self, hand: List[Card]) -> np.ndarray:
        """Encode hand as one-hot vector"""
        encoding = np.zeros(32, dtype=np.int32)
        for card in hand:
            card_idx = SUITS.index(card.suit) * 8 + RANKS.index(card.rank)
            encoding[card_idx] = 1
        return encoding
    
    def _calculate_reward(self) -> float:
        """Calculate reward for the current state"""
        if not self.game_state.game_over:
            return 0.0
        
        # Game is over, calculate final reward
        team = 'A' if self.player_id in TEAM_A else 'B'
        game_point = self.game_state.game_points[team]
        
        # Reward based on game point
        if game_point > 0:
            return 1.0
        elif game_point < 0:
            return -1.0
        else:
            return 0.0
    
    def render(self):
        """Render the current state"""
        print(f"Player {self.player_id} hand: {[str(card) for card in self.game_state.hands[self.player_id]]}")
        print(f"Current bid: {self.game_state.current_bid}")
        print(f"Bid history: {self.game_state.bid_history}")
        print(f"Phase: {self.game_state.phase}")
        if self.game_state.trump_suit:
            print(f"Trump: {self.game_state.trump_suit} (revealed: {self.game_state.trump_revealed})")
        print(f"Team scores: {self.game_state.team_scores}")
        print(f"Game points: {self.game_state.game_points}")


class Game28PlayEnv(gym.Env):
    """
    Gymnasium environment for Game 28 play (after bidding)
    """
    
    def __init__(self, player_id: int = 0):
        super().__init__()
        self.player_id = player_id
        self.game_state = None
        self.reset()
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(32)  # All possible cards
        
        # Observation space: hand + current trick + game state
        self.observation_space = spaces.Dict({
            'hand': spaces.MultiBinary(32),
            'current_trick': spaces.Box(low=0, high=4, shape=(4, 32), dtype=np.int32),
            'trick_leader': spaces.Discrete(4),
            'trump_suit': spaces.Discrete(5),
            'trump_revealed': spaces.Discrete(2),
            'phase': spaces.Discrete(3),
            'team_scores': spaces.Box(low=0, high=28, shape=(2,), dtype=np.int32),
            'legal_actions': spaces.MultiBinary(32)
        })
    
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment"""
        super().reset(seed=seed)
        self.game_state = Game28State()
        
        # Simulate bidding to get to play phase
        self._simulate_bidding()
        
        return self._get_observation(), {}
    
    def _simulate_bidding(self):
        """Simulate bidding to get to play phase"""
        # Simple bidding simulation - can be improved
        current_bid = MIN_BID
        for player in range(4):
            if player == self.player_id:
                continue
            
            # Simple heuristic bidding
            hand_strength = self._calculate_hand_strength(self.game_state.hands[player])
            if hand_strength > 0.6:
                current_bid = min(MAX_BID, current_bid + 2)
            elif hand_strength > 0.4:
                current_bid = min(MAX_BID, current_bid + 1)
            else:
                self.game_state.passed_players.append(player)
        
        # Set bidder and trump
        if len(self.game_state.passed_players) < 3:
            self.game_state.bidder = 0  # Assume player 0 wins
            self.game_state.winning_bid = current_bid
            self.game_state.set_trump('H')  # Assume hearts as trump
        else:
            self.game_state.game_over = True
    
    def _calculate_hand_strength(self, hand: List[Card]) -> float:
        """Calculate hand strength for bidding"""
        total_points = sum(CARD_VALUES[card.rank] for card in hand)
        return total_points / TOTAL_POINTS
    
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment"""
        # Convert action to card
        card = self._action_to_card(action)
        
        # Play the card
        self.game_state.play_card(self.player_id, card)
        
        # Get reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self.game_state.game_over
        
        # Get observation
        observation = self._get_observation()
        
        return observation, reward, done, False, {}
    
    def _action_to_card(self, action: int) -> Card:
        """Convert action index to card"""
        suit_idx = action // 8
        rank_idx = action % 8
        return Card(SUITS[suit_idx], RANKS[rank_idx])
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get the current observation"""
        obs = self.game_state.get_observation(self.player_id)
        
        # Encode hand
        hand_encoding = self._encode_hand(obs['hand'])
        
        # Encode current trick
        current_trick = np.zeros((4, 32), dtype=np.int32)
        for i, (player, card) in enumerate(obs['current_trick'].cards):
            card_idx = SUITS.index(card.suit) * 8 + RANKS.index(card.rank)
            current_trick[i, card_idx] = 1
        
        # Encode trump suit
        trump_suit_idx = 4  # None
        if obs['trump_suit']:
            trump_suit_idx = SUITS.index(obs['trump_suit'])
        
        # Get legal actions
        legal_actions = np.zeros(32, dtype=np.int32)
        legal_cards = self.game_state.get_legal_plays(self.player_id)
        for card in legal_cards:
            card_idx = SUITS.index(card.suit) * 8 + RANKS.index(card.rank)
            legal_actions[card_idx] = 1
        
        return {
            'hand': hand_encoding,
            'current_trick': current_trick,
            'trick_leader': obs['trick_leader'],
            'trump_suit': trump_suit_idx,
            'trump_revealed': int(obs['trump_revealed']),
            'phase': list(GamePhase).index(obs['phase']),
            'team_scores': np.array([obs['team_scores']['A'], obs['team_scores']['B']], dtype=np.int32),
            'legal_actions': legal_actions
        }
    
    def _encode_hand(self, hand: List[Card]) -> np.ndarray:
        """Encode hand as one-hot vector"""
        encoding = np.zeros(32, dtype=np.int32)
        for card in hand:
            card_idx = SUITS.index(card.suit) * 8 + RANKS.index(card.rank)
            encoding[card_idx] = 1
        return encoding
    
    def _calculate_reward(self) -> float:
        """Calculate reward for the current state"""
        if not self.game_state.game_over:
            return 0.0
        
        # Game is over, calculate final reward
        team = 'A' if self.player_id in TEAM_A else 'B'
        game_point = self.game_state.game_points[team]
        
        # Reward based on game point
        if game_point > 0:
            return 1.0
        elif game_point < 0:
            return -1.0
        else:
            return 0.0
