"""
Information Set Monte Carlo Tree Search for bidding decisions
"""

import math
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from ..game28.game_state import Game28State, Card
from ..game28.constants import *
from ..belief_model.belief_net import BeliefNetwork, BeliefState


@dataclass
class ISMCTSNode:
    """Node in the ISMCTS tree"""
    state: Game28State
    parent: Optional['ISMCTSNode'] = None
    action: Optional[int] = None  # The action that led to this node
    children: Dict[int, 'ISMCTSNode'] = None
    visits: int = 0
    value: float = 0.0
    player: int = 0
    
    def __post_init__(self):
        if self.children is None:
            self.children = {}


class ISMCTSBidding:
    """
    Information Set Monte Carlo Tree Search for bidding decisions
    """
    
    def __init__(self, belief_network: Optional[BeliefNetwork] = None, num_simulations: int = 1000):
        self.belief_network = belief_network
        self.num_simulations = num_simulations
        self.exploration_constant = math.sqrt(2)
    
    def select_action(self, game_state: Game28State, player_id: int) -> int:
        """
        Select the best bidding action using ISMCTS
        
        Args:
            game_state: Current game state
            player_id: ID of the player making the decision
            
        Returns:
            Best action (bid value or -1 for pass)
        """
        # Create root node
        root = ISMCTSNode(state=game_state.copy(), player=player_id)
        
        # Run simulations
        for _ in range(self.num_simulations):
            # Sample a determinized state
            determinized_state = self._sample_determinized_state(game_state, player_id)
            
            # Run UCT from root
            leaf = self._select(root, determinized_state)
            
            # Expand and simulate
            if leaf.visits > 0:
                leaf = self._expand(leaf, determinized_state)
            
            # Simulate to completion
            value = self._simulate(leaf.state, determinized_state)
            
            # Backpropagate
            self._backpropagate(leaf, value, player_id)
        
        # Select best action
        best_action = self._select_best_action(root)
        return best_action
    
    def _sample_determinized_state(self, game_state: Game28State, player_id: int) -> Game28State:
        """Sample a determinized state using belief network"""
        determinized_state = game_state.copy()
        
        if self.belief_network:
            # Use belief network to sample opponent hands
            belief_state = self.belief_network.predict_beliefs(game_state, player_id)
            opponent_samples = belief_state.opponent_hands
            
            # Sample opponent hands
            for opp_id in range(4):
                if opp_id != player_id:
                    # Sample cards based on belief probabilities
                    sampled_hand = self._sample_hand_from_belief(opponent_samples[opp_id], belief_state)
                    determinized_state.hands[opp_id] = sampled_hand
        else:
            # Uniform sampling if no belief network
            self._uniform_sample_opponent_hands(determinized_state, player_id)
        
        return determinized_state
    
    def _sample_hand_from_belief(self, card_probs: List[float], belief_state: BeliefState) -> List[Card]:
        """Sample a hand from belief distribution"""
        # Get available cards
        available_cards = []
        for suit in SUITS:
            for rank in RANKS:
                card = Card(suit, rank)
                if card not in belief_state.known_cards and card not in belief_state.played_cards:
                    available_cards.append(card)
        
        # Sample 8 cards
        sampled_hand = []
        for _ in range(8):
            if not available_cards:
                break
            
            # Calculate probabilities for available cards
            available_probs = []
            for card in available_cards:
                card_idx = SUITS.index(card.suit) * 8 + RANKS.index(card.rank)
                available_probs.append(card_probs[card_idx])
            
            # Normalize probabilities
            total_prob = sum(available_probs)
            if total_prob > 0:
                available_probs = [p / total_prob for p in available_probs]
            else:
                # Uniform if all probabilities are 0
                available_probs = [1.0 / len(available_cards)] * len(available_cards)
            
            # Sample card
            chosen_idx = np.random.choice(len(available_cards), p=available_probs)
            chosen_card = available_cards.pop(chosen_idx)
            sampled_hand.append(chosen_card)
        
        return sampled_hand
    
    def _uniform_sample_opponent_hands(self, game_state: Game28State, player_id: int):
        """Uniformly sample opponent hands"""
        # Get all cards not in player's hand
        all_cards = []
        for suit in SUITS:
            for rank in RANKS:
                all_cards.append(Card(suit, rank))
        
        # Remove player's cards
        for card in game_state.hands[player_id]:
            if card in all_cards:
                all_cards.remove(card)
        
        # Randomly distribute remaining cards to opponents
        random.shuffle(all_cards)
        cards_per_player = len(all_cards) // 3  # 3 opponents
        
        for i, opp_id in enumerate(range(4)):
            if opp_id != player_id:
                start_idx = i * cards_per_player
                end_idx = start_idx + cards_per_player
                game_state.hands[opp_id] = all_cards[start_idx:end_idx]
    
    def _select(self, node: ISMCTSNode, determinized_state: Game28State) -> ISMCTSNode:
        """Select a leaf node using UCT"""
        while node.children and not determinized_state.game_over:
            # Find best child using UCT
            best_child = None
            best_uct = float('-inf')
            
            for action, child in node.children.items():
                if child.visits == 0:
                    return child
                
                # Calculate UCT value
                exploitation = child.value / child.visits
                exploration = self.exploration_constant * math.sqrt(math.log(node.visits) / child.visits)
                uct = exploitation + exploration
                
                if uct > best_uct:
                    best_uct = uct
                    best_child = child
            
            if best_child:
                node = best_child
                # Update determinized state
                self._apply_action(determinized_state, node.action, node.player)
            else:
                break
        
        return node
    
    def _expand(self, node: ISMCTSNode, determinized_state: Game28State) -> ISMCTSNode:
        """Expand a leaf node"""
        if determinized_state.game_over:
            return node
        
        # Get legal actions
        legal_actions = determinized_state.get_legal_bids(node.player)
        
        # Find unexpanded action
        for action in legal_actions:
            if action not in node.children:
                # Create new state
                new_state = determinized_state.copy()
                new_state.make_bid(node.player, action)
                
                # Create child node
                child = ISMCTSNode(
                    state=new_state,
                    parent=node,
                    action=action,
                    player=(node.player + 1) % 4
                )
                node.children[action] = child
                return child
        
        return node
    
    def _simulate(self, state: Game28State, determinized_state: Game28State) -> float:
        """Simulate from current state to completion"""
        sim_state = determinized_state.copy()
        
        # Continue simulation until game is over
        while not sim_state.game_over:
            current_player = sim_state.current_player
            legal_actions = sim_state.get_legal_bids(current_player)
            
            if legal_actions:
                # Random action selection
                action = random.choice(legal_actions)
                sim_state.make_bid(current_player, action)
            else:
                break
        
        # Calculate reward
        return self._calculate_reward(sim_state, state.player)
    
    def _backpropagate(self, node: ISMCTSNode, value: float, player_id: int):
        """Backpropagate value up the tree"""
        while node:
            node.visits += 1
            node.value += value
            node = node.parent
    
    def _select_best_action(self, root: ISMCTSNode) -> int:
        """Select the best action from root node"""
        best_action = None
        best_value = float('-inf')
        
        for action, child in root.children.items():
            if child.visits > 0:
                value = child.value / child.visits
                if value > best_value:
                    best_value = value
                    best_action = action
        
        return best_action if best_action is not None else -1  # Default to pass
    
    def _apply_action(self, state: Game28State, action: int, player: int):
        """Apply action to state"""
        if action == -1:  # Pass
            state.passed_players.append(player)
            if len(state.passed_players) == 3:
                # Bidding ends
                active_players = [p for p in range(4) if p not in state.passed_players]
                if active_players:
                    state.bidder = active_players[0]
                    state.winning_bid = state.current_bid
                    state.game_over = True
                else:
                    state.game_over = True
        else:
            state.current_bid = action
            state.bid_history.append((player, action))
            state.passed_players = []
        
        # Move to next player
        state.current_player = (state.current_player + 1) % 4
        while state.current_player in state.passed_players:
            state.current_player = (state.current_player + 1) % 4
    
    def _calculate_reward(self, state: Game28State, player_id: int) -> float:
        """Calculate reward for the final state"""
        if not state.game_over:
            return 0.0
        
        # Calculate team-based reward
        team = 'A' if player_id in TEAM_A else 'B'
        game_point = state.game_points[team]
        
        if game_point > 0:
            return 1.0
        elif game_point < 0:
            return -1.0
        else:
            return 0.0


class BeliefAwareISMCTS(ISMCTSBidding):
    """
    Enhanced ISMCTS that uses belief network for better sampling
    """
    
    def __init__(self, belief_network: BeliefNetwork, num_simulations: int = 1000):
        super().__init__(belief_network, num_simulations)
        self.belief_updater = None  # Could be added for dynamic belief updates
    
    def select_action_with_confidence(self, game_state: Game28State, player_id: int) -> Tuple[int, float]:
        """
        Select action with confidence score
        
        Returns:
            Tuple of (action, confidence)
        """
        action = self.select_action(game_state, player_id)
        
        # Calculate confidence based on visit counts
        root = ISMCTSNode(state=game_state.copy(), player=player_id)
        
        # Run a few simulations to get visit distribution
        for _ in range(min(100, self.num_simulations)):
            determinized_state = self._sample_determinized_state(game_state, player_id)
            leaf = self._select(root, determinized_state)
            if leaf.visits > 0:
                leaf = self._expand(leaf, determinized_state)
            value = self._simulate(leaf.state, determinized_state)
            self._backpropagate(leaf, value, player_id)
        
        # Calculate confidence
        if action in root.children:
            child = root.children[action]
            confidence = child.visits / root.visits if root.visits > 0 else 0.0
        else:
            confidence = 0.0
        
        return action, confidence
