#!/usr/bin/env python3
"""
Simple Game 28 Simulation
4 AI agents play a complete game using different strategies:
- Agent 0: Advanced Heuristic + Belief-like reasoning
- Agent 1: Basic Heuristic + MCTS-like planning
- Agent 2: Conservative Heuristic
- Agent 3: Aggressive Heuristic
"""

import sys
import os
import time
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game28.game_state import Game28State, Card
from game28.constants import SUITS, RANKS, CARD_VALUES, TOTAL_POINTS, MIN_BID, MAX_BID, BID_RANGE


@dataclass
class AgentConfig:
    """Configuration for each agent"""
    agent_id: int
    name: str
    strategy: str
    aggressiveness: float = 0.5  # 0.0 = very conservative, 1.0 = very aggressive


class SimpleGameAgent:
    """Simple game agent with different heuristic strategies"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent_id = config.agent_id
        self.name = config.name
        self.strategy = config.strategy
        self.aggressiveness = config.aggressiveness
        
        # Track opponent behavior for belief-like reasoning
        self.opponent_bidding_patterns = {0: [], 1: [], 2: [], 3: []}
        self.opponent_card_patterns = {0: [], 1: [], 2: [], 3: []}
    
    def decide_bid(self, game_state: Game28State, current_bid: int) -> int:
        """Decide whether to bid or pass"""
        hand = game_state.hands[self.agent_id]
        hand_strength = self._calculate_hand_strength(hand[:4])  # First 4 cards only
        
        if self.strategy == "advanced_heuristic":
            return self._advanced_bidding_strategy(game_state, current_bid, hand_strength)
        elif self.strategy == "basic_heuristic":
            return self._basic_bidding_strategy(game_state, current_bid, hand_strength)
        elif self.strategy == "conservative":
            return self._conservative_bidding_strategy(game_state, current_bid, hand_strength)
        elif self.strategy == "aggressive":
            return self._aggressive_bidding_strategy(game_state, current_bid, hand_strength)
        else:
            return self._basic_bidding_strategy(game_state, current_bid, hand_strength)
    
    def choose_trump(self, game_state: Game28State) -> str:
        """Choose trump suit if we won the bidding"""
        hand = game_state.hands[self.agent_id]
        
        if self.strategy == "advanced_heuristic":
            return self._advanced_trump_strategy(game_state, hand)
        else:
            return self._heuristic_trump_strategy(hand)
    
    def choose_card(self, game_state: Game28State, legal_cards: List[Card]) -> Card:
        """Choose which card to play"""
        if self.strategy == "advanced_heuristic":
            return self._advanced_card_strategy(game_state, legal_cards)
        elif self.strategy == "basic_heuristic":
            return self._basic_card_strategy(game_state, legal_cards)
        else:
            return self._heuristic_card_strategy(game_state, legal_cards)
    
    def _calculate_hand_strength(self, hand: List[Card]) -> float:
        """Calculate hand strength (0.0 to 1.0)"""
        if not hand:
            return 0.0
        total_points = sum(CARD_VALUES[card.rank] for card in hand)
        return total_points / TOTAL_POINTS
    
    def _calculate_suit_strength(self, hand: List[Card], suit: str) -> Tuple[int, float]:
        """Calculate strength of a specific suit"""
        suit_cards = [card for card in hand if card.suit == suit]
        suit_count = len(suit_cards)
        suit_points = sum(CARD_VALUES[card.rank] for card in suit_cards)
        return suit_count, suit_points
    
    def _advanced_bidding_strategy(self, game_state: Game28State, current_bid: int, hand_strength: float) -> int:
        """Advanced bidding with opponent modeling"""
        # Analyze opponent bidding patterns
        opponent_aggression = self._analyze_opponent_aggression()
        
        # Adjust strategy based on opponent behavior
        adjusted_strength = hand_strength
        if opponent_aggression > 0.7:  # Opponents are aggressive
            adjusted_strength *= 0.8  # Be more conservative
        elif opponent_aggression < 0.3:  # Opponents are conservative
            adjusted_strength *= 1.2  # Be more aggressive
        
        # Consider position
        position_bonus = 0.1 if game_state.current_player == 3 else 0.0  # Last to bid
        
        # Make decision
        if adjusted_strength + position_bonus > 0.6:
            new_bid = min(current_bid + 2, MAX_BID)
            return new_bid if new_bid > current_bid else -1
        elif adjusted_strength + position_bonus > 0.4:
            new_bid = min(current_bid + 1, MAX_BID)
            return new_bid if new_bid > current_bid else -1
        else:
            return -1  # Pass
    
    def _basic_bidding_strategy(self, game_state: Game28State, current_bid: int, hand_strength: float) -> int:
        """Basic heuristic bidding strategy"""
        if hand_strength > 0.6:
            new_bid = min(current_bid + 2, MAX_BID)
            return new_bid if new_bid > current_bid else -1
        elif hand_strength > 0.4:
            new_bid = min(current_bid + 1, MAX_BID)
            return new_bid if new_bid > current_bid else -1
        else:
            return -1  # Pass
    
    def _conservative_bidding_strategy(self, game_state: Game28State, current_bid: int, hand_strength: float) -> int:
        """Conservative bidding strategy"""
        if hand_strength > 0.7:  # Only bid with very strong hands
            new_bid = min(current_bid + 1, MAX_BID)
            return new_bid if new_bid > current_bid else -1
        else:
            return -1  # Pass
    
    def _aggressive_bidding_strategy(self, game_state: Game28State, current_bid: int, hand_strength: float) -> int:
        """Aggressive bidding strategy"""
        if hand_strength > 0.3:  # Bid with moderately strong hands
            new_bid = min(current_bid + 2, MAX_BID)
            return new_bid if new_bid > current_bid else -1
        elif hand_strength > 0.2:  # Even bid with weak hands sometimes
            if random.random() < 0.3:  # 30% chance
                new_bid = min(current_bid + 1, MAX_BID)
                return new_bid if new_bid > current_bid else -1
        return -1  # Pass
    
    def _advanced_trump_strategy(self, game_state: Game28State, hand: List[Card]) -> str:
        """Advanced trump selection with opponent modeling"""
        # Analyze which suits opponents are likely weak in
        opponent_weakness = self._analyze_opponent_weakness()
        
        # Find our strongest suit
        suit_scores = {}
        for suit in SUITS:
            count, points = self._calculate_suit_strength(hand, suit)
            suit_scores[suit] = (count, points)
        
        # Choose trump that maximizes our advantage
        best_suit = None
        best_score = float('-inf')
        
        for suit in SUITS:
            count, points = suit_scores[suit]
            opponent_weakness_bonus = opponent_weakness.get(suit, 0.5)
            
            # Score = our strength + opponent weakness
            score = (count * 2 + points) * opponent_weakness_bonus
            
            if score > best_score:
                best_score = score
                best_suit = suit
        
        return best_suit
    
    def _heuristic_trump_strategy(self, hand: List[Card]) -> str:
        """Simple heuristic trump selection"""
        suit_counts = {suit: 0 for suit in SUITS}
        suit_points = {suit: 0 for suit in SUITS}
        
        for card in hand:
            suit_counts[card.suit] += 1
            suit_points[card.suit] += CARD_VALUES[card.rank]
        
        # Choose suit with most cards and highest points
        best_suit = max(suit_counts, key=lambda s: (suit_counts[s], suit_points[s]))
        return best_suit
    
    def _advanced_card_strategy(self, game_state: Game28State, legal_cards: List[Card]) -> Card:
        """Advanced card selection with opponent modeling"""
        if not legal_cards:
            return None
        
        # If leading, consider opponent patterns
        if not game_state.current_trick.cards:
            return self._choose_leading_card(game_state, legal_cards)
        
        # If following, try to win if possible
        return self._choose_following_card(game_state, legal_cards)
    
    def _basic_card_strategy(self, game_state: Game28State, legal_cards: List[Card]) -> Card:
        """Basic card selection strategy"""
        if not legal_cards:
            return None
        
        # If leading, play highest card
        if not game_state.current_trick.cards:
            return max(legal_cards, key=lambda c: CARD_VALUES[c.rank])
        
        # If following, try to win if possible
        return self._choose_following_card(game_state, legal_cards)
    
    def _heuristic_card_strategy(self, game_state: Game28State, legal_cards: List[Card]) -> Card:
        """Simple heuristic card selection"""
        if not legal_cards:
            return None
        
        # Simple strategy: play highest card
        return max(legal_cards, key=lambda c: CARD_VALUES[c.rank])
    
    def _choose_leading_card(self, game_state: Game28State, legal_cards: List[Card]) -> Card:
        """Choose card when leading"""
        # Consider trump and high cards
        trump_cards = [c for c in legal_cards if c.suit == game_state.trump_suit]
        high_cards = [c for c in legal_cards if CARD_VALUES[c.rank] > 5]
        
        if trump_cards:
            return max(trump_cards, key=lambda c: CARD_VALUES[c.rank])
        elif high_cards:
            return max(high_cards, key=lambda c: CARD_VALUES[c.rank])
        else:
            return max(legal_cards, key=lambda c: CARD_VALUES[c.rank])
    
    def _choose_following_card(self, game_state: Game28State, legal_cards: List[Card]) -> Card:
        """Choose card when following"""
        current_trick = game_state.current_trick
        leading_suit = current_trick.lead_suit if current_trick.cards else None
        trump_suit = game_state.trump_suit
        
        # Find cards that can follow suit
        following_suit = [c for c in legal_cards if c.suit == leading_suit]
        
        if following_suit:
            # Must follow suit
            cards_to_play = following_suit
        else:
            # Can play any card
            cards_to_play = legal_cards
        
        # Find the best card to play
        best_card = None
        best_score = float('-inf')
        
        for card in cards_to_play:
            score = self._evaluate_card_heuristic(card, game_state)
            if score > best_score:
                best_score = score
                best_card = card
        
        return best_card if best_card else legal_cards[0]
    
    def _evaluate_card_heuristic(self, card: Card, game_state: Game28State) -> float:
        """Evaluate a card's value in the current game state"""
        score = CARD_VALUES[card.rank]
        
        # Bonus for trump cards
        if game_state.trump_suit and card.suit == game_state.trump_suit:
            score *= 1.5
        
        # Bonus for high cards in leading suit
        if game_state.current_trick.cards:
            leading_suit = game_state.current_trick.lead_suit
            if card.suit == leading_suit:
                score *= 1.2
        
        return score
    
    def _analyze_opponent_aggression(self) -> float:
        """Analyze how aggressive opponents have been"""
        if not self.opponent_bidding_patterns:
            return 0.5  # Neutral
        
        total_bids = 0
        total_passes = 0
        
        for player_id, bids in self.opponent_bidding_patterns.items():
            if player_id != self.agent_id:
                total_bids += len([b for b in bids if b > 0])
                total_passes += len([b for b in bids if b == -1])
        
        total_actions = total_bids + total_passes
        if total_actions == 0:
            return 0.5
        
        return total_bids / total_actions
    
    def _analyze_opponent_weakness(self) -> Dict[str, float]:
        """Analyze which suits opponents are weak in"""
        # Simple heuristic: assume opponents are weak in suits we have many of
        hand = self.game_state.hands[self.agent_id] if hasattr(self, 'game_state') else []
        
        suit_counts = {suit: 0 for suit in SUITS}
        for card in hand:
            suit_counts[card.suit] += 1
        
        # Normalize to 0-1 range
        max_count = max(suit_counts.values()) if suit_counts.values() else 1
        weakness = {suit: 1.0 - (count / max_count) for suit, count in suit_counts.items()}
        
        return weakness


class SimpleGameSimulator:
    """Simulates a complete Game 28 match with simple agents"""
    
    def __init__(self, agents: List[SimpleGameAgent]):
        self.agents = agents
        self.game_state = None
        
    def simulate_game(self) -> Dict[str, Any]:
        """Simulate a complete game and return results"""
        print("="*80)
        print("🎮 GAME 28 SIMULATION STARTING")
        print("="*80)
        
        # Initialize game
        self.game_state = Game28State()
        
        print(f"\n📋 AGENTS:")
        for agent in self.agents:
            print(f"  Player {agent.agent_id}: {agent.name} ({agent.strategy})")
        
        print(f"\n🃏 INITIAL HANDS:")
        for i, agent in enumerate(self.agents):
            hand = [str(card) for card in self.game_state.hands[i]]
            print(f"  {agent.name}: {hand}")
        
        # Phase 1: Bidding
        print(f"\n" + "="*50)
        print("🏆 BIDDING PHASE")
        print("="*50)
        
        bidding_result = self._simulate_bidding()
        
        if bidding_result['winner'] is None:
            print("❌ All players passed - no winner!")
            return self._get_game_results()
        
        winner_agent = self.agents[bidding_result['winner']]
        winning_bid = bidding_result['winning_bid']
        
        print(f"✅ {winner_agent.name} won the bidding with {winning_bid}")
        
        # Set the bidder in game state
        self.game_state.bidder = bidding_result['winner']
        self.game_state.winning_bid = winning_bid
        
        # Phase 2: Trump Selection
        print(f"\n" + "="*50)
        print("🎯 TRUMP SELECTION")
        print("="*50)
        
        trump_suit = winner_agent.choose_trump(self.game_state)
        self.game_state.set_trump(trump_suit)
        print(f"🎯 {winner_agent.name} chose {trump_suit} as trump")
        
        # Phase 3: Card Play
        print(f"\n" + "="*50)
        print("🃏 CARD PLAY PHASE")
        print("="*50)
        
        self._simulate_card_play()
        
        # Game Results
        print(f"\n" + "="*50)
        print("🏁 GAME RESULTS")
        print("="*50)
        
        results = self._get_game_results()
        self._print_results(results)
        
        return results
    
    def _simulate_bidding(self) -> Dict[str, Any]:
        """Simulate the bidding phase"""
        current_bid = MIN_BID
        passed_players = set()
        bid_history = []
        
        while len(passed_players) < 4:
            for agent in self.agents:
                if agent.agent_id in passed_players:
                    continue
                
                print(f"\n🎯 {agent.name}'s turn to bid (current bid: {current_bid})")
                
                # Get agent's decision
                bid_decision = agent.decide_bid(self.game_state, current_bid)
                
                if bid_decision == -1:  # Pass
                    print(f"  {agent.name} passes")
                    passed_players.add(agent.agent_id)
                    bid_history.append((agent.agent_id, -1))
                else:  # Bid
                    print(f"  {agent.name} bids {bid_decision}")
                    current_bid = bid_decision
                    bid_history.append((agent.agent_id, bid_decision))
                
                # Check if only one player hasn't passed
                if len(passed_players) == 3:
                    # Find the winner
                    for agent in self.agents:
                        if agent.agent_id not in passed_players:
                            return {
                                'winner': agent.agent_id,
                                'winning_bid': current_bid,
                                'bid_history': bid_history
                            }
        
        return {'winner': None, 'winning_bid': None, 'bid_history': bid_history}
    
    def _simulate_card_play(self):
        """Simulate the card play phase"""
        trick_number = 1
        
        while not self.game_state.game_over and len(self.game_state.tricks) < 8:
            print(f"\n🃏 Trick {trick_number}")
            print("-" * 30)
            
            current_trick = []
            
            # Each player plays a card
            for agent in self.agents:
                legal_cards = self.game_state.get_legal_plays(agent.agent_id)
                
                if not legal_cards:
                    print(f"  {agent.name} has no legal cards to play")
                    continue
                
                # Get agent's card choice
                chosen_card = agent.choose_card(self.game_state, legal_cards)
                
                if chosen_card:
                    print(f"  {agent.name} plays {chosen_card}")
                    self.game_state.play_card(agent.agent_id, chosen_card)
                    current_trick.append(chosen_card)
                
                # Update current player
                self.game_state.current_player = (self.game_state.current_player + 1) % 4
            
            # Determine trick winner
            if current_trick:
                winning_card = self._determine_trick_winner(current_trick)
                winning_player = current_trick.index(winning_card)
                winning_agent = self.agents[winning_player]
                
                print(f"  🏆 {winning_agent.name} wins the trick with {winning_card}")
                
                # Update game state
                self.game_state.current_player = winning_player
                self.game_state.tricks.append(current_trick)
            
            trick_number += 1
    
    def _determine_trick_winner(self, trick: List[Card]) -> Card:
        """Determine which card wins the trick"""
        if not trick:
            return None
        
        leading_suit = trick[0].suit
        trump_suit = self.game_state.trump_suit
        
        # Find highest trump card
        trump_cards = [card for card in trick if card.suit == trump_suit]
        if trump_cards:
            return max(trump_cards, key=lambda c: CARD_VALUES[c.rank])
        
        # Find highest card in leading suit
        leading_suit_cards = [card for card in trick if card.suit == leading_suit]
        if leading_suit_cards:
            return max(leading_suit_cards, key=lambda c: CARD_VALUES[c.rank])
        
        return trick[0]  # Fallback
    
    def _get_game_results(self) -> Dict[str, Any]:
        """Get final game results"""
        if not self.game_state:
            return {}
        
        # Calculate scores
        team_a_score = self.game_state.game_points.get('A', 0)
        team_b_score = self.game_state.game_points.get('B', 0)
        
        # Determine winner
        if team_a_score > team_b_score:
            winner = "Team A (Players 0 & 2)"
            winning_team = "A"
        elif team_b_score > team_a_score:
            winner = "Team B (Players 1 & 3)"
            winning_team = "B"
        else:
            winner = "Tie"
            winning_team = None
        
        return {
            'team_a_score': team_a_score,
            'team_b_score': team_b_score,
            'winner': winner,
            'winning_team': winning_team,
            'bidder': self.game_state.bidder,
            'winning_bid': self.game_state.winning_bid,
            'trump_suit': self.game_state.trump_suit,
            'tricks': self.game_state.tricks,
            'game_over': self.game_state.game_over
        }
    
    def _print_results(self, results: Dict[str, Any]):
        """Print game results"""
        print(f"\n📊 FINAL SCORES:")
        print(f"  Team A (Players 0 & 2): {results['team_a_score']}")
        print(f"  Team B (Players 1 & 3): {results['team_b_score']}")
        
        print(f"\n🏆 WINNER: {results['winner']}")
        
        if results['bidder'] is not None:
            bidder_agent = self.agents[results['bidder']]
            print(f"\n🎯 BIDDING RESULTS:")
            print(f"  Bidder: {bidder_agent.name}")
            print(f"  Winning Bid: {results['winning_bid']}")
            print(f"  Trump Suit: {results['trump_suit']}")
            
            # Check if bidder succeeded
            if results['winning_team'] == 'A' and results['bidder'] in [0, 2]:
                bid_success = results['team_a_score'] >= results['winning_bid']
            elif results['winning_team'] == 'B' and results['bidder'] in [1, 3]:
                bid_success = results['team_b_score'] >= results['winning_bid']
            else:
                bid_success = False
            
            print(f"  Bid Success: {'✅' if bid_success else '❌'}")
        
        print(f"\n🃏 TRICKS PLAYED: {len(results['tricks'])}")


def create_simple_agents() -> List[SimpleGameAgent]:
    """Create the 4 simple game agents with different strategies"""
    agents = []
    
    # Agent 0: Advanced Heuristic
    agent0_config = AgentConfig(
        agent_id=0,
        name="Advanced Bot",
        strategy="advanced_heuristic",
        aggressiveness=0.6
    )
    agents.append(SimpleGameAgent(agent0_config))
    
    # Agent 1: Basic Heuristic
    agent1_config = AgentConfig(
        agent_id=1,
        name="Basic Bot",
        strategy="basic_heuristic",
        aggressiveness=0.5
    )
    agents.append(SimpleGameAgent(agent1_config))
    
    # Agent 2: Conservative
    agent2_config = AgentConfig(
        agent_id=2,
        name="Conservative Bot",
        strategy="conservative",
        aggressiveness=0.3
    )
    agents.append(SimpleGameAgent(agent2_config))
    
    # Agent 3: Aggressive
    agent3_config = AgentConfig(
        agent_id=3,
        name="Aggressive Bot",
        strategy="aggressive",
        aggressiveness=0.8
    )
    agents.append(SimpleGameAgent(agent3_config))
    
    return agents


def main():
    """Main function to run the simple game simulation"""
    print("🎮 28Bot v2 - Simple Game Simulation")
    print("="*80)
    
    # Create agents
    print("🤖 Creating AI agents...")
    agents = create_simple_agents()
    
    # Create game simulator
    simulator = SimpleGameSimulator(agents)
    
    # Simulate the game
    results = simulator.simulate_game()
    
    print("\n" + "="*80)
    print("🎉 GAME SIMULATION COMPLETE!")
    print("="*80)
    
    # Summary
    print(f"\n📋 SUMMARY:")
    print(f"  Winner: {results['winner']}")
    print(f"  Final Score: Team A {results['team_a_score']} - {results['team_b_score']} Team B")
    
    if results['bidder'] is not None:
        bidder_name = agents[results['bidder']].name
        print(f"  Bidder: {bidder_name} (bid {results['winning_bid']})")
        print(f"  Trump: {results['trump_suit']}")
    
    print(f"\n🎯 This simulation demonstrates:")
    print(f"  • Different heuristic strategies competing")
    print(f"  • Complete Game 28 gameplay")
    print(f"  • Bidding, trump selection, and card play")
    print(f"  • Team-based scoring and winning conditions")


if __name__ == "__main__":
    main()
