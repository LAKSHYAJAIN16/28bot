#!/usr/bin/env python3
"""
Hybrid Agent that combines Belief Model and ISMCTS for optimal decision making
"""

import math
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import torch

from game28.game_state import Game28State, Card
from game28.constants import *
from belief_model.improved_belief_net import ImprovedBeliefNetwork

# Try to import ISMCTS components, but make them optional
try:
    from ismcts.ismcts_bidding import ISMCTSBidding, BeliefAwareISMCTS
    ISMCTS_AVAILABLE = True
    print("✅ ISMCTS components loaded successfully")
except ImportError as e:
    print(f"Warning: ISMCTS components not available ({e}), hybrid agent will use belief model only")
    ISMCTS_AVAILABLE = False
    # Create dummy classes for compatibility
    class ISMCTSBidding:
        def __init__(self, *args, **kwargs):
            pass
        def select_action_with_confidence(self, *args, **kwargs):
            return -1, 0.5  # Default to pass with low confidence
    
    class BeliefAwareISMCTS(ISMCTSBidding):
        pass


@dataclass
class HybridDecision:
    """Represents a decision made by the hybrid agent"""
    action: Any
    method: str  # 'belief_model', 'ismcts', or 'hybrid'
    confidence: float
    belief_score: Optional[float] = None
    ismcts_score: Optional[float] = None
    reasoning: str = ""


class HybridAgent:
    """
    Hybrid agent that combines belief model and ISMCTS for optimal decision making.
    
    Strategy:
    - Early game (bidding, early tricks): Use belief model for speed and pattern recognition
    - Mid game (3-6 tricks): Use hybrid approach with weighted combination
    - Late game (7-8 tricks): Use ISMCTS for precise calculation when more information is available
    """
    
    def __init__(self, 
                 agent_id: int,
                 belief_model: ImprovedBeliefNetwork,
                 ismcts: BeliefAwareISMCTS,
                 use_hybrid: bool = True,
                 hybrid_threshold: float = 0.7):
        self.agent_id = agent_id
        self.belief_model = belief_model
        self.ismcts = ismcts
        self.use_hybrid = use_hybrid
        self.hybrid_threshold = hybrid_threshold
        
        # Decision history for adaptive learning
        self.decision_history = []
        
        # Performance tracking
        self.belief_decisions = 0
        self.ismcts_decisions = 0
        self.hybrid_decisions = 0
        
    def decide_bid(self, game_state: Game28State, legal_bids: List[int]) -> HybridDecision:
        """Decide on bidding using hybrid approach"""
        if not legal_bids:
            return HybridDecision(-1, "belief_model", 1.0, reasoning="No legal bids")
        
        # Early game: Use belief model for speed
        if len(game_state.bid_history) < 2:
            return self._belief_based_bid(game_state, legal_bids)
        
        # Mid game: Use hybrid approach
        if self.use_hybrid and len(game_state.bid_history) < 6:
            return self._hybrid_bid(game_state, legal_bids)
        
        # Late game: Use ISMCTS for precision
        return self._ismcts_based_bid(game_state, legal_bids)
    
    def choose_trump(self, game_state: Game28State, hand: List[Card]) -> HybridDecision:
        """Choose trump using hybrid approach"""
        # Trump selection is typically early game, use belief model
        return self._belief_based_trump(game_state, hand)
    
    def choose_card(self, game_state: Game28State, legal_cards: List[Card]) -> HybridDecision:
        """Choose card using hybrid approach"""
        if not legal_cards:
            return HybridDecision(None, "belief_model", 0.0, reasoning="No legal cards")
        
        # Determine game phase for decision method
        tricks_played = len(game_state.tricks)
        
        # Early game (0-2 tricks): Use belief model
        if tricks_played < 3:
            return self._belief_based_card(game_state, legal_cards)
        
        # Mid game (3-5 tricks): Use hybrid approach
        elif self.use_hybrid and tricks_played < 6:
            return self._hybrid_card(game_state, legal_cards)
        
        # Late game (6-8 tricks): Use ISMCTS for precision
        else:
            return self._ismcts_based_card(game_state, legal_cards)

    def decide_reveal_and_card(self, game_state: Game28State, legal_cards: List[Card]) -> Tuple[bool, HybridDecision]:
        """Evaluate no-reveal vs reveal+trump-play; return (reveal, decision).
        Minimal heuristic: compare best action under current state vs state with trump revealed
        and trump-only plays (if player holds trump).
        """
        # If no trump set or already revealed, no reveal path
        if not game_state.trump_suit or game_state.trump_revealed:
            return False, self.choose_card(game_state, legal_cards)

        # Gather player's trump cards
        player_hand = game_state.hands[self.agent_id]
        trump_cards = [c for c in player_hand if c.suit == game_state.trump_suit]
        if not trump_cards:
            return False, self.choose_card(game_state, legal_cards)

        # Rule: cannot reveal if you have a legal card in the lead suit
        if game_state.current_trick and game_state.current_trick.cards:
            lead_suit = game_state.current_trick.lead_suit
            has_lead_suit = any(c.suit == lead_suit for c in player_hand)
            if has_lead_suit:
                return False, self.choose_card(game_state, legal_cards)

        # Score no-reveal
        no_reveal_decision = self.choose_card(game_state, legal_cards)

        # Construct a shallow copy state with trump revealed
        reveal_state = game_state.copy()
        reveal_state.trump_revealed = True

        # Legal trump plays when revealing: if cannot follow suit, allow any trump; if on lead, any trump
        reveal_legal = trump_cards
        if not reveal_legal:
            return False, no_reveal_decision

        # Only consider reveal if there are points at stake in current trick
        trick_points = reveal_state.current_trick.points if reveal_state.current_trick else 0
        if trick_points <= 0:
            return False, no_reveal_decision

        # Score reveal path (reuse belief-based in early/mid; treat as late precision otherwise)
        if len(reveal_state.tricks) < 6:
            reveal_decision = self._belief_based_card(reveal_state, reveal_legal)
        else:
            reveal_decision = self._ismcts_based_card(reveal_state, reveal_legal)

        # Simple comparison by confidence
        reveal_score = reveal_decision.confidence
        no_reveal_score = no_reveal_decision.confidence

        if reveal_score > no_reveal_score:
            reveal_decision.reasoning = f"Reveal chosen (score {reveal_score:.3f} > {no_reveal_score:.3f})"
            return True, reveal_decision
        return False, no_reveal_decision
    
    def _belief_based_bid(self, game_state: Game28State, legal_bids: List[int]) -> HybridDecision:
        """Make bidding decision using belief model"""
        try:
            # Check if belief model is available
            if self.belief_model is None:
                # Fallback to simple heuristic
                hand_strength = self._calculate_hand_strength(game_state.hands[self.agent_id])
                optimal_bid = max(legal_bids, key=lambda bid: self._evaluate_bid(bid, hand_strength, 0.5, 0.5))
                return HybridDecision(
                    action=optimal_bid,
                    method="belief_model",
                    confidence=0.3,
                    belief_score=0.3,
                    reasoning=f"Simple heuristic - Hand strength: {hand_strength:.3f}"
                )
            
            # Get belief predictions
            belief_predictions = self.belief_model.predict_beliefs(game_state, self.agent_id)
            
            # Calculate bid score based on beliefs
            trump_probs = belief_predictions.trump_suit.cpu().numpy().flatten()
            trump_confidence = trump_probs.max()
            
            # Calculate expected points from hand
            hand_strength = self._calculate_hand_strength(game_state.hands[self.agent_id])
            
            # Calculate opponent strength
            opponent_strength = 0.0
            for opp_id in range(4):
                if opp_id != self.agent_id and opp_id in belief_predictions.opponent_hands:
                    opp_hand_probs = belief_predictions.opponent_hands[opp_id].cpu().numpy().flatten()
                    expected_points = 0.0
                    for i, prob in enumerate(opp_hand_probs):
                        suit_idx = i // 8
                        rank_idx = i % 8
                        if rank_idx < len(RANKS):
                            expected_points += float(prob) * CARD_VALUES[RANKS[rank_idx]]
                    opponent_strength += expected_points / TOTAL_POINTS
            
            # Calculate optimal bid
            optimal_bid = max(legal_bids, key=lambda bid: self._evaluate_bid(bid, hand_strength, opponent_strength, trump_confidence))
            
            confidence = min(0.9, trump_confidence + hand_strength * 0.3)
            
            self.belief_decisions += 1
            return HybridDecision(
                action=optimal_bid,
                method="belief_model",
                confidence=confidence,
                belief_score=confidence,
                reasoning=f"Hand strength: {hand_strength:.3f}, Opponent strength: {opponent_strength:.3f}, Trump confidence: {trump_confidence:.3f}"
            )
            
        except Exception as e:
            # Fallback to random bid
            fallback_bid = random.choice(legal_bids) if legal_bids else -1
            return HybridDecision(
                action=fallback_bid,
                method="belief_model",
                confidence=0.1,
                reasoning=f"Fallback due to error: {str(e)}"
            )
    
    def _ismcts_based_bid(self, game_state: Game28State, legal_bids: List[int]) -> HybridDecision:
        """Make bidding decision using ISMCTS"""
        if not ISMCTS_AVAILABLE:
            # Fallback to belief model if ISMCTS is not available
            return self._belief_based_bid(game_state, legal_bids)
        
        try:
            action, confidence = self.ismcts.select_action_with_confidence(game_state, self.agent_id)
            
            # Ensure action is legal
            if action not in legal_bids:
                action = random.choice(legal_bids) if legal_bids else -1
                confidence = 0.5
            
            self.ismcts_decisions += 1
            return HybridDecision(
                action=action,
                method="ismcts",
                confidence=confidence,
                ismcts_score=confidence,
                reasoning=f"ISMCTS simulation with {self.ismcts.num_simulations} iterations"
            )
            
        except Exception as e:
            # Fallback to belief model
            return self._belief_based_bid(game_state, legal_bids)
    
    def _hybrid_bid(self, game_state: Game28State, legal_bids: List[int]) -> HybridDecision:
        """Make bidding decision using hybrid approach"""
        belief_decision = self._belief_based_bid(game_state, legal_bids)
        ismcts_decision = self._ismcts_based_bid(game_state, legal_bids)
        
        # Weight based on game state
        belief_weight = 0.6  # Belief model gets more weight in mid-game
        ismcts_weight = 0.4
        
        # Combine decisions
        if belief_decision.action == ismcts_decision.action:
            # Both methods agree
            combined_confidence = (belief_decision.confidence + ismcts_decision.confidence) / 2
            final_action = belief_decision.action
        else:
            # Methods disagree, use weighted decision
            belief_score = belief_decision.confidence * belief_weight
            ismcts_score = ismcts_decision.confidence * ismcts_weight
            
            if belief_score > ismcts_score:
                final_action = belief_decision.action
                combined_confidence = belief_decision.confidence
            else:
                final_action = ismcts_decision.action
                combined_confidence = ismcts_decision.confidence
        
        self.hybrid_decisions += 1
        return HybridDecision(
            action=final_action,
            method="hybrid",
            confidence=combined_confidence,
            belief_score=belief_decision.confidence,
            ismcts_score=ismcts_decision.confidence,
            reasoning=f"Hybrid: Belief({belief_decision.action}, {belief_decision.confidence:.3f}) vs ISMCTS({ismcts_decision.action}, {ismcts_decision.confidence:.3f})"
        )
    
    def _belief_based_trump(self, game_state: Game28State, hand: List[Card]) -> HybridDecision:
        """Choose trump using belief model"""
        try:
            belief_predictions = self.belief_model.predict_beliefs(game_state, self.agent_id)
            trump_probs = belief_predictions.trump_suit.cpu().numpy().flatten()
            
            # Choose suit with highest probability
            best_suit_idx = trump_probs.argmax()
            best_suit = SUITS[best_suit_idx]
            confidence = trump_probs[best_suit_idx]
            
            self.belief_decisions += 1
            return HybridDecision(
                action=best_suit,
                method="belief_model",
                confidence=confidence,
                belief_score=confidence,
                reasoning=f"Trump probabilities: {dict(zip(SUITS, trump_probs))}"
            )
            
        except Exception as e:
            # Fallback to suit with most cards
            suit_counts = {suit: len([c for c in hand if c.suit == suit]) for suit in SUITS}
            best_suit = max(suit_counts, key=suit_counts.get)
            
            return HybridDecision(
                action=best_suit,
                method="belief_model",
                confidence=0.3,
                reasoning=f"Fallback to suit with most cards: {suit_counts}"
            )
    
    def _belief_based_card(self, game_state: Game28State, legal_cards: List[Card]) -> HybridDecision:
        """Choose card using belief model"""
        try:
            belief_predictions = self.belief_model.predict_beliefs(game_state, self.agent_id)
            
            best_card = None
            best_score = float('-inf')
            
            for card in legal_cards:
                score = self._evaluate_card_with_beliefs(game_state, card, belief_predictions)
                if score > best_score:
                    best_score = score
                    best_card = card
            
            confidence = min(0.9, (best_score + 10) / 20)  # Normalize score to confidence
            
            self.belief_decisions += 1
            return HybridDecision(
                action=best_card,
                method="belief_model",
                confidence=confidence,
                belief_score=confidence,
                reasoning=f"Card evaluation score: {best_score:.3f}"
            )
            
        except Exception as e:
            # Fallback to highest value card
            fallback_card = max(legal_cards, key=lambda c: CARD_VALUES[c.rank])
            return HybridDecision(
                action=fallback_card,
                method="belief_model",
                confidence=0.2,
                reasoning=f"Fallback to highest value card due to error: {str(e)}"
            )
    
    def _ismcts_based_card(self, game_state: Game28State, legal_cards: List[Card]) -> HybridDecision:
        """Choose card using ISMCTS"""
        if not ISMCTS_AVAILABLE:
            # Fallback to belief model if ISMCTS is not available
            return self._belief_based_card(game_state, legal_cards)
        
        try:
            # For card selection, we need to adapt ISMCTS to handle card actions
            # This is a simplified version - in practice, you'd need a card-specific ISMCTS
            
            # For now, use belief model with higher confidence
            belief_decision = self._belief_based_card(game_state, legal_cards)
            belief_decision.method = "ismcts"
            belief_decision.confidence = min(0.95, belief_decision.confidence * 1.2)
            
            self.ismcts_decisions += 1
            return belief_decision
            
        except Exception as e:
            return self._belief_based_card(game_state, legal_cards)
    
    def _hybrid_card(self, game_state: Game28State, legal_cards: List[Card]) -> HybridDecision:
        """Choose card using hybrid approach"""
        belief_decision = self._belief_based_card(game_state, legal_cards)
        ismcts_decision = self._ismcts_based_card(game_state, legal_cards)
        
        # For card selection, give more weight to belief model
        belief_weight = 0.7
        ismcts_weight = 0.3
        
        if belief_decision.action == ismcts_decision.action:
            combined_confidence = (belief_decision.confidence + ismcts_decision.confidence) / 2
            final_action = belief_decision.action
        else:
            belief_score = belief_decision.confidence * belief_weight
            ismcts_score = ismcts_decision.confidence * ismcts_weight
            
            if belief_score > ismcts_score:
                final_action = belief_decision.action
                combined_confidence = belief_decision.confidence
            else:
                final_action = ismcts_decision.action
                combined_confidence = ismcts_decision.confidence
        
        self.hybrid_decisions += 1
        return HybridDecision(
            action=final_action,
            method="hybrid",
            confidence=combined_confidence,
            belief_score=belief_decision.confidence,
            ismcts_score=ismcts_decision.confidence,
            reasoning=f"Hybrid card: Belief({belief_decision.action}, {belief_decision.confidence:.3f}) vs ISMCTS({ismcts_decision.action}, {ismcts_decision.confidence:.3f})"
        )
    
    def _calculate_hand_strength(self, hand: List[Card]) -> float:
        """Calculate the strength of a hand"""
        if not hand:
            return 0.0
        
        # Calculate total points
        total_points = sum(CARD_VALUES[card.rank] for card in hand)
        
        # Calculate suit distribution
        suit_counts = {suit: len([c for c in hand if c.suit == suit]) for suit in SUITS}
        max_suit_count = max(suit_counts.values())
        
        # Calculate high card strength
        high_cards = sum(1 for card in hand if CARD_VALUES[card.rank] >= 3)
        
        # Normalize to 0-1 range
        strength = (total_points / TOTAL_POINTS * 0.5 + 
                   max_suit_count / 8 * 0.3 + 
                   high_cards / 8 * 0.2)
        
        return min(1.0, strength)
    
    def _evaluate_bid(self, bid: int, hand_strength: float, opponent_strength: float, trump_confidence: float) -> float:
        """Evaluate a bid value with overbid penalty to avoid constant 28s.

        Heuristic:
        - Compute an expected bid from hand strength, opponent strength, and trump confidence.
        - Score prefers bids near expected; penalize distance quadratically.
        """
        # Expected bid in [MIN_BID, MAX_BID]
        # Estimate effective strength (0..1)
        effective_strength = (
            0.6 * hand_strength +                # own hand points/suit shape
            0.3 * trump_confidence +             # how sure we are about trump
            0.1 * (1.0 - opponent_strength)      # weaker opponents → bid more
        )
        effective_strength = max(0.0, min(1.0, effective_strength))

        # Map to bid range with a mild floor above current min
        expected_bid = MIN_BID + effective_strength * (MAX_BID - MIN_BID)

        # Core utility: prefer bids closer to expected_bid, normalized
        distance = (bid - expected_bid) / (MAX_BID - MIN_BID)
        proximity_score = 1.0 - (distance * distance)  # in [-inf, 1], but practical >= -something

        # Slight preference to higher bids when strong
        high_bid_bonus = effective_strength * ((bid - MIN_BID) / (MAX_BID - MIN_BID)) * 0.15

        # Uncertainty damping: if trump_confidence is low, reduce aggressiveness
        uncertainty_damp = 0.5 + 0.5 * trump_confidence  # 0.5..1.0

        return (proximity_score + high_bid_bonus) * uncertainty_damp
    
    def _evaluate_card_with_beliefs(self, game_state: Game28State, card: Card, belief_predictions: Any) -> float:
        """Evaluate a card using belief predictions"""
        score = CARD_VALUES[card.rank]  # Base score
        
        # Extract trump prediction
        trump_probs = belief_predictions.trump_suit.cpu().numpy().flatten()
        trump_suit = SUITS[trump_probs.argmax()]
        trump_confidence = trump_probs.max()
        
        # Trump factor
        if trump_confidence > 0.8 and card.suit == trump_suit:
            score *= trump_confidence * 2.0
        
        # Lead suit factor
        if game_state.current_trick and game_state.current_trick.cards:
            lead_suit = game_state.current_trick.lead_suit
            if card.suit == lead_suit:
                # Check if we can win the trick
                current_high_card = None
                for _, trick_card in game_state.current_trick.cards:
                    if trick_card.suit == lead_suit:
                        if current_high_card is None or CARD_VALUES[trick_card.rank] > CARD_VALUES[current_high_card.rank]:
                            current_high_card = trick_card
                
                if current_high_card is None or CARD_VALUES[card.rank] > CARD_VALUES[current_high_card.rank]:
                    score += 5.0
        
        return score
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total_decisions = self.belief_decisions + self.ismcts_decisions + self.hybrid_decisions
        
        return {
            'total_decisions': total_decisions,
            'belief_decisions': self.belief_decisions,
            'ismcts_decisions': self.ismcts_decisions,
            'hybrid_decisions': self.hybrid_decisions,
            'belief_percentage': self.belief_decisions / total_decisions if total_decisions > 0 else 0,
            'ismcts_percentage': self.ismcts_decisions / total_decisions if total_decisions > 0 else 0,
            'hybrid_percentage': self.hybrid_decisions / total_decisions if total_decisions > 0 else 0,
        }
