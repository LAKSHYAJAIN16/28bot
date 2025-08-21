"""
Visualization and explanation components for Game 28 AI decisions
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
import pandas as pd

from game28.game_state import Game28State, Card
from game28.constants import *
from belief_model.belief_net import BeliefState


class BidExplanation:
    """
    Generates explanations for bidding decisions
    """
    
    def __init__(self):
        self.colors = {
            'hearts': '#ff6b6b',
            'diamonds': '#ffa726',
            'clubs': '#66bb6a',
            'spades': '#42a5f5'
        }
    
    def explain_bid(self, game_state: Game28State, player_id: int, bid: int, 
                   confidence: float = 0.0, belief_state: Optional[BeliefState] = None) -> Dict:
        """
        Generate explanation for a bidding decision
        
        Args:
            game_state: Current game state
            player_id: ID of the player making the bid
            bid: The bid that was made
            confidence: Confidence in the decision
            belief_state: Belief state for opponent hands
            
        Returns:
            Dictionary with explanation components
        """
        hand = game_state.hands[player_id]
        
        # Calculate hand strength
        hand_strength = self._calculate_hand_strength(hand)
        
        # Analyze suit distribution
        suit_analysis = self._analyze_suits(hand)
        
        # Calculate expected points
        expected_points = self._calculate_expected_points(hand, belief_state)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(hand_strength, suit_analysis, expected_points, bid)
        
        # Create explanation
        explanation = {
            'bid': bid,
            'confidence': confidence,
            'hand_strength': hand_strength,
            'suit_analysis': suit_analysis,
            'expected_points': expected_points,
            'reasoning': reasoning,
            'hand': [str(card) for card in hand],
            'current_bid': game_state.current_bid,
            'bid_history': game_state.bid_history
        }
        
        return explanation
    
    def _calculate_hand_strength(self, hand: List[Card]) -> Dict:
        """Calculate hand strength metrics"""
        total_points = sum(CARD_VALUES[card.rank] for card in hand)
        point_ratio = total_points / TOTAL_POINTS
        
        # Count high cards (J, 9, A, 10)
        high_cards = sum(1 for card in hand if CARD_VALUES[card.rank] > 0)
        
        # Calculate suit lengths
        suit_lengths = {}
        for suit in SUITS:
            suit_lengths[suit] = sum(1 for card in hand if card.suit == suit)
        
        return {
            'total_points': total_points,
            'point_ratio': point_ratio,
            'high_cards': high_cards,
            'suit_lengths': suit_lengths,
            'strength_category': self._categorize_strength(point_ratio, high_cards)
        }
    
    def _analyze_suits(self, hand: List[Card]) -> Dict:
        """Analyze suit distribution"""
        suit_analysis = {}
        
        for suit in SUITS:
            suit_cards = [card for card in hand if card.suit == suit]
            suit_points = sum(CARD_VALUES[card.rank] for card in suit_cards)
            
            suit_analysis[suit] = {
                'count': len(suit_cards),
                'points': suit_points,
                'high_cards': [card for card in suit_cards if CARD_VALUES[card.rank] > 0],
                'strength': self._categorize_suit_strength(len(suit_cards), suit_points)
            }
        
        return suit_analysis
    
    def _calculate_expected_points(self, hand: List[Card], belief_state: Optional[BeliefState]) -> float:
        """Calculate expected points based on belief state"""
        if not belief_state:
            return sum(CARD_VALUES[card.rank] for card in hand)
        
        # Simple heuristic: assume we can win tricks with high cards
        expected_points = sum(CARD_VALUES[card.rank] for card in hand)
        
        # Add bonus for strong suits
        for suit in SUITS:
            suit_cards = [card for card in hand if card.suit == suit]
            if len(suit_cards) >= 3:
                expected_points += 1  # Bonus for long suits
        
        return expected_points
    
    def _categorize_strength(self, point_ratio: float, high_cards: int) -> str:
        """Categorize hand strength"""
        if point_ratio > 0.6 and high_cards >= 4:
            return "Very Strong"
        elif point_ratio > 0.4 and high_cards >= 3:
            return "Strong"
        elif point_ratio > 0.25 and high_cards >= 2:
            return "Moderate"
        else:
            return "Weak"
    
    def _categorize_suit_strength(self, count: int, points: int) -> str:
        """Categorize suit strength"""
        if count >= 4 and points >= 3:
            return "Very Strong"
        elif count >= 3 and points >= 2:
            return "Strong"
        elif count >= 2 and points >= 1:
            return "Moderate"
        else:
            return "Weak"
    
    def _generate_reasoning(self, hand_strength: Dict, suit_analysis: Dict, 
                          expected_points: float, bid: int) -> List[str]:
        """Generate reasoning for the bid"""
        reasoning = []
        
        # Hand strength reasoning
        strength = hand_strength['strength_category']
        reasoning.append(f"Hand strength: {strength} ({hand_strength['total_points']} points)")
        
        # Suit analysis
        strong_suits = [suit for suit, analysis in suit_analysis.items() 
                       if analysis['strength'] in ['Strong', 'Very Strong']]
        if strong_suits:
            reasoning.append(f"Strong suits: {', '.join(strong_suits)}")
        
        # Expected points
        reasoning.append(f"Expected points: {expected_points:.1f}")
        
        # Bid reasoning
        if bid == -1:
            reasoning.append("Decision: Pass (hand too weak for current bid)")
        else:
            reasoning.append(f"Decision: Bid {bid} (confident in making the bid)")
        
        return reasoning
    
    def create_bid_visualization(self, explanation: Dict) -> go.Figure:
        """Create interactive visualization of bid explanation"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Hand Strength', 'Suit Distribution', 'Bid Confidence', 'Point Analysis'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "indicator"}, {"type": "bar"}]]
        )
        
        # Hand strength bar chart
        hand_strength = explanation['hand_strength']
        fig.add_trace(
            go.Bar(
                x=['Total Points', 'Point Ratio', 'High Cards'],
                y=[hand_strength['total_points'], hand_strength['point_ratio'], hand_strength['high_cards']],
                name='Hand Metrics',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # Suit distribution pie chart
        suit_lengths = hand_strength['suit_lengths']
        fig.add_trace(
            go.Pie(
                labels=list(suit_lengths.keys()),
                values=list(suit_lengths.values()),
                name='Suit Distribution'
            ),
            row=1, col=2
        )
        
        # Bid confidence gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=explanation['confidence'] * 100,
                title={'text': "Bid Confidence (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "green"}]}
            ),
            row=2, col=1
        )
        
        # Point analysis
        suit_analysis = explanation['suit_analysis']
        suits = list(suit_analysis.keys())
        points = [suit_analysis[suit]['points'] for suit in suits]
        
        fig.add_trace(
            go.Bar(
                x=suits,
                y=points,
                name='Points per Suit',
                marker_color=['red', 'orange', 'green', 'blue']
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"Bid Explanation: {explanation['bid'] if explanation['bid'] != -1 else 'Pass'}",
            height=600,
            showlegend=False
        )
        
        return fig


class BeliefVisualization:
    """
    Visualizes belief state and opponent hand predictions
    """
    
    def __init__(self):
        self.suit_colors = {
            'H': '#ff6b6b',  # Hearts - red
            'D': '#ffa726',  # Diamonds - orange
            'C': '#66bb6a',  # Clubs - green
            'S': '#42a5f5'   # Spades - blue
        }
    
    def visualize_belief_state(self, belief_state: BeliefState, player_id: int) -> go.Figure:
        """Create visualization of belief state"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Opponent 1 Belief', 'Opponent 2 Belief', 
                          'Opponent 3 Belief', 'Card Distribution'),
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
                   [{"type": "heatmap"}, {"type": "bar"}]]
        )
        
        # Create heatmaps for each opponent
        opponents = [opp_id for opp_id in range(4) if opp_id != player_id]
        
        for i, opp_id in enumerate(opponents):
            if opp_id in belief_state.opponent_hands:
                probs = np.array(belief_state.opponent_hands[opp_id]).reshape(4, 8)
                
                fig.add_trace(
                    go.Heatmap(
                        z=probs,
                        x=RANKS,
                        y=SUITS,
                        colorscale='Blues',
                        name=f'Player {opp_id}',
                        showscale=(i == 0)
                    ),
                    row=(i // 2) + 1, col=(i % 2) + 1
                )
        
        # Card distribution bar chart
        all_probs = []
        for opp_id in opponents:
            if opp_id in belief_state.opponent_hands:
                all_probs.extend(belief_state.opponent_hands[opp_id])
        
        if all_probs:
            fig.add_trace(
                go.Bar(
                    x=[f"{rank}{suit}" for suit in SUITS for rank in RANKS],
                    y=all_probs,
                    name='Average Probability',
                    marker_color='lightcoral'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title=f"Belief State for Player {player_id}",
            height=600
        )
        
        return fig
    
    def create_card_probability_heatmap(self, belief_state: BeliefState, 
                                      opponent_id: int) -> go.Figure:
        """Create heatmap of card probabilities for a specific opponent"""
        if opponent_id not in belief_state.opponent_hands:
            return go.Figure()
        
        probs = np.array(belief_state.opponent_hands[opponent_id]).reshape(4, 8)
        
        fig = go.Figure(data=go.Heatmap(
            z=probs,
            x=RANKS,
            y=SUITS,
            colorscale='Blues',
            text=probs.round(2),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Probability")
        ))
        
        fig.update_layout(
            title=f"Card Probabilities for Opponent {opponent_id}",
            xaxis_title="Rank",
            yaxis_title="Suit"
        )
        
        return fig
    
    def visualize_belief_evolution(self, belief_history: List[BeliefState], 
                                 player_id: int, opponent_id: int) -> go.Figure:
        """Visualize how beliefs evolve over time"""
        if not belief_history:
            return go.Figure()
        
        # Extract probabilities for specific opponent over time
        time_steps = list(range(len(belief_history)))
        card_probs = []
        
        for belief_state in belief_history:
            if opponent_id in belief_state.opponent_hands:
                card_probs.append(belief_state.opponent_hands[opponent_id])
            else:
                card_probs.append([0.0] * 32)
        
        card_probs = np.array(card_probs)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=card_probs.T,
            x=time_steps,
            y=[f"{rank}{suit}" for suit in SUITS for rank in RANKS],
            colorscale='Blues',
            colorbar=dict(title="Probability")
        ))
        
        fig.update_layout(
            title=f"Belief Evolution for Opponent {opponent_id}",
            xaxis_title="Time Step",
            yaxis_title="Cards"
        )
        
        return fig
    
    def create_summary_statistics(self, belief_state: BeliefState, 
                                player_id: int) -> Dict:
        """Create summary statistics of belief state"""
        summary = {
            'player_id': player_id,
            'known_cards': len(belief_state.known_cards),
            'played_cards': len(belief_state.played_cards),
            'opponents': {}
        }
        
        for opp_id in range(4):
            if opp_id != player_id and opp_id in belief_state.opponent_hands:
                probs = belief_state.opponent_hands[opp_id]
                
                # Calculate statistics
                high_prob_cards = sum(1 for p in probs if p > 0.5)
                avg_prob = np.mean(probs)
                max_prob = np.max(probs)
                
                summary['opponents'][opp_id] = {
                    'high_prob_cards': high_prob_cards,
                    'average_probability': avg_prob,
                    'max_probability': max_prob,
                    'uncertainty': 1.0 - max_prob
                }
        
        return summary


def create_game_state_visualization(game_state: Game28State, player_id: int) -> go.Figure:
    """Create comprehensive game state visualization"""
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Player Hand', 'Bidding History', 'Tricks Played', 
                       'Team Scores', 'Game Points', 'Current Trick'),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Player hand
    hand = game_state.hands[player_id]
    hand_points = [CARD_VALUES[card.rank] for card in hand]
    hand_labels = [str(card) for card in hand]
    
    fig.add_trace(
        go.Bar(
            x=hand_labels,
            y=hand_points,
            name='Card Points',
            marker_color='lightgreen'
        ),
        row=1, col=1
    )
    
    # Bidding history
    if game_state.bid_history:
        players, bids = zip(*game_state.bid_history)
        fig.add_trace(
            go.Scatter(
                x=players,
                y=bids,
                mode='lines+markers',
                name='Bids',
                line=dict(color='blue')
            ),
            row=1, col=2
        )
    
    # Team scores
    teams = list(game_state.team_scores.keys())
    scores = list(game_state.team_scores.values())
    
    fig.add_trace(
        go.Bar(
            x=teams,
            y=scores,
            name='Team Scores',
            marker_color=['red', 'blue']
        ),
        row=2, col=1
    )
    
    # Game points
    fig.add_trace(
        go.Bar(
            x=teams,
            y=list(game_state.game_points.values()),
            name='Game Points',
            marker_color=['darkred', 'darkblue']
        ),
        row=2, col=2
    )
    
    # Current trick
    if game_state.current_trick.cards:
        players, cards = zip(*game_state.current_trick.cards)
        card_labels = [str(card) for card in cards]
        
        fig.add_trace(
            go.Scatter(
                x=players,
                y=[1] * len(players),
                mode='markers+text',
                text=card_labels,
                textposition="top center",
                name='Current Trick',
                marker=dict(size=20, color='orange')
            ),
            row=3, col=2
        )
    
    fig.update_layout(
        title=f"Game State for Player {player_id}",
        height=800,
        showlegend=False
    )
    
    return fig
