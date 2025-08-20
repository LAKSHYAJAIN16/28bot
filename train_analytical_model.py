#!/usr/bin/env python3
"""
Train Analytical Model Coefficients using Game Log Data

This script parses the game logs to extract:
- Initial 4-card hands for each player
- Actual game results (team points)
- Bidding decisions and outcomes

Then optimizes the analytical model coefficients to minimize prediction error.
"""

import os
import re
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import minimize
import glob
from sklearn.model_selection import train_test_split

from mcts.constants import SUITS, RANKS, card_suit, card_rank, card_value, rank_index, trick_rank_index


class GameDataExtractor:
    """Extract training data from game logs."""
    
    def __init__(self, logs_dir: str = "logs/game28/mcts_games", training_log: str = "training.log"):
        self.logs_dir = logs_dir
        self.training_log = training_log
        
    def extract_game_data(self, log_file: str) -> Optional[Dict]:
        """Extract data from a single game log file."""
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Extract game info
            game_match = re.search(r'GAME (\d+)', content)
            if not game_match:
                return None
            game_id = game_match.group(1)
            
            # Extract initial 4-card hands
            hands_pattern = r'Player (\d+) \(auction 4 cards\): \[(.*?)\]'
            hands_matches = re.findall(hands_pattern, content)
            
            if len(hands_matches) != 4:
                return None
                
            hands = {}
            for player, cards_str in hands_matches:
                # Clean up the cards string and extract individual cards
                cards_str = cards_str.replace("'", "").replace('"', '')
                cards = [card.strip() for card in cards_str.split(', ') if card.strip()]
                hands[int(player)] = cards
            
            # Extract bidder and bid
            bidder_match = re.search(r'Auction winner \(bidder\): Player (\d+) with bid (\d+)', content)
            if not bidder_match:
                return None
            bidder = int(bidder_match.group(1))
            bid_value = int(bidder_match.group(2))
            
            # Extract trump suit
            trump_match = re.search(r'Bidder sets concealed trump suit: (\w)', content)
            if not trump_match:
                return None
            trump_suit = trump_match.group(1)
            
            # Extract individual game scores by summing trick points for each player
            trick_pattern = r'Player (\d+) won the hand: (\d+) points'
            trick_matches = re.findall(trick_pattern, content)
            
            # Initialize points for each player
            player_points = {0: 0, 1: 0, 2: 0, 3: 0}
            
            for player, points in trick_matches:
                player = int(player)
                points = int(points)
                player_points[player] += points
            
            # Calculate team points
            team_a_points = player_points[0] + player_points[2]  # Players 0 and 2
            team_b_points = player_points[1] + player_points[3]  # Players 1 and 3
            
            # Extract bidding analysis for the bidder
            bidding_pattern = rf'Player {bidder}.*?bidding analysis:(.*?)(?=Player \d+|AUCTION WINNER|$)'
            bidding_match = re.search(bidding_pattern, content, re.DOTALL)
            
            bidding_stats = {}
            if bidding_match:
                bidding_text = bidding_match.group(1)
                # Extract suit statistics
                suit_pattern = r'(\w): pts=\[([\d, ]+)\].*?avg=([\d.]+).*?p30=([\d.]+).*?std=([\d.]+)'
                suit_matches = re.findall(suit_pattern, bidding_text, re.DOTALL)
                for suit, pts_str, avg, p30, std in suit_matches:
                    pts = [int(x.strip()) for x in pts_str.split(',') if x.strip()]
                    bidding_stats[suit] = {
                        'points': pts,
                        'avg': float(avg),
                        'p30': float(p30),
                        'std': float(std)
                    }
            
            return {
                'game_id': game_id,
                'hands': hands,
                'bidder': bidder,
                'bid_value': bid_value,
                'trump_suit': trump_suit,
                'player_points': player_points,
                'team_a_points': team_a_points,
                'team_b_points': team_b_points,
                'bidding_stats': bidding_stats
            }
            
        except Exception as e:
            print(f"Error parsing {log_file}: {e}")
            return None
    
    def extract_all_games(self) -> List[Dict]:
        """Extract data from training.log which contains both game_start and game_end events."""
        games_data = {}
        
        try:
            with open(self.training_log, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        
                        if data.get('event') == 'game_start':
                            game_id = data['game_id']
                            games_data[game_id] = {
                                'game_id': game_id,
                                'hands': {
                                    0: data['first_four_hands'][0],
                                    1: data['first_four_hands'][1], 
                                    2: data['first_four_hands'][2],
                                    3: data['first_four_hands'][3]
                                },
                                'bidder': data['bidder'],
                                'bid_value': data['bid_value'],
                                'trump_suit': data['trump'],
                                'team_a_points': None,
                                'team_b_points': None
                            }
                        
                        elif data.get('event') == 'game_end' and 'scores' in data:
                            game_id = data['game_id']
                            if game_id in games_data:
                                team_a_final, team_b_final = data['scores']
                                games_data[game_id]['team_a_points'] = team_a_final
                                games_data[game_id]['team_b_points'] = team_b_final
                                
        except Exception as e:
            print(f"Error reading training.log: {e}")
        
        # Only keep games that have both start and end data
        complete_games = [game for game in games_data.values() 
                         if game['team_a_points'] is not None and game['team_b_points'] is not None]
        
        print(f"Extracted complete data from {len(complete_games)} games")
        return complete_games


class AnalyticalModelTrainer:
    """Train analytical model coefficients using game data."""
    
    def __init__(self, games_data: List[Dict], regularization_strength: float = 0.1):
        self.games_data = games_data
        self.regularization_strength = regularization_strength
        
        # Current coefficients (from bet_advisor.py) - more conservative initial values
        self.coefficients = {
            'base_points_multiplier': 0.1,
            'trump_base_value': 1.0,
            'high_trump_bonus': 0.5,
            'trump_length_bonus': 0.3,
            'suit_base_multiplier': 1.5,
            'suit_length_bonus': 0.5,
            'suit_high_card_bonus': 0.5,
            'non_trump_penalty': 0.8,
            'suit_flexibility': 0.2,
            'balance_bonus': 1.0,
            'trump_control_bonus': 0.5
        }
    
    def calculate_expected_points_analytical(self, hand_4_cards: List[str], trump_suit: str) -> float:
        """Calculate expected points using current coefficients."""
        # Calculate base points from cards in hand
        base_points = sum(card_value(card) for card in hand_4_cards) * self.coefficients['base_points_multiplier']
        
        # Calculate trump control value
        trump_cards = [card for card in hand_4_cards if card_suit(card) == trump_suit]
        trump_control_value = self._calculate_trump_control_value(trump_cards, trump_suit)
        
        # Calculate trick-winning potential for each suit
        suit_values = {}
        for suit in SUITS:
            suit_cards = [card for card in hand_4_cards if card_suit(card) == suit]
            if suit_cards:
                suit_values[suit] = self._calculate_suit_trick_potential(suit_cards, suit, trump_suit)
        
        # Calculate team coordination bonus
        coordination_bonus = self._calculate_team_coordination_bonus(hand_4_cards, trump_suit)
        
        # Sum up all components
        total_expected = base_points + trump_control_value + sum(suit_values.values()) + coordination_bonus
        
        return max(0, total_expected)
    
    def _calculate_trump_control_value(self, trump_cards: List[str], trump_suit: str) -> float:
        """Calculate the value of trump control based on trump cards in hand."""
        if not trump_cards:
            return 0.0
        
        # Count high trumps (J, 9, A, 10)
        high_trumps = [card for card in trump_cards if card_rank(card) in ['J', '9', 'A', '10']]
        
        # Base value: each trump card is worth extra points
        base_value = len(trump_cards) * self.coefficients['trump_base_value']
        
        # High trump bonus
        high_trump_bonus = sum(card_value(card) for card in high_trumps) * self.coefficients['high_trump_bonus']
        
        # Trump length bonus: having multiple trumps is very valuable
        length_bonus = len(trump_cards) * (len(trump_cards) - 1) * self.coefficients['trump_length_bonus']
        
        return base_value + high_trump_bonus + length_bonus
    
    def _calculate_suit_trick_potential(self, suit_cards: List[str], suit: str, trump_suit: str) -> float:
        """Calculate expected trick-winning potential for a specific suit."""
        if not suit_cards:
            return 0.0
        
        # Calculate card strength in this suit
        card_strengths = [trick_rank_index(card) for card in suit_cards]
        
        # Base value: probability of winning tricks in this suit
        base_value = sum(strength / 8.0 for strength in card_strengths) * self.coefficients['suit_base_multiplier']
        
        # Length bonus: having multiple cards in same suit is valuable
        length_bonus = len(suit_cards) * self.coefficients['suit_length_bonus']
        
        # High card bonus: J, 9, A, 10 are especially valuable
        high_cards = [card for card in suit_cards if card_rank(card) in ['J', '9', 'A', '10']]
        high_card_bonus = len(high_cards) * self.coefficients['suit_high_card_bonus']
        
        # Trump competition: if this isn't trump suit, we compete with trump
        if suit != trump_suit:
            base_value *= self.coefficients['non_trump_penalty']
            length_bonus *= self.coefficients['non_trump_penalty']
        
        return base_value + length_bonus + high_card_bonus
    
    def _calculate_team_coordination_bonus(self, hand_4_cards: List[str], trump_suit: str) -> float:
        """Calculate bonus from team coordination potential."""
        # Having multiple suits gives flexibility
        suits_present = set(card_suit(card) for card in hand_4_cards)
        suit_flexibility = len(suits_present) * self.coefficients['suit_flexibility']
        
        # Having balanced hand (not all in one suit) is good for coordination
        suit_counts = {}
        for card in hand_4_cards:
            suit = card_suit(card)
            suit_counts[suit] = suit_counts.get(suit, 0) + 1
        
        balance_bonus = 0.0
        if len(suit_counts) >= 2:
            balance_bonus = self.coefficients['balance_bonus']
        
        # Having trump control helps team coordination
        trump_control_bonus = 0.0
        if any(card_suit(card) == trump_suit for card in hand_4_cards):
            trump_control_bonus = self.coefficients['trump_control_bonus']
        
        return suit_flexibility + balance_bonus + trump_control_bonus
    
    def objective_function(self, params):
        """Objective function to minimize: MSE + L2 regularization."""
        # Update coefficients with current parameters
        param_names = list(self.coefficients.keys())
        for i, name in enumerate(param_names):
            self.coefficients[name] = params[i]
        
        total_error = 0.0
        num_predictions = 0
        
        for game in self.games_data:
            trump_suit = game['trump_suit']
            
            # Train on team performance - each player's hand predicts their team's total score
            for player in range(4):
                hand_4_cards = game['hands'][player]
                
                # Get the team this player belongs to
                if player % 2 == 0:  # Team A (players 0, 2)
                    actual_team_points = game['team_a_points']
                else:  # Team B (players 1, 3)
                    actual_team_points = game['team_b_points']
                
                # Calculate predicted team points based on this player's hand
                predicted_team_points = self.calculate_expected_points_analytical(hand_4_cards, trump_suit)
                
                # Add to error
                error = (predicted_team_points - actual_team_points) ** 2
                total_error += error
                num_predictions += 1
        
        mse = total_error / num_predictions if num_predictions > 0 else float('inf')
        
        # Add L2 regularization to prevent overfitting
        l2_penalty = self.regularization_strength * sum(param**2 for param in params)
        
        return mse + l2_penalty
    
    def train(self):
        """Train the model coefficients."""
        print("Training analytical model coefficients...")
        
        # Initial parameters
        initial_params = list(self.coefficients.values())
        param_names = list(self.coefficients.keys())
        
        # No bounds - let the model find optimal values naturally
        bounds = None
        
        # Optimize using SLSQP without bounds
        result = minimize(
            self.objective_function,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500}
        )
        
        if result.success:
            print("Training completed successfully!")
            print(f"Final MSE: {result.fun:.4f}")
            
            # Update coefficients with optimized values
            for i, name in enumerate(param_names):
                self.coefficients[name] = result.x[i]
            
            # Print optimized coefficients
            print("\nOptimized coefficients:")
            for name, value in self.coefficients.items():
                print(f"  {name}: {value:.4f}")
            
            return self.coefficients
        else:
            print(f"Training failed: {result.message}")
            return None
    
    def evaluate(self):
        """Evaluate the model on the training data."""
        print("\nEvaluating model performance...")
        
        errors = []
        predictions = []
        actuals = []
        
        for game in self.games_data:
            trump_suit = game['trump_suit']
            
            # Evaluate on team performance
            for player in range(4):
                hand_4_cards = game['hands'][player]
                
                # Get the team this player belongs to
                if player % 2 == 0:  # Team A (players 0, 2)
                    actual_team_points = game['team_a_points']
                else:  # Team B (players 1, 3)
                    actual_team_points = game['team_b_points']
                
                predicted_team_points = self.calculate_expected_points_analytical(hand_4_cards, trump_suit)
                
                error = abs(predicted_team_points - actual_team_points)
                errors.append(error)
                predictions.append(predicted_team_points)
                actuals.append(actual_team_points)
        
        mae = np.mean(errors)
        mse = np.mean([e**2 for e in errors])
        correlation = np.corrcoef(predictions, actuals)[0, 1]
        
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Correlation: {correlation:.3f}")
        
        # Show some sample predictions vs actuals
        print(f"\nSample predictions vs actuals (first 10):")
        for i in range(min(10, len(predictions))):
            print(f"  Predicted: {predictions[i]:.1f}, Actual: {actuals[i]}")
        
        # Show statistics of actual team scores
        actual_scores = []
        for game in self.games_data:
            actual_scores.append(game['team_a_points'])
            actual_scores.append(game['team_b_points'])
        print(f"\nActual score statistics:")
        print(f"  Min: {min(actual_scores)}")
        print(f"  Max: {max(actual_scores)}")
        print(f"  Mean: {np.mean(actual_scores):.2f}")
        print(f"  Std: {np.std(actual_scores):.2f}")
        
        return {
            'mae': mae,
            'mse': mse,
            'correlation': correlation
        }


def main():
    """Main training function."""
    print("Analytical Model Coefficient Training (with regularization)")
    print("==========================================================")
    
    # Extract game data
    extractor = GameDataExtractor()
    games_data = extractor.extract_all_games()
    
    if not games_data:
        print("No valid game data found!")
        return
    
    # Split data for validation
    train_data, val_data = train_test_split(games_data, test_size=0.2, random_state=42)
    print(f"Training on {len(train_data)} games, validating on {len(val_data)} games")
    
    # Train model with stronger regularization to prevent overfitting
    trainer = AnalyticalModelTrainer(train_data, regularization_strength=0.5)
    
    # Evaluate before training
    print("\nBefore training:")
    trainer.evaluate()
    
    # Train
    optimized_coefficients = trainer.train()
    
    if optimized_coefficients:
        # Evaluate after training
        print("\nAfter training:")
        trainer.evaluate()
        
        # Validate on held-out data
        print("\nValidation on held-out data:")
        val_trainer = AnalyticalModelTrainer(val_data)
        val_trainer.coefficients = optimized_coefficients
        val_metrics = val_trainer.evaluate()
        
        print(f"\nValidation MAE: {val_metrics['mae']:.2f}")
        print(f"Validation MSE: {val_metrics['mse']:.2f}")
        print(f"Validation Correlation: {val_metrics['correlation']:.3f}")
        
        # Save optimized coefficients
        with open('optimized_analytical_coefficients.json', 'w') as f:
            json.dump(optimized_coefficients, f, indent=2)
        
        print(f"\nOptimized coefficients saved to 'optimized_analytical_coefficients.json'")
        
        # Generate code snippet for bet_advisor.py
        print("\nCode snippet for bet_advisor.py:")
        print("```python")
        for name, value in optimized_coefficients.items():
            print(f"# {name}: {value:.4f}")
        print("```")


if __name__ == "__main__":
    main()
