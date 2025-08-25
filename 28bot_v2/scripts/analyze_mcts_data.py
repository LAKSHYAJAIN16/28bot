#!/usr/bin/env python3
"""
Analyze MCTS game data to extract bidding patterns and improve the RL model
"""

import sys
import os
import json
import re
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple
import glob

# Add the parent directory to Python path to access game28 module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game28.game_state import Card
from game28.constants import CARD_VALUES, TOTAL_POINTS

class MCTSDataAnalyzer:
    """Analyze MCTS game data to extract bidding patterns"""
    
    def __init__(self, mcts_logs_dir: str = "../../logs/game28/mcts_games"):
        self.mcts_logs_dir = mcts_logs_dir
        self.bidding_data = []
        self.game_outcomes = []
        
    def analyze_all_games(self):
        """Analyze all MCTS game logs"""
        print(f"Analyzing MCTS games from {self.mcts_logs_dir}...")
        
        # Get all log files
        log_files = glob.glob(os.path.join(self.mcts_logs_dir, "game_*.log"))
        print(f"Found {len(log_files)} game logs")
        
        successful_games = 0
        total_games = 0
        
        for log_file in log_files:
            try:
                game_data = self.analyze_single_game(log_file)
                if game_data:
                    self.bidding_data.append(game_data)
                    successful_games += 1
                total_games += 1
            except Exception as e:
                print(f"Error analyzing {log_file}: {e}")
        
        print(f"Successfully analyzed {successful_games}/{total_games} games")
        
    def analyze_single_game(self, log_file: str) -> Dict[str, Any]:
        """Analyze a single MCTS game log"""
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Extract game ID
        game_id_match = re.search(r'GAME (\d+)', content)
        game_id = game_id_match.group(1) if game_id_match else "unknown"
        
        # Extract bidding information
        bidding_info = self.extract_bidding_info(content)
        
        # Extract game outcome
        outcome_info = self.extract_game_outcome(content)
        
        if not bidding_info or not outcome_info:
            return None
        
        return {
            'game_id': game_id,
            'log_file': log_file,
            'bidding': bidding_info,
            'outcome': outcome_info
        }
    
    def extract_bidding_info(self, content: str) -> Dict[str, Any]:
        """Extract bidding information from game log"""
        bidding_data = {
            'auction_winner': None,
            'winning_bid': None,
            'trump_suit': None,
            'player_hands': {},
            'bidding_rounds': []
        }
        
        # Extract auction winner
        winner_match = re.search(r'AUCTION WINNER: Player (\d+) with bid (\d+)', content)
        if winner_match:
            bidding_data['auction_winner'] = int(winner_match.group(1))
            bidding_data['winning_bid'] = int(winner_match.group(2))
        
        # Extract trump suit
        trump_match = re.search(r'chooses trump ([HDCS])', content)
        if trump_match:
            bidding_data['trump_suit'] = trump_match.group(1)
        
        # Extract player hands (initial hands)
        hand_pattern = r'Player (\d+) hand\s*:\s*\[(.*?)\]'
        for match in re.finditer(hand_pattern, content):
            player = int(match.group(1))
            hand_str = match.group(2)
            cards = [card.strip().strip("'") for card in hand_str.split(',')]
            bidding_data['player_hands'][player] = cards
        
        # Extract bidding rounds
        bid_pattern = r'(?:Bid|Pass).*?Player (\d+).*?(?:proposes (\d+)|\(locked\)).*?bidding analysis:(.*?)(?=\n\n|\n[A-Z]|$)'
        for match in re.finditer(bid_pattern, content, re.DOTALL):
            player = int(match.group(1))
            bid_value = match.group(2)
            analysis = match.group(3)
            
            bidding_round = {
                'player': player,
                'action': 'bid' if bid_value else 'pass',
                'bid_value': int(bid_value) if bid_value else None,
                'analysis': analysis.strip()
            }
            bidding_data['bidding_rounds'].append(bidding_round)
        
        return bidding_data
    
    def extract_game_outcome(self, content: str) -> Dict[str, Any]:
        """Extract game outcome information"""
        outcome_data = {
            'final_scores': None,
            'bid_success': None,
            'game_score_delta': None,
            'game_complete': False,
            'player_points': {}  # Points scored by each player
        }
        
        # Look for final scores
        scores_match = re.search(r'Game \d+ final points: Team A=(\d+), Team B=(\d+)', content)
        if scores_match:
            outcome_data['final_scores'] = [int(scores_match.group(1)), int(scores_match.group(2))]
            outcome_data['game_complete'] = True
        
        # Extract individual player points from trick wins
        # Look for patterns like "Player X won the hand: Y points"
        trick_pattern = r'Player (\d+) won the hand: (\d+) points'
        for match in re.finditer(trick_pattern, content):
            player = int(match.group(1))
            points = int(match.group(2))
            if player not in outcome_data['player_points']:
                outcome_data['player_points'][player] = 0
            outcome_data['player_points'][player] += points
        
        # Check if bid was successful
        if outcome_data['final_scores']:
            # Extract bidding info again to get current game's data
            bidding_info = self.extract_bidding_info(content)
            winning_bid = bidding_info.get('winning_bid')
            auction_winner = bidding_info.get('auction_winner')
            
            if auction_winner is not None and winning_bid is not None:
                # Determine which team the winner is on
                winner_team = 'A' if auction_winner in [0, 2] else 'B'
                team_score = outcome_data['final_scores'][0] if winner_team == 'A' else outcome_data['final_scores'][1]
                
                outcome_data['bid_success'] = team_score >= winning_bid
                outcome_data['game_score_delta'] = 1 if outcome_data['bid_success'] else -1
        
        return outcome_data
    

    
    def analyze_bidding_patterns(self):
        """Analyze bidding patterns from the collected data"""
        print("\n" + "="*60)
        print("ANALYZING BIDDING PATTERNS")
        print("="*60)
        
        if not self.bidding_data:
            print("No bidding data available")
            return
        
        # Statistics
        total_games = len(self.bidding_data)
        successful_bids = sum(1 for game in self.bidding_data if game['outcome']['bid_success'])
        success_rate = successful_bids / total_games if total_games > 0 else 0
        
        print(f"Total games analyzed: {total_games}")
        print(f"Successful bids: {successful_bids}")
        print(f"Bid success rate: {success_rate:.3f}")
        
        # Analyze bid values
        bid_values = []
        for game in self.bidding_data:
            if game['bidding']['winning_bid']:
                bid_values.append(game['bidding']['winning_bid'])
        
        if bid_values:
            avg_bid = sum(bid_values) / len(bid_values)
            print(f"Average winning bid: {avg_bid:.2f}")
            print(f"Bid range: {min(bid_values)} - {max(bid_values)}")
        
        # Analyze bid values and success rates
        print("\n--- Bid Analysis ---")
        bid_success_data = []
        
        for game in self.bidding_data:
            winner = game['bidding']['auction_winner']
            winning_bid = game['bidding']['winning_bid']
            
            if winner is not None and winning_bid is not None:
                bid_success_data.append({
                    'bid': winning_bid,
                    'success': game['outcome']['bid_success'],
                    'player_points': game['outcome']['player_points'].get(winner, 0)
                })
        
        # Group by bid ranges
        bid_ranges = {
            '16-18': [],
            '19-21': [],
            '22-24': [],
            '25-28': []
        }
        
        for item in bid_success_data:
            bid = item['bid']
            if bid <= 18:
                bid_ranges['16-18'].append(item)
            elif bid <= 21:
                bid_ranges['19-21'].append(item)
            elif bid <= 24:
                bid_ranges['22-24'].append(item)
            else:
                bid_ranges['25-28'].append(item)
        
        for range_name, items in bid_ranges.items():
            if items:
                avg_bid = sum(item['bid'] for item in items) / len(items)
                success_rate = sum(1 for item in items if item['success']) / len(items)
                avg_points = sum(item['player_points'] for item in items) / len(items)
                print(f"{range_name}: avg_bid={avg_bid:.1f}, success_rate={success_rate:.3f}, avg_points={avg_points:.1f} ({len(items)} games)")
        
        # Analyze player points distribution
        print("\n--- Player Points Analysis ---")
        all_player_points = []
        for game in self.bidding_data:
            for player, points in game['outcome']['player_points'].items():
                all_player_points.append(points)
        
        if all_player_points:
            avg_points = sum(all_player_points) / len(all_player_points)
            max_points = max(all_player_points)
            min_points = min(all_player_points)
            print(f"Average points per player: {avg_points:.1f}")
            print(f"Points range: {min_points} - {max_points}")
            print(f"Total player-point observations: {len(all_player_points)}")
        
        # Analyze trump suit preferences
        print("\n--- Trump Suit Analysis ---")
        trump_counts = Counter()
        trump_success = defaultdict(list)
        
        for game in self.bidding_data:
            trump = game['bidding']['trump_suit']
            success = game['outcome']['bid_success']
            
            if trump:
                trump_counts[trump] += 1
                trump_success[trump].append(success)
        
        for trump, count in trump_counts.most_common():
            success_list = [s for s in trump_success[trump] if s is not None]
            if success_list:
                success_rate = sum(success_list) / len(success_list)
                print(f"Trump {trump}: {count} times, success_rate={success_rate:.3f}")
            else:
                print(f"Trump {trump}: {count} times, success_rate=N/A")
    
    def generate_training_data(self) -> List[Dict[str, Any]]:
        """Generate training data from MCTS games for point prediction"""
        training_data = []
        
        for game in self.bidding_data:
            game_complete = game['outcome']['game_complete']
            
            # Only include complete games
            if not game_complete:
                continue
            
            # Create training examples for each player
            for player_id, hand in game['bidding']['player_hands'].items():
                if len(hand) == 7:  # Ensure we have full hands
                    # Get points scored by this player
                    player_points = game['outcome']['player_points'].get(player_id, 0)
                    
                    # Create training example
                    training_example = {
                        'initial_hand': hand,  # Full 7-card hand
                        'player_id': player_id,
                        'actual_points': player_points,  # Points actually scored
                        'game_id': game['game_id'],
                        'trump_suit': game['bidding']['trump_suit'],
                        'auction_winner': game['bidding']['auction_winner'],
                        'winning_bid': game['bidding']['winning_bid'],
                        'bid_success': game['outcome']['bid_success']
                    }
                    
                    training_data.append(training_example)
        
        return training_data
    
    def save_analysis_results(self, output_file: str = "mcts_bidding_analysis.json"):
        """Save analysis results to file"""
        results = {
            'summary': {
                'total_games': len(self.bidding_data),
                'successful_bids': sum(1 for game in self.bidding_data if game['outcome']['bid_success']),
                'success_rate': sum(1 for game in self.bidding_data if game['outcome']['bid_success']) / len(self.bidding_data) if self.bidding_data else 0
            },
            'training_data': self.generate_training_data(),
            'detailed_games': self.bidding_data
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nAnalysis results saved to {output_file}")
        print(f"Generated {len(results['training_data'])} training examples")

def main():
    """Main analysis function"""
    analyzer = MCTSDataAnalyzer()
    
    # Analyze all games
    analyzer.analyze_all_games()
    
    # Analyze patterns
    analyzer.analyze_bidding_patterns()
    
    # Save results
    analyzer.save_analysis_results()
    
if __name__ == "__main__":
    main()
