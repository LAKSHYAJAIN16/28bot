"""
Main game runner for 28bot v2
"""

import argparse
import os
import sys
from typing import List, Dict, Optional
import json

from game28.game_state import Game28State
from game28.constants import *
from rl_bidding.env_adapter import Game28Env
from belief_model.belief_net import BeliefNetwork
from ismcts.ismcts_bidding import ISMCTSBidding, BeliefAwareISMCTS
from viz.render import BidExplanation, BeliefVisualization, create_game_state_visualization


class Game28Runner:
    """
    Main game runner for Game 28 with AI players
    """
    
    def __init__(self, 
                 use_rl_agent: bool = False,
                 use_belief_network: bool = False,
                 use_ismcts: bool = False,
                 num_simulations: int = 1000,
                 visualize: bool = False):
        
        self.use_rl_agent = use_rl_agent
        self.use_belief_network = use_belief_network
        self.use_ismcts = use_ismcts
        self.num_simulations = num_simulations
        self.visualize = visualize
        
        # Initialize components
        self.belief_network = None
        self.ismcts = None
        self.bid_explainer = None
        self.belief_viz = None
        
        if use_belief_network:
            self.belief_network = BeliefNetwork()
            # Load trained model if available
            model_path = "models/belief_model/belief_model_final.pt"
            if os.path.exists(model_path):
                import torch
                self.belief_network.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.belief_network.eval()
                print("Loaded trained belief network")
        
        if use_ismcts:
            self.ismcts = BeliefAwareISMCTS(
                belief_network=self.belief_network,
                num_simulations=num_simulations
            )
        
        if visualize:
            self.bid_explainer = BidExplanation()
            self.belief_viz = BeliefVisualization()
    
    def run_game(self, num_rounds: int = 1, log_game: bool = True) -> Dict:
        """
        Run a complete game of 28
        
        Args:
            num_rounds: Number of rounds to play
            log_game: Whether to log game details
            
        Returns:
            Game results
        """
        game_log = []
        total_game_points = {'A': 0, 'B': 0}
        
        for round_num in range(num_rounds):
            print(f"\n=== Round {round_num + 1} ===")
            
            # Initialize game state
            game_state = Game28State()
            round_log = {
                'round': round_num + 1,
                'bidding_phase': [],
                'play_phase': [],
                'final_state': {}
            }
            
            # Bidding phase
            print("Bidding phase...")
            bidding_result = self._run_bidding_phase(game_state, round_log)
            
            if game_state.game_over:
                print("Game ended during bidding (all passed)")
                continue
            
            # Play phase
            print("Play phase...")
            play_result = self._run_play_phase(game_state, round_log)
            
            # Record final state
            round_log['final_state'] = {
                'team_scores': game_state.team_scores.copy(),
                'game_points': game_state.game_points.copy(),
                'bidder': game_state.bidder,
                'winning_bid': game_state.winning_bid,
                'trump_suit': game_state.trump_suit
            }
            
            # Update total scores
            for team in ['A', 'B']:
                total_game_points[team] += game_state.game_points[team]
            
            game_log.append(round_log)
            
            # Print round summary
            self._print_round_summary(game_state, round_num + 1)
        
        # Print final results
        self._print_final_results(total_game_points, num_rounds)
        
        results = {
            'total_rounds': num_rounds,
            'total_game_points': total_game_points,
            'game_log': game_log if log_game else None
        }
        
        return results
    
    def _run_bidding_phase(self, game_state: Game28State, round_log: Dict) -> Dict:
        """Run the bidding phase"""
        bidding_log = []
        
        while not game_state.game_over and game_state.phase == GamePhase.BIDDING:
            current_player = game_state.current_player
            legal_bids = game_state.get_legal_bids(current_player)
            
            if not legal_bids:
                break
            
            # Get AI decision
            if current_player == 0 and self.use_ismcts:  # AI player
                action, confidence = self.ismcts.select_action_with_confidence(game_state, current_player)
                
                # Generate explanation if visualization is enabled
                if self.visualize and self.bid_explainer:
                    explanation = self.bid_explainer.explain_bid(
                        game_state, current_player, action, confidence
                    )
                    print(f"AI Player {current_player} bid: {action} (confidence: {confidence:.2f})")
                    print(f"Reasoning: {explanation['reasoning']}")
            else:
                # Simple heuristic for other players
                action = self._heuristic_bid(game_state, current_player)
                confidence = 0.5
                print(f"Player {current_player} bid: {action}")
            
            # Record bid
            bid_record = {
                'player': current_player,
                'action': action,
                'confidence': confidence,
                'legal_bids': legal_bids,
                'current_bid': game_state.current_bid
            }
            bidding_log.append(bid_record)
            
            # Apply bid
            game_state.make_bid(current_player, action)
        
        round_log['bidding_phase'] = bidding_log
        return {'bids_made': len(bidding_log)}
    
    def _run_play_phase(self, game_state: Game28State, round_log: Dict) -> Dict:
        """Run the play phase"""
        play_log = []
        tricks_played = 0
        
        # Set trump if not already set
        if game_state.bidder is not None and game_state.trump_suit is None:
            # Simple trump selection
            trump_suit = self._choose_trump(game_state.hands[game_state.bidder])
            game_state.set_trump(trump_suit)
            print(f"Trump suit set to: {trump_suit}")
        
        # Play tricks
        while not game_state.game_over and len(game_state.tricks) < 8:
            current_player = game_state.current_player
            legal_cards = game_state.get_legal_plays(current_player)
            
            if not legal_cards:
                break
            
            # Get AI decision for play
            if current_player == 0:  # AI player
                card = self._ai_play_card(game_state, current_player, legal_cards)
            else:
                # Simple heuristic for other players
                card = self._heuristic_play_card(game_state, current_player, legal_cards)
            
            # Record play
            play_record = {
                'player': current_player,
                'card': str(card),
                'trick_number': len(game_state.tricks) + 1,
                'legal_cards': [str(c) for c in legal_cards]
            }
            play_log.append(play_record)
            
            # Apply play
            game_state.play_card(current_player, card)
            
            # Check if trick is complete
            if len(game_state.current_trick.cards) == 4:
                winner = game_state.current_trick.get_winner(
                    game_state.trump_suit, game_state.trump_revealed
                )
                print(f"Trick {len(game_state.tricks)} won by Player {winner}")
                tricks_played += 1
        
        round_log['play_phase'] = play_log
        return {'tricks_played': tricks_played}
    
    def _heuristic_bid(self, game_state: Game28State, player_id: int) -> int:
        """Simple heuristic bidding strategy"""
        hand = game_state.hands[player_id]
        hand_strength = sum(CARD_VALUES[card.rank] for card in hand) / TOTAL_POINTS
        
        if hand_strength > 0.6:
            # Strong hand - bid aggressively
            return min(game_state.current_bid + 2, MAX_BID)
        elif hand_strength > 0.4:
            # Moderate hand - bid moderately
            return min(game_state.current_bid + 1, MAX_BID)
        else:
            # Weak hand - pass
            return -1
    
    def _choose_trump(self, hand: List) -> str:
        """Choose trump suit based on hand"""
        suit_points = {suit: 0 for suit in SUITS}
        
        for card in hand:
            suit_points[card.suit] += CARD_VALUES[card.rank]
        
        return max(suit_points, key=suit_points.get)
    
    def _ai_play_card(self, game_state: Game28State, player_id: int, legal_cards: List) -> 'Card':
        """AI card selection (simplified)"""
        # For now, use heuristic - can be enhanced with ISMCTS for play
        return self._heuristic_play_card(game_state, player_id, legal_cards)
    
    def _heuristic_play_card(self, game_state: Game28State, player_id: int, legal_cards: List) -> 'Card':
        """Simple heuristic card selection"""
        if not game_state.current_trick.cards:
            # Leading - play highest card
            return max(legal_cards, key=lambda c: TRICK_RANKINGS[c.rank])
        else:
            # Following - try to win if possible
            winning_cards = []
            for card in legal_cards:
                if self._can_win_trick(card, game_state.current_trick, game_state.trump_suit, game_state.trump_revealed):
                    winning_cards.append(card)
            
            if winning_cards:
                return max(winning_cards, key=lambda c: TRICK_RANKINGS[c.rank])
            else:
                # Can't win - play lowest card
                return min(legal_cards, key=lambda c: TRICK_RANKINGS[c.rank])
    
    def _can_win_trick(self, card, current_trick, trump_suit, trump_revealed):
        """Check if card can win the current trick"""
        if not current_trick.cards:
            return True
        
        winning_card = current_trick.cards[0][1]
        
        # If trump is not revealed, only same suit can win
        if not trump_revealed:
            if card.suit == current_trick.lead_suit and winning_card.suit != current_trick.lead_suit:
                return True
            if card.suit != current_trick.lead_suit and winning_card.suit == current_trick.lead_suit:
                return False
            if card.suit != current_trick.lead_suit and winning_card.suit != current_trick.lead_suit:
                return False
        
        # Trump revealed or same suit comparison
        if trump_suit and card.suit == trump_suit and winning_card.suit != trump_suit:
            return True
        if trump_suit and card.suit != trump_suit and winning_card.suit == trump_suit:
            return False
        
        # Same suit comparison
        if card.suit == winning_card.suit:
            return TRICK_RANKINGS[card.rank] > TRICK_RANKINGS[winning_card.rank]
        
        return False
    
    def _print_round_summary(self, game_state: Game28State, round_num: int):
        """Print summary of the round"""
        print(f"\nRound {round_num} Summary:")
        print(f"  Team A Score: {game_state.team_scores['A']}")
        print(f"  Team B Score: {game_state.team_scores['B']}")
        
        if game_state.bidder is not None:
            bidder_team = 'A' if game_state.bidder in TEAM_A else 'B'
            team_score = game_state.team_scores[bidder_team]
            winning_bid = game_state.winning_bid
            
            print(f"  Bidder: Player {game_state.bidder} (Team {bidder_team})")
            print(f"  Winning Bid: {winning_bid}")
            print(f"  Bid Success: {'Yes' if team_score >= winning_bid else 'No'}")
            print(f"  Trump Suit: {game_state.trump_suit}")
        
        print(f"  Game Points - Team A: {game_state.game_points['A']}, Team B: {game_state.game_points['B']}")
    
    def _print_final_results(self, total_game_points: Dict, num_rounds: int):
        """Print final game results"""
        print(f"\n=== Final Results ({num_rounds} rounds) ===")
        print(f"Team A Total Game Points: {total_game_points['A']}")
        print(f"Team B Total Game Points: {total_game_points['B']}")
        
        if total_game_points['A'] > total_game_points['B']:
            print("Team A wins!")
        elif total_game_points['B'] > total_game_points['A']:
            print("Team B wins!")
        else:
            print("It's a tie!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="28bot v2 - Game 28 AI")
    parser.add_argument("--rounds", type=int, default=1, help="Number of rounds to play")
    parser.add_argument("--rl-agent", action="store_true", help="Use RL agent for bidding")
    parser.add_argument("--belief-network", action="store_true", help="Use belief network")
    parser.add_argument("--ismcts", action="store_true", help="Use ISMCTS for decisions")
    parser.add_argument("--simulations", type=int, default=1000, help="Number of ISMCTS simulations")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    parser.add_argument("--log-game", action="store_true", help="Log detailed game information")
    parser.add_argument("--output", type=str, help="Output file for game results")
    
    args = parser.parse_args()
    
    # Create game runner
    runner = Game28Runner(
        use_rl_agent=args.rl_agent,
        use_belief_network=args.belief_network,
        use_ismcts=args.ismcts,
        num_simulations=args.simulations,
        visualize=args.visualize
    )
    
    # Run game
    results = runner.run_game(num_rounds=args.rounds, log_game=args.log_game)
    
    # Save results if requested
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
