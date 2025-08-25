"""
Advanced Game Log Parser for Rich Feature Extraction
Extracts comprehensive game state information from multiple log formats
"""

import re
import os
import glob
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm

from game28.game_state import Game28State, Card, Trick
from game28.constants import *


@dataclass
class Move:
    """Represents a single move in the game"""
    player_id: int
    card: Card
    trick_number: int
    position_in_trick: int
    won_trick: bool
    points_earned: int
    lead_suit: Optional[str]
    trump_played: bool
    high_card_played: bool
    low_card_played: bool
    forced_play: bool  # Had to follow suit


@dataclass
class GameState:
    """Comprehensive game state at any point"""
    game_id: str
    timestamp: str
    phase: str  # 'bidding', 'concealed', 'revealed'
    tricks_played: int
    cards_played: int
    current_trick: Optional[Trick]
    hands: Dict[int, List[Card]]
    played_cards: List[Move]
    bidding_history: List[Tuple[int, int]]
    trump_suit: Optional[str]
    trump_revealed: bool
    trump_revealer: Optional[int]
    face_down_trump: Optional[Card]
    team_a_score: int
    team_b_score: int
    current_bid: int
    bidder: Optional[int]
    current_player: int
    game_progress: float  # 0.0 to 1.0


class AdvancedGameParser:
    """Advanced parser for extracting rich game state information"""
    
    def __init__(self):
        # Enhanced card pattern matching
        self.card_pattern = re.compile(r'(10|[2-9TJQKA])([HDCS])')
        
        # Multiple hand patterns for different log formats
        self.hand_patterns = [
            r"Player (\d+) hand\s*:\s*\[(.*?)\]",
            r"Player (\d+) hand\s*:\s*\[(.*?)\]",
            r"Player (\d+) hand\s*:\s*\[(.*?)\]"
        ]
        
        # Trump patterns
        self.trump_patterns = [
            r"AUCTION WINNER:\s*Player (\d+) with bid (\d+); chooses trump ([HDCS])",
            r"Auction winner:\s*Player (\d+) with bid (\d+); chooses trump ([HDCS])",
            r"trump suit:\s*([HDCS])",
            r"chooses trump ([HDCS])",
            r"concealed trump suit:\s*([HDCS])",
            r"Bidder sets concealed trump suit:\s*([HDCS])"
        ]
        
        # Bidding patterns
        self.bid_patterns = [
            r"Bid:\s*Player (\d+) proposes (\d+)",
            r"Pass:\s*Player (\d+)",
            r"Pass \(locked\):\s*Player (\d+)",
            r"AUCTION WINNER:\s*Player (\d+) with bid (\d+)",
            r"Auction winner:\s*Player (\d+) with bid (\d+)"
        ]
        
        # Play patterns
        self.play_patterns = [
            r"Player (\d+) plays ([2-9TJQKA][HDCS])",
            r"Player (\d+) plays ([2-9TJQKA][HDCS])"
        ]
        
        # Trick outcome patterns
        self.trick_patterns = [
            r"Player (\d+) won the hand:\s*(\d+) points",
            r"Player (\d+) won the trick:\s*(\d+) points"
        ]
        
        # Game outcome patterns
        self.outcome_patterns = [
            r"Game \d+ final points:\s*Team A=(\d+),\s*Team B=(\d+)",
            r"Final score:\s*Team A=(\d+),\s*Team B=(\d+)"
        ]
    
    def parse_card(self, card_str: str) -> Card:
        """Parse card string to Card object"""
        match = self.card_pattern.match(card_str)
        if match:
            rank, suit = match.groups()
            return Card(suit, rank)
        raise ValueError(f"Invalid card format: {card_str}")
    
    def parse_hand(self, hand_str: str) -> List[Card]:
        """Parse hand string to list of Card objects"""
        cards = re.findall(r"'([^']+)'", hand_str)
        return [self.parse_card(card) for card in cards]
    
    def extract_game_states(self, log_path: str) -> List[GameState]:
        """Extract multiple game states from a single log file"""
        game_states = []
        
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Extract game metadata
        game_id = os.path.basename(log_path).replace('.log', '')
        
        # Extract initial hands
        hands = self._extract_hands(content)
        if len(hands) != 4:
            return game_states  # Skip incomplete games
        
        # Extract trump information
        trump_info = self._extract_trump_info(content)
        
        # Extract bidding history
        bidding_history = self._extract_bidding_history(content)
        
        # Extract all plays with context
        plays = self._extract_plays_with_context(content, hands)
        
        # Extract trick outcomes
        trick_outcomes = self._extract_trick_outcomes(content)
        
        # Extract game outcome
        game_outcome = self._extract_game_outcome(content)
        
        # Generate game states at different points
        game_states = self._generate_game_states(
            game_id, hands, trump_info, bidding_history, plays, trick_outcomes, game_outcome
        )
        
        return game_states
    
    def _extract_hands(self, content: str) -> Dict[int, List[Card]]:
        """Extract all player hands"""
        hands = {}
        
        for pattern in self.hand_patterns:
            matches = re.findall(pattern, content)
            if len(matches) == 4:
                for player_id, hand_str in matches:
                    try:
                        hands[int(player_id)] = self.parse_hand(hand_str)
                    except:
                        continue
                break
        
        return hands
    
    def _extract_trump_info(self, content: str) -> Dict:
        """Extract comprehensive trump information"""
        trump_info = {
            'suit': None,
            'revealed': False,
            'revealer': None,
            'face_down_card': None
        }
        
        for pattern in self.trump_patterns:
            match = re.search(pattern, content)
            if match:
                if len(match.groups()) == 3:  # Full pattern
                    trump_info['suit'] = match.group(3)
                else:  # Simple pattern
                    trump_info['suit'] = match.group(1)
                break
        
        # Check if trump was revealed
        if re.search(r"trump revealed", content, re.IGNORECASE):
            trump_info['revealed'] = True
        
        # Extract face-down trump card
        face_down_match = re.search(r"concealed card:\s*([2-9TJQKA][HDCS])", content)
        if face_down_match:
            trump_info['face_down_card'] = self.parse_card(face_down_match.group(1))
        
        return trump_info
    
    def _extract_bidding_history(self, content: str) -> List[Tuple[int, int]]:
        """Extract complete bidding history"""
        bidding_history = []
        
        for pattern in self.bid_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if len(match) == 2:  # Actual bid
                    player_id, bid = match
                    bidding_history.append((int(player_id), int(bid)))
                else:  # Pass
                    player_id = match[0]
                    bidding_history.append((int(player_id), 0))
        
        return bidding_history
    
    def _extract_plays_with_context(self, content: str, hands: Dict[int, List[Card]]) -> List[Move]:
        """Extract all plays with rich context"""
        plays = []
        lines = content.split('\n')
        
        current_trick = 0
        position_in_trick = 0
        trick_cards = []
        trick_players = []
        
        for i, line in enumerate(lines):
            # Check for play
            for pattern in self.play_patterns:
                match = re.search(pattern, line)
                if match:
                    player_id = int(match.group(1))
                    card = self.parse_card(match.group(2))
                    
                    # Determine if this starts a new trick
                    if position_in_trick == 0:
                        current_trick += 1
                        trick_cards = []
                        trick_players = []
                    
                    # Add to current trick
                    trick_cards.append(card)
                    trick_players.append(player_id)
                    position_in_trick += 1
                    
                    # Check if trick is complete
                    if position_in_trick == 4:
                        # Determine trick winner and context
                        winner_info = self._determine_trick_winner(
                            trick_cards, trick_players, current_trick, lines, i
                        )
                        
                        # Create moves for all players in this trick
                        for j, (card, player) in enumerate(zip(trick_cards, trick_players)):
                            move = Move(
                                player_id=player,
                                card=card,
                                trick_number=current_trick,
                                position_in_trick=j,
                                won_trick=(player == winner_info['winner']),
                                points_earned=winner_info['points'] if player == winner_info['winner'] else 0,
                                lead_suit=trick_cards[0].suit if j == 0 else None,
                                trump_played=self._is_trump_play(card, trick_cards[0].suit, winner_info['trump_suit']),
                                high_card_played=self._is_high_card_play(card, trick_cards, winner_info['trump_suit']),
                                low_card_played=self._is_low_card_play(card, trick_cards, winner_info['trump_suit']),
                                forced_play=self._was_forced_play(card, trick_cards[0].suit, player, hands)
                            )
                            plays.append(move)
                        
                        # Reset for next trick
                        position_in_trick = 0
                        trick_cards = []
                        trick_players = []
                    
                    break
        
        return plays
    
    def _determine_trick_winner(self, trick_cards: List[Card], trick_players: List[int], 
                               trick_number: int, lines: List[str], current_line: int) -> Dict:
        """Determine who won the trick and get context"""
        # Look ahead for trick outcome
        for i in range(current_line, min(current_line + 10, len(lines))):
            for pattern in self.trick_patterns:
                match = re.search(pattern, lines[i])
                if match:
                    winner = int(match.group(1))
                    points = int(match.group(2))
                    return {
                        'winner': winner,
                        'points': points,
                        'trump_suit': self._infer_trump_suit(trick_cards, lines)
                    }
        
        # Fallback: determine winner based on card rankings
        return self._calculate_trick_winner(trick_cards, trick_players)
    
    def _calculate_trick_winner(self, trick_cards: List[Card], trick_players: List[int]) -> Dict:
        """Calculate trick winner based on card rankings"""
        # This is a simplified version - in practice, you'd need full game rules
        # For now, assume first card wins (this should be improved)
        return {
            'winner': trick_players[0],
            'points': 0,
            'trump_suit': None
        }
    
    def _is_trump_play(self, card: Card, lead_suit: str, trump_suit: Optional[str]) -> bool:
        """Determine if a card is a trump play"""
        if not trump_suit:
            return False
        return card.suit == trump_suit
    
    def _is_high_card_play(self, card: Card, trick_cards: List[Card], trump_suit: Optional[str]) -> bool:
        """Determine if a card is a high card play"""
        # Simplified logic - in practice, need full ranking system
        high_cards = ['A', 'K', 'Q', 'J']
        return card.rank in high_cards
    
    def _is_low_card_play(self, card: Card, trick_cards: List[Card], trump_suit: Optional[str]) -> bool:
        """Determine if a card is a low card play"""
        low_cards = ['7', '8', '9']
        return card.rank in low_cards
    
    def _was_forced_play(self, card: Card, lead_suit: str, player_id: int, all_hands: Dict[int, List[Card]]) -> bool:
        """Determine if player was forced to play this card"""
        if card.suit == lead_suit:
            return False  # Could have played any card of lead suit
        
        # Check if player had lead suit cards
        if player_id in all_hands:
            player_hand = all_hands[player_id]
            has_lead_suit = any(c.suit == lead_suit for c in player_hand)
            return has_lead_suit  # Forced if they had lead suit but didn't play it
        
        return False
    
    def _extract_trick_outcomes(self, content: str) -> List[Dict]:
        """Extract all trick outcomes"""
        outcomes = []
        for pattern in self.trick_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                outcomes.append({
                    'winner': int(match[0]),
                    'points': int(match[1])
                })
        return outcomes
    
    def _extract_game_outcome(self, content: str) -> Dict:
        """Extract final game outcome"""
        for pattern in self.outcome_patterns:
            match = re.search(pattern, content)
            if match:
                return {
                    'team_a_score': int(match.group(1)),
                    'team_b_score': int(match.group(2))
                }
        return {'team_a_score': 0, 'team_b_score': 0}
    
    def _generate_game_states(self, game_id: str, hands: Dict[int, List[Card]], 
                            trump_info: Dict, bidding_history: List[Tuple[int, int]], 
                            plays: List[Move], trick_outcomes: List[Dict], 
                            game_outcome: Dict) -> List[GameState]:
        """Generate game states at different points in the game"""
        game_states = []
        
        # State 1: After initial deal (before any plays)
        if len(hands) == 4:
            initial_state = GameState(
                game_id=game_id,
                timestamp="initial",
                phase="bidding",
                tricks_played=0,
                cards_played=0,
                current_trick=None,
                hands=hands.copy(),
                played_cards=[],
                bidding_history=bidding_history,
                trump_suit=trump_info['suit'],
                trump_revealed=trump_info['revealed'],
                trump_revealer=trump_info.get('revealer'),
                face_down_trump=trump_info.get('face_down_card'),
                team_a_score=0,
                team_b_score=0,
                current_bid=bidding_history[-1][1] if bidding_history else 0,
                bidder=bidding_history[-1][0] if bidding_history else None,
                current_player=0,
                game_progress=0.0
            )
            game_states.append(initial_state)
        
        # Generate states after each trick
        current_hands = {pid: hand.copy() for pid, hand in hands.items()}
        current_plays = []
        
        for trick_num in range(1, 8):  # 7 tricks total
            # Get plays for this trick
            trick_plays = [p for p in plays if p.trick_number == trick_num]
            if not trick_plays:
                break
            
            # Update hands and plays
            for play in trick_plays:
                if play.player_id in current_hands:
                    current_hands[play.player_id] = [c for c in current_hands[play.player_id] if c != play.card]
                current_plays.append(play)
            
            # Calculate scores
            team_a_score = sum(p.points_earned for p in current_plays if p.player_id in [0, 2])
            team_b_score = sum(p.points_earned for p in current_plays if p.player_id in [1, 3])
            
            # Determine phase
            phase = "concealed"
            if trump_info['revealed']:
                phase = "revealed"
            
            # Create game state
            state = GameState(
                game_id=game_id,
                timestamp=f"trick_{trick_num}",
                phase=phase,
                tricks_played=trick_num,
                cards_played=len(current_plays),
                current_trick=None,  # Could be enhanced to include current trick
                hands=current_hands.copy(),
                played_cards=current_plays.copy(),
                bidding_history=bidding_history,
                trump_suit=trump_info['suit'],
                trump_revealed=trump_info['revealed'],
                trump_revealer=trump_info.get('revealer'),
                face_down_trump=trump_info.get('face_down_card'),
                team_a_score=team_a_score,
                team_b_score=team_b_score,
                current_bid=bidding_history[-1][1] if bidding_history else 0,
                bidder=bidding_history[-1][0] if bidding_history else None,
                current_player=(trick_plays[-1].player_id + 1) % 4 if trick_plays else 0,
                game_progress=trick_num / 7.0
            )
            game_states.append(state)
        
        return game_states
    
    def _infer_trump_suit(self, trick_cards: List[Card], lines: List[str]) -> Optional[str]:
        """Infer trump suit from context"""
        # Look for trump information in nearby lines
        for line in lines:
            for pattern in self.trump_patterns:
                match = re.search(pattern, line)
                if match:
                    if len(match.groups()) == 3:
                        return match.group(3)
                    else:
                        return match.group(1)
        return None


def extract_all_game_states(log_dirs: List[str], max_games: Optional[int] = None) -> List[GameState]:
    """Extract all game states from log directories"""
    parser = AdvancedGameParser()
    all_states = []
    
    for log_dir in log_dirs:
        if not os.path.exists(log_dir):
            print(f"Warning: Log directory {log_dir} not found")
            continue
        
        log_files = glob.glob(os.path.join(log_dir, "*.log"))
        print(f"Processing {len(log_files)} log files in {log_dir}")
        
        for log_file in tqdm(log_files, desc=f"Processing {os.path.basename(log_dir)}"):
            try:
                game_states = parser.extract_game_states(log_file)
                all_states.extend(game_states)
                
                if max_games and len(all_states) >= max_games:
                    break
            except Exception as e:
                print(f"Error processing {log_file}: {e}")
                continue
    
    print(f"Extracted {len(all_states)} game states from {len(log_dirs)} directories")
    return all_states
