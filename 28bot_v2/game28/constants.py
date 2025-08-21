"""
Constants and game rules for Game 28
"""

from typing import Dict, List, Tuple
from enum import Enum

# Card suits and ranks
SUITS = ['H', 'D', 'C', 'S']  # Hearts, Diamonds, Clubs, Spades
RANKS = ['7', '8', '9', '10', 'J', 'Q', 'K', 'A']

# Point values for each card
CARD_VALUES = {
    '7': 0, '8': 0, '9': 2, '10': 1,
    'J': 3, 'Q': 0, 'K': 0, 'A': 1
}

# Trick rankings (highest to lowest)
TRICK_RANKINGS = {
    '7': 0, '8': 1, 'Q': 2, 'K': 3,
    '10': 4, 'A': 5, '9': 6, 'J': 7
}

# Total points in the game
TOTAL_POINTS = 28

# Bidding constants
MIN_BID = 16
MAX_BID = 28
BID_RANGE = list(range(MIN_BID, MAX_BID + 1))

# Game phases
class GamePhase(Enum):
    BIDDING = "bidding"
    CONCEALED = "concealed"
    REVEALED = "revealed"

# Team definitions
TEAM_A = [0, 2]  # Players 0 and 2
TEAM_B = [1, 3]  # Players 1 and 3

def get_team(player: int) -> List[int]:
    """Get the team for a given player"""
    return TEAM_A if player in TEAM_A else TEAM_B

def is_partner(player1: int, player2: int) -> bool:
    """Check if two players are partners"""
    return get_team(player1) == get_team(player2)

def get_opponents(player: int) -> List[int]:
    """Get the opponents for a given player"""
    team = get_team(player)
    return [p for p in range(4) if p not in team]
