from typing import List

SUITS: List[str] = ["H", "D", "C", "S"]
RANKS: List[str] = ["7", "8", "9", "10", "J", "Q", "K", "A"]
CARD_POINTS = {"J": 3, "9": 2, "10": 1, "A": 1, "7": 0, "8": 0, "Q": 0, "K": 0}


def card_rank(card: str) -> str:
    return card[:-1]


def card_suit(card: str) -> str:
    return card[-1]


def card_value(card: str) -> int:
    return CARD_POINTS[card_rank(card)]


def rank_index(card: str) -> int:
    return RANKS.index(card_rank(card))


DECK_RANKS_FULL = RANKS
FULL_DECK = [r + s for r in DECK_RANKS_FULL for s in SUITS]
CARD_TO_INDEX = {c: i for i, c in enumerate(FULL_DECK)}


def suit_trump_strength(hand: list[str], suit: str) -> int:
    suit_cards = [c for c in hand if card_suit(c) == suit]
    if not suit_cards:
        return 0
    count = len(suit_cards)
    rank_power_sum = sum(rank_index(c) for c in suit_cards)
    point_sum = sum(card_value(c) for c in suit_cards)
    has_jack = any(card_rank(c) == "J" for c in suit_cards)
    has_nine = any(card_rank(c) == "9" for c in suit_cards)
    return 3 * count + rank_power_sum + 2 * point_sum + (5 if has_jack else 0) + (3 if has_nine else 0)

# Trick-taking strength order (highest first): J > 9 > A > 10 > K > Q > 8 > 7
_TRICK_STRENGTH_ORDER = ["7", "8", "Q", "K", "10", "A", "9", "J"]  # low -> high indices
_RANK_TO_TRICK_STRENGTH = {r: i for i, r in enumerate(_TRICK_STRENGTH_ORDER)}

def trick_rank_index(card: str) -> int:
    return _RANK_TO_TRICK_STRENGTH[card_rank(card)]


