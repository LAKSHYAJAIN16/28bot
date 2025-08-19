from .constants import SUITS, RANKS, CARD_POINTS, FULL_DECK, CARD_TO_INDEX
from .env28 import TwentyEightEnv
from .mcts_core import mcts_search, mcts_plan
from .ismcts import ismcts_plan
from .policy import policy_move
from .mcts_bidding import MonteCarloBiddingAgent
from .log_utils import log_event as log_event, Tee as Tee, open_game_log as open_game_log

if __name__=="__main__":
    from .runner import play_games
    play_games(num_games=100, iterations=10000)
