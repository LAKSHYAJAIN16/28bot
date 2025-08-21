import copy
from .ismcts import ismcts_plan
from .mcts_core import simulate_from_state, SearchConfig


def policy_move(env, iterations, search_mode: str = "regular", mcts_config: dict = None):
    state = env.get_state()
    
    # Check if game is done
    if state.get("done", False) or all(len(hand) == 0 for hand in state.get("hands", [])):
        raise ValueError("Game is already finished")
    
    # Check if current player has cards
    current_hand = state["hands"][state["turn"]]
    if not current_hand:
        # Current player has no cards but game isn't marked as done
        # This might be a game state inconsistency
        if all(len(hand) == 0 for hand in state.get("hands", [])):
            raise ValueError("Game is finished - all hands empty")
        else:
            raise ValueError(f"Current player {state['turn']} has no cards but other players do")
    
    # Use MCTS config if provided, otherwise use defaults
    if mcts_config:
        samples = mcts_config.get('mcts_samples', 24)
        iters_per_sample = mcts_config.get('mcts_iters_per_sample', max(20, iterations // 4))
        c_puct = mcts_config.get('mcts_c_puct', 1.5)
    else:
        samples = 24  # Increased from 16 for better exploration
        iters_per_sample = max(20, iterations // 4)  # More iterations per sample
        c_puct = 1.5  # Increased exploration
    
    cfg = SearchConfig(mode=search_mode, c_puct=c_puct)
    move, _ = ismcts_plan(env, state, iterations=iters_per_sample, samples=samples, config=cfg)
    
    # Safety check: ensure move is not None
    if move is None:
        # Fallback to first valid move
        valid_moves = env.valid_moves(state["hands"][state["turn"]])
        if valid_moves:
            move = valid_moves[0]
        else:
            # If no valid moves, check if game is finished
            if state.get("done", False) or all(len(hand) == 0 for hand in state.get("hands", [])):
                raise ValueError("Game is already finished")
            else:
                # This shouldn't happen - player has cards but no valid moves
                # Try to get any card from the hand as a last resort
                current_hand = state["hands"][state["turn"]]
                if current_hand:
                    move = current_hand[0]  # Just play the first card
                else:
                    raise ValueError("No valid moves available and move is None")
    
    return move


