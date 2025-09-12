#!/usr/bin/env python3
"""
Train the improved belief model using real game data
"""

import sys
import os
import torch
import numpy as np
import re
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from belief_model.improved_belief_net import ImprovedBeliefNetwork
from game28.game_state import Game28State, Card, Trick, GamePhase
from game28.constants import SUITS, RANKS, CARD_VALUES


@dataclass
class TrainingExample:
    """A single training example with proper targets"""
    game_state: Game28State
    player_id: int
    target_trump: Optional[str] = None
    target_opponent_hands: Optional[Dict[int, List[Card]]] = None
    target_void_suits: Optional[Dict[int, List[str]]] = None


def parse_game_log(log_file_path: str) -> List[TrainingExample]:
    """Parse a game log file to extract training examples"""
    examples = []
    
    with open(log_file_path, 'r') as f:
        content = f.read()
    
    # Extract initial hands
    initial_hands_match = re.search(r'Player (\d+) initial hand \(bidding\): \[(.*?)\]', content)
    if initial_hands_match:
        player_id = int(initial_hands_match.group(1))
        hand_str = initial_hands_match.group(2)
        # Parse cards properly: "8D', 'AH', 'KH', 'JD" -> [Card('D', '8'), Card('H', 'A'), ...]
        cards = []
        for card_str in hand_str.split("', '"):
            card_str = card_str.strip("'")
            if len(card_str) >= 2:
                rank = card_str[:-1]  # Everything except last character
                suit = card_str[-1]   # Last character
                if rank in RANKS and suit in SUITS:
                    cards.append(Card(suit, rank))
        initial_hand = cards
    
    # Extract full hands
    full_hands = {}
    full_hands_matches = re.findall(r'Player (\d+) full hand: \[(.*?)\]', content)
    for match in full_hands_matches:
        player_id = int(match[0])
        hand_str = match[1]
        # Parse cards properly
        cards = []
        for card_str in hand_str.split("', '"):
            card_str = card_str.strip("'")
            if len(card_str) >= 2:
                rank = card_str[:-1]  # Everything except last character
                suit = card_str[-1]   # Last character
                if rank in RANKS and suit in SUITS:
                    cards.append(Card(suit, rank))
        full_hands[player_id] = cards
    
    # Extract trump suit
    trump_match = re.search(r'Bidder sets concealed trump suit: ([CDHS])', content)
    trump_suit = trump_match.group(1) if trump_match else None
    
    # Extract auction winner
    winner_match = re.search(r'Auction winner: Player (\d+) with bid (\d+)', content)
    if winner_match:
        bidder = int(winner_match.group(1))
        winning_bid = int(winner_match.group(2))
    else:
        bidder = None
        winning_bid = None
    
    # Extract trick information for more training examples
    trick_matches = re.findall(r'Trick (\d+) order: (.*?)(?=\n|$)', content)
    trick_data = []
    for match in trick_matches:
        trick_num = int(match[0])
        trick_order = match[1]
        # Parse trick order: "P0:8D, P1:AH, P2:KH, P3:JD"
        cards_played = []
        for card_play in trick_order.split(', '):
            if ':' in card_play:
                player_card = card_play.split(':')
                if len(player_card) == 2:
                    player = int(player_card[0][1:])  # Remove 'P' prefix
                    card_str = player_card[1]
                    if len(card_str) >= 2:
                        rank = card_str[:-1]
                        suit = card_str[-1]
                        if rank in RANKS and suit in SUITS:
                            cards_played.append((player, Card(suit, rank)))
        trick_data.append((trick_num, cards_played))
    
    # Create training examples for each player at different game phases
    
    # Phase 1: Bidding phase (before trump is known)
    if initial_hands_match and full_hands:
        bidding_state = Game28State()
        bidding_state.phase = GamePhase.BIDDING
        bidding_state.hands = full_hands
        bidding_state.bidder = bidder
        bidding_state.winning_bid = winning_bid
        
        for player_id in range(4):
            # Target: predict trump suit
            example = TrainingExample(
                game_state=bidding_state,
                player_id=player_id,
                target_trump=trump_suit
            )
            examples.append(example)
    
    # Phase 2: After trump is revealed
    if trump_suit and full_hands:
        revealed_state = Game28State()
        revealed_state.phase = GamePhase.REVEALED
        revealed_state.hands = full_hands
        revealed_state.trump_suit = trump_suit
        revealed_state.bidder = bidder
        revealed_state.winning_bid = winning_bid
        
        for player_id in range(4):
            # Target: predict opponent hands and void suits
            target_opponent_hands = {}
            target_void_suits = {}
            
            for opp_id in range(4):
                if opp_id != player_id and opp_id in full_hands:
                    target_opponent_hands[opp_id] = full_hands[opp_id]
                    
                    # Calculate void suits (suits the opponent doesn't have)
                    opp_suits = set(card.suit for card in full_hands[opp_id])
                    void_suits = [suit for suit in SUITS if suit not in opp_suits]
                    target_void_suits[opp_id] = void_suits
            
            example = TrainingExample(
                game_state=revealed_state,
                player_id=player_id,
                target_trump=trump_suit,
                target_opponent_hands=target_opponent_hands,
                target_void_suits=target_void_suits
            )
            examples.append(example)
    
    # Phase 3: During trick play (create examples for each trick)
    if trump_suit and full_hands and trick_data:
        for trick_num, cards_played in trick_data:
            # Create game state at this point in the game
            trick_state = Game28State()
            trick_state.phase = GamePhase.REVEALED if trump_suit else GamePhase.CONCEALED
            trick_state.trump_suit = trump_suit
            trick_state.bidder = bidder
            trick_state.winning_bid = winning_bid
            
            # Remove played cards from hands
            updated_hands = {}
            for player_id, hand in full_hands.items():
                updated_hand = hand.copy()
                # Remove cards that have been played in previous tricks
                for prev_trick_num, prev_cards in trick_data:
                    if prev_trick_num < trick_num:
                        for player, card in prev_cards:
                            if player == player_id and card in updated_hand:
                                updated_hand.remove(card)
                updated_hands[player_id] = updated_hand
            
            trick_state.hands = updated_hands
            
            # Create examples for each player at this trick
            for player_id in range(4):
                target_opponent_hands = {}
                target_void_suits = {}
                
                for opp_id in range(4):
                    if opp_id != player_id and opp_id in updated_hands:
                        target_opponent_hands[opp_id] = updated_hands[opp_id]
                        
                        # Calculate void suits
                        opp_suits = set(card.suit for card in updated_hands[opp_id])
                        void_suits = [suit for suit in SUITS if suit not in opp_suits]
                        target_void_suits[opp_id] = void_suits
                
                example = TrainingExample(
                    game_state=trick_state,
                    player_id=player_id,
                    target_trump=trump_suit,
                    target_opponent_hands=target_opponent_hands,
                    target_void_suits=target_void_suits
                )
                examples.append(example)
    
    return examples


def create_training_data_from_logs(log_dir: str) -> List[TrainingExample]:
    """Create training data from all log files in a directory"""
    all_examples = []
    
    for filename in os.listdir(log_dir):
        if filename.endswith('.log'):
            log_path = os.path.join(log_dir, filename)
            try:
                examples = parse_game_log(log_path)
                all_examples.extend(examples)
                print(f"Parsed {len(examples)} examples from {filename}")
            except Exception as e:
                print(f"Error parsing {filename}: {e}")
    
    return all_examples


def train_belief_model(model: ImprovedBeliefNetwork, 
                      training_data: List[TrainingExample],
                      epochs: int = 100,
                      learning_rate: float = 0.001,
                      batch_size: int = 32) -> ImprovedBeliefNetwork:
    """Train the belief model with proper targets"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss functions
    trump_loss_fn = torch.nn.CrossEntropyLoss()
    hand_loss_fn = torch.nn.BCELoss()
    void_loss_fn = torch.nn.BCELoss()
    
    # Create models directory for saving checkpoints
    os.makedirs("models", exist_ok=True)
    
    print(f"Training on {len(training_data)} examples for {epochs} epochs")
    print(f"Using device: {device}")
    print(f"Saving model checkpoints to models/improved_belief_model_epoch_*.pt")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Shuffle training data
        np.random.shuffle(training_data)
        
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i+batch_size]
            
            batch_loss = 0.0
            
            for example in batch:
                try:
                    # Forward pass
                    predictions = model(example.game_state, example.player_id)
                    
                    # Calculate losses
                    loss = 0.0
                    
                    # Trump prediction loss
                    if example.target_trump is not None:
                        target_trump_idx = SUITS.index(example.target_trump)
                        target_trump_tensor = torch.tensor([target_trump_idx], device=device)
                        trump_loss = trump_loss_fn(predictions.trump_suit.unsqueeze(0), target_trump_tensor)
                        loss += trump_loss
                    
                    # Opponent hand prediction loss
                    if example.target_opponent_hands is not None:
                        for opp_id, target_hand in example.target_opponent_hands.items():
                            if opp_id in predictions.opponent_hands:
                                # Create target tensor for opponent hand
                                target_tensor = torch.zeros(32, device=device)
                                for card in target_hand:
                                    suit_idx = SUITS.index(card.suit)
                                    rank_idx = RANKS.index(card.rank)
                                    card_idx = suit_idx * 8 + rank_idx
                                    target_tensor[card_idx] = 1.0
                                
                                hand_loss = hand_loss_fn(predictions.opponent_hands[opp_id], target_tensor)
                                loss += hand_loss
                    
                    # Void suit prediction loss
                    if example.target_void_suits is not None:
                        for opp_id, target_voids in example.target_void_suits.items():
                            if opp_id in predictions.void_suits:
                                # Create target tensor for void suits
                                target_void_tensor = torch.zeros(4, device=device)
                                for void_suit in target_voids:
                                    suit_idx = SUITS.index(void_suit)
                                    target_void_tensor[suit_idx] = 1.0
                                
                                void_loss = void_loss_fn(predictions.void_suits[opp_id], target_void_tensor)
                                loss += void_loss
                    
                    batch_loss += loss
                    
                except Exception as e:
                    print(f"Error processing example: {e}")
                    continue
            
            if len(batch) > 0:
                avg_batch_loss = batch_loss / len(batch)
                total_loss += avg_batch_loss.item()  # Convert to scalar for logging
                num_batches += 1
                
                # Backward pass
                avg_batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        
        if num_batches > 0:
            avg_epoch_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_epoch_loss:.4f}")
            
            # Save model checkpoint after each epoch
            checkpoint_path = f"models/improved_belief_model_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
    
    return model


def test_model_predictions(model: ImprovedBeliefNetwork):
    """Test the trained model with various scenarios"""
    
    print("\n=== Testing Model Predictions ===")
    
    # Test 1: Bidding phase with no trump known
    print("\nTest 1: Bidding Phase")
    bidding_state = Game28State()
    bidding_state.phase = GamePhase.BIDDING
    
    predictions = model(bidding_state, 0)
    trump_probs = predictions.trump_suit.cpu().numpy()
    
    print("Trump probabilities (should be roughly equal):")
    for suit, prob in zip(SUITS, trump_probs):
        print(f"  {suit}: {prob:.4f}")
    
    # Test 2: After trump is revealed
    print("\nTest 2: Trump Revealed (Diamonds)")
    revealed_state = Game28State()
    revealed_state.phase = GamePhase.REVEALED
    revealed_state.trump_suit = "D"
    
    predictions = model(revealed_state, 0)
    trump_probs = predictions.trump_suit.cpu().numpy()
    
    print("Trump probabilities (should be high for Diamonds):")
    for suit, prob in zip(SUITS, trump_probs):
        print(f"  {suit}: {prob:.4f}")
    
    # Test 3: Check opponent hand predictions
    print("\nTest 3: Opponent Hand Predictions")
    for opp_id in range(1, 4):
        if opp_id in predictions.opponent_hands:
            opp_probs = predictions.opponent_hands[opp_id].cpu().numpy()
            
            # Find highest probability cards
            top_indices = np.argsort(opp_probs)[-5:]
            print(f"Opponent {opp_id} - Top 5 most likely cards:")
            for idx in reversed(top_indices):
                suit_idx = idx // 8
                rank_idx = idx % 8
                suit = SUITS[suit_idx]
                rank = RANKS[rank_idx]
                prob = opp_probs[idx]
                print(f"  {rank}{suit}: {prob:.4f}")


def main():
    """Main training function"""
    
    print("=== Improved Belief Model Training ===")
    
    # Create model
    print("Creating model...")
    model = ImprovedBeliefNetwork()
    
    # Load training data
    print("Loading training data...")
    # Try multiple possible log directories
    possible_log_dirs = [
        os.path.join("logs", "improved_games"),
        os.path.join("28bot_v2", "logs", "improved_games"), 
        os.path.join("..", "logs", "improved_games"),
        os.path.join("logs", "simulation", "comprehensive"),
        os.path.join("28bot_v2", "logs", "simulation", "comprehensive")
    ]
    
    log_dir = None
    for dir_path in possible_log_dirs:
        if os.path.exists(dir_path):
            log_dir = dir_path
            print(f"Found log directory: {log_dir}")
            break
    
    if not log_dir:
        print("Error: No log directory found. Tried:")
        for dir_path in possible_log_dirs:
            print(f"  - {dir_path}")
        print("\nPlease ensure game logs exist in one of these locations.")
        return
    
    training_data = create_training_data_from_logs(log_dir)
    
    print(f"Total training examples: {len(training_data)}")
    
    if len(training_data) == 0:
        print("No training data found! Check log directory.")
        return
    
    # Train model
    print("Training model...")
    trained_model = train_belief_model(model, training_data, epochs=50)
    
    # Test model
    test_model_predictions(trained_model)
    
    # Save final model
    print("\nSaving final model...")
    os.makedirs("models", exist_ok=True)
    torch.save(trained_model.state_dict(), "models/improved_belief_model_fixed.pt")
    print("Final model saved to models/improved_belief_model_fixed.pt")
    print("All epoch checkpoints saved to models/improved_belief_model_epoch_*.pt")


if __name__ == "__main__":
    main()
