"""
Training script for belief network using real game data from logs
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict, Optional
import re
import glob
from tqdm import tqdm

from .belief_net import BeliefNetwork
from game28.game_state import Game28State, Card
from game28.constants import *


class RealGameLogParser:
    """Parser for real game logs"""
    
    def __init__(self):
        self.card_pattern = re.compile(r'(10|[2-9TJQKA])([HDCS])')
    
    def parse_card(self, card_str: str) -> Card:
        """Parse card string to Card object"""
        match = self.card_pattern.match(card_str)
        if match:
            rank, suit = match.groups()
            return Card(suit, rank)  # Note: Card constructor expects (suit, rank)
        raise ValueError(f"Invalid card format: {card_str}")
    
    def parse_hand(self, hand_str: str) -> List[Card]:
        """Parse hand string to list of Card objects"""
        # Extract cards from format like "['JD', '10C', '8C', 'QS']"
        cards = re.findall(r"'([^']+)'", hand_str)
        return [self.parse_card(card) for card in cards]
    
    def parse_game_log(self, log_path: str) -> List[Dict]:
        """Parse a single game log file and extract training examples"""
        examples = []
        
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Extract initial hands - try multiple patterns
        hand_patterns = [
            r"Player (\d+) hand\s*:\s*\[(.*?)\]",  # Newer format
            r"Player (\d+) hand\s*:\s*\[(.*?)\]"   # Older format (same but different context)
        ]
        
        hand_matches = []
        for pattern in hand_patterns:
            matches = re.findall(pattern, content)
            if len(matches) == 4:  # Found all 4 players
                hand_matches = matches
                break
        
        if len(hand_matches) != 4:
            return examples  # Skip incomplete games
        
        # Parse all hands
        hands = {}
        for player_id, hand_str in hand_matches:
            try:
                hands[int(player_id)] = self.parse_hand(hand_str)
            except:
                continue
        
        if len(hands) != 4:
            return examples
        
        # Extract trump suit - try multiple patterns
        trump_patterns = [
            r"AUCTION WINNER:\s*Player (\d+) with bid (\d+); chooses trump ([HDCS])",  # Newer format
            r"Auction winner:\s*Player (\d+) with bid (\d+); chooses trump ([HDCS])",  # Older format
            r"trump suit:\s*([HDCS])",  # Alternative
            r"chooses trump ([HDCS])",  # Alternative
            r"concealed trump suit:\s*([HDCS])"  # Alternative
        ]
        
        trump_suit = None
        for pattern in trump_patterns:
            trump_match = re.search(pattern, content)
            if trump_match:
                if len(trump_match.groups()) == 3:  # First two patterns
                    trump_suit = trump_match.group(3)
                else:  # Other patterns
                    trump_suit = trump_match.group(1)
                break
        
        # Extract played cards
        play_pattern = r"Player (\d+) plays ([2-9TJQKA][HDCS])"
        play_matches = re.findall(play_pattern, content)
        
        played_cards = []
        for player_id, card_str in play_matches:
            try:
                card = self.parse_card(card_str)
                played_cards.append((int(player_id), card))
            except:
                continue
        
        # Extract bidding information - try multiple patterns
        bid_patterns = [
            r"Bid:\s*Player (\d+) proposes (\d+)",  # Both formats
            r"Pass:\s*Player (\d+)",  # Pass bids
            r"Pass \(locked\):\s*Player (\d+)"  # Locked passes
        ]
        
        bidding_history = []
        for pattern in bid_patterns:
            bid_matches = re.findall(pattern, content)
            for match in bid_matches:
                if len(match) == 2:  # Actual bid
                    player_id, bid = match
                    bidding_history.append((int(player_id), int(bid)))
                else:  # Pass
                    player_id = match[0]
                    # For passes, we'll use 0 as the bid value
                    bidding_history.append((int(player_id), 0))
        
        # Also extract auction winner
        winner_patterns = [
            r"AUCTION WINNER:\s*Player (\d+) with bid (\d+)",  # Newer format
            r"Auction winner:\s*Player (\d+) with bid (\d+)"   # Older format
        ]
        
        for pattern in winner_patterns:
            winner_match = re.search(pattern, content)
            if winner_match:
                winner_id, winner_bid = winner_match.groups()
                bidding_history.append((int(winner_id), int(winner_bid)))
                break
        
        # Create training examples for each player at different game stages
        for player_id in range(4):
            # Example 1: After initial deal (before any plays)
            if len(played_cards) >= 0:
                example = self._create_example(
                    player_id, hands, trump_suit, played_cards[:0], bidding_history, 0
                )
                if example:
                    examples.append(example)
            
            # Example 2: After first trick (4 cards played)
            if len(played_cards) >= 4:
                example = self._create_example(
                    player_id, hands, trump_suit, played_cards[:4], bidding_history, 1
                )
                if example:
                    examples.append(example)
            
            # Example 3: After half the tricks (16 cards played)
            if len(played_cards) >= 16:
                example = self._create_example(
                    player_id, hands, trump_suit, played_cards[:16], bidding_history, 4
                )
                if example:
                    examples.append(example)
        
        return examples
    
    def _create_example(self, player_id: int, hands: Dict[int, List[Card]], 
                       trump_suit: Optional[str], played_cards: List[Tuple[int, Card]], 
                       bidding_history: List[Tuple[int, int]], num_tricks: int) -> Optional[Dict]:
        """Create a training example from game state"""
        try:
            # Create input features
            input_features = self._encode_game_state(player_id, hands, trump_suit, played_cards, bidding_history, num_tricks)
            
            # Create target labels
            target_hands = {}
            for opp_id in range(4):
                if opp_id != player_id:
                    target_hands[str(opp_id)] = self._encode_hand(hands[opp_id])  # Use string keys
            
            # Create trump target
            trump_target = np.zeros(4)
            if trump_suit:
                trump_target[SUITS.index(trump_suit)] = 1.0
            
            return {
                'input': input_features,
                'target_hands': target_hands,
                'target_trump': trump_target,
                'player_id': player_id,
                'num_tricks': num_tricks
            }
        except Exception as e:
            print(f"Error creating example: {e}")
            return None
    
    def _encode_game_state(self, player_id: int, hands: Dict[int, List[Card]], 
                          trump_suit: Optional[str], played_cards: List[Tuple[int, Card]], 
                          bidding_history: List[Tuple[int, int]], num_tricks: int) -> np.ndarray:
        """Encode game state into features"""
        # Hand encoding
        hand_encoding = np.zeros(32)
        for card in hands[player_id]:
            card_idx = SUITS.index(card.suit) * 8 + RANKS.index(card.rank)
            hand_encoding[card_idx] = 1
        
        # Bidding history encoding (last 4 bids)
        bidding_history_encoding = np.zeros(4)
        for i, (player, bid) in enumerate(bidding_history[-4:]):
            bidding_history_encoding[i] = bid
        
        # Played cards encoding
        played_encoding = np.zeros(32)
        for _, card in played_cards:
            card_idx = SUITS.index(card.suit) * 8 + RANKS.index(card.rank)
            played_encoding[card_idx] = 1
        
        # Game state encoding
        game_state_encoding = np.array([
            bidding_history[-1][1] if bidding_history else 0,  # current bid
            player_id,
            0,  # phase (concealed)
            4 if trump_suit is None else SUITS.index(trump_suit),
            0,  # trump revealed
            4 if not bidding_history else bidding_history[-1][0],  # bidder
            bidding_history[-1][1] if bidding_history else 0,  # winning bid
            0,  # team A score
            0,  # team B score
            num_tricks
        ])
        
        return np.concatenate([
            hand_encoding,
            bidding_history_encoding,
            played_encoding,
            game_state_encoding
        ])
    
    def _encode_hand(self, hand: List[Card]) -> np.ndarray:
        """Encode hand as one-hot vector"""
        encoding = np.zeros(32)
        for card in hand:
            card_idx = SUITS.index(card.suit) * 8 + RANKS.index(card.rank)
            encoding[card_idx] = 1
        return encoding


class RealGameDataset(Dataset):
    """Dataset for training belief network using real game data"""
    
    def __init__(self, log_dirs: List[str], max_games: Optional[int] = None):
        self.data = []
        self.parser = RealGameLogParser()
        self._load_data(log_dirs, max_games)
    
    def _load_data(self, log_dirs: List[str], max_games: Optional[int]):
        """Load training data from log directories"""
        print("Loading real game data from logs...")
        
        total_examples = 0
        for log_dir in log_dirs:
            if not os.path.exists(log_dir):
                print(f"Warning: Log directory {log_dir} not found")
                continue
            
            # Find all log files
            log_files = glob.glob(os.path.join(log_dir, "*.log"))
            print(f"Found {len(log_files)} log files in {log_dir}")
            
            for log_file in tqdm(log_files, desc=f"Processing {os.path.basename(log_dir)}"):
                try:
                    examples = self.parser.parse_game_log(log_file)
                    self.data.extend(examples)
                    total_examples += len(examples)
                    
                    if max_games and total_examples >= max_games:
                        break
                except Exception as e:
                    print(f"Error processing {log_file}: {e}")
                    continue
        
        print(f"Loaded {len(self.data)} training examples from real games")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def train_belief_model(
    log_dirs: List[str] = ["logs/game28/mcts_games", "logs/improved_games"],
    max_games: Optional[int] = None,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    num_epochs: int = 50,
    save_dir: str = "models/belief_model"
):
    """
    Train the belief network using real game data
    
    Args:
        log_dirs: List of log directories to load data from
        max_games: Maximum number of games to load (None for all)
        batch_size: Batch size for training
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        save_dir: Directory to save the trained model
    """
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = RealGameDataset(log_dirs, max_games)
    
    def custom_collate(batch):
        """Custom collate function to handle variable target_hands keys"""
        # Get all unique keys from target_hands across the batch
        all_keys = set()
        for item in batch:
            all_keys.update(item['target_hands'].keys())
        
        # Create batched data
        batched = {
            'input': torch.stack([torch.tensor(item['input'], dtype=torch.float32) for item in batch]),
            'target_trump': torch.stack([torch.tensor(item['target_trump'], dtype=torch.float32) for item in batch]),
            'player_id': torch.tensor([item['player_id'] for item in batch]),
            'target_hands': {}
        }
        
        # Handle target_hands for each key
        for key in all_keys:
            target_hands_list = []
            for item in batch:
                if key in item['target_hands']:
                    target_hands_list.append(torch.tensor(item['target_hands'][key], dtype=torch.float32))
                else:
                    # Create zero tensor if key doesn't exist
                    target_hands_list.append(torch.zeros(32, dtype=torch.float32))
            batched['target_hands'][key] = torch.stack(target_hands_list)
        
        return batched
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    
    if len(dataset) == 0:
        print("No training data found! Check log directories.")
        return None
    
    # Create model
    model = BeliefNetwork().to(device)
    
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    hand_loss_fn = nn.BCELoss()
    trump_loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs with {len(dataset)} examples...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Move batch to device
            input_features = batch['input'].float().to(device)
            target_hands = {k: v.float().to(device) for k, v in batch['target_hands'].items()}
            target_trump = batch['target_trump'].float().to(device)
            player_ids = batch['player_id']
            
            # Forward pass
            predictions = model(input_features, player_ids[0].item())
            
            # Calculate losses
            hand_loss = 0.0
            for opp_id in target_hands:
                if str(opp_id) in predictions['opponent_hands']:
                    hand_loss += hand_loss_fn(
                        predictions['opponent_hands'][str(opp_id)],
                        target_hands[opp_id]
                    )
            
            trump_loss = trump_loss_fn(predictions['trump_suit'], target_trump)
            
            total_loss = hand_loss + trump_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        # Print epoch results
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f"belief_model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
    
    # Save final model
    final_path = os.path.join(save_dir, "belief_model_final.pt")
    torch.save(model.state_dict(), final_path)
    
    print(f"Training completed. Model saved to {save_dir}")
    
    return model


def evaluate_belief_model(model_path: str, log_dirs: List[str] = ["logs/game28/mcts_games"], num_test_games: int = 100):
    """
    Evaluate the trained belief model using real game data
    
    Args:
        model_path: Path to the trained model
        log_dirs: List of log directories to load test data from
        num_test_games: Number of test games to use
    """
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BeliefNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create test dataset
    test_dataset = RealGameDataset(log_dirs, max_games=num_test_games)
    
    if len(test_dataset) == 0:
        print("No test data found!")
        return {}
    
    # Evaluation metrics
    hand_accuracy = 0.0
    trump_accuracy = 0.0
    total_examples = 0
    
    with torch.no_grad():
        for example in tqdm(test_dataset, desc="Evaluating"):
            # Prepare input
            input_features = torch.tensor(example['input'], dtype=torch.float32).unsqueeze(0).to(device)
            player_id = example['player_id']
            
            # Get predictions
            predictions = model(input_features, player_id)
            
            # Calculate hand accuracy
            for opp_id in example['target_hands']:
                if str(opp_id) in predictions['opponent_hands']:
                    pred_hand = predictions['opponent_hands'][str(opp_id)].cpu().numpy()
                    target_hand = example['target_hands'][opp_id]
                    
                    # Convert to binary predictions
                    pred_binary = (pred_hand > 0.5).astype(float)
                    accuracy = np.mean(pred_binary == target_hand)
                    hand_accuracy += accuracy
            
            # Calculate trump accuracy
            pred_trump = torch.argmax(predictions['trump_suit'], dim=-1).cpu().numpy()
            target_trump = np.argmax(example['target_trump'])
            trump_accuracy += (pred_trump == target_trump).astype(float)
            
            total_examples += 1
    
    # Calculate final metrics
    hand_accuracy /= total_examples
    trump_accuracy /= total_examples
    
    print(f"Evaluation Results:")
    print(f"  Hand Prediction Accuracy: {hand_accuracy:.3f}")
    print(f"  Trump Prediction Accuracy: {float(trump_accuracy):.3f}")
    
    return {
        'hand_accuracy': hand_accuracy,
        'trump_accuracy': trump_accuracy
    }


if __name__ == "__main__":
    # Train the belief model using real game data
    model = train_belief_model(
        log_dirs=["../logs/game28/mcts_games"],  # Correct path from belief_model directory
        max_games=10,  # Test with 10 games
        batch_size=32,
        learning_rate=1e-3,
        num_epochs=5
    )
    
    # Evaluate the trained model
    if model:
        results = evaluate_belief_model("models/belief_model/belief_model_final.pt", 
                                      log_dirs=["../logs/game28/mcts_games"], 
                                      num_test_games=20)
