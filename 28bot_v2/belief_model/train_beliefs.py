"""
Training script for belief network
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict
import random
from tqdm import tqdm

from .belief_net import BeliefNetwork
from ..game28.game_state import Game28State
from ..game28.constants import *


class BeliefDataset(Dataset):
    """
    Dataset for training belief network
    """
    
    def __init__(self, num_games: int = 10000):
        self.data = []
        self._generate_data(num_games)
    
    def _generate_data(self, num_games: int):
        """Generate training data from simulated games"""
        print(f"Generating {num_games} training games...")
        
        for _ in tqdm(range(num_games)):
            # Create game state
            game_state = Game28State()
            
            # Simulate bidding
            self._simulate_bidding(game_state)
            
            # Simulate play
            self._simulate_play(game_state)
            
            # Extract training examples
            self._extract_examples(game_state)
    
    def _simulate_bidding(self, game_state: Game28State):
        """Simulate bidding phase"""
        current_bid = MIN_BID
        
        for player in range(4):
            if game_state.game_over:
                break
            
            # Simple heuristic bidding
            hand_strength = self._calculate_hand_strength(game_state.hands[player])
            
            if hand_strength > 0.6:
                # Strong hand - bid aggressively
                if current_bid < MAX_BID:
                    current_bid = min(MAX_BID, current_bid + 2)
                    game_state.make_bid(player, current_bid)
                else:
                    game_state.make_bid(player, -1)  # Pass
            elif hand_strength > 0.4:
                # Medium hand - bid moderately
                if current_bid < MAX_BID - 1:
                    current_bid = min(MAX_BID, current_bid + 1)
                    game_state.make_bid(player, current_bid)
                else:
                    game_state.make_bid(player, -1)  # Pass
            else:
                # Weak hand - pass
                game_state.make_bid(player, -1)
        
        # Set trump if bidding completed
        if game_state.bidder is not None:
            # Choose trump based on bidder's hand
            trump_suit = self._choose_trump(game_state.hands[game_state.bidder])
            game_state.set_trump(trump_suit)
    
    def _simulate_play(self, game_state: Game28State):
        """Simulate play phase"""
        while not game_state.game_over and len(game_state.tricks) < 8:
            current_player = game_state.current_player
            legal_cards = game_state.get_legal_plays(current_player)
            
            if legal_cards:
                # Simple heuristic play
                card = self._choose_card(legal_cards, game_state)
                game_state.play_card(current_player, card)
    
    def _calculate_hand_strength(self, hand: List) -> float:
        """Calculate hand strength for bidding"""
        total_points = sum(CARD_VALUES[card.rank] for card in hand)
        return total_points / TOTAL_POINTS
    
    def _choose_trump(self, hand: List) -> str:
        """Choose trump suit based on hand"""
        suit_counts = {suit: 0 for suit in SUITS}
        suit_points = {suit: 0 for suit in SUITS}
        
        for card in hand:
            suit_counts[card.suit] += 1
            suit_points[card.suit] += CARD_VALUES[card.rank]
        
        # Choose suit with most points
        best_suit = max(suit_points, key=suit_points.get)
        return best_suit
    
    def _choose_card(self, legal_cards: List, game_state: Game28State):
        """Choose card to play"""
        if not game_state.current_trick.cards:
            # Leading - play highest card
            return max(legal_cards, key=lambda c: TRICK_RANKINGS[c.rank])
        else:
            # Following - try to win if possible
            lead_suit = game_state.current_trick.lead_suit
            trump_suit = game_state.trump_suit
            
            # Find highest card that can win
            winning_cards = []
            for card in legal_cards:
                if self._can_win(card, game_state.current_trick, trump_suit, game_state.trump_revealed):
                    winning_cards.append(card)
            
            if winning_cards:
                return max(winning_cards, key=lambda c: TRICK_RANKINGS[c.rank])
            else:
                # Can't win - play lowest card
                return min(legal_cards, key=lambda c: TRICK_RANKINGS[c.rank])
    
    def _can_win(self, card, current_trick, trump_suit, trump_revealed):
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
    
    def _extract_examples(self, game_state: Game28State):
        """Extract training examples from game state"""
        # Extract examples at different points in the game
        for player_id in range(4):
            # Example 1: After initial deal
            if len(game_state.tricks) == 0 and game_state.phase == GamePhase.CONCEALED:
                self._add_example(game_state, player_id)
            
            # Example 2: After first trick
            if len(game_state.tricks) == 1:
                self._add_example(game_state, player_id)
            
            # Example 3: After half the tricks
            if len(game_state.tricks) == 4:
                self._add_example(game_state, player_id)
    
    def _add_example(self, game_state: Game28State, player_id: int):
        """Add training example"""
        # Create input features
        input_features = self._encode_game_state(game_state, player_id)
        
        # Create target labels
        target_hands = {}
        for opp_id in range(4):
            if opp_id != player_id:
                target_hands[opp_id] = self._encode_hand(game_state.hands[opp_id])
        
        # Create trump target
        trump_target = np.zeros(4)
        if game_state.trump_suit:
            trump_target[SUITS.index(game_state.trump_suit)] = 1.0
        
        self.data.append({
            'input': input_features,
            'target_hands': target_hands,
            'target_trump': trump_target,
            'player_id': player_id
        })
    
    def _encode_game_state(self, game_state: Game28State, player_id: int) -> np.ndarray:
        """Encode game state into features"""
        # Hand encoding
        hand_encoding = np.zeros(32)
        for card in game_state.hands[player_id]:
            card_idx = SUITS.index(card.suit) * 8 + RANKS.index(card.rank)
            hand_encoding[card_idx] = 1
        
        # Bidding history encoding
        bidding_history = np.zeros(4)
        for i, (player, bid) in enumerate(game_state.bid_history[-4:]):
            bidding_history[i] = bid if bid != -1 else 0
        
        # Played cards encoding
        played_encoding = np.zeros(32)
        for trick in game_state.tricks:
            for _, card in trick.cards:
                card_idx = SUITS.index(card.suit) * 8 + RANKS.index(card.rank)
                played_encoding[card_idx] = 1
        for _, card in game_state.current_trick.cards:
            card_idx = SUITS.index(card.suit) * 8 + RANKS.index(card.rank)
            played_encoding[card_idx] = 1
        
        # Game state encoding
        game_state_encoding = np.array([
            game_state.current_bid,
            player_id,
            list(GamePhase).index(game_state.phase),
            4 if game_state.trump_suit is None else SUITS.index(game_state.trump_suit),
            int(game_state.trump_revealed),
            4 if game_state.bidder is None else game_state.bidder,
            game_state.winning_bid if game_state.winning_bid else 0,
            game_state.team_scores['A'],
            game_state.team_scores['B'],
            len(game_state.tricks)
        ])
        
        return np.concatenate([
            hand_encoding,
            bidding_history,
            played_encoding,
            game_state_encoding
        ])
    
    def _encode_hand(self, hand: List) -> np.ndarray:
        """Encode hand as one-hot vector"""
        encoding = np.zeros(32)
        for card in hand:
            card_idx = SUITS.index(card.suit) * 8 + RANKS.index(card.rank)
            encoding[card_idx] = 1
        return encoding
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def train_belief_model(
    num_games: int = 10000,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    num_epochs: int = 50,
    save_dir: str = "models/belief_model"
):
    """
    Train the belief network
    
    Args:
        num_games: Number of games to generate for training
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
    dataset = BeliefDataset(num_games=num_games)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = BeliefNetwork().to(device)
    
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    hand_loss_fn = nn.BCELoss()
    trump_loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs...")
    
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


def evaluate_belief_model(model_path: str, num_test_games: int = 1000):
    """
    Evaluate the trained belief model
    
    Args:
        model_path: Path to the trained model
        num_test_games: Number of test games
    """
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BeliefNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create test dataset
    test_dataset = BeliefDataset(num_games=num_test_games)
    
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
    print(f"  Trump Prediction Accuracy: {trump_accuracy:.3f}")
    
    return {
        'hand_accuracy': hand_accuracy,
        'trump_accuracy': trump_accuracy
    }


if __name__ == "__main__":
    # Train the belief model
    model = train_belief_model(
        num_games=5000,
        batch_size=64,
        learning_rate=1e-3,
        num_epochs=30
    )
    
    # Evaluate the trained model
    results = evaluate_belief_model("models/belief_model/belief_model_final.pt", num_test_games=500)
