"""
Advanced Training Pipeline for Belief Network
State-of-the-art training with curriculum learning and advanced loss functions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
from tqdm import tqdm

from .simple_advanced_belief_net import SimpleAdvancedBeliefNetwork, BeliefPrediction
from .advanced_parser import GameState, extract_all_game_states
from game28.constants import *


class AdvancedBeliefDataset(Dataset):
    """Advanced dataset with rich feature extraction"""
    
    def __init__(self, game_states: List[GameState], augment: bool = True):
        self.game_states = game_states
        self.augment = augment
        self.examples = self._create_examples()
    
    def _create_examples(self) -> List[Dict]:
        """Create training examples from game states"""
        examples = []
        
        for game_state in self.game_states:
            # Create examples for each player
            for player_id in range(4):
                if player_id in game_state.hands:
                    example = self._create_example(game_state, player_id)
                    if example:
                        examples.append(example)
        
        return examples
    
    def _create_example(self, game_state: GameState, player_id: int) -> Optional[Dict]:
        """Create a single training example"""
        try:
            # Encode features
            card_features = self._encode_card_features(game_state, player_id)
            player_features = self._encode_player_features(game_state, player_id)
            game_features = self._encode_game_features(game_state)
            temporal_features = self._encode_temporal_features(game_state)
            
            # Create targets
            targets = self._create_targets(game_state, player_id)
            
            return {
                'card_features': card_features,
                'player_features': player_features,
                'game_features': game_features,
                'temporal_features': temporal_features,
                'player_id': player_id,
                'targets': targets
            }
        except Exception as e:
            return None
    
    def _encode_card_features(self, game_state: GameState, player_id: int) -> torch.Tensor:
        """Encode card features for all 32 cards"""
        features = torch.zeros(32, 6)
        
        for card_idx in range(32):
            suit_idx = card_idx // 8
            rank_idx = card_idx % 8
            
            # Basic card features
            features[card_idx, 0] = suit_idx  # Suit
            features[card_idx, 1] = rank_idx  # Rank
            features[card_idx, 2] = CARD_VALUES[RANKS[rank_idx]]  # Point value
            features[card_idx, 3] = TRICK_RANKINGS[RANKS[rank_idx]]  # Trick ranking
            
            # Trump ranking
            if game_state.trump_suit:
                trump_rank = TRICK_RANKINGS[RANKS[rank_idx]] if SUITS[suit_idx] == game_state.trump_suit else 0
                features[card_idx, 4] = trump_rank
            else:
                features[card_idx, 4] = 0
            
            # Playability
            features[card_idx, 5] = 1.0
        
        return features
    
    def _encode_player_features(self, game_state: GameState, player_id: int) -> torch.Tensor:
        """Encode player features for all 4 players"""
        features = torch.zeros(4, 15)
        
        for pid in range(4):
            if pid in game_state.hands:
                hand = game_state.hands[pid]
                
                # Basic hand features
                features[pid, 0] = len(hand)  # Hand size
                
                # Suit counts
                for suit_idx, suit in enumerate(SUITS):
                    suit_count = sum(1 for card in hand if card.suit == suit)
                    features[pid, 1 + suit_idx] = suit_count
                
                # Void suits
                for suit_idx, suit in enumerate(SUITS):
                    has_suit = any(card.suit == suit for card in hand)
                    features[pid, 5 + suit_idx] = 0.0 if has_suit else 1.0
                
                # High cards
                high_cards = sum(1 for card in hand if card.rank in ['A', 'K', 'Q', 'J'])
                features[pid, 9] = high_cards
                
                # Trump cards
                if game_state.trump_suit:
                    trump_cards = sum(1 for card in hand if card.suit == game_state.trump_suit)
                    features[pid, 10] = trump_cards
                else:
                    features[pid, 10] = 0
                
                # Bidding features
                if game_state.bidding_history:
                    player_bids = [bid for player, bid in game_state.bidding_history if player == pid]
                    features[pid, 11] = max(player_bids) if player_bids else 0
                else:
                    features[pid, 11] = 0
                
                # Role encoding
                if pid == game_state.bidder:
                    features[pid, 12] = 0  # Bidder
                elif pid in [0, 2] and player_id in [0, 2] or pid in [1, 3] and player_id in [1, 3]:
                    features[pid, 12] = 1  # Partner
                elif pid == player_id:
                    features[pid, 12] = 2  # Self
                else:
                    features[pid, 12] = 3  # Opponent
                
                # Score features
                if pid in [0, 2]:
                    features[pid, 13] = game_state.team_a_score
                else:
                    features[pid, 13] = game_state.team_b_score
                
                # Tricks won
                tricks_won = sum(1 for play in game_state.played_cards if play.player_id == pid and play.won_trick)
                features[pid, 14] = tricks_won
        
        return features
    
    def _encode_game_features(self, game_state: GameState) -> torch.Tensor:
        """Encode game-level features"""
        features = torch.zeros(20)
        
        # Phase encoding
        phase_map = {'bidding': 0, 'concealed': 1, 'revealed': 2}
        features[0] = phase_map.get(game_state.phase, 0)
        
        # Trump encoding
        if game_state.trump_suit:
            trump_idx = SUITS.index(game_state.trump_suit)
            features[1] = trump_idx
        else:
            features[1] = 4  # Unknown
        
        # Game progress
        features[2] = game_state.game_progress
        
        # Scores
        features[3] = game_state.team_a_score
        features[4] = game_state.team_b_score
        
        # Bidding information
        features[5] = game_state.current_bid
        features[6] = game_state.bidder if game_state.bidder is not None else -1
        
        # Trick information
        features[7] = game_state.tricks_played
        features[8] = game_state.cards_played
        
        # Trump revelation
        features[9] = 1.0 if game_state.trump_revealed else 0.0
        features[10] = game_state.trump_revealer if game_state.trump_revealer is not None else -1
        
        # Game pressure
        features[11] = game_state.game_progress * 10
        
        # Risk level
        features[12] = abs(game_state.team_a_score - game_state.team_b_score) / 28.0
        
        # Expected value
        features[13] = (game_state.team_a_score + game_state.team_b_score) / 28.0
        
        # Variance
        features[14] = 0.5
        
        # Information entropy
        features[15] = 1.0 - game_state.game_progress
        
        # Time remaining
        features[16] = 1.0 - game_state.game_progress
        
        # Lead suit
        if game_state.current_trick and game_state.current_trick.cards:
            lead_suit = game_state.current_trick.cards[0][1].suit
            features[17] = SUITS.index(lead_suit)
        else:
            features[17] = -1
        
        # Current player
        features[18] = game_state.current_player
        
        # Game completion
        features[19] = game_state.game_progress
        
        return features
    
    def _encode_temporal_features(self, game_state: GameState) -> torch.Tensor:
        """Encode temporal features from move history"""
        max_moves = 20  # Fixed sequence length
        
        if not game_state.played_cards:
            return torch.zeros(max_moves, 10)  # Empty sequence padded
        
        # Take last 20 moves
        recent_moves = game_state.played_cards[-max_moves:]
        
        features = torch.zeros(max_moves, 10)
        
        for i, move in enumerate(recent_moves):
            # Move features
            features[i, 0] = move.player_id
            features[i, 1] = SUITS.index(move.card.suit)
            features[i, 2] = RANKS.index(move.card.rank)
            features[i, 3] = move.trick_number
            features[i, 4] = move.position_in_trick
            features[i, 5] = 1.0 if move.won_trick else 0.0
            features[i, 6] = move.points_earned
            features[i, 7] = 1.0 if move.trump_played else 0.0
            features[i, 8] = 1.0 if move.high_card_played else 0.0
            features[i, 9] = 1.0 if move.forced_play else 0.0
        
        return features
    
    def _create_targets(self, game_state: GameState, player_id: int) -> Dict:
        """Create target labels for training"""
        targets = {}
        
        # Hand targets for each opponent
        for opp_id in range(4):
            if opp_id != player_id and opp_id in game_state.hands:
                hand = game_state.hands[opp_id]
                hand_target = torch.zeros(32)
                
                for card in hand:
                    card_idx = SUITS.index(card.suit) * 8 + RANKS.index(card.rank)
                    hand_target[card_idx] = 1.0
                
                targets[f'hand_{opp_id}'] = hand_target
        
        # Trump target
        trump_target = torch.zeros(4)
        if game_state.trump_suit:
            trump_idx = SUITS.index(game_state.trump_suit)
            trump_target[trump_idx] = 1.0
        targets['trump'] = trump_target
        
        # Void suit targets
        for opp_id in range(4):
            if opp_id != player_id and opp_id in game_state.hands:
                hand = game_state.hands[opp_id]
                void_target = torch.zeros(4)
                
                for suit_idx, suit in enumerate(SUITS):
                    has_suit = any(card.suit == suit for card in hand)
                    void_target[suit_idx] = 0.0 if has_suit else 1.0
                
                targets[f'void_{opp_id}'] = void_target
        
        return targets
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


class AdvancedLossFunction(nn.Module):
    """Advanced loss function with multiple components"""
    
    def __init__(self, 
                 hand_weight: float = 1.0,
                 trump_weight: float = 0.5,
                 void_weight: float = 0.3):
        super().__init__()
        
        self.hand_weight = hand_weight
        self.trump_weight = trump_weight
        self.void_weight = void_weight
        
        # Loss functions
        self.bce_loss = nn.BCELoss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, predictions: BeliefPrediction, targets: Dict) -> Dict[str, torch.Tensor]:
        """Compute comprehensive loss"""
        total_loss = 0.0
        loss_components = {}
        
        # Hand prediction loss
        hand_loss = 0.0
        for opp_id, pred_hand in predictions.opponent_hands.items():
            target_key = f'hand_{opp_id}'
            if target_key in targets:
                target_hand = targets[target_key]
                hand_loss += self.hand_weight * self.bce_loss(pred_hand, target_hand).mean()
        
        loss_components['hand'] = hand_loss
        total_loss += hand_loss
        
        # Trump prediction loss
        if 'trump' in targets:
            trump_loss = self.trump_weight * self.ce_loss(predictions.trump_suit, targets['trump'])
            loss_components['trump'] = trump_loss.mean()
            total_loss += trump_loss.mean()
        
        # Void suit loss
        void_loss = 0.0
        for opp_id, pred_void in predictions.void_suits.items():
            target_key = f'void_{opp_id}'
            if target_key in targets:
                void_loss += self.void_weight * self.bce_loss(pred_void, targets[target_key]).mean()
        
        loss_components['void'] = void_loss
        total_loss += void_loss
        
        loss_components['total'] = total_loss
        
        return loss_components


def train_advanced_belief_model(log_dirs: List[str],
                               max_games: Optional[int] = None,
                               batch_size: int = 32,
                               learning_rate: float = 1e-3,
                               num_epochs: int = 50,
                               device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> SimpleAdvancedBeliefNetwork:
    """Train the advanced belief model"""
    
    print("Extracting game states...")
    game_states = extract_all_game_states(log_dirs, max_games)
    
    if len(game_states) == 0:
        print("No game states extracted. Check log directories and parsing.")
        return None
    
    print(f"Extracted {len(game_states)} game states")
    
    # Split into train/val
    split_idx = int(0.8 * len(game_states))
    train_states = game_states[:split_idx]
    val_states = game_states[split_idx:]
    
    # Create datasets
    train_dataset = AdvancedBeliefDataset(train_states, augment=True)
    val_dataset = AdvancedBeliefDataset(val_states, augment=False)
    
    # Create model
    model = SimpleAdvancedBeliefNetwork()
    
    def custom_collate(batch):
        """Custom collate function to handle variable target keys"""
        # Get all unique keys from targets across the batch
        all_target_keys = set()
        for item in batch:
            all_target_keys.update(item['targets'].keys())
        
        # Create batched data
        batched = {
            'card_features': torch.stack([item['card_features'] for item in batch]),
            'player_features': torch.stack([item['player_features'] for item in batch]),
            'game_features': torch.stack([item['game_features'] for item in batch]),
            'temporal_features': torch.stack([item['temporal_features'] for item in batch]),
            'player_id': torch.tensor([item['player_id'] for item in batch]),
            'targets': {}
        }
        
        # Handle targets with variable keys
        for key in all_target_keys:
            target_tensors = []
            for item in batch:
                if key in item['targets']:
                    target_tensors.append(item['targets'][key])
                else:
                    # Create a zero tensor with the same shape as other targets
                    if target_tensors:
                        target_tensors.append(torch.zeros_like(target_tensors[0]))
                    else:
                        # If no targets exist yet, create a default shape
                        if 'hand_' in key:
                            target_tensors.append(torch.zeros(32))
                        elif 'void_' in key:
                            target_tensors.append(torch.zeros(4))
                        elif key == 'trump':
                            target_tensors.append(torch.zeros(4))
                        else:
                            target_tensors.append(torch.zeros(1))
            
            batched['targets'][key] = torch.stack(target_tensors)
        
        return batched
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    
    # Training components
    loss_function = AdvancedLossFunction()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            # Move data to device
            card_features = batch['card_features'].to(device)
            player_features = batch['player_features'].to(device)
            game_features = batch['game_features'].to(device)
            temporal_features = batch['temporal_features'].to(device)
            player_ids = batch['player_id']
            targets = {k: v.to(device) for k, v in batch['targets'].items()}
            
            # Forward pass
            optimizer.zero_grad()
            
            predictions = model(
                card_features,
                player_features,
                game_features,
                temporal_features,
                player_ids[0].item()
            )
            
            # Compute loss
            loss_components = loss_function(predictions, targets)
            loss = loss_components['total']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
                # Move data to device
                card_features = batch['card_features'].to(device)
                player_features = batch['player_features'].to(device)
                game_features = batch['game_features'].to(device)
                temporal_features = batch['temporal_features'].to(device)
                player_ids = batch['player_id']
                targets = {k: v.to(device) for k, v in batch['targets'].items()}
                
                # Forward pass
                predictions = model(
                    card_features,
                    player_features,
                    game_features,
                    temporal_features,
                    player_ids[0].item()
                )
                
                # Compute loss
                loss_components = loss_function(predictions, targets)
                loss = loss_components['total']
                
                val_loss += loss.item()
                val_batches += 1
        
        # Update learning rate
        scheduler.step()
        
        # Log progress
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        
        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'models/belief_model/advanced_belief_model_best.pt')
            print(f"New best model saved!")
    
    # Save final model
    torch.save(model.state_dict(), 'models/belief_model/advanced_belief_model_final.pt')
    
    print("Training completed!")
    return model
