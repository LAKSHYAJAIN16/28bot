# Belief Model Training Guide

## Overview

The belief model is a neural network that learns to predict opponent hand distributions and trump suit probabilities in Game 28. It uses a probabilistic approach to model what cards opponents might have based on the current game state.

## What the Belief Model Does

### 1. **Opponent Hand Prediction**
- Predicts probability distributions for each opponent's hand
- Uses information from bidding, played cards, and game state
- Outputs 32 probabilities (one for each card) for each opponent

### 2. **Trump Suit Prediction**
- Predicts the likelihood of each suit being trump
- Uses bidding patterns and hand information
- Outputs 4 probabilities (one for each suit)

### 3. **Belief State Management**
- Maintains probabilistic beliefs about opponent hands
- Updates beliefs as cards are played
- Provides uncertainty-aware opponent modeling

## Architecture

The belief network consists of:

```
Input Features (78 dimensions):
├── Your hand (32) - binary encoding of your cards
├── Bidding history (4) - current bid, position, etc.
├── Played cards (32) - binary encoding of played cards
└── Game state (10) - phase, scores, etc.

Feature Extractor:
├── 3 hidden layers (256 units each)
├── ReLU activation
└── Dropout (0.1)

Output Heads:
├── Opponent Hand Heads (3) - one for each opponent
│   ├── Hidden layer (128 units)
│   └── Output layer (32 units) with Sigmoid
└── Trump Head (1)
    ├── Hidden layer (128 units)
    └── Output layer (4 units) with Softmax
```

## Training Process

### 1. **Data Generation**
The model generates training data by simulating thousands of Game 28 games:

```python
# Generate training data
dataset = BeliefDataset(num_games=10000)
```

**Data Generation Steps:**
1. Create random game states
2. Simulate bidding with heuristic strategies
3. Simulate card play with simple heuristics
4. Extract training examples at each step
5. Create input features and target outputs

### 2. **Training Configuration**

```python
# Training parameters
num_games = 5000          # Number of games to simulate
batch_size = 64           # Batch size for training
learning_rate = 1e-3      # Learning rate
num_epochs = 30           # Number of training epochs
```

### 3. **Loss Function**
- **Hand Prediction Loss**: Binary cross-entropy for each card
- **Trump Prediction Loss**: Cross-entropy for suit prediction
- **Total Loss**: Weighted combination of both losses

## How to Train the Belief Model

### Quick Start

```bash
# Navigate to the project directory
cd 28bot_v2

# Train the belief model
python belief_model/train_beliefs.py
```

### Custom Training

```python
from belief_model.train_beliefs import train_belief_model, evaluate_belief_model

# Train with custom parameters
model = train_belief_model(
    num_games=10000,      # More games for better training
    batch_size=128,       # Larger batch size
    learning_rate=5e-4,   # Lower learning rate
    num_epochs=50         # More epochs
)

# Evaluate the model
results = evaluate_belief_model(
    "models/belief_model/belief_model_final.pt",
    num_test_games=1000
)
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_games` | 5000 | Number of games to simulate for training |
| `batch_size` | 64 | Batch size for training |
| `learning_rate` | 1e-3 | Learning rate for optimizer |
| `num_epochs` | 30 | Number of training epochs |
| `hidden_dim` | 256 | Hidden layer dimension |
| `num_layers` | 3 | Number of hidden layers |

## Model Outputs

### 1. **Opponent Hand Predictions**
```python
# For each opponent, get card probabilities
opponent_hands = {
    '1': [0.1, 0.9, 0.2, ...],  # 32 probabilities
    '2': [0.3, 0.1, 0.8, ...],  # 32 probabilities  
    '3': [0.5, 0.2, 0.1, ...]   # 32 probabilities
}
```

### 2. **Trump Suit Predictions**
```python
# Probability of each suit being trump
trump_suit = [0.1, 0.6, 0.2, 0.1]  # [H, D, C, S]
```

## Using the Trained Model

### 1. **Load the Model**
```python
from belief_model.belief_net import BeliefNetwork
import torch

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BeliefNetwork().to(device)
model.load_state_dict(torch.load("models/belief_model/belief_model_final.pt"))
model.eval()
```

### 2. **Get Predictions**
```python
from game28.game_state import Game28State

# Create game state
game_state = Game28State()

# Get belief predictions
belief_state = model.predict_beliefs(game_state, player_id=0)

# Access predictions
opponent_hands = belief_state.opponent_hands
trump_probability = belief_state.trump_suit
```

### 3. **Integration with Other Models**
```python
# Use belief model with bidding model
from examples.use_improved_bidding_model import ImprovedBiddingModel

bidding_model = ImprovedBiddingModel()
belief_model = BeliefNetwork()

# Get both predictions
bid_suggestion = bidding_model.get_bid_suggestion(hand, current_bid, bid_history, position)
belief_state = belief_model.predict_beliefs(game_state, position)
```

## Evaluation Metrics

The model is evaluated on:

### 1. **Hand Prediction Accuracy**
- Binary accuracy for each card prediction
- Average accuracy across all opponents and cards

### 2. **Trump Prediction Accuracy**
- Accuracy of trump suit prediction
- Cross-entropy loss for suit probabilities

### 3. **Expected Results**
- **Hand Accuracy**: ~0.7-0.8 (70-80% accuracy)
- **Trump Accuracy**: ~0.6-0.7 (60-70% accuracy)

## Training Tips

### 1. **Data Quality**
- Increase `num_games` for better training data
- Use more diverse bidding strategies
- Include edge cases and rare scenarios

### 2. **Model Architecture**
- Experiment with different `hidden_dim` values
- Try different numbers of layers
- Adjust dropout rates

### 3. **Training Stability**
- Use learning rate scheduling
- Monitor validation loss
- Save checkpoints regularly

### 4. **Hyperparameter Tuning**
```python
# Example hyperparameter search
learning_rates = [1e-2, 1e-3, 1e-4]
batch_sizes = [32, 64, 128]
hidden_dims = [128, 256, 512]

for lr in learning_rates:
    for bs in batch_sizes:
        for hd in hidden_dims:
            model = train_belief_model(
                learning_rate=lr,
                batch_size=bs,
                hidden_dim=hd
            )
```

## Troubleshooting

### Common Issues

1. **Low Accuracy**
   - Increase training data (`num_games`)
   - Adjust learning rate
   - Check data generation quality

2. **Training Instability**
   - Reduce learning rate
   - Increase batch size
   - Add gradient clipping

3. **Memory Issues**
   - Reduce batch size
   - Use smaller hidden dimensions
   - Enable gradient checkpointing

### Performance Optimization

1. **GPU Training**
   - Use CUDA if available
   - Increase batch size on GPU
   - Use mixed precision training

2. **Data Loading**
   - Use multiple workers for data loading
   - Pre-generate and cache training data
   - Use memory-mapped files for large datasets

## Integration with Game 28 Bot

The belief model can be integrated with other components:

1. **Bidding Advisor**: Use opponent hand predictions for better bidding
2. **MCTS**: Provide prior probabilities for opponent hands
3. **Card Play**: Use beliefs to make better card choices
4. **Evaluation**: Assess game state with uncertainty

## Next Steps

After training the belief model:

1. **Evaluate Performance**: Run evaluation on test games
2. **Integrate with Bidding**: Use beliefs to improve bidding decisions
3. **Combine with MCTS**: Use beliefs as priors in MCTS search
4. **Fine-tune**: Adjust parameters based on performance
5. **Deploy**: Use in actual Game 28 games

The belief model provides a probabilistic foundation for understanding opponent hands, which can significantly improve the overall performance of the Game 28 bot.
