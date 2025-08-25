# Improved Bidding Model Usage Guide

## Overview

The improved bidding model is an enhanced version of the basic bidding model that incorporates:
- **MCTS-based opponent strategies** - Opponents use patterns learned from 279 MCTS games
- **Enhanced reward functions** - Better reward signals for learning optimal bidding
- **Improved training data** - Training on more realistic game scenarios
- **Better network architecture** - Deeper networks for better learning

## Quick Start

### 1. Train the Improved Model

First, make sure you have the MCTS data available, then train the model:

```bash
python improved_bidding_trainer.py
```

This will:
- Load MCTS analysis data from `mcts_bidding_analysis.json`
- Create an improved environment with MCTS-based opponents
- Train the model for 32,000 timesteps
- Save the model to `models/improved_bidding_model`

### 2. Use the Improved Model

#### Option A: Simple Example
```bash
python simple_improved_bidding_example.py
```

#### Option B: Full Usage Script
```bash
python use_improved_bidding_model.py
```

#### Option C: Interactive Mode
```bash
python use_bidding_model.py --mode improved
```

## Code Integration

### Basic Usage

```python
from stable_baselines3 import PPO
from improved_bidding_trainer import ImprovedBiddingTrainer
from game28.constants import BID_RANGE

# Load the improved model
model = PPO.load("models/improved_bidding_model")

# Create improved environment
trainer = ImprovedBiddingTrainer()
env = trainer.create_improved_environment()

# Get a prediction
obs, _ = env.reset()
action, _ = model.predict(obs, deterministic=True)

# Convert action to bid
if action == len(BID_RANGE):
    bid = -1  # Pass
else:
    bid = BID_RANGE[action]
```

### Advanced Usage with Custom Game State

```python
from use_improved_bidding_model import ImprovedBiddingModel

# Create model wrapper
bidding_model = ImprovedBiddingModel()

# Get bid suggestion for specific game state
hand = ['H7', 'H9', 'H10', 'DJ', 'DQ', 'DK', 'C8', 'C9']
current_bid = 18
bid_history = [(1, 16), (2, 17), (3, 18)]
position = 0

suggestion = bidding_model.get_bid_suggestion(hand, current_bid, bid_history, position)
print(f"Suggested bid: {suggestion}")
```

## Model Features

### 1. MCTS-Based Opponent Strategies
The improved model uses opponents that follow patterns learned from MCTS analysis:
- **Suit-based strategies**: More aggressive with Clubs and Spades (higher success rates)
- **Hand strength adaptation**: Different strategies based on hand strength
- **Realistic bidding patterns**: Based on actual MCTS game data

### 2. Enhanced Reward Function
- **Bid efficiency rewards**: Rewards for successful bids relative to bid value
- **Intermediate rewards**: Small rewards for good decisions during bidding
- **MCTS-based success patterns**: Rewards based on patterns from successful MCTS games

### 3. Better Training Data
- **901 MCTS games**: Training data from real MCTS analysis
- **First 4 cards focus**: Analysis focuses on the first 4 cards which are relevant for bidding decisions
- **Success rate patterns**: Incorporates success rates for different suits and strategies
- **Realistic scenarios**: More diverse and challenging training scenarios

## Performance Comparison

The improved model should show:
- **Higher win rates** compared to the basic model
- **Better bid efficiency** - more successful bids relative to bid values
- **More realistic opponent behavior** - opponents that follow MCTS patterns
- **Improved decision making** - better choices in complex bidding situations

## Troubleshooting

### Model Not Found
If you get an error about the model not being found:
```bash
# Make sure you've trained the model first
python improved_bidding_trainer.py
```

### MCTS Data Missing
If you get an error about MCTS data:
```bash
# Make sure mcts_bidding_analysis.json exists
ls mcts_bidding_analysis.json
```

### Environment Issues
If you encounter environment-related errors:
```bash
# Test the environment first
python test_improved_env.py
```

## File Structure

```
28bot_v2/
├── improved_bidding_trainer.py      # Main training script
├── use_improved_bidding_model.py    # Full usage examples
├── simple_improved_bidding_example.py # Simple usage example
├── test_improved_env.py             # Environment testing
├── models/
│   └── improved_bidding_model/      # Trained model files
├── mcts_bidding_analysis.json       # MCTS training data
└── IMPROVED_MODEL_USAGE.md          # This guide
```

## Next Steps

1. **Evaluate Performance**: Run the model evaluation to see improvement metrics
2. **Compare Models**: Compare performance between basic and improved models
3. **Fine-tune**: Adjust hyperparameters if needed for your specific use case
4. **Integrate**: Use the improved model in your game application

## Example Output

When running the improved model, you should see output like:
```
Loading improved bidding model...
✓ Model loaded successfully!

Playing a game with the improved model...
The improved model features:
• MCTS-based opponent strategies
• Enhanced reward functions
• Better training data from 901 MCTS games

Step 1: Model chose action 3 -> Bid: 19
Step 2: Model chose action 13 -> Bid: PASS
Game ended. Final reward: 0.500
```

This indicates the model is working correctly and making informed bidding decisions based on the improved training.
