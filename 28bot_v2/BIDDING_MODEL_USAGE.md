# Using the Trained Bidding Model

This guide explains how to use the trained bidding model for Game 28.

## Quick Start

### 1. Train the Model (if not already done)
```bash
python run_training.py
```

### 2. Use the Model

#### Option A: Simple Usage Script
```bash
# Play a single game with the model
python use_bidding_model.py --mode play

# Evaluate model performance over 100 games
python use_bidding_model.py --mode evaluate --games 100

# Interactive mode (see model suggestions)
python use_bidding_model.py --mode interactive
```

#### Option B: Programmatic Usage
```python
from bidding_advisor import BiddingAdvisor
from game28.game_state import Card

# Create advisor
advisor = BiddingAdvisor()

# Example hand
hand = [
    Card('H', 'A'), Card('H', 'K'), Card('D', 'J'), Card('C', '9'),
    Card('S', '10'), Card('H', '8'), Card('D', '7'), Card('C', 'Q')
]

# Get suggestion
suggestion = advisor.get_bid_suggestion(
    hand=hand,
    current_bid=16,
    bid_history=[],
    position=0
)

print(advisor.format_suggestion(suggestion))
```

## Model Files

The trained model is saved in:
- `models/bidding_policy/best_model/best_model.zip` - Best performing model
- `models/bidding_policy/bidding_model_*.zip` - Checkpoint models

## Integration Examples

### 1. Simple Integration
```python
from bidding_advisor import BiddingAdvisor

advisor = BiddingAdvisor()

def make_bid_decision(hand, current_bid, bid_history, position):
    suggestion = advisor.get_bid_suggestion(hand, current_bid, bid_history, position)
    return suggestion['suggested_bid']
```

### 2. With Confidence Threshold
```python
def make_bid_decision_with_confidence(hand, current_bid, bid_history, position, min_confidence=0.5):
    suggestion = advisor.get_bid_suggestion(hand, current_bid, bid_history, position)
    
    if suggestion['confidence'] >= min_confidence:
        return suggestion['suggested_bid']
    else:
        # Use fallback strategy or pass
        return -1  # Pass
```

### 3. Multiple Models
```python
# Load different models for different scenarios
advisor_aggressive = BiddingAdvisor("models/bidding_policy/aggressive_model.zip")
advisor_conservative = BiddingAdvisor("models/bidding_policy/conservative_model.zip")

def choose_model_strategy(hand_strength):
    if hand_strength > 0.6:
        return advisor_aggressive
    else:
        return advisor_conservative
```

## Model Output

The model returns a dictionary with:
- `suggested_bid`: The recommended bid (-1 for pass, or bid value)
- `confidence`: Confidence score (0.0 to 1.0)
- `method`: How the suggestion was generated ("trained_model" or "heuristic")
- `hand_strength`: Calculated hand strength

## Troubleshooting

### Model Not Found
If you get an error about the model not being found:
1. Make sure you've trained the model first: `python run_training.py`
2. Check that the model file exists in the expected location
3. The advisor will fall back to heuristic bidding if the model can't be loaded

### Performance Issues
- The model works best with the same environment setup used during training
- For CPU-only usage, consider using smaller model architectures
- The heuristic fallback is always available if the model is slow

### Training Issues
If training stops early (like at 8196 steps):
1. This was a known issue with the environment - we've fixed it
2. The environment now properly simulates 4-player bidding
3. Re-train the model with the updated environment

## Advanced Usage

### Custom Model Path
```python
advisor = BiddingAdvisor("path/to/your/custom/model.zip")
```

### Batch Predictions
```python
# For multiple hands
hands = [hand1, hand2, hand3]
suggestions = [advisor.get_bid_suggestion(h, 16, [], 0) for h in hands]
```

### Model Comparison
```python
# Compare different models
models = {
    "best": BiddingAdvisor("models/bidding_policy/best_model/best_model.zip"),
    "checkpoint_1000": BiddingAdvisor("models/bidding_policy/bidding_model_1000_steps.zip"),
    "checkpoint_2000": BiddingAdvisor("models/bidding_policy/bidding_model_2000_steps.zip")
}

for name, advisor in models.items():
    suggestion = advisor.get_bid_suggestion(hand, 16, [], 0)
    print(f"{name}: {advisor.format_suggestion(suggestion)}")
```
