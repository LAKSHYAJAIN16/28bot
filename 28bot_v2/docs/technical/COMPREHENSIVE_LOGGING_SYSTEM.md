# Comprehensive Logging System

## Overview

The Game 28 simulation now features a **dual logging system** that provides both condensed and comprehensive game analysis. This system captures every detail of the belief model's decision-making process, making it perfect for research, debugging, and understanding how the AI agents think.

## Log Structure

### 1. Condensed Logs (`logs/condensed_games/`)
- **Purpose**: Quick overview of game flow
- **Content**: 
  - Basic game events (bids, card plays, trick winners)
  - Hand information and card selections
  - Final scores and game results
  - Similar to the original log format

### 2. Comprehensive Logs (`logs/comprehensive_games/`)
- **Purpose**: Detailed analysis of every decision
- **Content**:
  - Raw model outputs and predictions
  - Step-by-step scoring breakdowns
  - Belief model analysis (trump predictions, opponent modeling)
  - Detailed card evaluation factors
  - Trump revelation events
  - Adaptive model capabilities

## Key Features

### 🔍 **Raw Model Outputs**
Every belief model prediction is logged with complete raw data:
```python
=== RAW OUTPUT: Belief Model Outputs - Player 0 ===
{
    'trump_suit_probs': [0.1299, 0.2486, 0.1749, 0.4465],
    'uncertainty': 0.4740,
    'opponent_hands_shape': {1: torch.Size([1, 32]), ...},
    'void_suits_shape': {1: torch.Size([1, 4]), ...}
}
```

### 📊 **Detailed Scoring Breakdown**
Each decision shows exactly how scores are calculated:

**Bidding Evaluation:**
```
[DEBUG]     Evaluating bid 17 with belief model:
[DEBUG]       Starting score: 0.0
[DEBUG]       1. Trump prediction factor (confidence: 0.4465):
[DEBUG]         Low confidence, no trump factor
[DEBUG]       2. Opponent strength factor (avg: 0.4145):
[DEBUG]         Weak opponents + bidding: +1.1711
[DEBUG]       3. Void suit factor:
[DEBUG]         Opponent 1: no void suits
[DEBUG]       4. Uncertainty factor (uncertainty: 0.4740):
[DEBUG]         Low uncertainty, no factor
[DEBUG]       5. Point prediction factor (prediction: -0.012):
[DEBUG]         Predicted points (-0.012) < bid (17): -2.0000
[DEBUG]       Final score for bid 17: -0.8289
```

**Card Evaluation:**
```
[DEBUG]      Evaluating card 9H with belief model:
[DEBUG]        Base score from card value: 2
[DEBUG]        1. Trump factor (confidence: 0.4465):
[DEBUG]          Not a high-confidence trump card
[DEBUG]        2. Lead suit factor:
[DEBUG]          Leading with high card bonus: +2.0000
[DEBUG]        3. Opponent void factor:
[DEBUG]          Opponent 1: no void bonus for H
[DEBUG]        4. Opponent strength factor (avg: 0.4145):
[DEBUG]          Weak opponents, low card bonus: +1.0000
[DEBUG]        Final score for 9H: 5.0000
```

### 🎯 **Trump Revelation Tracking**
The system tracks when trump is revealed and logs detailed analysis:
```
[DEBUG] TRUMP REVEALED!
[DEBUG]   Player 2 revealed trump S
[DEBUG]   Trick number: 2
[DEBUG]   Card played: JS
[DEBUG]   Game phase changed from CONCEALED to REVEALED
```

### 🧠 **Belief Model Analysis**
Comprehensive analysis of opponent modeling:
```
[DEBUG]   Trump Analysis:
[DEBUG]     Probabilities: [0.1299 0.2486 0.1749 0.4465]
[DEBUG]     Predicted trump: S
[DEBUG]     Confidence: 0.4465
[DEBUG]   Opponent Strength Analysis:
[DEBUG]     Opponent 1: 0.4145
[DEBUG]     Opponent 2: 0.4145
[DEBUG]     Opponent 3: 0.4145
[DEBUG]   Void Suit Analysis:
[DEBUG]     Opponent 1 void in: []
[DEBUG]     Opponent 2 void in: []
[DEBUG]     Opponent 3 void in: []
[DEBUG]   Uncertainty: 0.4740
```

## File Naming Convention

- **Condensed**: `game_{id}_{timestamp}_condensed.log`
- **Comprehensive**: `game_{id}_{timestamp}_comprehensive.log`

Example:
- `game_1_20250825_102313_707_condensed.log`
- `game_1_20250825_102313_709_comprehensive.log`

## Usage

### Running with Comprehensive Logging
```python
from main_game_simulation import create_agents, GameSimulator

# Create agents
agents = create_agents()

# Create simulator (automatically creates both log types)
simulator = GameSimulator(agents, game_id=1)

# Run game
results = simulator.simulate_game()

# Access log paths
condensed_log = results['condensed_log_file']
comprehensive_log = results['comprehensive_log_file']
```

### Running Multiple Games
```python
from main_game_simulation import run_multiple_games

# Run 10 games with comprehensive logging
results = run_multiple_games(num_games=10)

# All logs saved to:
# - logs/condensed_games/
# - logs/comprehensive_games/
```

## Analysis Capabilities

### 1. **Decision Transparency**
- See exactly why each card was chosen
- Understand bidding decisions step-by-step
- Track how belief model predictions influence choices

### 2. **Model Performance Analysis**
- Compare predicted vs actual outcomes
- Analyze uncertainty levels and their impact
- Study opponent modeling accuracy

### 3. **Strategic Insights**
- Identify patterns in trump selection
- Analyze void suit predictions
- Study adaptive behavior across game phases

### 4. **Debugging and Research**
- Isolate specific decision factors
- Track model behavior changes
- Validate belief model predictions

## Example Analysis

### Understanding a Card Choice
From the comprehensive log:
```
[DEBUG]      Evaluating card JS with belief model:
[DEBUG]        Base score from card value: 3
[DEBUG]        1. Trump factor (confidence: 0.4465):
[DEBUG]          Not a high-confidence trump card
[DEBUG]        2. Lead suit factor:
[DEBUG]          Void in C, trump bonus: +3.0000
[DEBUG]        3. Opponent void factor:
[DEBUG]          Opponent 1: no void bonus for S
[DEBUG]        4. Opponent strength factor (avg: 0.4145):
[DEBUG]          Weak opponents, but not a low card
[DEBUG]        5. Uncertainty factor (uncertainty: 0.4740):
[DEBUG]          Low uncertainty, no uncertainty factor
[DEBUG]        6. Game phase factor (phase: CONCEALED):
[DEBUG]          Concealed phase, but not a high card
[DEBUG]        7. Trick position factor:
[DEBUG]          Middle position, no position factor
[DEBUG]        Final score for JS: 6.0000
```

**Analysis**: The JS (Jack of Spades) was chosen because:
1. It's a trump card (S) played when void in the lead suit (C)
2. This earned a +3.0 bonus for trump when void
3. Base card value of 3.0
4. Total score: 6.0 (highest among all cards)

## Benefits

1. **Complete Transparency**: Every decision is fully explained
2. **Research Ready**: Perfect for academic analysis and publication
3. **Debugging Power**: Quickly identify model issues or unexpected behavior
4. **Performance Analysis**: Track how well the belief model performs
5. **Strategic Insights**: Understand the AI's reasoning process

This comprehensive logging system makes the Game 28 AI completely transparent and analyzable, providing unprecedented insight into neural network decision-making in card games.
