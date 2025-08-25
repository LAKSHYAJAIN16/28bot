# Belief Model Scoring System Explained

## Overview

The belief model uses a sophisticated scoring system that evaluates different aspects of the game state to make optimal decisions. Here's exactly how the scoring works and where those 2-3 scores come from.

## Bidding Evaluation Scoring

### Score Components:

1. **Trump Prediction Factor** (0 to ~2.0)
   - **High confidence (>0.8) + Have trump cards**: `+trump_confidence × trump_cards × 2.0`
   - **High confidence + No trump cards**: `-trump_confidence × 1.0`
   - **Low confidence**: No factor

2. **Opponent Strength Factor** (0 to ~3.0)
   - **Strong opponents (>0.6) + Bidding**: `-avg_opponent_strength × 3.0`
   - **Strong opponents + Passing**: `+avg_opponent_strength × 1.0`
   - **Weak opponents + Bidding**: `+(1.0 - avg_opponent_strength) × 2.0`

3. **Void Suit Factor** (0 to ~6.0)
   - For each opponent void suit we have cards in: `+cards_in_void_suit × 1.5`
   - Example: 2 cards in opponent's void suit = `+3.0`

4. **Uncertainty Factor** (0 to ~2.0)
   - **High uncertainty (>0.7) + Bidding**: `-uncertainty × 2.0`
   - **High uncertainty + Passing**: `+uncertainty × 1.0`

5. **Point Prediction Factor** (0 to 3.0)
   - **Predicted points >= bid**: `+3.0`
   - **Predicted points < bid**: `-2.0`
   - **High predicted points (>20) + Passing**: `-1.0`

### Example Bidding Scores:
- **Pass**: Usually 0.0 to 1.0 (mostly uncertainty bonus)
- **Bid 16**: 4.0 to 6.0 (opponent weakness + point prediction bonuses)
- **Bid 20**: -1.0 to 2.0 (point prediction penalty)
- **Bid 24**: -1.0 to 2.0 (point prediction penalty)

## Card Evaluation Scoring

### Base Score:
- **Card value**: 0 (7,8), 1 (10,A), 2 (9), 3 (J), 0 (Q,K)

### Score Components:

1. **Trump Factor** (Multiplier)
   - **High confidence (>0.8) + Trump card**: `score *= trump_confidence × 2.0`

2. **Lead Suit Factor** (+5.0 to -2.0)
   - **Following suit + Can win**: `+5.0`
   - **Following suit + Can't win**: `+1.0`
   - **Void + Trump**: `+3.0`
   - **Void + Discard**: `+0.5`
   - **Not following when could**: `-2.0`

3. **Opponent Void Factor** (+2.0 per opponent)
   - **Playing in opponent's void suit**: `+2.0`

4. **Opponent Strength Factor** (+3.0 to +1.0)
   - **Strong opponents + High card**: `+3.0`
   - **Weak opponents + Low card**: `+1.0`

5. **Uncertainty Factor** (+2.0)
   - **High uncertainty (>0.8) + High card**: `+2.0`

6. **Game Phase Factor** (+2.0 to +1.0)
   - **Concealed + High card**: `+1.0`
   - **Revealed + Trump**: `+2.0`

7. **Trick Position Factor** (+4.0 to +2.0)
   - **Leading + High card**: `+2.0`
   - **Last + Can win**: `+4.0`

### Example Card Scores:
- **9D following suit**: 2 (base) + 1 (follow) + 1 (weak opp) = **4.0**
- **7D following suit**: 0 (base) + 1 (follow) + 1 (weak opp) = **2.0**
- **AC not following**: 1 (base) - 2 (not follow) + 1 (concealed) = **0.0**

## Where the 2-3 Scores Come From

The **2-3 scores** you see in the simulation come from these specific combinations:

### Bidding Scores:
- **+1.1694**: Opponent weakness bonus `(1.0 - 0.4153) × 2.0`
- **+3.0000**: Point prediction bonus (when predicted points >= bid)
- **Total: ~4.17** for good bids

### Card Scores:
- **+1.0000**: Following suit bonus
- **+1.0000**: Weak opponent bonus for low cards
- **+1.0000**: Concealed phase bonus for high cards
- **Total: ~3.0** for good card plays

## Key Insights

1. **Opponent Modeling**: The model predicts opponent hand strengths and void suits
2. **Uncertainty Handling**: High uncertainty favors conservative play (passing)
3. **Context Awareness**: Scores adapt to game phase, trick position, and lead suit
4. **Balanced Scoring**: Multiple factors prevent any single aspect from dominating
5. **Realistic Ranges**: Scores typically range from -2.0 to +6.0 for bidding, 0.0 to +8.0 for cards

## Debug Output Example

From the debug output:
```
Final score for Bid 16: 4.1694
  - Opponent weakness bonus: +1.1694
  - Point prediction bonus: +3.0000

Final score for 9D: 4.0000
  - Base card value: +2.0000
  - Following suit: +1.0000
  - Weak opponent bonus: +1.0000
```

This scoring system ensures that the belief model makes intelligent, context-aware decisions based on its predictions about opponent hands and game state.
