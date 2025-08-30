# 28Bot v2 - Advanced Multi-Agent AI System for Imperfect Information Games

### Couldn't beat my grandfather at 28 normally, so I decided to build a comprehensive AI system that combines belief networks, Monte Carlo Tree Search, and reinforcement learning.

### Highlights
- **Advanced Belief Network**: Can predict what cards your opponent is the most likely to have + what the trump suit might be, based on bids and card play.
- **Bidding Bot**: Model that accurately predicts the points that might come out of a hand, leading to accurate bids.
- **Hybrid Decision Framework**: Dynamically combines multiple AI methods based on game phase (a RL model, a MCTS, the belief model, etc)
- **Information Imperfect MCTS**: A Monte Carlo Tree Search without perfect information.
---

## System Architecture

**Core Components:**
- `belief_model/` - Advanced belief networks for opponent modeling
- `rl_bidding/` - Reinforcement Learning environment and training for bidding
- `ismcts/` - Monte Carlo Tree Search implementation
- `hybrid_agent.py` - Hybrid decision-making system

**Key Features:**
- 78-dimensional feature encoding for game state representation
- Multi-head architecture predicting hands, trump suits, void suits, and uncertainty
- Attention mechanisms for complex game relationships
---

## Performance Results

- **Win Rate**: 52.8% (25.4% improvement over baselines)s
- **Trump Prediction**: 67.5% accuracy for suit selection
- **Bidding**: 62.1% accuracy for bid accomplishment.
- **Training**: 17,768 example  game states (all self-generated)
---

## Project Structure
**Models & Data:**
- `models/` - Trained model files 
- `data/` - Training data and analysis
- `logs/` - Comprehensive logging and evaluation
---
