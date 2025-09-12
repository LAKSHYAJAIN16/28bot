# 28Bot v2 - Advanced Multi-Agent AI System for Imperfect Information Games

### Couldn't beat my grandfather at 28 normally, so I decided to build a comprehensive AI system that combines belief networks, Monte Carlo Tree Search, and reinforcement learning.

This project presents a comprehensive artificial intelligence system for imperfect information games, demonstrated through the complex card game Game 28. The system introduces several novel contributions including a hybrid decision-making framework, innovative point prediction approach, advanced belief network architecture, and Information Set MCTS implementation.

### Key Innovations
- **Hybrid Decision-Making Framework**: Dynamically combines belief networks, Monte Carlo Tree Search (MCTS), and reinforcement learning based on game phase and information availability
- **Advanced Belief Network**: Multi-task learning architecture predicting opponent hands (77.5% accuracy), trump suits (67.5% accuracy), and game state uncertainty with well-calibrated confidence estimates
- **Point Prediction Model**: Custom RL architecture for accurate bidding with 73% bidding accuracy
- **Information Set MCTS**: Handles imperfect information scenarios with sophisticated belief state management
- **State Canonicalization**: Treats functionally equivalent game states identically, improving learning efficiency and generalization
---

## System Architecture

Our system employs a modular architecture that combines multiple AI paradigms through a hybrid decision-making framework, with the belief network serving as the foundational component.

**Core Components:**
- `belief_model/` - Advanced belief networks for opponent modeling with multi-task learning
- `rl_bidding/` - Reinforcement Learning environment and training for bidding
- `ismcts/` - Information Set Monte Carlo Tree Search implementation
- `hybrid_agent.py` - Hybrid decision-making system with dynamic method selection

**Belief Network Architecture:**
- **Multi-Task Learning**: Simultaneously predicts opponent hands, trump suits, void suits, and uncertainty
- **78-Dimensional Feature Vector**: Comprehensive game state encoding including hand features (32), bidding features (4), played cards (32), and game state (10)
- **Multi-Head Architecture**: 
  - Feature extractor with 4 hidden layers (512, 256, 256, 128 units)
  - Multi-head self-attention mechanism for capturing game relationships
  - 3 opponent hand prediction heads (32 probabilities each)
  - Trump suit prediction head (4 suit probabilities)
  - Void suit detection heads and uncertainty quantification
- **Real-time Inference**: Sub-10ms inference time for practical deployment

**Hybrid Decision Framework:**
- **Phase-based Selection**: Different methods for early game (belief network), mid-game (hybrid), and late game (ISMCTS)
- **Dynamic Weighting**: Combines methods based on confidence and game context
- **Computational Efficiency**: Balances speed vs. accuracy trade-offs
---

## Performance Results

The hybrid system significantly outperforms all individual methods, achieving substantial improvements across multiple metrics:

### Overall System Performance
- **Win Rate**: 52.8% (25.4% improvement over baselines)
- **Bidding Accuracy**: 73% (vs. 12% random, 45% heuristic)
- **Decision Time**: 25ms (optimal balance of speed and quality)
- **Training Data**: 873 MCTS-generated games with comprehensive analysis

### Belief Network Performance
- **Opponent Hand Prediction**: 77.5% average accuracy across all players
- **Trump Suit Prediction**: 67.5% accuracy for suit selection
- **Void Suit Detection**: 72.1% accuracy for identifying opponent voids
- **Uncertainty Calibration**: 0.024 calibration error (well-calibrated confidence)

### Component Analysis
- **Base System**: 42.1% win rate
- **+ Point Prediction**: 46.3% win rate (10.0% improvement)
- **+ Belief Network**: 49.7% win rate (18.1% improvement)
- **+ ISMCTS**: 51.2% win rate (21.6% improvement)
- **Full Hybrid System**: 52.8% win rate (25.4% improvement)

### Comparison with Individual Methods
| Method | Win Rate (%) | Bidding Acc. | Decision Time (ms) |
|--------|-------------|--------------|-------------------|
| Random Agent | 25.0 | 0.12 | 1 |
| Heuristic Agent | 35.2 | 0.45 | 5 |
| Belief Network | 42.1 | 0.58 | 8 |
| RL Agent | 38.7 | 0.52 | 15 |
| MCTS | 45.3 | 0.61 | 150 |
| **Hybrid System** | **52.8** | **0.73** | **25** |
---

## Game 28 Overview

Game 28 is a four-player team-based trick-taking card game that serves as an excellent testbed for imperfect information AI systems. The game features:

- **32-card deck**: 7, 8, 9, 10, J, Q, K, A of each suit
- **Point values**: J (3), 9 (2), A (1), 10 (1) - totaling 28 points
- **Complex bidding**: Players bid on points they believe their team can make
- **Trump selection**: Highest bidder sets trump suit (concealed initially)
- **Imperfect information**: Players must infer opponent hands and trump suit
- **Team coordination**: Partners must work together strategically

The game's combination of bidding, trump selection, and imperfect information makes it an ideal domain for testing advanced AI techniques.

## Project Structure

**Core Implementation:**
- `28bot_v2/` - Main system implementation
  - `belief_model/` - Advanced belief networks for opponent modeling
  - `rl_bidding/` - Reinforcement learning for bidding strategies
  - `ismcts/` - Information Set Monte Carlo Tree Search
  - `hybrid_agent.py` - Hybrid decision-making framework
  - `game28/` - Game logic and state management

**Models & Data:**
- `models/` - Trained model files (122+ model files)
- `data/` - Training data and analysis
- `logs/` - Comprehensive logging and evaluation (1400+ log files)

**Documentation & Research:**
- `docs/` - Technical documentation and analysis
- `examples/` - Usage examples and demonstrations
- `tests/` - Comprehensive test suite
- `research_paper.tex` - Complete research paper with detailed methodology

**Additional Components:**
- `frontend/` - Web interface for card detection and game visualization
- `legacy/` - Previous implementations for comparison
- `mcts/` - Core MCTS implementation and training

## Getting Started

1. **Installation**: Follow setup instructions in the repository
2. **Quick Start**: Run `quick_start.py` for immediate demonstration
3. **Training**: Use provided training scripts for custom model development
4. **Evaluation**: Comprehensive test suite available in `tests/`

## Research Contributions

This project represents a significant contribution to AI research in imperfect information games, with novel approaches to:
- Hybrid multi-agent decision making
- Belief network architecture for opponent modeling
- Information Set Monte Carlo Tree Search
- State canonicalization for improved generalization
- Real-time inference in complex game environments

The system's modular architecture and general design principles make it suitable for adaptation to other imperfect information games including poker, bridge, and strategic board games.

---
