# 28Bot - A Multi-Agent AI System for the Card Game 28

This project presents a comprehensive artificial intelligence system for imperfect information games, demonstrated through the card game 28. The system introduces several novel contributions including a hybrid decision-making framework, advanced belief network architecture, and Information Set MCTS implementation.

### Key Innovations
- **Hybrid Decision-Making Framework**: Combines belief networks, Monte Carlo Tree Search (MCTS), and reinforcement learning based on information availability
- **Advanced Belief Network**: Multi-task learning architecture that aims to predict opponent hands, opponent play and trump suits.
- **Point Prediction Model**: Custom RL architecture that predicts the number of points one will score, leaeding to accurate bidding.
- **Information Set MCTS**: Handles imperfect information simulations with belief state management
- **State Canonicalization**: Treats functionally equivalent game states identically, improving learning efficiency and generalization
---

## System Architecture

The system employs a modular architecture that combines multiple AI paradigms through a hybrid decision-making framework.

**Core Components:**
- `belief_model/` - Advanced belief networks for opponent modeling with multi-task learning
- `rl_bidding/` - Reinforcement Learning environment and training for bidding
- `ismcts/` - Information Set Monte Carlo Tree Search implementation
- `hybrid_agent.py` - Hybrid decision-making system with dynamic method selection
--

## Game 28 Overview

Game 28 is a four-player team-based trick-taking card game that serves as an excellent testbed for imperfect information AI systems. The game features:

- **32-card deck**: 7, 8, 9, 10, J, Q, K, A of each suit
- **Point values**: J (3), 9 (2), A (1), 10 (1) - totaling 28 points
- **Complex bidding**: Players bid on points they believe their team can make
- **Trump selection**: Highest bidder sets trump suit (concealed initially)
- **Imperfect information**: Players must infer opponent hands and trump suit
- **Team coordination**: Partners must work together strategically

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

## Getting Started

1. **Installation**: Follow setup instructions in the repository
2. **Quick Start**: Run `quick_start.py` for immediate demonstration

## Research Contributions

This project represents a significant contribution to AI research in imperfect information games, with novel approaches to:
- Hybrid multi-agent decision making
- Belief network architecture for opponent modeling
- Information Set Monte Carlo Tree Search

---
