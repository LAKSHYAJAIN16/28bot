# 28Bot v2 - System Organization

This directory contains the main implementation of the 28Bot v2 system, organized for easy navigation and maintenance.

## Quick Start

```bash
# Run a quick demonstration
python main.py quick-start

# Train the belief network
python main.py train-belief

# Play a game simulation
python main.py play-game
```

## Directory Structure

### Core Components
- **`agents/`** - AI agent implementations (hybrid decision-making)
- **`belief_model/`** - Belief network for opponent modeling
- **`game28/`** - Game logic and state management
- **`ismcts/`** - Information Set Monte Carlo Tree Search
- **`rl_bidding/`** - Reinforcement learning for bidding strategies

### Scripts & Tools
- **`scripts/`** - Training scripts and game simulations
- **`examples/`** - Usage examples and demonstrations
- **`utils/`** - Utility functions and analysis tools
- **`viz/`** - Visualization and rendering tools

### Data & Models
- **`models/`** - Trained model files (122+ models)
- **`data/`** - Training data and analysis files
- **`logs/`** - Comprehensive logging and evaluation

### Documentation & Research
- **`docs/`** - Technical documentation and guides
  - **`research/`** - Research papers and technical explanations
  - **`notebooks/`** - Jupyter notebooks for training and analysis
  - **`technical/`** - Detailed technical documentation

### Testing
- **`tests/`** - Comprehensive test suite
  - **`unit/`** - Unit tests
  - **`integration/`** - Integration tests
  - **`debug/`** - Debug utilities

## Main Entry Points

- **`main.py`** - Main entry point with command-line interface
- **`quick_start.py`** - Quick demonstration script
- **`config.py`** - Central configuration file

## Key Features

- **Hybrid AI System**: Combines belief networks, MCTS, and RL
- **Real-time Inference**: Sub-10ms decision making
- **Comprehensive Testing**: 25+ test files covering all components
- **Extensive Documentation**: Research papers, technical docs, and examples
- **Modular Architecture**: Easy to extend and adapt to other games

## Performance

- **Win Rate**: 52.8% (25.4% improvement over baselines)
- **Bidding Accuracy**: 73%
- **Opponent Hand Prediction**: 77.5% accuracy
- **Trump Prediction**: 67.5% accuracy

For detailed usage instructions, see the examples in the `examples/` folder and documentation in `docs/`.