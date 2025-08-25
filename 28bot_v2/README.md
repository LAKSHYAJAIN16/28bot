# 28Bot v2 - Game 28 AI Bot

A comprehensive AI bot for the Game 28 card game, featuring multiple AI approaches including Reinforcement Learning, MCTS, and Point Prediction models.

## 📁 Project Structure

```
28bot_v2/
├── 📚 docs/                    # Documentation and guides
│   ├── README.md              # Main project documentation
│   ├── QUICKSTART.md          # Quick start guide
│   ├── BIDDING_MODEL_USAGE.md # Basic bidding model usage
│   ├── IMPROVED_MODEL_USAGE.md # Improved bidding model usage
│   ├── POINT_PREDICTION_APPROACH.md # Point prediction approach
│   ├── FIRST_4_CARDS_ANALYSIS.md # First 4 cards analysis
│   └── ...                    # Other documentation files
├── 🔧 scripts/                 # Main training and analysis scripts
│   ├── improved_bidding_trainer.py # Train improved bidding model
│   ├── analyze_mcts_data.py   # Analyze MCTS game data
│   ├── point_prediction_model.py # Point prediction model
│   ├── bidding_advisor.py     # Bidding advisor utility
│   ├── run_training.py        # Run training pipeline
│   └── run_game.py            # Run game simulation
├── 💡 examples/                # Usage examples and demos
│   ├── use_improved_bidding_model.py # Use improved model
│   ├── use_bidding_model.py   # Use basic bidding model
│   ├── use_point_prediction.py # Use point prediction
│   ├── simple_improved_bidding_example.py # Simple example
│   └── example_usage.py       # General usage examples
├── 🧪 tests/                   # Testing and debugging scripts
│   ├── test_env.py            # Environment testing
│   ├── test_improved_env.py   # Improved environment testing
│   ├── test_env_minimal.py    # Minimal environment test
│   ├── debug_observation.py   # Debug observation space
│   └── debug_model_behavior.py # Debug model behavior
├── 📊 data/                    # Data files
│   └── mcts_bidding_analysis.json # MCTS analysis data
├── 🤖 models/                  # Trained model files
│   ├── bidding_policy/        # Bidding models
│   └── point_prediction_model.pth # Point prediction model
├── 📈 logs/                    # Training logs and TensorBoard data
├── 🎮 game28/                  # Core game logic
├── 🧠 rl_bidding/             # RL environment and training
├── 🌳 ismcts/                 # MCTS implementation
├── 🧮 belief_model/           # Belief network models
├── 🔬 experiments/            # Experimental code
├── 🎨 viz/                    # Visualization tools
├── requirements.txt           # Python dependencies
└── setup.py                   # Package setup
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Models
```bash
# Train the improved bidding model
python scripts/improved_bidding_trainer.py

# Train the belief model
python scripts/train_belief_model.py
```

### 3. Use the Models
```bash
# Use the improved bidding model
python examples/use_improved_bidding_model.py

# Use the belief model
python examples/use_belief_model.py

# Run a complete game simulation
python simple_game_simulation.py
```

## 🎯 Main Features

### 🤖 AI Models
- **Improved Bidding Model**: MCTS-enhanced RL model for bidding decisions
- **Point Prediction Model**: Neural network for predicting game outcomes
- **MCTS Bot**: Monte Carlo Tree Search implementation
- **Belief Network**: Probabilistic modeling of opponent hands

### 📊 Analysis Tools
- **MCTS Data Analysis**: Analyze patterns from MCTS games
- **First 4 Cards Analysis**: Focus on bidding-relevant cards only
- **Model Evaluation**: Comprehensive evaluation scripts

### 🎮 Game Features
- **Complete Game 28 Implementation**: Full game logic
- **Multiple AI Opponents**: Different AI strategies
- **Interactive Play**: Play against AI bots
- **Performance Analysis**: Win rates and statistics

## 📖 Documentation

### Getting Started
- [Quick Start Guide](docs/QUICKSTART.md) - Get up and running quickly
- [Main README](docs/README.md) - Detailed project overview

### Model Usage
- [Basic Bidding Model](docs/BIDDING_MODEL_USAGE.md) - Use the basic RL bidding model
- [Improved Bidding Model](docs/IMPROVED_MODEL_USAGE.md) - Use the MCTS-enhanced model
- [Belief Model Training](docs/BELIEF_MODEL_TRAINING.md) - Train and use the belief network
- [Point Prediction](docs/POINT_PREDICTION_APPROACH.md) - Use point prediction models

### Technical Details
- [First 4 Cards Analysis](docs/FIRST_4_CARDS_ANALYSIS.md) - Why only first 4 cards matter
- [MCTS Integration](docs/IMPROVING_BIDDING_MODEL_WITH_MCTS.md) - How MCTS improves RL
- [Model Issues](docs/ANALYSIS_BIDDING_MODEL_ISSUES.md) - Analysis of model problems

## 🔧 Scripts Overview

### Training Scripts
- `scripts/improved_bidding_trainer.py` - Train the main improved bidding model
- `scripts/train_belief_model.py` - Train the belief network model
- `scripts/point_prediction_model.py` - Train point prediction models
- `scripts/run_training.py` - Run complete training pipeline

### Analysis Scripts
- `scripts/analyze_mcts_data.py` - Analyze MCTS game patterns
- `scripts/bidding_advisor.py` - Get bidding advice for specific hands

### Game Scripts
- `scripts/run_game.py` - Run full game simulations
- `simple_game_simulation.py` - Complete 4-agent game simulation

## 💡 Examples

### Basic Usage
```python
# Use improved bidding model
from examples.use_improved_bidding_model import ImprovedBiddingModel

model = ImprovedBiddingModel()
suggestion = model.get_bid_suggestion(hand, current_bid, bid_history, position)
```

### Training
```python
# Train improved model
from scripts.improved_bidding_trainer import ImprovedBiddingTrainer

trainer = ImprovedBiddingTrainer()
model = trainer.train_with_mcts_data()
```

## 🧪 Testing

### Environment Tests
```bash
python tests/test_env.py              # Test basic environment
python tests/test_improved_env.py     # Test improved environment
python tests/test_env_minimal.py      # Minimal environment test
```

### Debug Scripts
```bash
python tests/debug_observation.py     # Debug observation space
python tests/debug_model_behavior.py  # Debug model behavior
```

## 📊 Data

The project uses MCTS analysis data stored in `data/mcts_bidding_analysis.json` which contains:
- 901 analyzed MCTS games
- Bidding patterns and success rates
- Hand strength analysis (first 4 cards only)
- Trump suit preferences

## 🤝 Contributing

1. Follow the organized structure
2. Put new scripts in appropriate directories
3. Update documentation in `docs/`
4. Add tests in `tests/`
5. Update this README if needed

## 📝 License

This project is for educational and research purposes.

---

**Note**: This project focuses on the first 4 cards for bidding decisions, as the remaining 4 cards are dealt after bidding in Game 28.
