# 28bot v2 - Quick Start Guide

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/28bot/28bot-v2.git
   cd 28bot-v2
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package (optional):**
   ```bash
   pip install -e .
   ```

## Quick Examples

### Basic Game Play

```python
from game28.game_state import Game28State

# Create a new game
game_state = Game28State()

# Check current state
print(f"Phase: {game_state.phase}")
print(f"Current player: {game_state.current_player}")

# Get legal bids
legal_bids = game_state.get_legal_bids(game_state.current_player)
print(f"Legal bids: {legal_bids}")
```

### Run a Complete Game

```bash
# Run a simple game
python run_game.py --rounds 1

# Run with AI components
python run_game.py --rounds 1 --ismcts --belief-network --visualize

# Run multiple rounds
python run_game.py --rounds 5 --log-game --output results.json
```

### Train Models

```python
# Train bidding policy
from rl_bidding.train_policy import train_bidding_policy
model = train_bidding_policy(num_episodes=1000)

# Train belief network
from belief_model.train_beliefs import train_belief_model
model = train_belief_model(num_games=5000, num_epochs=20)
```

### Use ISMCTS for Decisions

```python
from ismcts.ismcts_bidding import BeliefAwareISMCTS
from belief_model.belief_net import BeliefNetwork

# Create ISMCTS with belief network
belief_network = BeliefNetwork()
ismcts = BeliefAwareISMCTS(belief_network=belief_network, num_simulations=1000)

# Get decision
action, confidence = ismcts.select_action_with_confidence(game_state, player_id=0)
print(f"Action: {action}, Confidence: {confidence:.3f}")
```

### Generate Explanations

```python
from viz.render import BidExplanation

# Create bid explainer
explainer = BidExplanation()

# Generate explanation
explanation = explainer.explain_bid(game_state, player_id=0, bid=20, confidence=0.75)
print(f"Reasoning: {explanation['reasoning']}")
```

## Command Line Usage

### Basic Game Runner

```bash
# Run a single round
python run_game.py

# Run multiple rounds
python run_game.py --rounds 5

# Enable visualization
python run_game.py --visualize

# Save results
python run_game.py --output game_results.json
```

### AI Components

```bash
# Use ISMCTS for decisions
python run_game.py --ismcts --simulations 1000

# Use belief network
python run_game.py --belief-network

# Use RL agent (requires trained model)
python run_game.py --rl-agent
```

### Evaluation

```bash
# Run exploitability evaluation
python -c "from experiments.exploitability import evaluate_exploitability; evaluate_exploitability(strategies, num_games=100)"
```

## Project Structure

```
28bot_v2/
├── game28/              # Core game implementation
│   ├── constants.py     # Game rules and constants
│   └── game_state.py    # Game state management
├── rl_bidding/          # Reinforcement learning components
│   ├── env_adapter.py   # RL environment wrapper
│   └── train_policy.py  # Training scripts
├── belief_model/        # Belief modeling
│   ├── belief_net.py    # Neural network for opponent modeling
│   └── train_beliefs.py # Training scripts
├── ismcts/              # Information Set MCTS
│   └── ismcts_bidding.py # ISMCTS implementation
├── viz/                 # Visualization and explanation
│   └── render.py        # Bid explanations and visualizations
├── experiments/         # Evaluation and experiments
│   └── exploitability.py # Exploitability evaluation
├── run_game.py          # Main game runner
├── example_usage.py     # Comprehensive examples
└── requirements.txt     # Dependencies
```

## Key Features

- **Complete Game Implementation**: Full Game 28 rules with bidding and play phases
- **Reinforcement Learning**: PPO-based bidding policy training
- **Belief Modeling**: Neural network for opponent hand inference
- **ISMCTS**: Information Set Monte Carlo Tree Search for decision making
- **Explanations**: Human-readable bid explanations and reasoning
- **Visualization**: Interactive game state and belief visualizations
- **Evaluation**: Exploitability analysis and strategy comparison

## Next Steps

1. **Train Models**: Run the training scripts to create your own models
2. **Experiment**: Try different combinations of AI components
3. **Evaluate**: Use the exploitability evaluation to compare strategies
4. **Extend**: Add new features or modify existing components

## Troubleshooting

- **Import Errors**: Make sure you're in the correct directory and dependencies are installed
- **CUDA Issues**: Models will automatically use CPU if CUDA is not available
- **Memory Issues**: Reduce the number of ISMCTS simulations for lower memory usage

## Support

For issues and questions:
- Check the main README.md for detailed documentation
- Run `python example_usage.py` for comprehensive examples
- Review the code comments for implementation details
