
28bot v2 — Deep Learning Bidding AI for 28

A research-ready system to explore bidding in imperfect information trick-taking games.

🎯 Project Objective

Build a high-quality, modular AI system for the game of 28 with a focus on bidding, using:

Reinforcement Learning (PPO, NFSP)

Belief Modeling (opponent hand inference)

Joint Policy Optimization

Auction Theory Concepts (bid shading, confidence)

Monte Carlo Tree Search (belief-aware)

Visual and Explainable AI for strategy interpretation

Theoretical rigor (exploitability, convergence, generalization)

🎮 Game: What is "28"?
Game 28 - A Trick-Taking Card Game
Game 28 is a 4-player trick-taking card game with bidding and trump mechanics. Here's how it works:
Basic Setup
Players: 4 players (teams of 2, players 0&2 vs 1&3)
Deck: 32 cards (7, 8, 9, 10, J, Q, K, A of each suit: Hearts, Diamonds, Clubs, Spades)
Dealing: Two-stage deal - first 4 cards to each player, then bidding, then 4 more cards
Card Values & Rankings
Point Values: J=3, 9=2, 10=1, A=1, 7=0, 8=0, Q=0, K=0
Trick Strength (highest to lowest): J > 9 > A > 10 > K > Q > 8 > 7
Total Points: 28 points per round (hence the name "28")
Game Phases
1. Bidding Phase
Players bid starting from 16 points (minimum bid)
Bids range from 16-28
Players can pass or raise the current bid
Winner becomes the "bidder" and chooses trump suit
Stakes: Bids ≥20 double the round value
2. Concealed Phase
Trump suit is chosen but not revealed to other players
Bidder must place their highest trump card face-down
Bidder cannot lead trump unless they only have trump cards
Game continues with trump hidden
3. Revealed Phase
Triggered when a player cannot follow suit and plays trump
Trump suit is now revealed to all players
Face-down trump card is returned to bidder's hand
Trump now functions normally in trick-taking
Trick-Taking Rules
Follow Suit
Players must follow the lead suit if possible
If unable to follow suit, can play any card (can choose to reveal trump if this is suitable)
Winning Tricks
Concealed Phase: Only same-suit cards can win (no trump)
Revealed Phase: Trump cards beat non-trump; highest trump wins
Scoring
Each trick awards points based on card values played
Team scores accumulate throughout the round
Bid Success: If bidder's team meets/exceeds their bid → +1 game point
Bid Failure: If bidder's team falls short → -1 game point
Special Rules
Invalid Rounds
If trump is never exposed by the 7th trick, round is invalid
No game points awarded, round is replayed
Bidder Restrictions
Cannot lead trump during concealed phase (unless only trump remains)
Must place highest trump face-down at start
Team Play
Players 0&2 form Team A
Players 1&3 form Team B
Teams compete for points and game wins
Game Flow
Deal 4 cards each → Bidding → Choose trump → Place face-down trump
Deal 4 more cards each → Play begins
8 tricks total (all 32 cards played)
Determine bid success/failure
Award game points
Repeat for multiple rounds
Key Strategic Elements
Trump Management: When to expose trump vs. when to conserve
Bidding Strategy: Balancing hand strength with risk
Team Coordination: Supporting partner's leads and plays
Hand Reading: Inferring opponent holdings from plays
Point Management: Timing high-value card plays
This creates a complex strategic game where information asymmetry (concealed trump) and team coordination play crucial roles in success.

📂 Project Structure
28bot_v2/
├── rl_bidding/
│   ├── train_policy.py        # PPO/NFSP training
│   └── env_adapter.py         # RL environment wrapper
│
├── belief_model/
│   ├── belief_net.py          # Predicts unseen cards
│   └── train_beliefs.py
│
├── ismcts/
│   ├── ismcts_bidding.py      # Belief-aware MCTS bidding
│   └── ismcts_play.py
│
├── jps/
│   └── refine_bids.py         # Joint Policy Search optimizer
│
├── viz/
│   └── render.py              # Bid + belief explanation
│
├── experiments/
│   ├── exploitability.py
│   └── compare_methods.py
│
├── data/
│   ├── logs/
│   └── saved_models/
└── README.md

🛠️ Development Plan (Step-by-Step)
✅ 1. Train a Deep RL Bidding Agent

Use self-play with PPO/NFSP

Action space = legal bids (14–28, pass)

Observations = 4-card hand, bidding history, position

Reward = +1 if win, −1 if lose

✅ 2. Build Opponent Belief Model

Input: cards played, auction behavior, lead patterns

Output: likelihood of each card in each opponent's hand

Train via simulation

Use this for sampling in MCTS and bidding confidence

✅ 3. Belief-Aware MCTS

Modify ISMCTS to sample using belief net

Adjust rollouts using inferred hands

Use to validate bids and play moves

✅ 4. Joint Policy Search (JPS)

Refine partner coordination

Alternate optimizing partner/bidder

Use JPS to improve team bidding synergy

✅ 5. Auction-Theoretic Bidding Logic

Add risk thresholds & confidence-based shading

Avoid overbidding when belief is uncertain

Experiment with bluffing / bid manipulation

✅ 6. Bid Explanation Visualization

Show:

Inferred hand strength

Trump estimate

Conservative vs. aggressive threshold

Why a bid was chosen

✅ 7. Experimental Evaluation

Compare:

RL vs. analytical vs. hybrid bidding

With/without belief

Metrics:

Win rate

Overbid frequency

Average score margin

Exploitability under best-response opponents

📚 References

Deep Bidding Policies in Skat

Joint Policy Search for Bridge

ReBeL (Search + Learning for Poker)

NFSP in Poker

📦 Deliverables

policy.pt — RL-trained bidding policy

belief_model.pt — Belief network for hidden cards

render.py — Visual explanation of bids

exploitability.py — Evaluation scripts

Full game logs with bidding decisions