#!/usr/bin/env python3
"""
train_28_qlearning.py

Tabular Q-learning for the Indian card game "28".
Agent learns as player 0. Other players play a simple random policy.
Legal moves enforced: must follow suit if possible.
Demo output is readable: shows tricks, cards, and final scores.
"""

import random
import pickle
from collections import defaultdict
import numpy as np
import argparse

# -------------------------
# Game constants
# -------------------------
SUITS = ["Hearts", "Diamonds", "Clubs", "Spades"]
RANKS = [14, 13, 12, 11, 10, 9, 8, 7]  # A=14, K=13, Q=12, J=11, 10=10...
POINTS = {14: 1, 13: 0, 12: 0, 11: 3, 10: 1, 9: 0, 8: 0, 7: 0}

def create_deck():
    return [(r, s) for s in SUITS for r in RANKS]

def card_str(card):
    r, s = card
    name = {14: "A", 13: "K", 12: "Q", 11: "J", 10: "10", 9: "9", 8: "8", 7: "7"}[r]
    return f"{name}-{s[0]}"

# -------------------------
# Environment
# -------------------------
class TwentyEightEnv:
    def __init__(self, seed=None):
        self.players = [0, 1, 2, 3]
        self.trump = None
        self.hands = {}
        self.scores = [0, 0]  # team 0(0&2) vs team 1(1&3)
        self.current_trick = []
        self.leader = 0
        self.current_player = 0
        self.done = False
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def reset(self):
        deck = create_deck()
        random.shuffle(deck)
        self.hands = {p: deck[p*8:(p+1)*8] for p in self.players}
        self.trump = random.choice(SUITS)
        self.scores = [0, 0]
        self.current_trick = []
        self.leader = 0
        self.current_player = 0
        self.done = False
        return self.get_obs_for_player(0)

    def legal_moves(self, player):
        hand = self.hands[player]
        if not hand:
            return []
        if not self.current_trick:
            return list(range(len(hand)))
        lead_suit = self.current_trick[0][0][1]
        same_suit_indices = [i for i, c in enumerate(hand) if c[1] == lead_suit]
        return same_suit_indices if same_suit_indices else list(range(len(hand)))

    def play_card(self, player, card_idx):
        card = self.hands[player].pop(card_idx)
        self.current_trick.append((card, player))
        return card

    def trick_winner_and_points(self):
        winning_card, winning_player = self.current_trick[0]
        for card, player in self.current_trick[1:]:
            if card[1] == winning_card[1] and card[0] > winning_card[0]:
                winning_card, winning_player = card, player
            elif card[1] == self.trump and winning_card[1] != self.trump:
                winning_card, winning_player = card, player
        trick_points = sum(POINTS[c[0]] for c, _ in self.current_trick)
        return winning_player, trick_points

    def advance_after_trick(self, winner, trick_points):
        if winner % 2 == 0:
            self.scores[0] += trick_points
        else:
            self.scores[1] += trick_points
        self.current_trick = []
        self.leader = winner
        self.current_player = winner

    def step(self, action_idx):
        if self.done:
            raise RuntimeError("Step on terminated environment")
        player = self.current_player
        legal = self.legal_moves(player)
        if action_idx not in legal:
            action_idx = random.choice(legal)
        played = self.play_card(player, action_idx)
        reward = 0.0
        if len(self.current_trick) == 4:
            winner, trick_points = self.trick_winner_and_points()
            self.advance_after_trick(winner, trick_points)
        else:
            self.current_player = (self.current_player + 1) % 4
        if all(len(self.hands[p]) == 0 for p in self.players):
            self.done = True
            reward = float(self.scores[0] - self.scores[1])
        return self.get_obs_for_player(0), reward, self.done

    def step_until_agent(self, action_idx, opponent_policy):
        obs, reward, done = self.step(action_idx)
        if done:
            return obs, reward, done
        while not done and self.current_player != 0:
            player = self.current_player
            legal = self.legal_moves(player)
            act = opponent_policy(self, player)
            if act not in legal:
                act = random.choice(legal)
            obs, reward, done = self.step(act)
        return obs, reward, done

    def get_obs_for_player(self, player):
        hand = self.hands[player]
        hand_enc = tuple(sorted([RANKS.index(r)+SUITS.index(s)*8 for r,s in hand]))
        trump_idx = SUITS.index(self.trump)
        trick_len = len(self.current_trick)
        return (hand_enc, trump_idx, trick_len, self.leader, self.current_player)

    def render(self):
        print("Trump:", self.trump)
        for p in self.players:
            print(f"P{p} hand:", " ".join(card_str(c) for c in self.hands[p]))
        print("Scores:", self.scores)
        if self.current_trick:
            print("Current trick:", [(card_str(c), pl) for c,pl in self.current_trick])
        print("Leader:", self.leader, "Current:", self.current_player)
        print("-"*40)

# -------------------------
# Opponent policy
# -------------------------
def random_opponent_policy(env, player):
    legal = env.legal_moves(player)
    return random.choice(legal)

# -------------------------
# Q-learning agent
# -------------------------
class QAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, seed=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(float)
        if seed: random.seed(seed)

    def choose_action(self, env, state, legal_indices, exploit_only=False):
        hand = env.hands[env.current_player]
        card_ids = [RANKS.index(hand[i][0]) + SUITS.index(hand[i][1])*8 for i in legal_indices]
        if (not exploit_only) and random.random() < self.epsilon:
            choice = random.choice(range(len(legal_indices)))
            return legal_indices[choice]
        q_vals = [self.Q[(state, cid)] for cid in card_ids]
        max_q = max(q_vals)
        best_idxs = [i for i,q in enumerate(q_vals) if q==max_q]
        choice = random.choice(best_idxs)
        return legal_indices[choice]

    def update(self, state, action_card_int, reward, next_state, next_legal_card_ids, done):
        key = (state, action_card_int)
        if done:
            target = reward
        else:
            future = max([self.Q[(next_state, cid)] for cid in next_legal_card_ids]) if next_legal_card_ids else 0.0
            target = reward + self.gamma*future
        self.Q[key] += self.alpha*(target - self.Q[key])

    def save(self, path):
        with open(path,"wb") as f:
            pickle.dump(dict(self.Q), f)

    def load(self, path):
        with open(path,"rb") as f:
            raw = pickle.load(f)
            self.Q = defaultdict(float, raw)

# -------------------------
# Training
# -------------------------
def train(episodes=20000, alpha=0.1, gamma=0.95, epsilon=0.2, epsilon_decay=0.9999, opponent_policy=random_opponent_policy):
    env = TwentyEightEnv()
    agent = QAgent(alpha=alpha, gamma=gamma, epsilon=epsilon)
    for ep in range(1, episodes+1):
        state = env.reset()
        while not env.done:
            if env.current_player==0:
                legal = env.legal_moves(0)
                action_idx = agent.choose_action(env, state, legal)
                card = env.hands[0][action_idx]
                card_id = RANKS.index(card[0])+SUITS.index(card[1])*8
                next_state, reward, done = env.step_until_agent(action_idx, opponent_policy)
                if done:
                    next_ids=[]
                else:
                    next_legal = env.legal_moves(0)
                    next_ids = [RANKS.index(env.hands[0][i][0])+SUITS.index(env.hands[0][i][1])*8 for i in next_legal]
                agent.update(state, card_id, reward, next_state, next_ids, done)
                state = next_state
            else:
                player = env.current_player
                legal = env.legal_moves(player)
                act = opponent_policy(env, player)
                if act not in legal: act=random.choice(legal)
                state, reward, done = env.step(act)
        agent.epsilon = max(agent.epsilon*epsilon_decay,0.01)
        if ep%5000==0: print(f"Episode {ep}/{episodes} complete")
    return agent

# -------------------------
# Demo
# -------------------------
def play_demo(agent, games=3):
    env = TwentyEightEnv()
    for g in range(1,games+1):
        env.reset()
        print(f"\n=== Demo Game {g} ===")
        print("Trump:", env.trump)
        env.render()

        while not env.done:
            player = env.current_player
            legal = env.legal_moves(player)
            if not legal:
                # Skip player if no cards
                env.current_player = (env.current_player + 1) % 4
                continue

            # Select card
            if player == 0:
                action_idx = agent.choose_action(env, env.get_obs_for_player(0), legal, exploit_only=True)
            else:
                action_idx = random.choice(legal)

            card_played = env.hands[player][action_idx]
            env.play_card(player, action_idx)
            print(f"Player {player} plays {card_str(card_played)}")

            # Advance to next player or resolve trick
            if len(env.current_trick) == 4:
                winner, trick_points = env.trick_winner_and_points()
                env.advance_after_trick(winner, trick_points)
                print(f"Trick won by Player {winner} (+{trick_points} points)")
                print(f"Scores: {env.scores}\n")
            else:
                env.current_player = (env.current_player + 1) % 4

        print("=== Final Scores ===")
        print(f"Team 0 & 2: {env.scores[0]}")
        print(f"Team 1 & 3: {env.scores[1]}")
        if env.scores[0] > env.scores[1]:
            print("AI's team wins! 🏆")
        elif env.scores[0] < env.scores[1]:
            print("Opponents win! 😢")
        else:
            print("Draw!")

# -------------------------
# CLI
# -------------------------
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--save", type=str, default="q28.pkl")
    parser.add_argument("--load", type=str, default=None)
    args=parser.parse_args()

    if args.load:
        agent = QAgent()
        agent.load(args.load)
        print(f"Loaded Q-table from {args.load}")
    else:
        agent = train(episodes=args.episodes)
        if args.save:
            agent.save(args.save)
            print(f"Saved Q-table to {args.save}")

    if args.demo:
        play_demo(agent)

if __name__=="__main__":
    main()
