import numpy as np
import gymnasium as gym
from gymnasium import spaces

class Game28Env(gym.Env):
    """
    Simplified 28 card game environment for RL training.
    Turn-based: single-agent control for now.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        self.num_cards = 32  # Full deck after removing low cards
        self.max_hand_size = 8

        # Observation: binary vector showing what cards the player holds
        self.observation_space = spaces.MultiBinary(self.num_cards)

        # Action: choose a card index from 0 to 31
        self.action_space = spaces.Discrete(self.num_cards)

        self.render_mode = render_mode
        self.reset()

    def _deal_cards(self):
        deck = np.arange(self.num_cards)
        np.random.shuffle(deck)
        self.hands = deck[:self.max_hand_size]  # player’s hand
        self.table_cards = []
        self.cards_left = set(self.hands)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._deal_cards()
        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        obs = np.zeros(self.num_cards, dtype=int)
        obs[list(self.cards_left)] = 1
        return obs

    def step(self, action):
        terminated = False
        truncated = False
        reward = 0

        valid_actions = list(self.cards_left)

        if action not in valid_actions:
            # Invalid action handling
            if len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
            reward -= 5  # penalty for illegal move
            if self.render_mode == "human":
                print(f"[Invalid Move] Played illegal card! Penalty applied.")

        # Play the chosen card
        self.cards_left.remove(action)
        self.table_cards.append(action)

        # Reward logic (example: random)
        reward += np.random.randint(-1, 2)

        if len(self.cards_left) == 0:
            terminated = True
            if self.render_mode == "human":
                print("[Round Ended] No cards left.")

        obs = self._get_obs()
        info = {"cards_left": len(self.cards_left)}

        return obs, reward, terminated, truncated, info

    def render(self):
        print(f"Hand: {sorted(list(self.cards_left))}, Table: {self.table_cards}")
