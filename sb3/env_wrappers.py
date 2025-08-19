import gymnasium as gym

class SB3SingleAgentWrapper(gym.Wrapper):
    """
    Wraps the Game28Env to be used with Stable-Baselines3.
    Handles Gymnasium reset() and step() formats.
    """

    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info
