import gymnasium as gym
from stable_baselines3 import PPO
from env_wrappers import SB3SingleAgentWrapper
from game28_pz import Game28Env

env = Game28Env(render_mode=None)
env = SB3SingleAgentWrapper(env)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
