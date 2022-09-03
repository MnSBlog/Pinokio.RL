import gym
from agents import REGISTRY as AgentRegistry


class GymRunner:
    def __init__(self, config: dict, env: gym.Env):
        self.config = config
        self.agent = AgentRegistry['tf_PPO'](parameters=self.config['agent'], )
        self.env = env

        self.obs = self.env.reset()
        self.action = None
        self.reward = 0


