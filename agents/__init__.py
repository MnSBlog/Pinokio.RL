from functools import partial
from general_agent import GeneralAgent
from ppo_agent import PPO


def get_agent_fn(env, **kwargs) -> GeneralAgent:
    return env(kwargs)


REGISTRY = {'PPO': partial(get_agent_fn, env=PPO)}
