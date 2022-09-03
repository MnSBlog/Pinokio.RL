from functools import partial
from agents.general_agent import GeneralAgent
from agents.tf.ppo_agent import PpoAgent
from agents.tf.a2c_agent import A2cAgent


def agent_fn(env, **kwargs) -> GeneralAgent:
    return env(**kwargs)


REGISTRY = {'tf_PPO': partial(agent_fn, env=PpoAgent),
            'tf_A2C': partial(agent_fn, env=A2cAgent)}
