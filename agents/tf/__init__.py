from functools import partial
from agents.general_agent import GeneralAgent, PolicyAgent
from agents.tf.ppo_agent import PpoAgent
from agents.tf.a2c_agent import A2cAgent


def policy_fn(agent, **kwargs) -> PolicyAgent:
    return agent(**kwargs)


def agent_fn(agent, **kwargs) -> GeneralAgent:
    return agent(**kwargs)


REGISTRY = {'tf_PPO': partial(policy_fn, agent=PpoAgent),
            'tf_A2C': partial(policy_fn, agent=A2cAgent)}
