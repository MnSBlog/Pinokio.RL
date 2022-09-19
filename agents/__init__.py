from agents.general_agent import PolicyAgent
from agents.tf.ppo_agent import PPO as TF_PPO
from agents.tf.a2c_agent import A2C as TF_A2C
from agents.pytorch.ppo_agent import PPO as TORCH_PPO
from functools import partial


def get_policy_fn(agent, **kwargs) -> PolicyAgent:
    return agent(**kwargs)


REGISTRY = {'tfPPO': partial(get_policy_fn, agent=TF_PPO),
            'tfA2C': partial(get_policy_fn, agent=TF_A2C),
            'torchPPO': partial(get_policy_fn, agent=TORCH_PPO)}
