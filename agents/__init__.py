from agents.general_agent import PolicyAgent
from agents.pytorch.discrete_sac_agent import Discrete_SAC as TORCH_SAC
from agents.pytorch.ppo_agent import PPO as TORCH_PPO
from functools import partial


def get_policy_fn(agent, **kwargs) -> PolicyAgent:
    return agent(**kwargs)

REGISTRY = {'torchPPO': partial(get_policy_fn, agent=TORCH_PPO),
            'torchSAC': partial(get_policy_fn, agent=TORCH_SAC)}

# REGISTRY = {'tfPPO': partial(get_policy_fn, agent=TF_PPO),
#             'tfA2C': partial(get_policy_fn, agent=TF_A2C),
#             'torchPPO': partial(get_policy_fn, agent=TORCH_PPO)}
