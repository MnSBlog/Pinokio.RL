from agents.general_agent import PolicyAgent, GeneralAgent
from agents.pytorch.sac_agent import SAC as TORCH_SAC
from agents.pytorch.ppo_agent import PPO as TORCH_PPO
from agents.pytorch.a2c_agent import A2C as TORCH_A2C
from agents.pytorch.apex_agent import ApeX as TorchAPEX
from agents.pytorch.ddpg_agent import DDPG as TORCH_DDPG
from agents.pytorch.dqn_agent import DQN as TORCH_DQN
from agents.pytorch.r2d2_agent import R2D2 as TORCH_R2D2
from agents.pytorch.rainbow_agent import Rainbow as TorchRainbow
from agents.pytorch.td3_agent import TD3 as TORCH_TD3
from agents.pytorch.c51_agent import C51 as TORCH_C51
from functools import partial


def get_policy_fn(agent, **kwargs) -> PolicyAgent:
    return agent(**kwargs)


def get_value_fn(agent, **kwargs) -> GeneralAgent:
    return agent(**kwargs)


REGISTRY = {'torchPPO': partial(get_policy_fn, agent=TORCH_PPO),
            'torchSAC': partial(get_policy_fn, agent=TORCH_SAC),
            'torchA2C': partial(get_policy_fn, agent=TORCH_A2C),
            'torchAPEX': partial(get_value_fn, agent=TorchAPEX),
            'torchDDPG': partial(get_value_fn, agent=TORCH_DDPG),
            'torchDQN': partial(get_value_fn, agent=TORCH_DQN),
            'torchR2D2': partial(get_policy_fn, agent=TORCH_R2D2),
            'torchRainbow': partial(get_value_fn, agent=TorchRainbow),
            'torchC51': partial(get_value_fn, agent=TORCH_C51),
            'torchTD3': partial(get_value_fn, agent=TORCH_TD3)}
