from envs.rlfps.rlfps_v4 import RLFPSv4
from envs.smat.smat_ocs import OHTRouting
from envs.fightersim.combat_strategy import CombatStrategy
from envs.paintshop.paintshop_v1 import PaintShop
from envs.graph_gym.graphification import GraphEnv
from functools import partial
import gym


def get_env_fn(env, **kwargs) -> gym.Env:
    return env(kwargs)


REGISTRY = {'RLFPSv4': partial(get_env_fn, env=RLFPSv4),
            'CombatStrategy': partial(get_env_fn, env=CombatStrategy),
            'OHTRouting': partial(get_env_fn, env=OHTRouting),
            'PaintShop-v1': partial(get_env_fn, env=PaintShop),
            'Graph': partial(get_env_fn, env=GraphEnv)}
