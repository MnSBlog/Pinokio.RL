from envs.rlfps.rlfps_v4 import RLFPSv4
from envs.ohtsim.oht_routing import OhtBase
from envs.pathfindSim.pathfind_v1 import PathFindSim
from envs.fightersim.combat_strategy import CombatStrategy
from functools import partial
import gym


def get_env_fn(env, **kwargs) -> gym.Env:
    return env(kwargs)

REGISTRY = {'RLFPSv4': partial(get_env_fn, env=RLFPSv4),
			'CombatStrategy': partial(get_env_fn, env=CombatStrategy),
			'OHTRouting': partial(get_env_fn, env=OhtBase),
			'PathFindSim': partial(get_env_fn, env=PathFindSim)}
