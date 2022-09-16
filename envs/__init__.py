from envs.rlfps.rlfps_v4 import RLFPSv4
from envs.ohtsim.oht_routing import OhtBase
from functools import partial
import gym


def get_env_fn(env, **kwargs) -> gym.Env:
    return env(kwargs)


REGISTRY = {'RLFPSv4': partial(get_env_fn, env=RLFPSv4)}