from envs.rlfps.rlfps_v4 import RLFPSv4
<<<<<<< HEAD
from envs.dogfight.dogfight_v1 import Dogfightv1
=======
from envs.ohtsim.oht_routing import OhtBase
>>>>>>> 8bb0add5a694264079f7cdbe3732d9a7bb4b4514
from functools import partial
import gym


def get_env_fn(env, **kwargs) -> gym.Env:
    return env(kwargs)


REGISTRY = {'RLFPSv4': partial(get_env_fn, env=RLFPSv4),
<<<<<<< HEAD
            'Dogfightv1': partial(get_env_fn, env=Dogfightv1)}
=======
            'OHTRouting': partial(get_env_fn, env=OhtBase)}
>>>>>>> 8bb0add5a694264079f7cdbe3732d9a7bb4b4514
