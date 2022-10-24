import copy
import time
import gym
import torch
import numpy as np
from typing import Optional, Union, Tuple
import threading

from gym.core import ObsType
from utils.zero_mq import ZmqServer
from utils.comm_manger import CommunicationManager


class PathFindSim(gym.Env):
    def __init__(self, env_config):
        self.__envConfig = env_config
        self.__zmq_server = ZmqServer(self.__envConfig['port'], func=self.on_receive)
        self.__state = None
        self.initialize()

    def initialize(self):
        t = threading.Thread(target=self.__zmq_server.listen)
        t.start()

    def on_receive(self, msg):
        _, state, mask = CommunicationManager.deserialize_info(msg)
        self.__state = (state, mask)

    def step(self, action: torch.tensor = None):
        # Send action list
        action_buffer = CommunicationManager.serialize_action("Action", action.cpu())
        self.__zmq_server.send(action_buffer)
        return self.__get_observation()

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:

        # To gathering spatial-feature
        state, _, _, _, _ = self.__get_observation()

        return state

    def __get_observation(self):
        while self.__state is None:
            time.sleep(0.001)

        reward = self.__calculate_reward()
        state = copy.deepcopy(self.__state[0])
        mask = copy.deepcopy(self.__state[1])
        done = torch.tensor(False, dtype=torch.bool)
        trunc = False
        self.__state = None

        return (state.squeeze(), mask), reward, done, trunc, None

    def __calculate_reward(self):
        # reward += 1 * step_result[:, 1]  # team win
        # reward += 1 * step_result[:, 2]  # kill score
        # reward -= 1 * step_result[:, 3]  # dead score
        # # reward += 1 * step_result[:, 4]  # damage score
        # # reward -= 1 * step_result[:, 5]  # hitted score
        # reward += 1 * step_result[:, 6]  # healthy ratio
        reward = 1
        return reward

    def render(self, mode="human", **kwargs):
        return

    def seed(self, seed=None):
        return

    def close(self):
        pass
