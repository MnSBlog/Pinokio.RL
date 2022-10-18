import time
import gym
import torch
import numpy as np
from typing import Optional, Union, Tuple
import threading

from gym.core import ObsType
from matplotlib import pyplot as plt
from pyviz_comms import Comm
from utils.zero_mq import ZmqServer
from utils.comm_manger import CommunicationManager


class PathFindSim(gym.Env):
    def __init__(self, env_config):
        self.__envConfig = env_config
        self.__zmq_server = ZmqServer(self.__envConfig['port'], func=self.on_receive)
        self.__period = self.__envConfig['period']
        self.__state = None
        self.observation_space = np.array(self.__envConfig['observation_space'])
        self.action_space = self.__envConfig['action_space']
        self.initialize()
    def initialize(self):
        t = threading.Thread(target=self.__zmq_server.listen)
        t.start()

    def on_receive(self, msg):
        _, state, mask = CommunicationManager.deserialize_info(msg)
        self.__state = (state, mask)

    def step(self, action: torch.tensor = None):
        # Send action list

        # 받아온 액션을 넘겨준다.
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
        while True:
            print(self.__state)
            if self.__state is not None:
                reward = self.__calculate_reward()
                temp = self.__state
                done = torch.tensor(True, dtype=torch.bool)
                trunc = False
                self.__state = None

                return temp[0], reward, done, trunc, None



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

