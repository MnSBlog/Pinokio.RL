import torch
import copy
from abc import ABC
from typing import SupportsFloat, Any
from agents.pytorch.utilities import summary_graph
import gymnasium as gym
from gymnasium.core import ActType, ObsType
from torch_geometric.data import Data


class GraphEnv(gym.Env, ABC):
    def __init__(self, kwargs):
        if kwargs['multi_step'] <= 1:
            raise Exception("Increase a parameter of multi step more than 1.")
        env = gym.make(kwargs['name'], render_mode='human', autoreset=True)
        state = env.reset()[0]

        self.features = []
        for feature in range(state.shape[0]):
            self.features.append(state[feature])

        self.edge = []
        self.node = [copy.deepcopy(self.features)]
        for index in range(1, kwargs['multi_step']):
            edge = [index, index-1]
            self.edge.append(edge)
            self.node.append(copy.deepcopy(self.features))

        graph = self.__update_graph()
        summary_graph(graph, draw=True)

        self.gym_env = env
        self.__params = kwargs

    def step(self, action: ActType):
        state, reward, done, truncate, info = self.gym_env.step(action)

    def __update_graph(self):
        edge_index = torch.tensor(self.edge, dtype=torch.long)
        x = torch.tensor(self.node, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index.t().contiguous())
        return data
