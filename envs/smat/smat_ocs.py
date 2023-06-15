from __future__ import annotations
import copy
import time

import numpy as np

from envs.smat.Flatbuffer import MsgType, MsgBuilder
import gymnasium as gym
import torch
import subprocess
from abc import ABC
from typing import Any
from gymnasium.core import ActType, ObsType
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from agents.pytorch.utilities import summary_graph

from utils.zero_mq import ZmqServer


class OHTRouting(gym.Env, ABC):
    def __init__(self, kwargs):
        kwargs['args']['id'] = "root"
        kwargs['args']['pw'] = "151212kyhASH@"
        kwargs['env_path'] = r"D:\MnS\Projects\Pinokio.V2\PinokioRL\PinokioRL\bin\Debug\PinokioRL.exe"
        self.__params = kwargs
        self.id = -1
        # Communication manager
        self.server = ZmqServer(self.__params['args']['zmqport'], self.get_state)
        self.builder = MsgBuilder.OHTMsgBuilder()
        # Environment 실행
        self.start_env()
        self.episode_length = 80
        self.episode_counter = 0
        # 젠장..
        self.reset_counter = 0

    # region local custom
    def start_env(self):
        args = [self.__params['env_path']]
        for key, value in self.__params['args'].items():
            arg = key + ":" + value
            args.append(arg)
        subprocess.Popen(args, shell=True, stdin=None, stdout=subprocess.DEVNULL, stderr=None, close_fds=True)

    def get_state(self, buff: bytearray):
        request = self.builder.GetRequestMsg(buff)
        state = request.StateAsNumpy()
        reward = request.Reward()
        done = False
        truncate = False
        self.id = request.OhtId()
        info = {"OHTID": self.id,
                "StateEmpty": request.StateIsNone(),
                "StateLength": request.PrevStateLength()}

        # state = np.nan_to_num(state, copy=True)
        if np.isnan(state).any():
            adj_matrix = copy.deepcopy(torch.tensor(state[:169]))
            adj_matrix = adj_matrix.view(13, 13)
            feature_nodes = torch.tensor(copy.deepcopy(state[169:]))
            feature_nodes = feature_nodes.view(13, 4)
            adj_numpy = adj_matrix.numpy()
            feature_numpy = feature_nodes.numpy()

        adj_matrix = copy.deepcopy(state[:169])
        adj_matrix = adj_matrix.reshape(13, 13)
        feature_nodes = torch.tensor(copy.deepcopy(state[169:]))
        feature_nodes = feature_nodes.view(13, 4)

        edge_indices, edge_attributes = dense_to_sparse(torch.tensor(adj_matrix))
        state = Data(x=feature_nodes, edge_index=edge_indices, edge_attr=edge_attributes)
        # print("Reward: ", reward)
        return state, reward, done, truncate, info

    def __achilles(self):
        import psutil
        del self.server
        del self.builder
        for proc in psutil.process_iter():
            if proc.name() == "PinokioRL.exe":
                proc.kill()
        self.reset_counter = 0
        time.sleep(5)
        # Communication manager
        self.server = ZmqServer(self.__params['args']['zmqport'], self.get_state)
        self.builder = MsgBuilder.OHTMsgBuilder()
        # Environment 실행
        self.start_env()
        return self.server.listen()

    # endregion
    # region Gym environment override
    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        graph, reward, done, truncate, info = self.server.listen()
        state = {'matrix': torch.empty(0),
                 'vector': torch.empty(0),
                 'graph': graph,
                 'action_mask': torch.empty(0)}
        return state, info

    def render(self, mode="human"):
        pass

    def step(self, action):
        # print("step in")
        msg = self.builder.BuildReplyMessage(self.id, MsgType.MsgType.DoneCheck, action)
        self.server.send(msg)
        # print("step send")
        if self.reset_counter > 500:
            graph, reward, done, truncate, info = self.__achilles()
        else:
            self.reset_counter += 1
            begin = time.time()
            graph, reward, done, truncate, info = self.server.listen()
            # print("step listen done")
            # print("interaction_time: ", time.time() - begin)
        # print("action: ", action, " and reward: ", reward)
        reward = torch.clamp(torch.tensor(reward), min=-5, max=1).item()
        state = {'matrix': torch.empty(0),
                 'vector': torch.empty(0),
                 'graph': graph,
                 'action_mask': torch.empty(0)}
        self.episode_counter += 1
        if self.episode_length == self.episode_counter:
            self.episode_counter = 0
            done = True
            truncate = True
        return state, reward, done, truncate, info

    def close(self):
        pass

    # endregion
