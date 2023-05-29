from abc import ABC
import torch
from typing import Optional
import gym
import pandas as pd
import numpy as np


class PaintShop(gym.Env, ABC):
    def __init__(self, kwargs):
        self.cost = pd.read_csv(kwargs['cost_path'], header=None)
        self.upstream = 3 # action space
        self.car_stream = 5
        self.state, _ = self.reset()

    def step(self, action):
        stream = self.streams[action]
        car = stream.pop(0)
        reward = 0
        stream.append(-1)
        self.streams[action] = stream
        if len(self.downstream) > 0:
            old_car = self.downstream[-1]
            cost = self.cost.iloc[car, old_car]
            self.total_cost += cost
            reward = -1 * cost
        self.downstream.append(car)
        done = len(self.downstream) == (self.upstream * self.car_stream)
        if done:
            print("total cost: " + str(self.total_cost))
            print("[downstream]")
            print(self.downstream)
        return self.__update(), reward, done, False, None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        self.total_cost = 0
        self.streams = []
        self.downstream = []
        for i in range(self.upstream):
            stream = []
            for j in range(self.car_stream):
                stream.append(j + (i * self.car_stream))
            self.streams.append(stream)
        return self.__update(), None

    def __update(self):
        mask = []
        for i in range(self.upstream):
            stream = self.streams[i]
            if stream[0] == -1:
                mask.append(0)
            else:
                mask.append(1)

        cost_map = self.cost.to_numpy() # 15, 15
        down_stream = np.array(self.streams) # 3, 5
        down_stream = down_stream.flatten()
        cost_map = cost_map.flatten()
        mask = np.array(mask)
        state = np.concatenate((cost_map, down_stream))

        state = {'matrix': torch.empty(0),
                 'vector': torch.tensor(state, dtype=torch.float).unsqueeze(dim=0),
                 'action_mask': torch.tensor(mask, dtype=torch.float)}
        return state

class SelfPlayPS(gym.Env):
    def __init__(self, kwargs):
        self.envs = []
        for _ in range(kwargs['players']):
            self.envs.append(PaintShop(kwargs))

    def step(self, actions):
        rewards = []
        states = []
        done = False
        for index, action in enumerate(actions):
            ret = self.envs[index].step(action)
            state, reward, done, _, _ = ret
            rewards.append(reward)
            states.append(state)

        sum_reward = sum(rewards) -10
        for index, reward in enumerate(rewards):
            rewards[index] = sum_reward
