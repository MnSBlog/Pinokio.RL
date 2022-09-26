import copy
import numpy as np
import torch
import torch.nn as nn
from agents.pytorch.utilities import get_device
from agents.general_agent import GeneralAgent

class DQN(GeneralAgent):
    def __init__(self, parameters: dict):
        super(DQN, self).__init__(parameters=parameters)

    def select_action(self, state):
        raise NotImplementedError

    def update(self, next_state=None, done=None):
        raise NotImplementedError

    def save(self, checkpoint_path: str):
        raise NotImplementedError

    def load(self, checkpoint_path: str):
        raise NotImplementedError
    