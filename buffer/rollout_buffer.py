import copy

import numpy as np
from buffer.base import BaseBuffer


class RolloutBuffer(BaseBuffer):
    def __init__(self):
        super(RolloutBuffer, self).__init__()
        self.buffer = list()

    def store(self, transitions):
        if self.first_store:
            self.check_dim(transitions[0])
        self.buffer += copy.deepcopy(transitions)

    def sample(self, batch_size=None):
        transitions = self.stack_transition(self.buffer)

        self.buffer.clear()
        return transitions

    def clear(self):
        self.buffer = list()

    @property
    def size(self):
        return len(self.buffer)
