import copy

import numpy as np
from buffer.base import BaseBuffer


class ReplayBuffer(BaseBuffer):
    def __init__(self, buffer_size=50000):
        super(ReplayBuffer, self).__init__()
        self.buffer = np.zeros(buffer_size, dtype=dict)  # define replay buffer
        self.buffer_index = 0
        self.buffer_size = buffer_size
        self.buffer_counter = 0

    def store(self, transitions):
        transitions = super(ReplayBuffer, self).store(transitions)
        if self.first_store:
            self.check_dim(transitions[0])

        for transition in transitions:
            self.buffer[self.buffer_index] = copy.deepcopy(transition)
            self.buffer_index = (self.buffer_index + 1) % self.buffer_size
            self.buffer_counter = min(self.buffer_counter + 1, self.buffer_size)

    def sample(self, batch_size):
        batch_idx = np.random.randint(self.buffer_counter, size=batch_size)
        batch = self.buffer[batch_idx]

        transitions = self.stack_transition(batch)

        return transitions

    @property
    def size(self):
        return self.buffer_counter

    def clear(self):
        self.buffer = np.zeros(self.buffer_size, dtype=dict)  # define replay buffer
        self.buffer_index = 0
        self.buffer_counter = 0
