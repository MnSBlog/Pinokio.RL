from abc import abstractmethod, ABC
import numpy as np
import torch


class BaseBuffer(ABC):
    def __init__(self):
        self.first_store = True

    def check_dim(self, transition):
        print("########################################")
        print("You should check dimension of transition")
        for key, val in transition.items():
            if len(val) > 1:
                for i in range(len(val)):
                    print(f"{key}{i}: {val[i].shape}")
            else:
                print(f"{key}: {val.shape}")
        print("########################################")
        self.first_store = False

    @abstractmethod
    def store(self, transitions):
        """
        Store transitions into buffer.

        Parameter Type
        - transitions: List[Dict]
        """

    @abstractmethod
    def sample(self, batch_size):
        """
        Sample transition data from buffer as much as the batch size.

        Parameter Type
        - batch_size:  int
        - transitions: List[Dict]
        """
        transitions = [{}]
        return transitions

    def clear(self):
        return

    @staticmethod
    def stack_transition(self, batch):
        transitions = {}

        for key in batch[0].keys():
            if len(batch[0][key]) > 1:
                # Multimodal
                b_list = []
                for i in range(len(batch[0][key])):
                    tmp_transition = np.stack([b[key][i][0] for b in batch], axis=0)
                    b_list.append(tmp_transition)
                transitions[key] = b_list
            else:
                transitions[key] = np.stack([b[key][0] for b in batch], axis=0)

        return transitions

    def _clear_state_memory(self):
        self.memory_q = {'matrix': [], 'vector': [], 'action_mask': []}

    def _update_memory(self, state=None):
        matrix_obs, vector_obs, mask_obs = [], [], []

        if state is not None:
            self._insert_q(state)

        if self.torch_state:
            if len(self.memory_q['matrix']) > 0:
                matrix_obs = torch.cat(self.memory_q['matrix'], dim=2).detach()
                shape = matrix_obs.shape
                matrix_obs = matrix_obs.view(shape[0], -1, shape[-2], shape[-1])
                self.memory_q['matrix'].pop(0)

            if len(self.memory_q['vector']) > 0:
                vector_obs = torch.cat(self.memory_q['vector'], dim=1).detach()
                shape = vector_obs.shape
                vector_obs = vector_obs.view(shape[0], -1, shape[-1])
                self.memory_q['vector'].pop(0)

            if len(self.memory_q['action_mask']) > 0:
                mask_obs = self.memory_q['action_mask'][-1]
                self.memory_q['action_mask'].pop(0)
        else:
            if len(self.memory_q['matrix']) > 0:
                matrix_obs = np.concatenate(self.memory_q['matrix'], axis=1)
                self.memory_q['matrix'].pop(0)

            if len(self.memory_q['vector']) > 0:
                vector_obs = np.concatenate(self.memory_q['vector'], axis=1)
                self.memory_q['vector'].pop(0)

            if len(self.memory_q['action_mask']) > 0:
                mask_obs = self.memory_q['action_mask'][-1]
                self.memory_q['action_mask'].pop(0)

        state = {'matrix': matrix_obs, 'vector': vector_obs, 'action_mask': mask_obs}
        return state
