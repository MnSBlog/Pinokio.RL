import copy
from abc import abstractmethod, ABC
from torch_geometric.loader import DataLoader
import numpy as np
import torch


class BaseBuffer(ABC):
    def __init__(self):
        self.first_store = True

    def check_dim(self, transition):
        print("########################################")
        print("You should check dimension of transition")
        for key, val in transition.items():
            if "graph" in key:
                print(f"Graph nodes: {val.num_nodes}, and feature {val.num_features} ")
            else:
                print(f"{key}: {val.shape}")
        print("########################################")
        self.first_store = False

    def store(self, transitions):
        transitions = copy.deepcopy(transitions)
        del_keys = set()
        for index, object_transition in enumerate(transitions):
            for key, transition in object_transition.items():
                if len(transition) == 0:
                    del_keys.add(key)
                else:
                    if torch.is_tensor(transition) or "graph" in key:
                        transitions[index][key] = transition.to("cpu")
                    else:
                        transitions[index][key] = torch.FloatTensor(transition).to("cpu")

        for index, _ in enumerate(transitions):
            for element in del_keys:
                del transitions[index][element]
        return transitions

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

    @abstractmethod
    def clear(self):
        return

    @abstractmethod
    def size(self):
        return

    @staticmethod
    def stack_transition(batch):
        transitions = {}

        for key, sample in batch[0].items():
            if "graph" in key:
                loader = DataLoader([b[key] for b in batch], batch_size=len(batch))
                transitions[key] = next(iter(loader))
                continue

            if len(batch[0][key]) > 1:
                # Multimodal
                b_list = []
                for i in range(len(batch[0][key])):
                    tmp_transition = np.stack([b[key][i][0] for b in batch], axis=0)
                    b_list.append(tmp_transition)
                transitions[key] = b_list
            else:
                if torch.is_tensor(sample):
                    transitions[key] = torch.cat([b[key] for b in batch], dim=0).detach()
                else:
                    dump = np.stack([b[key][0] for b in batch], axis=0)
                    if len(dump.shape) == 1:
                        dump = np.expand_dims(dump, axis=1)
                    transitions[key] = dump

        return transitions


class DummyBuffer(BaseBuffer):
    def store(self, transitions):
        super(DummyBuffer, self).store(transitions)
        pass

    def sample(self, batch_size):
        pass

    def clear(self):
        pass

    def size(self):
        pass
