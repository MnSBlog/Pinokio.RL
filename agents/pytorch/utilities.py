import torch
import numpy as np
from typing import Union
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
import networkx as nx


def get_device(device: Union[torch.device, str] = "auto") -> torch.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return:
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to th.device
    device = torch.device(device)

    # Cuda not available
    if device.type == torch.device("cuda").type and not torch.cuda.is_available():
        return torch.device("cpu")

    return device


class OuNoise:
    def __init__(self, action_size, mu, theta, sigma):
        self.action_size = action_size

        self.mu = mu
        self.theta = theta
        self.sigma = sigma

        self.reset()

    def reset(self):
        self.X = np.ones((1, self.action_size), dtype=np.float32) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X) + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X


# Reference: m-rl official repository (stable_scaled_log_softmax, stable_softmax)
# https://github.com/google-research/google-research/blob/master/munchausen_rl/common/utils.py
def stable_scaled_log_softmax(x, tau):
    max_x, max_indices = torch.max(x, -1, keepdim=True)
    y = x - max_x
    tau_lse = max_x + tau * torch.log(torch.sum(torch.exp(y / tau), -1, keepdim=True))
    return x - tau_lse


def stable_softmax(x, tau):
    max_x, max_indices = torch.max(x, -1, keepdim=True)
    y = x - max_x
    return torch.exp(F.log_softmax(y / tau, -1))


def draw_graph(data: Data, unidirectional=False):
    g = torch_geometric.utils.to_networkx(data, to_undirected=unidirectional)
    nx.draw(g, with_labels=True)


def summary_graph(graph: Data, draw=False):
    print("=========SUMMARY GRAPH=========")
    print("CUDA: ", graph.is_cuda, " Undirected: ", graph.is_undirected(), " Coalesced ", graph.is_coalesced())
    print("=====Node:")
    print("# of node:", graph.num_nodes, "# of node feature", graph.num_node_features)
    print("node types: ", graph.num_node_types)
    print("=====Edges:")
    print("# of edges:", graph.num_edges, "# of edge feature", graph.num_edge_features)
    print("node types: ", graph.num_edge_types)
    print("=====Others:")

    if draw:
        draw_graph(graph)