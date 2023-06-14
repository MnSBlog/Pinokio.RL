import copy
import struct
from agents.pytorch.utilities import summary_graph
from torch_geometric.data import Data
import torch
import Batch
from Batch.FlatData import FlatData
from Batch.DataArray import DataArray
import numpy as np


f = open("file2.txt", 'rb')
total_matrix = np.fromfile(f, np.float32)
total_matrix = np.reshape(total_matrix, (17, 13))
f.close()

adj_matrix = copy.deepcopy(total_matrix[:13, :])
feature_nodes = copy.deepcopy(total_matrix[13:, :])

from torch_geometric.utils import dense_to_sparse
edge_indices, edge_attributes = dense_to_sparse(torch.tensor(adj_matrix))
data = Data(x=feature_nodes.T, edge_index=edge_indices, edge_attr=edge_attributes)
summary_graph(data, draw=True)

