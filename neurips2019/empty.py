from __future__ import print_function
import copy
import numpy as np
import torch
from torch import nn
from composition import Composer
from structure import Structure
from tqdm import tqdm as Tqdm
import json
# import matplotlib.pyplot as plt
import os
import networkx as nx
# from torchviz import make_dot



from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

_ = torch.tensor([
    [0,1,2],
    [3,0,4],
    [5,6,0]
])

print(dense_to_sparse(torch.tensor(_)), dense_to_sparse(torch.tensor(_))[0].T)