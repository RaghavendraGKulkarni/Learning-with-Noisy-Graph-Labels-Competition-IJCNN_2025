import numpy as np
import torch
import heapq
import logging
from sklearn.metrics import accuracy_score, f1_score

from torch_geometric.nn import global_mean_pool
from source.conv import GNN_Node

class myGNN(torch.nn.Module):
    
    def __init__(self, num_classes, num_layers, dim, dropout, residual):
        super(myGNN, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dim = dim
        self.gnn_node = GNN_Node(self.num_layers, self.dim, dropout, residual)
        self.pooler = global_mean_pool
        self.predictor = torch.nn.Linear(self.dim, self.num_classes)
        pass
    
    def forward(self, batched_data):
        node_embedding = self.gnn_node(batched_data)
        graph_embedding = self.pooler(node_embedding, batched_data.batch)
        return self.predictor(graph_embedding)