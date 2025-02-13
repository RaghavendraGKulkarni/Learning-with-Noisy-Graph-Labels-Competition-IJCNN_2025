import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import degree

import math

class GINConv(MessagePassing):
    
    def __init__(self, emb_dim):
        super(GINConv, self).__init__(aggr = "add")
        self.convs = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim//2), 
                                       torch.nn.BatchNorm1d(emb_dim//2), 
                                       torch.nn.ReLU(), 
                                       torch.nn.Linear(emb_dim//2, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.edge_encoder = torch.nn.Linear(7, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)
        out = self.convs((1 + self.eps) * x + self.propagate(edge_index, x = x, edge_attr = edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

class GNN_Node(torch.nn.Module):
    
    def __init__(self, num_layers, dim, dropout, residual):
        super(GNN_Node, self).__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.dropout = dropout
        self.residual = residual
        self.node_encoder = torch.nn.Embedding(1, self.dim)
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.convs.append(GINConv(self.dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(self.dim))
    
    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        h_list = [self.node_encoder(x)]
        for layer in range(self.num_layers):
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                h = F.dropout(h, self.dropout, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training = self.training)
            if self.residual:
                h += h_list[layer]
            h_list.append(h)
        node_representation = 0
        for layer in range(self.num_layers + 1):
            node_representation += h_list[layer]
        return node_representation