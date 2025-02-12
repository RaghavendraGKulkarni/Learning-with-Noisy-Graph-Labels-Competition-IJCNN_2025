import json
import gzip
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data
from tqdm import tqdm

class myDataset(Dataset):
    
    def __init__(self, graph_dicts, weights = None, transform = None, pre_transform = None):
        self.graphs = self.convert_to_data_object(graph_dicts, weights)
        super().__init__(None, transform, pre_transform)
    
    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]
    
    @staticmethod
    def convert_to_data_object(graph_dicts, weights):
        graphList = []
        for i in range(len(graph_dicts)):
            edges = torch.tensor(graph_dicts[i]['edge_index'], dtype = torch.long)
            attributes = torch.tensor(graph_dicts[i]['edge_attr'], dtype = torch.float32) if graph_dicts[i]['edge_attr'] else None
            num_nodes = graph_dicts[i]['num_nodes']
            if  weights is not None:
                labels = torch.tensor([weights[i]], dtype = torch.float32)
            else:
                labels = torch.tensor(graph_dicts[i]['y'][0], dtype = torch.long) if graph_dicts[i]['y'] else None
            graphList.append(Data(edge_index = edges, edge_attr = attributes, num_nodes = num_nodes, y = labels))
        return graphList