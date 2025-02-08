import json
import gzip
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data
from tqdm import tqdm

class myDataset(Dataset):
    
    def __init__(self, filename, transform, pre_transform = None):
        self.filename = filename
        self.graphs = self.load_graphs_from_json(self.filename)
        super().__init__(None, transform, pre_transform)
    
    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]
    
    @staticmethod
    def load_graphs_from_json(filename):
        with gzip.open(filename, 'rt', encoding = 'utf-8') as file:
            graph_dicts = json.load(file)
        graphList = []
        for graph_dict in tqdm(graph_dicts, desc = 'Loading Input Graphs', unit = 'graph'):
            edges = torch.tensor(graph_dict['edge_index'], dtype = torch.long)
            attributes = torch.tensor(graph_dict['edge_attr'], dtype = torch.float32) if graph_dict['edge_attr'] else None
            num_nodes = graph_dict['num_nodes']
            labels = torch.tensor(graph_dict['y'][0], dtype = torch.long) if graph_dict['y'] is not None else None
            graphList.append(Data(edge_index = edges, edge_attr = attributes, num_nodes = num_nodes, y = labels))
        return graphList