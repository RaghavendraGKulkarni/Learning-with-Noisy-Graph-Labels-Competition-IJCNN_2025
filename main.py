import argparse

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import json
import gzip
import random
import pickle
from tqdm import tqdm

import torch
from torch_geometric.loader import DataLoader

from source.models import myGNN, add_zeros
from source.train import train
from source.loadData import myDataset
import networkx as nx
from torch_geometric.utils import to_networkx

def set_device_and_seed():
    
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return device

def load_graphs_from_json(filename):
    with gzip.open(filename, 'rt', encoding = 'utf-8') as file:
        graphs = json.load(file)
    return graphs

def compute_graph_features(graphs_dict):
    for graph in graphs_dict:
        G = nx.Graph()
        G.add_edges_from(np.asarray(graph['edge_index']).T.tolist())
        centrality = nx.degree_centrality(G)
        centrality_values = list(centrality.values())
        max_centrality = max(centrality_values) if centrality_values else 0.0
        min_centrality = min(centrality_values) if centrality_values else 0.0
        avg_centrality = sum(centrality_values) / len(centrality_values) if centrality_values else 0.0
        assortativity = nx.degree_assortativity_coefficient(G) if len(G.nodes) > 1 else 0.0
        graph['graph_features'] = [max_centrality, min_centrality, avg_centrality, assortativity]
    return graphs_dict

def load_and_test(test_path, batch_size, device):
    
    dataset = list(map(str, test_path.split('/')))[-2]
    with open('./source/filepaths.json', 'r') as f:
        filepaths = json.load(f)
    
    test_graphs = load_graphs_from_json(test_path)
    test_graphs = compute_graph_features(test_graphs)
    
    test_dataset = myDataset(test_graphs, weights = None, transform = add_zeros)
    test_loader = DataLoader(test_dataset, batch_size = batch_size)
    
    model = myGNN(num_classes = 6, num_layers = 2, dim = 128, dropout = 0.5, residual = True).to(device)
    model.load_state_dict(torch.load(filepaths[dataset]['prediction_model']))
    model.eval()
    
    test_pred_labels = []
    with torch.no_grad():
        for data in tqdm(test_loader, desc = "Evaluating testing graphs", unit = "batch"):
            data = data.to(device)
            output = model(data)
            test_pred_labels += output.cpu().numpy().tolist()
    test_pred_labels = np.argmax(np.asarray(test_pred_labels), axis = 1)
    
    ids = np.asarray([x for x in range(len(test_pred_labels))])
    pred_df = pd.DataFrame({'id':ids, 'pred':test_pred_labels})
    
    pred_df.to_csv(filepaths[dataset]['test_predictions'], index = False)
    
    return
        

def train_and_test(train_path, test_path, batch_size, device):
    
    train_graphs = load_graphs_from_json(train_path)
    test_graphs = load_graphs_from_json(test_path)
    
    train_graphs = compute_graph_features(train_graphs)
    test_graphs = compute_graph_features(test_graphs)
    
    test_dataset = myDataset(test_graphs, weights = None, transform = add_zeros)
    test_loader = DataLoader(test_dataset, batch_size = batch_size)
    print("Data loading completed")
    
    best_model = train(train_graphs, device, batch_size)
    print("Training completed")
    
    test_pred_labels = []
    best_model = best_model.to(device)
    with torch.no_grad():
        for data in tqdm(test_loader, desc = "Evaluating testing graphs", unit = "batch"):
            data = data.to(device)
            output = best_model(data)
            test_pred_labels += output.cpu().numpy().tolist()
    test_pred_labels = np.argmax(np.asarray(test_pred_labels), axis = 1)
    
    ids = np.asarray([x for x in range(len(test_pred_labels))])
    pred_df = pd.DataFrame({'id':ids, 'pred':test_pred_labels})
    
    pred_df.to_csv('./submission/testset_E.csv', index = False)

    return


def main(args):
    
    device = set_device_and_seed()
    
    if not args.train_path:
        load_and_test(args.test_path, args.batch_size, device)
    
    else:
        train_and_test(args.train_path, args.test_path, args.batch_size, device)

    return
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--test_path", type = str, required = True)
    parser.add_argument("--batch_size", type = int, default = 128)
    parser.add_argument("--train_path", type = str)
    
    args = parser.parse_args()
    
    main(args)