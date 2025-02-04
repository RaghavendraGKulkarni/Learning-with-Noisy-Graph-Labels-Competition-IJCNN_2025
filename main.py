import argparse

import numpy as np
import pandas as pd
import json
import pickle

import networkx as nx
from karateclub import Graph2Vec

import torch
from source.models import classifier, myModel
from source.train import train

def set_device_and_seed():
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return device


def create_networkx_graph(graph):
    
    graphDF = pd.DataFrame({'from':graph['edge_index'][0], 'to':graph['edge_index'][1], 'edge_attr':graph['edge_attr']})
    
    G = nx.from_pandas_edgelist(graphDF, source = 'from', target = 'to', edge_attr = 'edge_attr')
    
    degrees = dict(G.degree())
    deg_cent = dict(nx.degree_centrality(G))
    
    node_features = {}
    for node in G.nodes():
        feature_vector = np.zeros(7)
        total = 0.0
        for edge in G.edges(node, data = True):
            adj = (edge[0] + edge[1]) - node
            feature_vector += (np.asarray(edge[2].get('edge_attr', np.zeros(7))) * degrees[adj])
            total += degrees[adj]
        feature_vector /= total
        feature_vector = feature_vector.tolist() + [deg_cent[node]]
        node_features[node] = {str(i) : v for i, v in enumerate(feature_vector)}
    nx.set_node_attributes(G, node_features)
    
    return G


def load_and_test(test_path, device):
    
    with open('./source/filepaths.json', 'r') as f:
        filepaths = json.load(f)
    
    dataset = list(map(str, test_path.split('/')))[-2]
    
    test_graphs = pd.read_json(test_path, compression = 'gzip')
    test_graphs['nx_graph'] = test_graphs.apply(create_networkx_graph, axis = 1)
    
    with open(filepaths[dataset]['features_model'], 'rb') as f:
        g2v = pickle.load(f)
    
    with open(filepaths[dataset]['scaler'], 'rb') as f:
        scaler = pickle.load(f)
    
    test_features = pd.DataFrame(g2v.infer(list(test_graphs['nx_graph'])), columns = [str(x) for x in range(128)])
    test_features = scaler.transform(test_features)
    test_tensor = torch.tensor(test_features, dtype = torch.float32).to(device)
    
    model = classifier().to(device)
    model.load_state_dict(torch.load(filepaths[dataset]['prediction_model'], weights_only = True))
    
    with torch.no_grad():
        pred_probs = model(test_tensor)
    
    pred_labels = torch.argmax(pred_probs, dim = 1).cpu().numpy()
    ids = np.asarray([x for x in range(len(pred_labels))])
    pred_df = pd.DataFrame({'id':ids, 'pred':pred_labels})
    
    pred_df.to_csv(filepaths[dataset]['test_predictions'], index = False)
    
    return
        

def train_and_test(train_path, test_path, device):
    
    train_graphs = pd.read_json(train_path, compression = 'gzip')
    train_graphs['nx_graph'] = train_graphs.apply(create_networkx_graph, axis = 1)
    
    test_graphs = pd.read_json(test_path, compression = 'gzip')
    test_graphs['nx_graph'] = test_graphs.apply(create_networkx_graph, axis = 1)
    
    print("Networkx Conversion completed")
    
    g2v = Graph2Vec(wl_iterations = 5, epochs = 20, learning_rate = 0.2)
    g2v.fit(list(train_graphs['nx_graph']))
    
    with open('./checkpoints/E/features_model.pkl', 'wb') as f:
        pickle.dump(g2v, f)
    
    train_features = pd.DataFrame(g2v.get_embedding(), columns = [str(x) for x in range(128)])
    train_features['y'] = train_features.index.map(lambda x : train_graphs.iloc[x]['y'][0][0])
    test_features = pd.DataFrame(g2v.infer(list(test_graphs['nx_graph'])), columns = [str(x) for x in range(128)])
    
    print("Feature extraction completed")
    
    train(train_features, device)
    
    print("Training completed")
    
    with open('./checkpoints/E/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    test_features = scaler.transform(test_features)
    test_tensor = torch.tensor(test_features, dtype = torch.float32).to(device)
    
    model = classifier().to(device)
    model.load_state_dict(torch.load('./checkpoints/E/prediction_model.pth', weights_only = True))
    
    with torch.no_grad():
        pred_probs = model(test_tensor)
    
    pred_labels = torch.argmax(pred_probs, dim = 1).cpu().numpy()
    ids = np.asarray([x for x in range(len(pred_labels))])
    pred_df = pd.DataFrame({'id':ids, 'pred':pred_labels})
    
    pred_df.to_csv('./submission/testset_E.csv', index = False)

    return


def main(args):
    
    device = set_device_and_seed()
    
    if not args.train_path:
        load_and_test(args.test_path, device)
    
    else:
        train_and_test(args.train_path, args.test_path, device)

    return
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--test_path", type = str, required = True)
    parser.add_argument("--train_path", type = str)
    
    args = parser.parse_args()
    
    main(args)