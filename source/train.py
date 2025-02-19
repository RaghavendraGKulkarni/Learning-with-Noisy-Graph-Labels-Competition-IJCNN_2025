import os
import pickle
import numpy as np

import torch
from sklearn.metrics import accuracy_score, f1_score
import logging

from source.models import myGNN, myModel

def train(train_graphs, device, batch_size):
    
    classifier = myGNN(num_classes = 6, num_layers = 2, dim = 128, dropout = 0.5, residual = True)
    model = myModel(model = classifier, num_classes = 6, device = device)
    
    model.fit(train_graphs, num_epochs = 10, batch_size = batch_size)
    
    model.bestEpochs.sort()
    best_model = myGNN(num_classes = 6, num_layers = 2, dim = 128, dropout = 0.5, residual = True)
    best_model.load_state_dict(model.bestEpochs[0][3])
    
    parent = './checkpoints/E/'
    for filename in os.listdir(parent):
        if filename.endswith(".pth"):
            file_path = os.path.join(parent, filename)
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error deleting {file_path}: {e}")
    for epoch in model.bestEpochs[:10]:
        name = f'model_E_epoch_{epoch[2]}.pth'
        torch.save({
            'epoch':epoch[2],
            'train_log_likelihood':-epoch[1],
            'validation_log_likelihood':-epoch[0],
            'model_state_dict': epoch[3]
        }, parent + name)
    torch.save(best_model.state_dict(), './checkpoints/E/prediction_model.pth')
    return best_model