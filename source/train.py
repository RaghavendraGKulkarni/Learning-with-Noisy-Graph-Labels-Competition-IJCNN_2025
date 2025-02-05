import os
import pickle
import numpy as np

import torch

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import heapq
import logging

from .models import classifier, myModel

def train(train_features, device):
    
    logging.basicConfig(filename = './logs/E.log', filemode = 'w', level = logging.INFO, format = '%(asctime)s - %(message)s')
    logging_frequency = 10
    
    train_X, validation_X, train_y, validation_y = train_test_split(train_features.drop(columns = ['y']), train_features['y'], test_size = 0.1, random_state = 42)
    
    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    validation_X = scaler.transform(validation_X)
    
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    classifier_model = classifier()
    model = myModel(classifier_model, num_classes = 6, device = device)
    model.fit(train_X, train_y, validation_X, validation_y, num_epochs = 20)
    
    model.bestEpochs.sort()
    best_model = classifier()
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
            'test_log_likelihood':-epoch[0],
            'model_state_dict': epoch[3]
        }, parent + name)
    torch.save(best_model.state_dict(), './checkpoints/E/prediction_model.pth')
    with open('./checkpoints/E/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return