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
    model = myModel(classifier_model, num_classes = 6, device = device, bestEpochsNum = 10)
    model.fit(train_X, train_y, validation_X, validation_y, num_epochs = 10)
    
    parent = './checkpoints/E/'
    for filename in os.listdir(parent):
        if filename.endswith(".pth"):
            file_path = os.path.join(parent, filename)
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error deleting {file_path}: {e}")
    
    for epoch in model.bestEpochs:
        name = f'model_E_epoch_{epoch[3]}.pth'
        torch.save({
            'epoch':epoch[3],
            'test_log_likelihood':epoch[2],
            'test_accuracy':epoch[0],
            'test_f1':epoch[1],
            'model_state_dict': epoch[4]
        }, parent + name)
    
    model.bestEpochs.sort(reverse = True)
    best_model = classifier().to(device)
    best_model.load_state_dict(model.bestEpochs[0][4])
    
    torch.save(best_model.state_dict(), './checkpoints/E/prediction_model.pth')
    with open('./checkpoints/E/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return