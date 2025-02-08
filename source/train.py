import os
import pickle
import numpy as np

import torch
from sklearn.metrics import accuracy_score, f1_score
import logging

from source.models import myGNN

def train(train_loader, device):
    
    logging.basicConfig(filename = './logs/E.log', filemode = 'w', level = logging.INFO, format = '%(asctime)s - %(message)s')
    logging_frequency = 10
    
    model = myGNN(num_classes = 6, num_layers = 5, dim = 128, dropout = 0.5, residual = True).to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    
    num_epochs = 100
    bestEpochs = []
    for epoch in range(1, num_epochs + 1):
        total_loss = 0
        train_true_labels, train_pred_labels = [], []
        for data in train_loader:
            data = data.to(device) 
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_pred_labels += output.cpu().detach().numpy().tolist()
            train_true_labels += data.y.cpu().detach().numpy().tolist()
        train_true_labels, train_pred_labels = np.asarray(train_true_labels), np.argmax(np.asarray(train_pred_labels), axis = 1)
        acc, f1 = accuracy_score(train_true_labels, train_pred_labels), f1_score(train_true_labels, train_pred_labels, average = 'weighted')
        epoch_info = (total_loss, 1.0 - acc, 1.0 - f1, epoch, model.state_dict())
        bestEpochs.append(epoch_info)
        if epoch % logging_frequency == 0:
            print(f"{epoch}/{num_epochs} epochs completed")
            logging.info(f"Epoch: {epoch}/{num_epochs} ---> Training Loss: {total_loss:.4f}, Training Accuracy: {acc:.4f}, Training F1: {f1:.4f}")
    
    bestEpochs.sort()
    best_model = myGNN(num_classes = 6, num_layers = 5, dim = 128, dropout = 0.5, residual = True)
    best_model.load_state_dict(bestEpochs[0][4])
    
    parent = './checkpoints/E/'
    for filename in os.listdir(parent):
        if filename.endswith(".pth"):
            file_path = os.path.join(parent, filename)
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error deleting {file_path}: {e}")
    for epoch in bestEpochs[:10]:
        name = f'model_E_epoch_{epoch[3]}.pth'
        torch.save({
            'epoch':epoch[3],
            'train_loss':epoch[0],
            'train_accuracy':1 - epoch[1],
            'train_f1':1 - epoch[2],
            'model_state_dict': epoch[4]
        }, parent + name)
    torch.save(best_model.state_dict(), './checkpoints/E/prediction_model.pth')
    return best_model