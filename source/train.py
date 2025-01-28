import os
import pickle

import torch

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

import heapq
import logging

from .models import myModel, myLoss

def train(train_features, device):
    
    logging.basicConfig(filename = './logs/E.log', filemode = 'w', level = logging.INFO, format = '%(asctime)s - %(message)s')
    logging_frequency = 10
    
    train_X, validation_X, train_y, validation_y = train_test_split(train_features.drop(columns = ['y']), train_features['y'], test_size = 0.1, random_state = 42)
    
    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    validation_X = scaler.transform(validation_X)
    
    counts = train_y.value_counts().to_dict()
    total_sum = sum(list(counts.values()))
    for label in counts.keys():
        counts[label] = (total_sum - counts[label]) / total_sum
    weights = [counts[label] for label in range(6)]
    weights = torch.tensor(weights, dtype = torch.float32).to(device)
    
    train_X_tensor, train_y_tensor = torch.tensor(train_X, dtype = torch.float32).to(device), torch.tensor(train_y.values.reshape(-1, 1), dtype = torch.long).view(-1).to(device)
    validation_X_tensor, validation_y_tensor = torch.tensor(validation_X, dtype = torch.float32).to(device), torch.tensor(validation_y.values.reshape(-1, 1), dtype = torch.long).view(-1).to(device)

    model = myModel().to(device)
    loss = myLoss(weights)
    optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.01, weight_decay = 1e-5)
    
    num_epochs = 1000
    bestEpochsNum, bestEpochs = 10, []
    for epoch in range(1, num_epochs + 1):

        train_noisy_probs = model(train_X_tensor)

        optimizer.zero_grad()
        train_loss = loss(train_noisy_probs, train_y_tensor, model.noise_matrix)
        train_loss.backward()
        optimizer.step()

        train_noisy_labels = torch.argmax(train_noisy_probs, dim = 1).cpu().numpy()
        train_accuracy = accuracy_score(train_y, train_noisy_labels)
        train_f1 = f1_score(train_y, train_noisy_labels, average = 'weighted')

        with torch.no_grad():
            validation_noisy_probs = model(validation_X_tensor)

        validation_loss = loss(validation_noisy_probs, validation_y_tensor, model.noise_matrix)
        validation_noisy_labels = torch.argmax(validation_noisy_probs, dim = 1).cpu().numpy()
        validation_accuracy = accuracy_score(validation_y, validation_noisy_labels)
        validation_f1 = f1_score(validation_y, validation_noisy_labels, average = 'weighted')

        epoch_info = (train_accuracy, train_f1, train_loss.item(), validation_accuracy, validation_f1, validation_loss.item(), epoch, model.state_dict())

        if len(bestEpochs) >= bestEpochsNum:
            heapq.heappushpop(bestEpochs, epoch_info)
        else:
            heapq.heappush(bestEpochs, epoch_info)

        if epoch % logging_frequency == 0:
            logging.info(f"Epoch: {epoch}/{num_epochs}")
            logging.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}")
            logging.info(f"Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.4f}, Validation F1: {validation_f1:.4f}")
    
    parent = './checkpoints/E/'
    for filename in os.listdir(parent):
        if filename.endswith(".pth"):
            file_path = os.path.join(parent, filename)
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error deleting {file_path}: {e}")
    
    for epoch in bestEpochs:
        name = f'model_E_epoch_{epoch[6]}.pth'
        torch.save({
            'epoch':epoch[6],
            'train_loss':epoch[2],
            'train_accuracy':epoch[0],
            'train_f1':epoch[1],
            'test_loss':epoch[5],
            'test_accuracy':epoch[3],
            'test_f1':epoch[4],
            'model_state_dict': epoch[7]
        }, parent + name)
    
    bestEpochs.sort(reverse = True)
    best_model = myModel().to(device)
    best_model.load_state_dict(bestEpochs[0][7])
    
    torch.save(best_model.state_dict(), './checkpoints/E/prediction_model.pth')
    with open('./checkpoints/E/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return