import numpy as np
import torch
import heapq
import logging
from sklearn.metrics import accuracy_score, f1_score

class classifier(torch.nn.Module):
    
    def __init__(self):
        super(classifier, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(in_features = 128, out_features = 64))
        self.layers.append(torch.nn.Linear(in_features = 64, out_features = 32))
        self.layers.append(torch.nn.Linear(in_features = 32, out_features = 16))
        self.layers.append(torch.nn.Linear(in_features = 16, out_features = 8))
        self.layers.append(torch.nn.Linear(in_features = 8, out_features = 6))
        #self.dropout = torch.nn.Dropout(0.1)
    
    def forward(self, X):
        for layer in self.layers[:-1]:
            X = layer(X)
            X = X.tanh()
            #X = self.dropout(X)
        true_probs = self.layers[-1](X)
        true_probs = torch.softmax(true_probs, dim = 1)
        return true_probs

class myModel:
    
    def __init__(self, model, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        self.model = model.to(self.device)
        self.transition_matrix = np.full((self.num_classes, self.num_classes), 0.1)
        np.fill_diagonal(self.transition_matrix, 0.5)
        self.bestEpochs = []
        self.loss, self.optimizer = None, None
        pass
    
    def estimate_true_probs(self, preds, noisy_labels):
        num_samples = noisy_labels.shape[0]
        true_probs = np.zeros((num_samples, self.num_classes))
        for i in range(num_samples):
            for true_label in range(self.num_classes):
                likelihood = self.transition_matrix[true_label, noisy_labels[i]]
                prior = preds[i, true_label]
                numerator = np.exp(np.log(likelihood) + np.log(prior))
                denominator = 0
                for other_true_label in range(self.num_classes):
                    denominator += np.exp(np.log(self.transition_matrix[other_true_label, noisy_labels[i]]) + np.log(preds[i, other_true_label]))
                if denominator > 0:
                    true_probs[i, true_label] = numerator / denominator
                else:
                    true_probs[i, true_label] = 1.0/self.num_classes
        return true_probs
    
    def update_transition_matrix(self, noisy_labels, true_probs):
        num_samples = true_probs.shape[0]
        transition_matrix = np.zeros((self.num_classes, self.num_classes))
        for i in range(num_samples):
            for true_label in range(self.num_classes):
                for noisy_label in range(self.num_classes):
                    transition_matrix[true_label, noisy_label] += true_probs[i, true_label] * (noisy_labels[i] == noisy_label)
        for true_label in range(self.num_classes):
            row_sum = np.sum(transition_matrix[true_label, :])
            if row_sum > 0:
                transition_matrix[true_label, :] /= row_sum
            else:
                transition_matrix[true_label, :] = 1.0 / self.num_classes
        return transition_matrix
    
    def train(self, train_X_tensor, soft_labels):
        train_X_tensor, soft_labels = train_X_tensor.to(self.device), torch.tensor(soft_labels, dtype = torch.float32).to(self.device)
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params = self.model.parameters(), lr = 0.01, weight_decay = 1e-5)
        self.model.train()
        num_epochs = 1000
        for _ in range(num_epochs):
            train_probs = self.model(train_X_tensor)
            self.optimizer.zero_grad()
            train_loss = self.loss(train_probs, soft_labels)
            train_loss.backward()
            self.optimizer.step()
        return train_loss.item()
    
    def predict(self, X_tensor):
        X_tensor = X_tensor.to(self.device)
        self.model.eval()
        with torch.no_grad():
            true_probs = self.model(X_tensor)
        true_probs = true_probs.cpu().numpy()
        return true_probs
    
    def calculate_log_likelihood(self, X_tensor, y):
        num_samples = X_tensor.shape[0]
        log_likelihood = 0
        preds = self.predict(X_tensor)
        for i in range(num_samples):
            likelihood_i = 0
            for true_label in range(self.num_classes):
                classifier_prob = preds[i, true_label]
                noise_prob = self.transition_matrix[true_label, y[i]]
                likelihood_i += np.exp(np.log(noise_prob) + np.log(classifier_prob))
            log_likelihood += np.log(likelihood_i)
        return log_likelihood
    
    def fit(self, train_X, train_y, test_X, test_y, num_epochs = 10, tol = 1e-4):
        
        num_train_samples = train_X.shape[0]
        train_X_tensor, test_X_tensor = torch.tensor(train_X, dtype = torch.float32), torch.tensor(test_X, dtype = torch.float32)
        
        logging.basicConfig(filename = './logs/E.log', filemode = 'w', level = logging.INFO, format = '%(asctime)s - %(message)s')
        logging_frequency = 1
        
        train_labels = np.eye(self.num_classes)[train_y.values]
        train_loss = self.train(train_X_tensor, train_labels)
        
        for epoch in range(1, num_epochs + 1):
            
            # E - Step
            train_preds = self.predict(train_X_tensor)
            train_true_probs = self.estimate_true_probs(train_preds, train_y.values)
            
            # M - Step
            new_transition_matrix = self.update_transition_matrix(train_y.values, train_true_probs)
            self.transition_matrix = new_transition_matrix
            train_loss = self.train(train_X_tensor, train_true_probs)

            test_preds = self.predict(test_X_tensor)
            test_labels = np.argmax(test_preds, axis = 1)
            test_true_probs = self.estimate_true_probs(test_preds, test_y.values)
            true_labels = np.argmax(test_true_probs, axis = 1)
            
            train_log_likelihood = self.calculate_log_likelihood(train_X_tensor, train_y.values)
            test_log_likelihood = self.calculate_log_likelihood(test_X_tensor, test_y.values)
            
            test_accuracy = accuracy_score(true_labels, test_labels)
            test_f1 = f1_score(true_labels, test_labels, average = 'weighted')
            
            epoch_info = (-test_log_likelihood, -train_log_likelihood, epoch, self.model.state_dict())
            self.bestEpochs.append(epoch_info)
            
            if epoch % logging_frequency == 0:
                #print(f"Train Log Likelihood: {train_log_likelihood}, Test Log Likelihood: {test_log_likelihood}")
                logging.info(f"Epoch: {epoch}/{num_epochs}")
                logging.info(f"Train Log Likelihood: {train_log_likelihood:.4f}, Test Log Likelihood: {test_log_likelihood:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")

        return