import numpy as np
import torch
import logging
from sklearn.metrics import accuracy_score, f1_score

from torch.nn import TransformerDecoderLayer, TransformerDecoder
from torch_geometric.nn import SAGPooling, global_mean_pool
from torch_geometric.nn.conv import GATConv
from torch_geometric.loader import DataLoader

from source.conv import GNN_Node
from source.loadData import myDataset

from kan import KANLayer

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

class myGNN(torch.nn.Module):
    
    def __init__(self, num_classes, num_layers, dim, dropout, residual):
        super(myGNN, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dim = dim
        self.gnn_node = GNN_Node(self.num_layers, self.dim, dropout, residual)
        self.pooler1 = SAGPooling(in_channels = self.dim, GNN = GATConv)
        self.pooler2 = global_mean_pool
        self.layer1 = KANLayer(self.dim, self.dim)
        self.layer2 = KANLayer(self.dim, self.dim//2)
        self.layer3 = KANLayer(self.dim//2, self.num_classes)
        transformer_layer = TransformerDecoderLayer(d_model = self.dim, nhead = 4)
        self.transformer_decoder = TransformerDecoder(transformer_layer, num_layers = num_layers)
        pass
    
    def forward(self, batched_data):
        node_embedding = self.gnn_node(batched_data)
        out = self.pooler1(x = node_embedding, edge_index = batched_data.edge_index, batch = batched_data.batch)
        graph_embedding = self.pooler2(out[0], out[3])
        x = self.layer1(graph_embedding)[0]
        x = self.transformer_decoder(x, x)
        x = self.transformer_decoder(x, x)
        hidden = self.layer2(x)[0]
        prediction = self.layer3(hidden)[0]
        return prediction

class myModel:
    
    def __init__(self, model, num_classes, device):
        self.num_classes = num_classes
        self.model = model.to(device)
        self.device = device
        self.criterion, self.optimizer = None, None
        self.transition_matrix = np.full((self.num_classes, self.num_classes), 0.1)
        np.fill_diagonal(self.transition_matrix, 0.5)
        self.bestEpochs = []
        pass
    
    def estimate_true_probs(self, preds, noisy_labels):
        num_samples = noisy_labels.shape[0]
        true_probs = np.zeros((num_samples, self.num_classes))
        for i in range(num_samples):
            for true_label in range(self.num_classes):
                likelihood = self.transition_matrix[true_label, noisy_labels[i]]
                prior = preds[i, true_label]
                numerator = likelihood * prior
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
    
    def predict_proba(self, loader):
        preds = []
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device) 
                output = self.model(data)
                output = torch.softmax(output, dim = 1)
                preds += output.cpu().numpy().tolist()
        return np.asarray(preds)
    
    def train(self, train_loader, num_epochs, one_hot):
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.01)
        bestEpochs = []
        for epoch in range(1, num_epochs + 1):
            total_loss = 0
            train_true_labels, train_pred_labels = [], []
            for data in train_loader:
                data = data.to(self.device) 
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, data.y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                train_pred_labels += np.argmax(output.cpu().detach().numpy(), axis = 1).tolist()
                if one_hot:
                    train_true_labels += np.argmax(data.y.cpu().detach().numpy(), axis = 1).tolist()
                else:
                    train_true_labels += data.y.cpu().detach().numpy().tolist()
            acc, f1 = accuracy_score(train_true_labels, train_pred_labels), f1_score(train_true_labels, train_pred_labels, average = 'weighted')
            bestEpochs.append((total_loss, 1 - acc, 1 - f1, self.model.state_dict()))
        bestEpochs.sort(key = lambda x : x[0])
        self.model = myGNN(num_classes = 6, num_layers = 2, dim = 128, dropout = 0.5, residual = True).to(self.device)
        self.model.load_state_dict(bestEpochs[0][3])
        return bestEpochs[0][0], 1 - bestEpochs[0][1], 1 - bestEpochs[0][2]
    
    def calculate_log_likelihood(self, loader, y):
        num_samples = y.shape[0]
        log_likelihood = 0
        preds = self.predict_proba(loader)
        for i in range(num_samples):
            likelihood_i = 0
            for true_label in range(self.num_classes):
                classifier_prob = preds[i, true_label]
                noise_prob = self.transition_matrix[true_label, y[i]]
                likelihood_i += np.exp(np.log(noise_prob) + np.log(classifier_prob))
            log_likelihood += np.log(likelihood_i)
        return log_likelihood
    
    def fit(self, train_graphs, num_epochs = 10, batch_size = 128, tol = 1e-4):
        
        logging.basicConfig(filename = './logs/E.log', filemode = 'w', level = logging.INFO, format = '%(asctime)s - %(message)s')
        logging_frequency = 1
        
        train_size = (4 * len(train_graphs)) // 5
        train_graphs, validation_graphs = train_graphs[:train_size], train_graphs[train_size:]
        
        train_y, validation_y = [], []
        for i in range(len(train_graphs)):
            train_y.append(train_graphs[i]['y'][0][0])
        for i in range(len(validation_graphs)):
            validation_y.append(validation_graphs[i]['y'][0][0])
        train_y, validation_y = np.asarray(train_y), np.asarray(validation_y)
        
        train_dataset = myDataset(train_graphs, weights = None, transform = add_zeros)
        train_loader = DataLoader(train_dataset, batch_size = batch_size)
        
        validation_dataset = myDataset(validation_graphs, weights = None, transform = add_zeros)
        validation_loader = DataLoader(validation_dataset, batch_size = batch_size)
        
        print("Initial Training on Noisy Labels for 100 epochs")
        train_loss, acc, f1 = self.train(train_loader, 100, False)
        print(f"Loss: {train_loss:.4f}, Accuracy: {acc:.4f}, F1 - Score: {f1:.4f}")
        
        print(f"EM Iteration begins for {num_epochs} iterations")
        
        for epoch in range(1, num_epochs + 1):
            
            # E - Step
            train_preds = self.predict_proba(train_loader)
            train_true_probs = self.estimate_true_probs(train_preds, train_y)
            
            # M - Step
            self.transition_matrix = self.update_transition_matrix(train_y, train_true_probs)
            train_dataset = myDataset(train_graphs, weights = train_true_probs, transform = add_zeros)
            train_loader = DataLoader(train_dataset, batch_size = batch_size)
            
            train_loss, acc, f1 = self.train(train_loader, 10, True)
            
            # Progress checking
            train_log_likelihood = self.calculate_log_likelihood(train_loader, train_y)
            validation_log_likelihood = self.calculate_log_likelihood(validation_loader, validation_y)
            
            epoch_info = (-validation_log_likelihood, -train_log_likelihood, epoch, self.model.state_dict())
            self.bestEpochs.append(epoch_info)
            
            if epoch % logging_frequency == 0:
                print(f"EM Iteration {epoch}/{num_epochs} completed..!!")
                logging.info(f"Epoch: {epoch}/{num_epochs}")
                logging.info(f"Train Log Likelihood: {train_log_likelihood:.4f}, Validation Log Likelihood: {validation_log_likelihood:.4f}")
        return