import torch

class myModel(torch.nn.Module):
    
    def __init__(self):
        super(myModel, self).__init__()
        self.classifier = torch.nn.ModuleList()
        self.classifier.append(torch.nn.Linear(in_features = 128, out_features = 32))
        self.classifier.append(torch.nn.Linear(in_features = 32, out_features = 8))
        self.classifier.append(torch.nn.Linear(in_features = 8, out_features = 6))
        self.noise_matrix = torch.nn.Parameter(torch.eye(6))
    
    def forward(self, X):
        for layer in self.classifier[:-1]:
            X = layer(X)
            X = X.tanh()
        X = self.classifier[-1](X)
        true_probs = torch.softmax(X, dim = 1)
        noisy_probs = torch.matmul(true_probs, self.noise_matrix)
        return noisy_probs
    
    def predict(self, X):
        for layer in self.classifier[:-1]:
            X = layer(X)
            X = X.tanh()
        X = self.classifier[-1](X)
        true_probs = torch.softmax(X, dim = 1)
        return true_probs

    
class myLoss(torch.nn.Module):
    def __init__(self, weights):
        super(myLoss, self).__init__()
        self.loss = torch.nn.NLLLoss(weight = weights)
        self.alpha = 1
    
    def forward(self, pred, true, noise_matrix):
        l1 = self.loss(pred, true)
        neg_penalty = torch.relu(-noise_matrix).sum() + torch.relu(1 - noise_matrix).sum()
        row_sums = torch.sum(noise_matrix, dim = 1)
        row_sum_penalty = torch.sum((row_sums - 1) ** 2)
        symmetric_penalty = torch.sum((noise_matrix - noise_matrix.t()) ** 2)
        total_loss = l1 + self.alpha * (neg_penalty + row_sum_penalty + symmetric_penalty)
        return total_loss