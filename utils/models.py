import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConnected(nn.Module):
    def __init__(self, input_size = 100, n_classes = 2, dropout = 0.2):
        super(FullyConnected, self).__init__()
        
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.bn4 = torch.nn.BatchNorm1d(128)
        self.bn5 = torch.nn.BatchNorm1d(128)
        self.bn6 = torch.nn.BatchNorm1d(64)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, n_classes)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.relu(self.bn5(self.fc5(x)))
        x = self.dropout(x)
        x = self.relu(self.bn6(self.fc6(x)))
        x = self.fc7(x)
        return x

class FullyConnected2(nn.Module):
    def __init__(self, input_size = 3341, n_classes = 2, dropout = 0.2):
        super(FullyConnected2, self).__init__()
        
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(64)
        self.bn4 = torch.nn.BatchNorm1d(16)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, n_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.dropout(self.relu(x))
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.dropout(self.relu(x))
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.dropout(self.relu(x))
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.dropout(self.relu(x))
        x = self.fc5(x)
        return x
    



class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2, n_classes=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, 128)  # hidden_size * 2 because it's bidirectional
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_classes)
        
        # Activation function and dropout
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x):
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Take the output of the last time step
        out = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.relu(self.fc1(out))
        out = self.dropout_layer(out)
        out = self.relu(self.fc2(out))
        out = self.dropout_layer(out)
        out = self.fc3(out)
        
        return out

