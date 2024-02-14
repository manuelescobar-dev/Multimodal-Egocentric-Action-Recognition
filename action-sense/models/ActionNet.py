import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class ActionNet(nn.Module):
    def __init__(self, num_classes, model_config, **kwargs):
        super().__init__()
        self.model_config = model_config
        self.num_classes = num_classes

        #self.lstm = nn.LSTM(16, 5, batch_first=True)
        self.lstm = nn.LSTM(16, self.model_config.hidden_size, batch_first=True)
        self.dropout = nn.Dropout(self.model_config.dropout)  # Dropout layer
        self.fc = nn.Linear(
            self.model_config.hidden_size, self.num_classes
        )  # Fully connected layer

    def forward(self, x):
        # x = self.avgpool(x)  # Average pooling
        x, _ = self.lstm(x)
        # x: (batch_size, seq_length, hidden_size)
        x = x[:, -1, :]  # Get the last output for the sequence
        x = self.dropout(x)
        x = self.fc(x)
        # Softmax activation
        x = F.softmax(x, dim=1)
        return x, {}
        
        
        
        
