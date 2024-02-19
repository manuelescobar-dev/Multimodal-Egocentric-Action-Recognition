import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM_Classifier(nn.Module):
    def __init__(self, num_classes, model_config, **kwargs):
        super().__init__()
        self.model_config = model_config
        self.num_classes = num_classes

        self.lstm = nn.LSTM(1024, self.model_config.hidden_size, batch_first=True)
        # Readout layer
        # self.avgpool = nn.AvgPool2d(7)  # Average pooling layer 7x7
        self.dropout = nn.Dropout(0.7)  # Dropout layer
        self.fc = nn.Linear(
            self.model_config.hidden_size, self.num_classes
        )  # Fully connected layer
        self.classifier = nn.Sequential(self.dropout, self.fc)

    def forward(self, x):
        # x = self.avgpool(x)  # Average pooling
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.classifier(x)
        return x, {}
