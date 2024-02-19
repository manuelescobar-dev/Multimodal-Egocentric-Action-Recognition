import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import math
import torch
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
import torch.nn.init as init


class ActionNet(nn.Module):
    def __init__(self, num_classes, modality, model_config, **kwargs):
        super().__init__()
        self.model_config = model_config
        self.num_classes = num_classes

        self.lstm = nn.LSTM(
            16, self.model_config.hidden_size, batch_first=True
        )  # LSTM layer
        self.dropout = nn.Dropout(self.model_config.dropout)  # Dropout layer
        self.fc = nn.Linear(
            self.model_config.hidden_size, self.num_classes
        )  # Fully connected layer

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                init.xavier_uniform_(param)

        # Initialize FC layer weights
        init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        # x = self.avgpool(x)  # Average pooling
        assert x.shape == (x.size(0), 100, 16)
        (h0, c0) = (
            torch.zeros(1, x.size(0), self.model_config.hidden_size).to(x.device),
            torch.zeros(1, x.size(0), self.model_config.hidden_size).to(x.device),
        )
        x, _ = self.lstm(x, (h0, c0)) # x: (batch_size, seq_length, hidden_size)
        x = x[:, -1, :]  # Get the last output for the sequence
        x = self.dropout(x) # Dropout
        x = self.fc(x) # Fully connected layer
        return x, {}

    def get_augmentation(self, modality):
        return None, None
