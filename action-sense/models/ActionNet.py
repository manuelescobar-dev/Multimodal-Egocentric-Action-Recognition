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


class ActionNet(nn.Module):
    def __init__(self, num_classes, modality, model_config, **kwargs):
        super().__init__()
        self.model_config = model_config
        self.num_classes = num_classes
        self.fs = self.model_config.fs
        self.cutoff = self.model_config.cutoff
        self.num_channels = self.model_config.num_channels

        # self.lstm = nn.LSTM(16, 5, batch_first=True)
        self.lstm = nn.LSTM(
            16, self.model_config.hidden_size, batch_first=True, dtype=torch.float32
        )  # LSTM layer
        self.dropout = nn.Dropout(self.model_config.dropout)  # Dropout layer
        self.fc = nn.Linear(
            self.model_config.hidden_size, self.num_classes
        )  # Fully connected layer

    def forward(self, x):
        # x = self.avgpool(x)  # Average pooling
        (h0, c0) = (
            torch.zeros(1, x.size(0), self.model_config.hidden_size).to(x.device),
            torch.zeros(1, x.size(0), self.model_config.hidden_size).to(x.device),
        )
        x, _ = self.lstm(x, (h0, c0))
        # x: (batch_size, seq_length, hidden_size)
        x = x[:, -1, :]  # Get the last output for the sequence
        x = self.dropout(x)
        x = self.fc(x)
        # Softmax activation
        x = F.softmax(x, dim=1)
        return x, {}

    def rectify_signal(self, data):
        return np.abs(data)

    def filter_signal(self, data):
        return self.low_pass_filter(data, self.fs, self.cutoff)

    def low_pass_filter(self, data, fs, cutoff, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        return filtfilt(b, a, data, padlen=12)

    def normalization(self, data):
        return 2 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1

    def preprocessing(self, data, steps):
        for step in steps:
            data.apply(step)

    def get_augmentation(self, modality):
        return self.compose, self.compose

    def compose(self, data):
        for i in range(self.num_channels):
            # Probably can be done in a more efficient way
            data[:, i] = self.rectify_signal(data[:, i])
            data[:, i] = self.filter_signal(data[:, i])
            # data[:, i] = self.normalization(data[:, i])
        return data
