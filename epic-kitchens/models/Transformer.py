import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
import torch.utils.data as data
import math
import copy


class TransformerEncoder(nn.Module):
    def __init__(self, num_classes, model_config, **kwargs):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=5)
        self.dropout = nn.Dropout(model_config.dropout)
        self.fc1 = nn.Linear(1024, num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc1)

    def forward(self, x: torch.Tensor, mask = None, src_key_padding_mask = None) -> Tensor:
        output = self.transformer_encoder(x)
        output = self.classifier(output)
        return output, {}

