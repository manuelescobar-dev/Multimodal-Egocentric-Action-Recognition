import torch
import torch.nn as nn
import torch
import torch.nn.init as init


class MidlevelActionNet_P(nn.Module):
    def __init__(self, num_classes, modality, model_config, **kwargs):
        super().__init__()
        self.model_config = model_config
        self.num_classes = num_classes

        # For the EMG modality
        self.lstm = nn.LSTM(16, 50, batch_first=True)

        self.dropout = nn.Dropout(self.model_config.dropout)  # Dropout layer

        # Adaptive average pooling for the RGB modality 1024 -> 50
        self.avgpool = nn.AdaptiveAvgPool1d(50)

        self.fc1 = nn.Linear(
            100,
            self.num_classes,
        )
        
        # Initialize weights
        self.init_weights()

    def init_weights(self):
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                init.xavier_uniform_(param)

        # Initialize FC weights
        for name, param in self.fc1.named_parameters():
            if "weight" in name:
                init.xavier_uniform_(param)

        # for name, param in self.fc2.named_parameters():
        #     if "weight" in name:
        #         init.xavier_uniform_(param)

    def forward(self, x: dict):
        """
        x: dictionary containing the input data of rgb and emg modalities
        """
        x_emg = x["EMG"]
        x_rgb = x["RGB"]

        assert x_emg.shape == (x_emg.size(0), 100, 16)
        assert x_rgb.shape == (x_rgb.size(0), 1024)
        # x = self.avgpool(x)  # Average pooling
        (h0, c0) = (
            torch.zeros(1, x_emg.size(0), 50).to(
                x_emg.device
            ),
            torch.zeros(1, x_emg.size(0), 50).to(
                x_emg.device
            ),
        )
        x_emg, _ = self.lstm(x_emg, (h0, c0))
        x_emg = x_emg[:, -1, :]  # x_emg: (batch_size, 50)
        # Concatenate the EMG and RGB features
        x_rgb= self.avgpool(x_rgb) # x_rgb: (batch_size, 50)
        x = torch.cat((x_emg, x_rgb), dim=1)  # (batch_size, 50 + 50)
        x = self.dropout(x)
        x = self.fc1(x)
        return x, {}
        

    def get_augmentation(self, modality):
        return None, None
