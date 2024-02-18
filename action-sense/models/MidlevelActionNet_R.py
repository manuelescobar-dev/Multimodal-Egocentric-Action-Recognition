import torch
import torch.nn as nn
import torch
import torch.nn.init as init


class MidlevelActionNet_R(nn.Module):
    def __init__(self, num_classes, modality, model_config, **kwargs):
        super().__init__()
        self.model_config = model_config
        self.num_classes = num_classes

        # For the EMG modality
        self.lstm = nn.LSTM(16, self.model_config.emg_feature_size, batch_first=True)

        self.dropout1 = nn.Dropout(self.model_config.dropout)  # Dropout layer
        self.dropout2 = nn.Dropout(self.model_config.dropout)  # Dropout layer

        # Concatenate the EMG and RGB features
        self.fc1 = nn.Linear(
            self.model_config.emg_feature_size + self.model_config.rgb_feature_size,
            self.model_config.hidden_size,
        )

        self.fc2 = nn.Linear(
            self.model_config.hidden_size,
            self.num_classes,
        )

        self.classifier = nn.Sequential(
            self.dropout1,
            self.fc1,
            nn.ReLU(),
            self.dropout2,
            self.fc2,
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

        for name, param in self.fc2.named_parameters():
            if "weight" in name:
                init.xavier_uniform_(param)

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
            torch.zeros(1, x_emg.size(0), self.model_config.emg_feature_size).to(
                x_emg.device
            ),
            torch.zeros(1, x_emg.size(0), self.model_config.emg_feature_size).to(
                x_emg.device
            ),
        )
        x_emg, _ = self.lstm(x_emg, (h0, c0))
        x_emg = x_emg[:, -1, :]  # x_emg: (batch_size, 1024)
        # Concatenate the EMG and RGB features
        x = torch.cat((x_emg, x_rgb), dim=1)  # (batch_size, 1024 + 1024)
        return self.classifier(x), {}

    def get_augmentation(self, modality):
        return None, None
