from torch import nn


class MLP_Classifier_2(nn.Module):
    def __init__(self, num_classes, m, model_config, **kwargs):
        super().__init__()
        self.model_config = model_config
        self.dropout = nn.Dropout(self.model_config.dropout) # Dropout layer
        self.linear = nn.Linear(1024, 512) # Linear layer
        self.relu = nn.ReLU() # ReLU activation function
        self.linear2 = nn.Linear(512, num_classes) # Linear layer for classification
        self.classifier = nn.Sequential(
            self.dropout,
            self.linear,
            self.relu,
            self.linear2
        )

    def forward(self, x):
        return self.classifier(x), {}
    
    def get_augmentation(self, modality):
        return None, None