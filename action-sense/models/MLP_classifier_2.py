from torch import nn


class MLP_Classifier_2(nn.Module):
    def __init__(self, num_classes, m, model_config, **kwargs):
        super().__init__()
        self.model_config = model_config
        self.dropout = nn.Dropout(self.model_config.dropout)
        self.linear = nn.Linear(1024, 512)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(512, num_classes)
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