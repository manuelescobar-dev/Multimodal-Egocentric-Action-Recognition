from torch import nn


class MLP_Classifier(nn.Module):
    def __init__(self, num_classes, m, model_config, **kwargs):
        super().__init__()
        self.model_config = model_config
        self.dropout = nn.Dropout(self.model_config.dropout)
        self.linear = nn.Linear(1024, num_classes)
        self.classifier = nn.Sequential(self.dropout, self.linear)

    def forward(self, x):
        assert x.shape == (x.size(0), 1024)
        return self.classifier(x), {}

    def get_augmentation(self, modality):
        return None, None
