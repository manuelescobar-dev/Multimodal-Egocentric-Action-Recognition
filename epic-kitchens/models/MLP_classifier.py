from torch import nn


class MLP_Classifier(nn.Module):
    def __init__(self, num_classes, model_config, **kwargs):
        super().__init__()
        """
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """
        self.model_config = model_config
        self.dropout = nn.Dropout(self.model_config.dropout)
        self.linear = nn.Linear(1024, num_classes)
        self.classifier = nn.Sequential(self.dropout, self.linear)

    def forward(self, x):
        assert x.shape == (x.size(0), 1024)
        return self.classifier(x), {}
