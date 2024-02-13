from torch import nn


class MLP_Classifier(nn.Module):
    def __init__(self, num_classes, model_config, **kwargs):
        super().__init__()
        """
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """
        self.model_config = model_config
        self.avg_pool = nn.AvgPool1d(kernel_size=1, stride=1)
        self.dropout = nn.Dropout(self.model_config.dropout)
        self.linear = nn.Linear(1024, num_classes)
        self.classifier = nn.Sequential(
            self.avg_pool,
            self.dropout,
            self.linear
        )

    def forward(self, x):
        return self.classifier(x), {}
