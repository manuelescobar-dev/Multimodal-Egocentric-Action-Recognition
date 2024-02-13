from torch import nn
from models.I3D import InceptionI3d


class OriginalClassifier(nn.Module):
    def __init__(self, num_classes, model_config, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.model_config = model_config

        self.dropout = nn.Dropout(self.model_config.dropout)
        self.logits = InceptionI3d.Unit3D(
            in_channels=1024,
            output_channels=self.num_classes,
            kernel_shape=[1, 1, 1],
            padding=0,
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
            name="classifier",
        )
        InceptionI3d.truncated_normal_(self.logits.conv3d.weight, std=1 / 32)

    def forward(self, x):
        x = self.dropout(x).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        logits = self.logits(x).squeeze(3).squeeze(3).squeeze(2)
        return logits, {}
