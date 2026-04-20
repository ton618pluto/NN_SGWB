import torch
import torch.nn as nn

try:
    from nflows.distributions import StandardNormal
    from nflows.flows import Flow
    from nflows.transforms import (
        CompositeTransform,
        MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
        RandomPermutation,
    )
    NFLOWS_IMPORT_ERROR = None
except ModuleNotFoundError as error:
    StandardNormal = None
    Flow = None
    CompositeTransform = None
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform = None
    RandomPermutation = None
    NFLOWS_IMPORT_ERROR = error


class OldSimpleCNN1D(nn.Module):
    # 这里直接对齐 origin_model/model/CNNModel.py。
    def __init__(self, num_outputs=10, in_channels=1, input_length=524288):
        super(OldSimpleCNN1D, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=64, stride=4, padding=32),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4, 4),
            nn.Conv1d(16, 32, kernel_size=16, stride=2, padding=8),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4, 4),
            nn.Conv1d(32, 64, kernel_size=8, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4, 4),
        )

        self.feature_dim = self._get_final_flattened_size(in_channels, input_length)

        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_outputs),
        )

    def _get_final_flattened_size(self, channels, length):
        with torch.no_grad():
            x = torch.zeros(1, channels, length)
            x = self.conv_layers(x)
            return x.numel()

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class OldGWFlowModel(nn.Module):
    # 这里直接对齐 origin_model/model/GWFlowModel.py。
    def __init__(self, param_dim=10, context_dim=512, in_channels=1):
        super().__init__()
        if NFLOWS_IMPORT_ERROR is not None:
            raise ModuleNotFoundError(
                "OldGWFlowModel requires the 'nflows' package."
            ) from NFLOWS_IMPORT_ERROR

        self.embedding_net = OldSimpleCNN1D(
            num_outputs=context_dim,
            in_channels=in_channels,
        )

        num_layers = 8
        transforms = []
        for _ in range(num_layers):
            transforms.append(RandomPermutation(features=param_dim))
            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=param_dim,
                    hidden_features=256,
                    context_features=context_dim,
                    num_blocks=2,
                    tails="linear",
                    tail_bound=3.0,
                    num_bins=8,
                )
            )

        self.transform = CompositeTransform(transforms)
        self.distribution = StandardNormal((param_dim,))
        self.flow = Flow(self.transform, self.distribution)

    def forward(self, theta, x):
        context = self.embedding_net(x)
        return -self.flow.log_prob(inputs=theta, context=context).mean()

    def sample(self, x, num_samples=1000):
        context = self.embedding_net(x)
        return self.flow.sample(num_samples, context=context)
