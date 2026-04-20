import torch
import torch.nn as nn
from nflows.distributions import StandardNormal
from nflows.flows import Flow
from nflows.transforms import (
    CompositeTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
    RandomPermutation,
)

from .CNNModel import SimpleCNN1D


class GWFlowModel(nn.Module):
    def __init__(self, param_dim=10, context_dim=512, in_channels=1):
        super().__init__()

        self.embedding_net = SimpleCNN1D(
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
                    tails='linear',
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
