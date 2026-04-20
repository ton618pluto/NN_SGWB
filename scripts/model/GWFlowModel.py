import torch
import torch.nn as nn
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform, RandomPermutation, MaskedAffineAutoregressiveTransform,MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from .CNNModel import SimpleCNN1D


# 如果你想使用更强大的样条插值变换（推荐用于 PE）：
# from nflows.transforms import MaskedPiecewiseRationalQuadraticAutoregressiveTransform

class GWFlowModel(nn.Module):
    # param_dim为超参数的维度，
    def __init__(self, param_dim=10, context_dim=512):
        super().__init__()

        # 1. 特征提取网络 (复用之前的 CNN)
        # 确保 SimpleCNN1D 的 num_outputs 等于 context_dim

        self.embedding_net = SimpleCNN1D(num_outputs=context_dim)

        # 2. 构建流层
        num_layers = 8
        transforms = []
        for _ in range(num_layers):
            # 随机排列通道，确保参数之间充分耦合
            transforms.append(RandomPermutation(features=param_dim))

            # 使用标准的掩码仿射自回归变换
            transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=param_dim,
                hidden_features=256,
                context_features=context_dim,
                num_blocks=2,  # 内部残差块数量
                tails='linear',  # 尾部处理
                tail_bound=3.0,  # 边界范围
                num_bins=8  # 样条分箱数
            ))

        self.transform = CompositeTransform(transforms)
        self.distribution = StandardNormal((param_dim,))
        self.flow = Flow(self.transform, self.distribution)

    def forward(self, theta, x):
        """计算负对数似然 (NLL) 作为 Loss"""
        context = self.embedding_net(x)
        # log_prob 返回的是对数似然，取负号变成 Loss
        return -self.flow.log_prob(inputs=theta, context=context).mean()

    def sample(self, x, num_samples=1000):
        """推理阶段：给定波形，生成参数采样"""
        context = self.embedding_net(x)
        return self.flow.sample(num_samples, context=context)
