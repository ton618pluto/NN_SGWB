import torch
import torch.nn as nn

try:
    # nflows 负责条件归一化流部分的联合后验建模。
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


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        kernel_size: int = 7,
        groups: int = 8,
    ) -> None:
        super().__init__()
        # 通过 dilation 扩大感受野，但仍保持时域卷积结构。
        padding = dilation * (kernel_size // 2)
        norm_groups = min(groups, out_channels)

        # 主分支：两层卷积 + 归一化 + 激活。
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(norm_groups, out_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(norm_groups, out_channels)
        self.act2 = nn.GELU()

        if stride != 1 or in_channels != out_channels:
            # 通道数或步长不一致时，用 1x1 卷积对残差分支对齐。
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(norm_groups, out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        return self.act2(x)


class TemporalEncoder1D(nn.Module):
    def __init__(self, in_channels: int = 1, context_dim: int = 256) -> None:
        super().__init__()
        # in_channels 决定输入是单通道还是双通道：
        # - in_channels = 1 -> 只输入 H1
        # - in_channels = 2 -> 输入 H1 + L1
        # stem 先做一次较温和的降采样，降低后续计算量。
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 24, kernel_size=15, stride=2, padding=7, bias=False),
            nn.GroupNorm(8, 24),
            nn.GELU(),
        )
        # 多层残差块逐步增加通道、压缩长度，并扩大有效感受野。
        self.blocks = nn.Sequential(
            ResidualConvBlock(24, 32, stride=2, dilation=1),
            ResidualConvBlock(32, 48, stride=2, dilation=1),
            ResidualConvBlock(48, 64, stride=2, dilation=2),
            ResidualConvBlock(64, 96, stride=2, dilation=2),
            ResidualConvBlock(96, 128, stride=2, dilation=4),
            ResidualConvBlock(128, 160, stride=2, dilation=4),
            ResidualConvBlock(160, 192, stride=2, dilation=8),
        )
        # 用全局池化替代大规模 flatten，减少参数量并保留全局统计信息。
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        # 最终投影到 flow 需要的 context 向量。
        self.projector = nn.Sequential(
            nn.Linear(384, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, context_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        avg_features = self.avg_pool(x).squeeze(-1)
        max_features = self.max_pool(x).squeeze(-1)
        # 拼接平均池化和最大池化的特征，提高表达能力。
        features = torch.cat([avg_features, max_features], dim=1)
        return self.projector(features)


class GWFlowModelV2(nn.Module):
    def __init__(
        self,
        param_dim: int = 10,
        in_channels: int = 1,
        context_dim: int = 256,
        flow_layers: int = 6,
        flow_hidden_features: int = 256,
        flow_bins: int = 8,
        flow_tail_bound: float = 3.0,
    ) -> None:
        super().__init__()
        if NFLOWS_IMPORT_ERROR is not None:
            raise ModuleNotFoundError(
                "GWFlowModelV2 requires the 'nflows' package. "
                "Install it in your training environment before running train_flow.py."
            ) from NFLOWS_IMPORT_ERROR
        # 先用时域编码器把原始波形映射成条件向量 context。
        # 这里的 in_channels 由数据集自动决定，因此可以无缝支持 H1 单通道或 H1+L1 双通道。
        self.encoder = TemporalEncoder1D(in_channels=in_channels, context_dim=context_dim)

        transforms = []
        for _ in range(flow_layers):
            # 先随机打乱参数顺序，再做自回归样条变换，增强参数耦合建模能力。
            transforms.append(RandomPermutation(features=param_dim))
            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=param_dim,
                    hidden_features=flow_hidden_features,
                    context_features=context_dim,
                    num_blocks=2,
                    use_residual_blocks=True,
                    tails="linear",
                    tail_bound=flow_tail_bound,
                    num_bins=flow_bins,
                    dropout_probability=0.0,
                )
            )

        # 标准高斯基分布 + 多层可逆变换，共同构成条件 flow。
        self.transform = CompositeTransform(transforms)
        self.base_distribution = StandardNormal((param_dim,))
        self.flow = Flow(self.transform, self.base_distribution)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def log_prob(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        context = self.encode(x)
        return self.flow.log_prob(inputs=theta, context=context)

    def forward(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # 训练时直接返回负对数似然，作为优化目标。
        return -self.log_prob(theta, x).mean()

    def sample(self, x: torch.Tensor, num_samples: int = 1000) -> torch.Tensor:
        # 推理时从 p(theta | x) 中采样，用来近似后验分布。
        context = self.encode(x)
        return self.flow.sample(num_samples=num_samples, context=context)
