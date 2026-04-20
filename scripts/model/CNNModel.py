import torch
import torch.nn as nn


class SimpleCNN1D(nn.Module):
    def __init__(self, num_outputs=10):  # 修改为 10 个输出
        super(SimpleCNN1D, self).__init__()

        # 卷积层部分保持不变，利用 stride 和 pooling 快速降维
        self.conv_layers = nn.Sequential(
            # 第一层：快速压缩长度 524288 -> 16*131073
            nn.Conv1d(1, 16, kernel_size=64, stride=4, padding=32),
            nn.BatchNorm1d(16),     # 16*131073 -> 16*131073 （批量归一化）
            nn.ReLU(),  # 16*131073 -> 16*131073
            nn.MaxPool1d(4, 4),   # 16*131073 -> 16*32768

            # 第二层：16*32768 -> 32*16385
            nn.Conv1d(16, 32, kernel_size=16, stride=2, padding=8),
            nn.BatchNorm1d(32), # 32*16385 -> 32*16385
            nn.ReLU(),  # 32*16385 -> 32*216385
            nn.MaxPool1d(4, 4),  # 32*16385 -> 32*4096

            # 第三层：32*4096 -> 64 * 2049
            nn.Conv1d(32, 64, kernel_size=8, stride=2, padding=4),
            nn.BatchNorm1d(64),   # 32*4096 -> 64 * 2049
            nn.ReLU(),    # 32*4096 -> 64 * 2049
            nn.MaxPool1d(4, 4)   # 64 * 512
        )

        # 自动计算经过卷积层self.conv_layers 展平后的维度：也就是上面的64*512=32768
        self.feature_dim = self._get_final_flattened_size(1, 524288)

        # 全连接层：最终输出 11 个值
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),   # 防止过拟合，将50%的神经元输出变为0
            nn.Linear(256, num_outputs)  # 输出为 10
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