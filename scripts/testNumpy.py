import torch
import torch.nn as nn

# 定义卷积层
conv = nn.Conv1d(1, 16, kernel_size=64, stride=4, padding=32)

# 创建输入张量 (Batch, Channel, Length)
input_tensor = torch.randn(1, 1, 524288)

# 计算输出
output = conv(input_tensor)
print(output.shape) # 输出: torch.Size([1, 16, 131073])

output = output.view(output.size(0), -1)
print(output.numel())
print(output.shape)