import torch
import torch.nn as nn


class SimpleCNN1D(nn.Module):
    def __init__(self, num_outputs=10, in_channels=1, input_length=524288):
        super(SimpleCNN1D, self).__init__()

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
            nn.MaxPool1d(4, 4)
        )

        self.feature_dim = self._get_final_flattened_size(in_channels, input_length)

        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_outputs)
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
