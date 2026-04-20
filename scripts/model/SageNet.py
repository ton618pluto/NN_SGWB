import torch.nn as nn
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # 参数编码器
        self.encoder = nn.Sequential(
            nn.Linear(9, 128),  # 9->128
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 256),  # 128->256
            nn.GELU(),
            nn.LayerNorm(256)
        )

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=3,
            bidirectional=False,
            batch_first=True
        )

        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        # encode param [B,9] -> [B,256]
        encoded = self.encoder(x)

        # expand sequence [B,256] -> [B,256,256]
        repeated = encoded.unsqueeze(1).repeat(1, 256, 1)

        # LSTM [B,256,256] -> [B,256,256]
        lstm_out, _ = self.lstm(repeated)

        # decoded [B,256,512] -> [B,256,2]
        return self.decoder(lstm_out)