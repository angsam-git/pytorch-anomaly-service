import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [64, 32],
        latent_dim: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        enc_layers, prev = [], input_dim
        for h in hidden_dims:
            enc_layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                enc_layers.append(nn.Dropout(dropout))
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers, prev = [], latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                dec_layers.append(nn.Dropout(dropout))
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
