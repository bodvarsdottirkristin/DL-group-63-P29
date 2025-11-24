import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationRNN(nn.Module):
    """
    Bidirectional 5-layer GRU encoder + 3-layer MLP classifier head.

    Input: numeric AIS features for 30-minute trajectory windows.
    Output: logits over num_classes (clusters).
    """

    def __init__(
        self,
        input_size: int = 5,      # all numeric features per timestep
        hidden_size: int = 20,    
        num_layers: int = 5,
        num_classes: int = 50,
        bidirectional: bool = True,
        rnn_dropout: float = 0.1  # internal GRU dropout between layers
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        # ---- GRU encoder ----
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else rnn_dropout,
            bidirectional=bidirectional
        )

        # ---- Classification head (3-layer MLP) ----
        encoder_out_dim = hidden_size * (2 if bidirectional else 1)

        fc1_dim = num_classes // 4     # first: N/4
        fc2_dim = num_classes // 2     # second: N/2
        fc3_dim = num_classes          # third: N

        self.classifier = nn.Sequential(
            nn.Linear(encoder_out_dim, fc1_dim),
            nn.ReLU(),
            nn.Linear(fc1_dim, fc2_dim),
            nn.ReLU(),
            nn.Linear(fc2_dim, fc3_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, input_size) numeric features only
        """
        out, h_n = self.gru(x)

        if self.bidirectional:
            # last layer forward/backward
            h_forward = h_n[-2, :, :]   # (B, H)
            h_backward = h_n[-1, :, :]  # (B, H)
            h_last = torch.cat([h_forward, h_backward], dim=1)  # (B, 2H)
        else:
            h_last = h_n[-1, :, :]      # (B, H)

        logits = self.classifier(h_last)  # (B, num_classes)
        return logits
