import torch
import torch.nn as nn


class NatureDQN(nn.Module):
    """Nature DQN CNN: 3 conv layers + 2 FC layers."""

    def __init__(self, n_actions: int, in_channels: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # 84x84 input → (84-8)/4+1=20 → (20-4)/2+1=9 → (9-3)/1+1=7 → 64*7*7=3136
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: uint8 [0,255] → float32 [0,1]
        x = x.float() / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
