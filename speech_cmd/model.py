import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    def __init__(self, n_maps: int):
        super().__init__()
        self.conv1 = nn.Conv2d(n_maps, n_maps, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(n_maps)
        self.conv2 = nn.Conv2d(n_maps, n_maps, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(n_maps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + x)


class Res15(nn.Module):
    """
    ResNet-15 keyword spotter.
    Input : (B, 1, 40, 101) log-mel spectrogram
    Output: (B, num_classes) logits
    ~239 K parameters with default n_maps=45.
    """

    def __init__(self, num_classes: int = 26, n_maps: int = 45):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, n_maps, 3, padding=1, bias=False),
            nn.BatchNorm2d(n_maps),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[_ResBlock(n_maps) for _ in range(6)])
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.head   = nn.Linear(n_maps, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.blocks(self.stem(x))).flatten(1))
