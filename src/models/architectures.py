from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from src.constants import IMG_SIZE, NUM_CLASSES, NUM_PIXELS


class MLPUnderfit(nn.Module):
    """Small fully-connected network — expected to underfit (limited capacity)."""

    def __init__(self, hidden_dims: List[int] | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [128]
        layers: List[nn.Module] = []
        in_dim = NUM_PIXELS
        for hidden in hidden_dims:
            layers.extend([nn.Linear(in_dim, hidden), nn.ReLU(inplace=True)])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, NUM_CLASSES))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = False) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ]
        if use_bn:
            layers.insert(1, nn.BatchNorm2d(out_ch))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNSmall(nn.Module):
    """2 conv blocks — baseline CNN."""

    def __init__(
        self,
        channels: List[int] | None = None,
        fc_dim: int = 128,
        dropout: float = 0.25,
        use_bn: bool = False,
    ) -> None:
        super().__init__()
        channels = channels or [16, 32]
        convs: List[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            convs.append(_ConvBlock(in_ch, out_ch, use_bn=use_bn))
            in_ch = out_ch
        self.features = nn.Sequential(*convs)
        # 48 -> 24 -> 12
        flat_dim = channels[-1] * (IMG_SIZE // (2 ** len(channels))) ** 2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class CNNMedium(nn.Module):
    """3 conv blocks — stronger feature extractor."""

    def __init__(
        self,
        channels: List[int] | None = None,
        fc_dim: int = 256,
        dropout: float = 0.3,
        use_bn: bool = False,
    ) -> None:
        super().__init__()
        channels = channels or [32, 64, 128]
        convs: List[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            convs.append(_ConvBlock(in_ch, out_ch, use_bn=use_bn))
            in_ch = out_ch
        self.features = nn.Sequential(*convs)
        flat_dim = channels[-1] * (IMG_SIZE // (2 ** len(channels))) ** 2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class CNNDeepOverfit(nn.Module):
    """Deep CNN with weak regularization — intended to overfit."""

    def __init__(
        self,
        channels: List[int] | None = None,
        fc_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        channels = channels or [64, 128, 256, 512]
        convs: List[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            convs.append(_ConvBlock(in_ch, out_ch, use_bn=False))
            in_ch = out_ch
        self.features = nn.Sequential(*convs)
        flat_dim = channels[-1] * (IMG_SIZE // (2 ** len(channels))) ** 2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class CNNBest(nn.Module):
    """Best architecture: BatchNorm + deeper CNN + strong dropout."""

    def __init__(
        self,
        channels: List[int] | None = None,
        fc_dim: int = 512,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        channels = channels or [32, 64, 128, 256]
        convs: List[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            convs.append(_ConvBlock(in_ch, out_ch, use_bn=True))
            in_ch = out_ch
        self.features = nn.Sequential(*convs)
        flat_dim = channels[-1] * (IMG_SIZE // (2 ** len(channels))) ** 2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, fc_dim),
            nn.BatchNorm1d(fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


MODEL_REGISTRY = {
    "mlp_underfit": MLPUnderfit,
    "cnn_small": CNNSmall,
    "cnn_medium": CNNMedium,
    "cnn_deep_overfit": CNNDeepOverfit,
    "cnn_best": CNNBest,
}


def build_model(model_name: str, **kwargs: Any) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_name](**kwargs)
