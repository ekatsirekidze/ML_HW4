from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.constants import IMG_SIZE, NUM_PIXELS


def parse_pixels(pixel_str: str) -> np.ndarray:
    """Convert space-separated pixel string to (48, 48) float32 array in [0, 1]."""
    values = np.fromstring(pixel_str, sep=" ", dtype=np.float32)
    if values.size != NUM_PIXELS:
        raise ValueError(f"Expected {NUM_PIXELS} pixels, got {values.size}")
    return (values.reshape(IMG_SIZE, IMG_SIZE) / 255.0).astype(np.float32)


def load_train_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "emotion" not in df.columns or "pixels" not in df.columns:
        raise ValueError("train.csv must contain columns: emotion, pixels")
    return df


def load_test_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "pixels" not in df.columns:
        raise ValueError("test.csv must contain column: pixels")
    return df


def get_train_transforms(augment: bool = True) -> transforms.Compose:
    if augment:
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


def get_eval_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


class FERDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: Optional[Callable] = None,
        has_labels: bool = True,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.has_labels = has_labels

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = parse_pixels(row["pixels"])
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).unsqueeze(0)

        if self.has_labels:
            label = int(row["emotion"])
            return image, label
        return image


def create_dataloaders(
    train_csv: str,
    batch_size: int = 64,
    val_ratio: float = 0.2,
    seed: int = 42,
    use_augmentation: bool = True,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, pd.DataFrame, pd.DataFrame]:
    df = load_train_dataframe(train_csv)
    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=seed,
        stratify=df["emotion"],
    )

    train_ds = FERDataset(train_df, transform=get_train_transforms(use_augmentation), has_labels=True)
    val_ds = FERDataset(val_df, transform=get_eval_transforms(), has_labels=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_df, val_df


def create_test_loader(
    test_csv: str,
    batch_size: int = 64,
    num_workers: int = 2,
) -> DataLoader:
    df = load_test_dataframe(test_csv)
    test_ds = FERDataset(df, transform=get_eval_transforms(), has_labels=False)
    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
