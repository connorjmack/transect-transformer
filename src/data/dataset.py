"""
PyTorch Dataset for CliffCast training.

Loads pre-processed training data from NPZ files and prepares batches
for training with proper collation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class CliffCastDataset(Dataset):
    """
    PyTorch Dataset for CliffCast multi-task learning.

    Loads data from NPZ files containing aligned transects, wave data,
    atmospheric data, and labels. Supports train/val/test splitting.

    Args:
        data_path: Path to training_data.npz file
        split: One of 'train', 'val', or 'test'
        split_ratios: Tuple of (train, val, test) ratios (default (0.7, 0.15, 0.15))
        seed: Random seed for splitting (default 42)
        temporal_split: If True, use temporal split instead of random (default False)
    """

    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
        temporal_split: bool = False,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.split = split
        self.split_ratios = split_ratios
        self.seed = seed
        self.temporal_split = temporal_split

        # Validate split
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

        # Validate split ratios
        if not np.isclose(sum(split_ratios), 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratios)}")

        # Load data
        self.data = self._load_data()

        # Split indices
        self.indices = self._get_split_indices()

    def _load_data(self) -> Dict[str, np.ndarray]:
        """Load data from NPZ file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        data = np.load(self.data_path, allow_pickle=True)

        # Convert to dict
        return {key: data[key] for key in data.files}

    def _get_split_indices(self) -> np.ndarray:
        """Get indices for current split."""
        n_samples = len(self.data['risk_index'])

        if self.temporal_split:
            # Temporal split: sort by target_epoch, split chronologically
            target_epochs = self.data['target_epoch']
            indices = np.argsort(target_epochs)
        else:
            # Random split
            rng = np.random.RandomState(self.seed)
            indices = np.arange(n_samples)
            rng.shuffle(indices)

        # Split
        train_ratio, val_ratio, _ = self.split_ratios
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        if self.split == 'train':
            return indices[:n_train]
        elif self.split == 'val':
            return indices[n_train:n_train + n_val]
        else:  # test
            return indices[n_train + n_val:]

    def __len__(self) -> int:
        """Return number of samples in split."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get sample by index.

        Returns:
            Dictionary containing:
                Inputs:
                - point_features: (T, N, F) where T=10, N=128, F=12
                - metadata: (T, 12)
                - distances: (T, N)
                - context_mask: (T,) boolean mask for valid context epochs
                - wave_features: (T_w, 4) where T_w=360
                - wave_doy: (T_w,) day of year for wave data
                - atmos_features: (T_a, 24) where T_a=90
                - atmos_doy: (T_a,) day of year for atmos data

                Labels:
                - risk_index: scalar [0, 1]
                - total_volume: scalar (m³)
                - event_class: scalar {0, 1, 2, 3}
                - collapse_labels: (4,) binary labels per horizon
                - confidence: scalar [0, 1]
                - label_source: scalar {0=derived, 1=observed}
        """
        # Map to actual data index
        data_idx = self.indices[idx]

        # Extract sample
        sample = {
            # Transect inputs
            'point_features': torch.from_numpy(
                self.data['point_features'][data_idx].astype(np.float32)
            ),
            'metadata': torch.from_numpy(
                self.data['metadata'][data_idx].astype(np.float32)
            ),
            'distances': torch.from_numpy(
                self.data['distances'][data_idx].astype(np.float32)
            ),
            'context_mask': torch.from_numpy(
                self.data['context_mask'][data_idx].astype(np.bool_)
            ),
            # Wave inputs
            'wave_features': torch.from_numpy(
                self.data['wave_features'][data_idx].astype(np.float32)
            ),
            'wave_doy': torch.from_numpy(
                self.data['wave_doy'][data_idx].astype(np.int64)
            ),
            # Atmospheric inputs
            'atmos_features': torch.from_numpy(
                self.data['atmos_features'][data_idx].astype(np.float32)
            ),
            'atmos_doy': torch.from_numpy(
                self.data['atmos_doy'][data_idx].astype(np.int64)
            ),
            # Labels
            'risk_index': torch.tensor(
                self.data['risk_index'][data_idx], dtype=torch.float32
            ),
            'total_volume': torch.tensor(
                self.data['total_volume'][data_idx], dtype=torch.float32
            ),
            'event_class': torch.tensor(
                self.data['event_class'][data_idx], dtype=torch.long
            ),
            'collapse_labels': torch.from_numpy(
                self.data['collapse_labels'][data_idx].astype(np.float32)
            ),
            'confidence': torch.tensor(
                self.data['confidence'][data_idx], dtype=torch.float32
            ),
            'label_source': torch.tensor(
                self.data['label_source'][data_idx], dtype=torch.long
            ),
        }

        return sample

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for handling imbalanced event classes.

        Returns:
            Tensor of shape (4,) with inverse frequency weights
        """
        event_classes = self.data['event_class'][self.indices]
        class_counts = np.bincount(event_classes, minlength=4)
        # Avoid division by zero
        class_counts = np.maximum(class_counts, 1)
        weights = 1.0 / class_counts
        # Normalize so weights sum to num_classes
        weights = weights * len(weights) / weights.sum()
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> torch.Tensor:
        """
        Compute per-sample weights based on event class and label source.

        Observed events get higher weight than derived labels.

        Returns:
            Tensor of shape (n_samples,) with sample weights
        """
        event_classes = self.data['event_class'][self.indices]
        label_sources = self.data['label_source'][self.indices]

        # Base weight from class frequency
        class_counts = np.bincount(event_classes, minlength=4)
        class_counts = np.maximum(class_counts, 1)
        class_weights = 1.0 / class_counts

        weights = class_weights[event_classes]

        # Boost observed events by 1.5x
        weights[label_sources == 1] *= 1.5

        return torch.tensor(weights, dtype=torch.float32)


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching samples.

    Args:
        batch: List of samples from CliffCastDataset

    Returns:
        Batched dictionary with tensors
    """
    batch_dict = {}
    keys = batch[0].keys()

    for key in keys:
        tensors = [sample[key] for sample in batch]
        batch_dict[key] = torch.stack(tensors)

    return batch_dict


def create_dataloaders(
    data_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    temporal_split: bool = False,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train/val/test dataloaders.

    Args:
        data_path: Path to training_data.npz
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        split_ratios: Train/val/test ratios
        seed: Random seed
        temporal_split: Use temporal instead of random split

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = CliffCastDataset(
        data_path, split='train', split_ratios=split_ratios,
        seed=seed, temporal_split=temporal_split
    )
    val_dataset = CliffCastDataset(
        data_path, split='val', split_ratios=split_ratios,
        seed=seed, temporal_split=temporal_split
    )
    test_dataset = CliffCastDataset(
        data_path, split='test', split_ratios=split_ratios,
        seed=seed, temporal_split=temporal_split
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
