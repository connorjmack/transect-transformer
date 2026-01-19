"""
PyTorch Dataset for CliffCast training.

Loads transect cubes, wave data, and atmospheric data, and prepares batches
for training with proper collation and padding.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class CliffCastDataset(Dataset):
    """
    PyTorch Dataset for CliffCast multi-task learning.

    Loads data from NPZ files containing transects, wave data, atmospheric data,
    and targets. Supports train/val/test splitting.

    Args:
        data_path: Path to NPZ file or directory containing data
        split: One of 'train', 'val', or 'test'
        split_ratios: Tuple of (train, val, test) ratios (default (0.7, 0.15, 0.15))
        seed: Random seed for splitting (default 42)
    """

    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.split = split
        self.split_ratios = split_ratios
        self.seed = seed

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

        # Create shuffled indices
        rng = np.random.RandomState(self.seed)
        indices = np.arange(n_samples)
        rng.shuffle(indices)

        # Split
        train_ratio, val_ratio, test_ratio = self.split_ratios
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
                - point_features: (T, N, 12)
                - metadata: (T, 12)
                - distances: (T, N)
                - wave_features: (T_w, 4)
                - wave_doy: (T_w,)
                - atmos_features: (T_a, 24)
                - atmos_doy: (T_a,)
                - risk_index: scalar
                - retreat_m: scalar
                - collapse_labels: (4,)
                - failure_mode: scalar (long)
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
            # Targets
            'risk_index': torch.tensor(
                self.data['risk_index'][data_idx], dtype=torch.float32
            ),
            'retreat_m': torch.tensor(
                self.data['retreat_m'][data_idx], dtype=torch.float32
            ),
            'collapse_labels': torch.from_numpy(
                self.data['collapse_labels'][data_idx].astype(np.float32)
            ),
            'failure_mode': torch.tensor(
                self.data['failure_mode'][data_idx], dtype=torch.long
            ),
        }

        return sample


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching samples.

    Handles different sequence lengths by padding if needed.

    Args:
        batch: List of samples from CliffCastDataset

    Returns:
        Batched dictionary with tensors
    """
    # Stack all samples
    batch_dict = {}

    # Get first sample to check keys
    keys = batch[0].keys()

    for key in keys:
        tensors = [sample[key] for sample in batch]

        if key in ['risk_index', 'retreat_m', 'failure_mode']:
            # Scalars: stack into (B,)
            batch_dict[key] = torch.stack(tensors)
        else:
            # Sequences/arrays: stack into batch dimension
            batch_dict[key] = torch.stack(tensors)

    return batch_dict
