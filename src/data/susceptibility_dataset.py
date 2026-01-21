"""
PyTorch Dataset for susceptibility classification training.

Loads pre-processed training data from NPZ files containing:
- Transect point features and metadata
- Wave and atmospheric environmental features
- Erosion mode labels (5-class)
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class SusceptibilityDataset(Dataset):
    """
    Dataset for cliff erosion susceptibility classification.

    Loads training samples from NPZ file with structure:
        - point_features: (N, T, 128, 7) transect point features
        - metadata: (N, T, 12) transect metadata per epoch
        - distances: (N, T, 128) distance from cliff toe
        - context_mask: (N, T) valid epoch mask
        - wave_features: (N, 360, 4) wave time series
        - wave_doy: (N, 360) wave day-of-year
        - atmos_features: (N, 90, 24) atmospheric time series
        - atmos_doy: (N, 90) atmospheric day-of-year
        - event_class: (N,) erosion mode labels [0-4]
        - risk_index: (N,) derived risk scores

    Args:
        data_path: Path to NPZ file with training data
        transform: Optional transform to apply to samples
        subset_fraction: Optional fraction of data to use (for debugging)
        seed: Random seed for subset selection
    """

    # Class names for reference
    CLASS_NAMES = ['stable', 'beach_erosion', 'toe_erosion', 'small_rockfall', 'large_failure']

    def __init__(
        self,
        data_path: Union[str, Path],
        transform: Optional[callable] = None,
        subset_fraction: Optional[float] = None,
        seed: int = 42,
    ):
        self.data_path = Path(data_path)
        self.transform = transform

        # Load data
        data = np.load(self.data_path, allow_pickle=True)

        # Core features
        self.point_features = data['point_features']  # (N, T, 128, 7)
        self.metadata = data['metadata']  # (N, T, 12)
        self.distances = data['distances']  # (N, T, 128)
        self.context_mask = data['context_mask']  # (N, T)

        # Environmental features
        self.wave_features = data['wave_features']  # (N, 360, 4)
        self.wave_doy = data['wave_doy']  # (N, 360)
        self.atmos_features = data['atmos_features']  # (N, 90, 24)
        self.atmos_doy = data['atmos_doy']  # (N, 90)

        # Labels
        self.event_class = data['event_class']  # (N,) 0-4
        self.risk_index = data['risk_index']  # (N,)

        # Optional metadata (for analysis)
        self.transect_idx = data.get('transect_idx', None)
        self.target_epoch = data.get('target_epoch', None)
        self.mop_id = data.get('mop_id', None)
        self.beach = data.get('beach', None)
        self.label_source = data.get('label_source', None)
        self.confidence = data.get('confidence', None)

        # Apply subset if requested
        self.indices = np.arange(len(self.event_class))
        if subset_fraction is not None and subset_fraction < 1.0:
            rng = np.random.RandomState(seed)
            n_samples = int(len(self.indices) * subset_fraction)
            self.indices = rng.choice(self.indices, size=n_samples, replace=False)
            self.indices.sort()

        # Compute class statistics
        self._compute_class_stats()

    def _compute_class_stats(self):
        """Compute class distribution statistics."""
        labels = self.event_class[self.indices]
        self.class_counts = np.bincount(labels, minlength=5)
        self.class_weights = 1.0 / (self.class_counts + 1e-6)
        self.class_weights = self.class_weights / self.class_weights.sum() * 5

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - point_features: (T, 128, 7)
                - metadata: (T, 12)
                - distances: (T, 128)
                - context_mask: (T,)
                - wave_features: (360, 4)
                - wave_doy: (360,)
                - atmos_features: (90, 24)
                - atmos_doy: (90,)
                - event_class: scalar
                - risk_index: scalar
        """
        real_idx = self.indices[idx]

        sample = {
            'point_features': torch.from_numpy(self.point_features[real_idx]).float(),
            'metadata': torch.from_numpy(self.metadata[real_idx]).float(),
            'distances': torch.from_numpy(self.distances[real_idx]).float(),
            'context_mask': torch.from_numpy(self.context_mask[real_idx]).bool(),
            'wave_features': torch.from_numpy(self.wave_features[real_idx]).float(),
            'wave_doy': torch.from_numpy(self.wave_doy[real_idx]).long(),
            'atmos_features': torch.from_numpy(self.atmos_features[real_idx]).float(),
            'atmos_doy': torch.from_numpy(self.atmos_doy[real_idx]).long(),
            'event_class': torch.tensor(self.event_class[real_idx], dtype=torch.long),
            'risk_index': torch.tensor(self.risk_index[real_idx], dtype=torch.float32),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def get_class_weights(self) -> torch.Tensor:
        """Get inverse frequency class weights for loss function."""
        return torch.from_numpy(self.class_weights).float()

    def get_sample_weights(self) -> torch.Tensor:
        """Get per-sample weights for weighted random sampler."""
        labels = self.event_class[self.indices]
        weights = self.class_weights[labels]
        return torch.from_numpy(weights).float()


def create_data_loaders(
    train_path: Union[str, Path],
    val_path: Optional[Union[str, Path]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
    subset_fraction: Optional[float] = None,
) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
    """
    Create training and validation data loaders.

    Args:
        train_path: Path to training data NPZ
        val_path: Optional path to validation data NPZ
        batch_size: Batch size for data loaders
        num_workers: Number of data loading workers
        use_weighted_sampler: Whether to use weighted random sampling
        subset_fraction: Optional fraction of data to use

    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader, WeightedRandomSampler

    # Create training dataset
    train_dataset = SusceptibilityDataset(
        train_path, subset_fraction=subset_fraction
    )

    # Create sampler
    if use_weighted_sampler:
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    # Create validation loader
    val_loader = None
    if val_path is not None:
        val_dataset = SusceptibilityDataset(val_path)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader
