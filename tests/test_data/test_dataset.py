"""Tests for CliffCastDataset."""

import pytest
import torch
import tempfile
from pathlib import Path

from src.data.dataset import CliffCastDataset, collate_fn
from src.data.synthetic import SyntheticDataGenerator


class TestCliffCastDataset:
    """Test CliffCastDataset."""

    @pytest.fixture
    def synthetic_data_path(self):
        """Create synthetic data file for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "synthetic.npz"

            # Generate small dataset
            gen = SyntheticDataGenerator(
                n_samples=20, n_timesteps=3, n_points=128
            )
            gen.save_dataset(str(output_path))

            yield str(output_path)

    def test_dataset_initialization(self, synthetic_data_path):
        """Test dataset initializes correctly."""
        dataset = CliffCastDataset(synthetic_data_path, split='train')

        assert dataset.split == 'train'
        assert len(dataset) > 0

    def test_dataset_splits(self, synthetic_data_path):
        """Test train/val/test splits."""
        train_dataset = CliffCastDataset(synthetic_data_path, split='train')
        val_dataset = CliffCastDataset(synthetic_data_path, split='val')
        test_dataset = CliffCastDataset(synthetic_data_path, split='test')

        # Splits should have different sizes
        assert len(train_dataset) > len(val_dataset)
        assert len(train_dataset) > len(test_dataset)

        # Total should equal original
        total = len(train_dataset) + len(val_dataset) + len(test_dataset)
        assert total == 20

    def test_getitem_returns_dict(self, synthetic_data_path):
        """Test __getitem__ returns dictionary."""
        dataset = CliffCastDataset(synthetic_data_path, split='train')

        sample = dataset[0]

        assert isinstance(sample, dict)
        assert 'point_features' in sample
        assert 'wave_features' in sample
        assert 'risk_index' in sample

    def test_sample_shapes(self, synthetic_data_path):
        """Test sample has correct tensor shapes."""
        dataset = CliffCastDataset(synthetic_data_path, split='train')

        sample = dataset[0]

        T, N = 3, 128
        T_w, T_a = 360, 90

        assert sample['point_features'].shape == (T, N, 12)
        assert sample['metadata'].shape == (T, 12)
        assert sample['distances'].shape == (T, N)
        assert sample['wave_features'].shape == (T_w, 4)
        assert sample['wave_doy'].shape == (T_w,)
        assert sample['atmos_features'].shape == (T_a, 24)
        assert sample['atmos_doy'].shape == (T_a,)
        assert sample['risk_index'].ndim == 0  # Scalar
        assert sample['retreat_m'].ndim == 0  # Scalar
        assert sample['collapse_labels'].shape == (4,)
        assert sample['failure_mode'].ndim == 0  # Scalar

    def test_invalid_split_raises(self, synthetic_data_path):
        """Test invalid split raises error."""
        with pytest.raises(ValueError):
            CliffCastDataset(synthetic_data_path, split='invalid')

    def test_missing_file_raises(self):
        """Test missing file raises error."""
        with pytest.raises(FileNotFoundError):
            CliffCastDataset('nonexistent.npz', split='train')


class TestCollateFunction:
    """Test collate function for batching."""

    @pytest.fixture
    def synthetic_data_path(self):
        """Create synthetic data file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "synthetic.npz"
            gen = SyntheticDataGenerator(n_samples=10)
            gen.save_dataset(str(output_path))
            yield str(output_path)

    def test_collate_creates_batch(self, synthetic_data_path):
        """Test collate creates proper batch."""
        dataset = CliffCastDataset(synthetic_data_path, split='train')

        batch_list = [dataset[i] for i in range(4)]
        batch = collate_fn(batch_list)

        assert isinstance(batch, dict)
        assert 'point_features' in batch

    def test_collate_batch_dimension(self, synthetic_data_path):
        """Test collate adds batch dimension."""
        dataset = CliffCastDataset(synthetic_data_path, split='train')

        batch_size = 4
        batch_list = [dataset[i] for i in range(batch_size)]
        batch = collate_fn(batch_list)

        assert batch['point_features'].shape[0] == batch_size
        assert batch['wave_features'].shape[0] == batch_size
        assert batch['risk_index'].shape[0] == batch_size
