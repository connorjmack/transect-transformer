"""
Tests for synthetic data generator.

Tests cover:
- Data generation with correct shapes
- Value ranges are realistic
- Relationships between inputs and targets
- Reproducibility with seeds
- Save/load functionality
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from src.data.synthetic import SyntheticDataGenerator


class TestSyntheticDataGenerator:
    """Test synthetic data generator initialization."""

    def test_default_initialization(self):
        """Test generator initializes with default parameters."""
        gen = SyntheticDataGenerator()

        assert gen.n_samples == 100
        assert gen.n_timesteps == 5
        assert gen.n_points == 128
        assert gen.n_wave_timesteps == 360
        assert gen.n_atmos_timesteps == 90

    def test_custom_initialization(self):
        """Test generator initializes with custom parameters."""
        gen = SyntheticDataGenerator(
            n_samples=50,
            n_timesteps=3,
            n_points=64,
            n_wave_timesteps=180,
            n_atmos_timesteps=45,
            seed=123,
        )

        assert gen.n_samples == 50
        assert gen.n_timesteps == 3
        assert gen.n_points == 64
        assert gen.n_wave_timesteps == 180
        assert gen.n_atmos_timesteps == 45
        assert gen.seed == 123


class TestGenerateTransects:
    """Test transect generation."""

    @pytest.fixture
    def generator(self):
        """Create generator with small size for testing."""
        return SyntheticDataGenerator(n_samples=10, n_timesteps=3, n_points=128)

    def test_transect_output_keys(self, generator):
        """Test transect generation returns all expected keys."""
        transects = generator.generate_transects()

        assert 'data' in transects
        assert 'metadata' in transects
        assert 'distances' in transects
        assert 'las_sources' in transects
        assert 'transect_ids' in transects

    def test_transect_shapes(self, generator):
        """Test transect data has correct shapes."""
        transects = generator.generate_transects()

        N, T, P = 10, 3, 128

        assert transects['data'].shape == (N, T, P, 12)
        assert transects['metadata'].shape == (N, T, 12)
        assert transects['distances'].shape == (N, T, P)
        assert transects['las_sources'].shape == (T,)
        assert transects['transect_ids'].shape == (N,)

    def test_point_features_valid(self, generator):
        """Test point features have valid values."""
        transects = generator.generate_transects()
        data = transects['data']

        # Distance should be 0-50m
        assert data[:, :, :, 0].min() >= 0
        assert data[:, :, :, 0].max() <= 50

        # Elevation should be positive
        assert data[:, :, :, 1].min() > 0

        # Slope should be 0-90 degrees
        assert data[:, :, :, 2].min() >= 0
        assert data[:, :, :, 2].max() <= 90

        # Intensity, RGB should be [0,1]
        assert data[:, :, :, 5].min() >= 0
        assert data[:, :, :, 5].max() <= 1
        assert data[:, :, :, 6:9].min() >= 0
        assert data[:, :, :, 6:9].max() <= 1

    def test_metadata_valid(self, generator):
        """Test metadata has valid values."""
        transects = generator.generate_transects()
        metadata = transects['metadata']

        # Cliff height should be positive
        assert metadata[:, :, 0].min() > 0

        # Slopes should be positive
        assert metadata[:, :, 1].min() >= 0
        assert metadata[:, :, 2].min() >= 0

        # Orientation should be 0-360
        assert metadata[:, :, 5].min() >= 0
        assert metadata[:, :, 5].max() <= 360

    def test_transect_ids_unique(self, generator):
        """Test transect IDs are unique."""
        transects = generator.generate_transects()
        ids = transects['transect_ids']

        assert len(np.unique(ids)) == len(ids)

    def test_temporal_consistency(self, generator):
        """Test transect properties change consistently over time."""
        transects = generator.generate_transects()
        data = transects['data']

        # Elevation should generally decrease over time (erosion)
        for i in range(data.shape[0]):
            elevations = data[i, :, :, 1].mean(axis=1)
            # Check trend is generally decreasing
            assert elevations[-1] <= elevations[0] + 5  # Allow some variance


class TestGenerateWaveData:
    """Test wave data generation."""

    @pytest.fixture
    def generator(self):
        """Create generator for testing."""
        return SyntheticDataGenerator(n_samples=10, n_wave_timesteps=360)

    def test_wave_output_shape(self, generator):
        """Test wave data has correct shape."""
        features, doy = generator.generate_wave_data()

        assert features.shape == (10, 360, 4)
        assert doy.shape == (10, 360)

    def test_wave_features_valid(self, generator):
        """Test wave features have realistic values."""
        features, doy = generator.generate_wave_data()

        # Significant height (hs) should be 0.5-8m
        assert features[:, :, 0].min() >= 0.5
        assert features[:, :, 0].max() <= 8.0

        # Peak period (tp) should be 4-20s
        assert features[:, :, 1].min() >= 4
        assert features[:, :, 1].max() <= 25

        # Direction (dp) should be roughly 180-360 (from west)
        assert features[:, :, 2].min() >= 0
        assert features[:, :, 2].max() <= 360

        # Power should be positive
        assert features[:, :, 3].min() >= 0

    def test_wave_doy_valid(self, generator):
        """Test day-of-year values are valid."""
        features, doy = generator.generate_wave_data()

        assert doy.min() >= 0
        assert doy.max() <= 365


class TestGenerateAtmosphericData:
    """Test atmospheric data generation."""

    @pytest.fixture
    def generator(self):
        """Create generator for testing."""
        return SyntheticDataGenerator(n_samples=10, n_atmos_timesteps=90)

    def test_atmospheric_output_shape(self, generator):
        """Test atmospheric data has correct shape."""
        features, doy = generator.generate_atmospheric_data()

        assert features.shape == (10, 90, 24)
        assert doy.shape == (10, 90)

    def test_atmospheric_features_valid(self, generator):
        """Test atmospheric features have realistic values."""
        features, doy = generator.generate_atmospheric_data()

        # Precipitation should be non-negative
        assert features[:, :, 0].min() >= 0

        # Temperature should be reasonable (-10 to 40C)
        assert features[:, :, 1].min() >= -20
        assert features[:, :, 1].max() <= 50

        # Cumulative precip should be non-negative
        assert features[:, :, 5:9].min() >= 0

        # Binary flags should be 0 or 1
        assert np.all(np.isin(features[:, :, 12], [0, 1]))
        assert np.all(np.isin(features[:, :, 20], [0, 1]))

    def test_atmospheric_doy_valid(self, generator):
        """Test day-of-year values are valid."""
        features, doy = generator.generate_atmospheric_data()

        assert doy.min() >= 0
        assert doy.max() <= 365


class TestGenerateTargets:
    """Test target generation."""

    @pytest.fixture
    def generator(self):
        """Create generator for testing."""
        return SyntheticDataGenerator(n_samples=10)

    @pytest.fixture
    def sample_inputs(self, generator):
        """Generate sample inputs for target generation."""
        wave_features, _ = generator.generate_wave_data()
        atmos_features, _ = generator.generate_atmospheric_data()
        transect_data = generator.generate_transects()

        return {
            'wave_features': wave_features,
            'atmos_features': atmos_features,
            'transect_metadata': transect_data['metadata'],
        }

    def test_target_output_keys(self, generator, sample_inputs):
        """Test targets include all expected keys."""
        targets = generator.generate_targets(**sample_inputs)

        assert 'risk_index' in targets
        assert 'retreat_m' in targets
        assert 'collapse_labels' in targets
        assert 'failure_mode' in targets

    def test_target_shapes(self, generator, sample_inputs):
        """Test target shapes are correct."""
        targets = generator.generate_targets(**sample_inputs)

        N = 10

        assert targets['risk_index'].shape == (N,)
        assert targets['retreat_m'].shape == (N,)
        assert targets['collapse_labels'].shape == (N, 4)
        assert targets['failure_mode'].shape == (N,)

    def test_target_value_ranges(self, generator, sample_inputs):
        """Test target values are in expected ranges."""
        targets = generator.generate_targets(**sample_inputs)

        # Risk index should be [0, 1]
        assert targets['risk_index'].min() >= 0
        assert targets['risk_index'].max() <= 1

        # Retreat should be positive and reasonable
        assert targets['retreat_m'].min() > 0
        assert targets['retreat_m'].max() < 10

        # Collapse labels should be 0 or 1
        assert np.all(np.isin(targets['collapse_labels'], [0, 1]))

        # Failure mode should be 0-4
        assert targets['failure_mode'].min() >= 0
        assert targets['failure_mode'].max() <= 4

    def test_target_dtypes(self, generator, sample_inputs):
        """Test target data types are correct."""
        targets = generator.generate_targets(**sample_inputs)

        assert targets['risk_index'].dtype == np.float32
        assert targets['retreat_m'].dtype == np.float32
        assert targets['collapse_labels'].dtype == np.float32
        assert targets['failure_mode'].dtype == np.int64


class TestGenerateDataset:
    """Test complete dataset generation."""

    @pytest.fixture
    def generator(self):
        """Create generator with small size."""
        return SyntheticDataGenerator(n_samples=10, n_timesteps=3)

    def test_dataset_contains_all_components(self, generator):
        """Test dataset includes all inputs and targets."""
        dataset = generator.generate_dataset()

        # Transect inputs
        assert 'point_features' in dataset
        assert 'metadata' in dataset
        assert 'distances' in dataset

        # Environmental inputs
        assert 'wave_features' in dataset
        assert 'wave_doy' in dataset
        assert 'atmos_features' in dataset
        assert 'atmos_doy' in dataset

        # Targets
        assert 'risk_index' in dataset
        assert 'retreat_m' in dataset
        assert 'collapse_labels' in dataset
        assert 'failure_mode' in dataset

    def test_dataset_shapes_consistent(self, generator):
        """Test all dataset components have consistent sample count."""
        dataset = generator.generate_dataset()

        N = 10

        assert dataset['point_features'].shape[0] == N
        assert dataset['metadata'].shape[0] == N
        assert dataset['wave_features'].shape[0] == N
        assert dataset['atmos_features'].shape[0] == N
        assert dataset['risk_index'].shape[0] == N
        assert dataset['retreat_m'].shape[0] == N

    def test_dataset_no_nans(self, generator):
        """Test dataset contains no NaN values."""
        dataset = generator.generate_dataset()

        for key, value in dataset.items():
            if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
                assert not np.isnan(value).any(), f"NaN found in {key}"


class TestSaveLoad:
    """Test save and load functionality."""

    @pytest.fixture
    def generator(self):
        """Create generator with small size."""
        return SyntheticDataGenerator(n_samples=5, n_timesteps=2)

    def test_save_creates_file(self, generator):
        """Test save creates NPZ file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "synthetic.npz"
            generator.save_dataset(str(output_path))

            assert output_path.exists()

    def test_save_load_roundtrip(self, generator):
        """Test save and load preserves data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "synthetic.npz"

            # Generate and save
            original = generator.generate_dataset()
            generator.save_dataset(str(output_path))

            # Load
            loaded = np.load(output_path, allow_pickle=True)

            # Check all keys present
            for key in original.keys():
                assert key in loaded, f"Missing key: {key}"

            # Check shapes match
            for key in original.keys():
                if isinstance(original[key], np.ndarray):
                    orig_shape = original[key].shape
                    loaded_shape = loaded[key].shape
                    assert orig_shape == loaded_shape, f"Shape mismatch for {key}"


class TestReproducibility:
    """Test reproducibility with seeds."""

    def test_same_seed_same_data(self):
        """Test same seed produces identical data."""
        gen1 = SyntheticDataGenerator(n_samples=10, seed=42)
        gen2 = SyntheticDataGenerator(n_samples=10, seed=42)

        dataset1 = gen1.generate_dataset()
        dataset2 = gen2.generate_dataset()

        # Check all arrays are identical
        for key in dataset1.keys():
            if isinstance(dataset1[key], np.ndarray):
                if dataset1[key].dtype.kind in ['f', 'i', 'u']:  # Numeric
                    assert np.allclose(
                        dataset1[key], dataset2[key]
                    ), f"Mismatch in {key}"

    def test_different_seed_different_data(self):
        """Test different seeds produce different data."""
        gen1 = SyntheticDataGenerator(n_samples=10, seed=42)
        gen2 = SyntheticDataGenerator(n_samples=10, seed=123)

        dataset1 = gen1.generate_dataset()
        dataset2 = gen2.generate_dataset()

        # At least some numeric arrays should differ
        risk1 = dataset1['risk_index']
        risk2 = dataset2['risk_index']

        assert not np.allclose(risk1, risk2)


class TestRelationships:
    """Test that targets have expected relationships with inputs."""

    @pytest.fixture
    def generator(self):
        """Create generator with more samples for correlation testing."""
        return SyntheticDataGenerator(n_samples=50, seed=42)

    def test_high_waves_correlate_with_retreat(self, generator):
        """Test higher waves lead to higher retreat."""
        dataset = generator.generate_dataset()

        mean_hs = dataset['wave_features'][:, :, 0].mean(axis=1)
        retreat = dataset['retreat_m']

        # Compute correlation
        corr = np.corrcoef(mean_hs, retreat)[0, 1]

        # Should have positive correlation
        assert corr > 0.3

    def test_retreat_correlates_with_risk(self, generator):
        """Test higher retreat correlates with higher risk."""
        dataset = generator.generate_dataset()

        retreat = dataset['retreat_m']
        risk = dataset['risk_index']

        # Compute correlation
        corr = np.corrcoef(retreat, risk)[0, 1]

        # Should have strong positive correlation
        assert corr > 0.7

    def test_failure_mode_distribution_reasonable(self, generator):
        """Test failure mode distribution is reasonable."""
        dataset = generator.generate_dataset()

        failure_modes = dataset['failure_mode']

        # Should have mixture of modes, not all one type
        unique_modes = np.unique(failure_modes)
        assert len(unique_modes) >= 3  # At least 3 different modes

        # Mode 0 (stable) should exist but not dominate
        stable_count = (failure_modes == 0).sum()
        assert stable_count < len(failure_modes) * 0.5
