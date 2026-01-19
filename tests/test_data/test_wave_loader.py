"""Unit tests for wave_loader module.

Tests the WaveLoader and WaveDataset classes that integrate CDIP wave data
into the CliffCast training pipeline.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.data.wave_loader import WaveLoader, WaveDataset
from src.data.cdip_wave_loader import WaveData


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_wave_data():
    """Create synthetic WaveData object."""
    n_hours = 365 * 24  # One year of hourly data
    start_date = datetime(2023, 1, 1)

    time = np.array([
        np.datetime64(start_date + timedelta(hours=i))
        for i in range(n_hours)
    ])

    # Generate realistic wave data
    np.random.seed(42)
    hs = np.abs(np.random.normal(1.5, 0.5, n_hours)).astype(np.float64)
    tp = np.abs(np.random.normal(10, 2, n_hours)).astype(np.float64)
    dp = np.random.uniform(0, 360, n_hours).astype(np.float64)
    ta = np.abs(np.random.normal(8, 1.5, n_hours)).astype(np.float64)

    # Compute wave power
    power = 0.49 * hs**2 * tp

    return WaveData(
        time=time,
        hs=hs,
        tp=tp,
        dp=dp,
        ta=ta,
        power=power,
        sxy=None,
        sxx=None,
        latitude=32.8,
        longitude=-117.2,
        water_depth=10.0,
        mop_id=582,
    )


@pytest.fixture
def mock_cdip_dir(tmp_path, mock_wave_data):
    """Create temporary directory with mock NetCDF files."""
    cdip_dir = tmp_path / "cdip"
    cdip_dir.mkdir()

    # Create mock NetCDF files for a few MOPs
    for mop_id in [582, 583, 584]:
        file_path = cdip_dir / f"D{mop_id:04d}_hindcast.nc"

        # Create minimal NetCDF file structure
        ds = xr.Dataset(
            {
                'waveTime': (['time'], np.arange(len(mock_wave_data.time))),
                'waveHs': (['time'], mock_wave_data.hs),
                'waveTp': (['time'], mock_wave_data.tp),
                'waveDp': (['time'], mock_wave_data.dp),
                'waveTa': (['time'], mock_wave_data.ta),
                'metaLatitude': mock_wave_data.latitude,
                'metaLongitude': mock_wave_data.longitude,
                'metaWaterDepth': mock_wave_data.water_depth,
            }
        )

        ds.to_netcdf(file_path)

    return cdip_dir


@pytest.fixture
def mock_cube_arrays():
    """Create synthetic cube arrays for WaveDataset testing."""
    n_transects = 10
    n_epochs = 5

    transect_ids = np.array([582, 583, 584, 585, 586, 587, 588, 589, 590, 591])

    # Create timestamps spanning 2023
    base_date = datetime(2023, 3, 1)
    timestamps = np.zeros((n_transects, n_epochs), dtype=int)

    for i in range(n_transects):
        for j in range(n_epochs):
            date = base_date + timedelta(days=j * 90)  # Quarterly scans
            timestamps[i, j] = date.toordinal()

    return transect_ids, timestamps


# ============================================================================
# Test WaveLoader Initialization
# ============================================================================


class TestWaveLoaderInit:
    """Test WaveLoader initialization."""

    def test_init_with_valid_directory(self, mock_cdip_dir):
        """Test initialization with valid directory."""
        loader = WaveLoader(mock_cdip_dir)

        assert loader.cdip_dir == mock_cdip_dir
        assert loader.lookback_days == 90
        assert loader.resample_hours == 6
        assert loader.n_features == 4
        assert len(loader._cache) == 0
        assert loader._cache_size == 50

    def test_init_with_custom_parameters(self, mock_cdip_dir):
        """Test initialization with custom parameters."""
        loader = WaveLoader(
            mock_cdip_dir,
            lookback_days=60,
            resample_hours=3,
            cache_size=100,
        )

        assert loader.lookback_days == 60
        assert loader.resample_hours == 3
        assert loader._cache_size == 100

    def test_init_with_nonexistent_directory(self):
        """Test initialization with nonexistent directory raises error."""
        with pytest.raises(FileNotFoundError, match="CDIP data directory not found"):
            WaveLoader("/nonexistent/path")

    def test_scan_available_mops(self, mock_cdip_dir):
        """Test scanning for available MOPs."""
        loader = WaveLoader(mock_cdip_dir)

        assert loader.available_mops == [582, 583, 584]

    def test_empty_directory_warning(self, tmp_path, caplog):
        """Test warning when directory is empty."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        loader = WaveLoader(empty_dir)

        assert "No CDIP data files found" in caplog.text
        assert loader.available_mops == []


# ============================================================================
# Test WaveLoader Methods
# ============================================================================


class TestWaveLoaderMethods:
    """Test WaveLoader data loading methods."""

    @patch('src.data.wave_loader.CDIPWaveLoader')
    def test_get_wave_for_scan_datetime(self, mock_cdip_class, mock_cdip_dir, mock_wave_data):
        """Test get_wave_for_scan with datetime input."""
        # Mock CDIPWaveLoader.load_mop to return our mock_wave_data
        mock_cdip_instance = MagicMock()
        mock_cdip_instance.load_mop.return_value = mock_wave_data
        mock_cdip_class.return_value = mock_cdip_instance

        loader = WaveLoader(mock_cdip_dir)

        # Get wave data for a scan date
        scan_date = datetime(2023, 12, 15)
        features, doy = loader.get_wave_for_scan(582, scan_date)

        # Check shapes
        expected_timesteps = int((90 * 24) / 6)  # 90 days at 6hr = 360
        assert features.shape == (expected_timesteps, 4)
        assert doy.shape == (expected_timesteps,)

        # Check feature types
        assert features.dtype == np.float32
        assert doy.dtype == np.int32

        # Check day-of-year values are in valid range
        assert np.all(doy >= 1) and np.all(doy <= 366)

    @patch('src.data.wave_loader.CDIPWaveLoader')
    def test_get_wave_for_scan_ordinal(self, mock_cdip_class, mock_cdip_dir, mock_wave_data):
        """Test get_wave_for_scan with ordinal date input."""
        mock_cdip_instance = MagicMock()
        mock_cdip_instance.load_mop.return_value = mock_wave_data
        mock_cdip_class.return_value = mock_cdip_instance

        loader = WaveLoader(mock_cdip_dir)

        # Use ordinal date
        scan_date = datetime(2023, 12, 15).toordinal()
        features, doy = loader.get_wave_for_scan(582, scan_date)

        assert features.shape[0] == 360
        assert doy.shape[0] == 360

    @patch('src.data.wave_loader.CDIPWaveLoader')
    def test_get_wave_for_scan_timestamp(self, mock_cdip_class, mock_cdip_dir, mock_wave_data):
        """Test get_wave_for_scan with pandas Timestamp input."""
        mock_cdip_instance = MagicMock()
        mock_cdip_instance.load_mop.return_value = mock_wave_data
        mock_cdip_class.return_value = mock_cdip_instance

        loader = WaveLoader(mock_cdip_dir)

        # Use pandas Timestamp
        scan_date = pd.Timestamp('2023-12-15')
        features, doy = loader.get_wave_for_scan(582, scan_date)

        assert features.shape[0] == 360
        assert doy.shape[0] == 360

    @patch('src.data.wave_loader.CDIPWaveLoader')
    def test_get_wave_for_scan_missing_data(self, mock_cdip_class, mock_cdip_dir, caplog):
        """Test graceful handling of missing data."""
        # Mock to raise FileNotFoundError
        mock_cdip_instance = MagicMock()
        mock_cdip_instance.load_mop.side_effect = FileNotFoundError("File not found")
        mock_cdip_class.return_value = mock_cdip_instance

        loader = WaveLoader(mock_cdip_dir)

        # Should return zeros instead of crashing
        scan_date = datetime(2023, 12, 15)
        features, doy = loader.get_wave_for_scan(999, scan_date)  # Non-existent MOP

        # Should get zeros
        assert features.shape == (360, 4)
        assert np.allclose(features, 0.0)
        assert "returning zeros" in caplog.text

    @patch('src.data.wave_loader.CDIPWaveLoader')
    def test_get_batch_wave(self, mock_cdip_class, mock_cdip_dir, mock_wave_data):
        """Test batch wave loading."""
        mock_cdip_instance = MagicMock()
        mock_cdip_instance.load_mop.return_value = mock_wave_data
        mock_cdip_class.return_value = mock_cdip_instance

        loader = WaveLoader(mock_cdip_dir)

        # Batch of 3 samples
        mop_ids = [582, 583, 584]
        scan_dates = [datetime(2023, 12, 15)] * 3

        features, doy = loader.get_batch_wave(mop_ids, scan_dates)

        # Check shapes
        assert features.shape == (3, 360, 4)
        assert doy.shape == (3, 360)

    def test_get_batch_wave_length_mismatch(self, mock_cdip_dir):
        """Test error when mop_ids and scan_dates lengths don't match."""
        loader = WaveLoader(mock_cdip_dir)

        mop_ids = [582, 583]
        scan_dates = [datetime(2023, 12, 15)]

        with pytest.raises(ValueError, match="Length mismatch"):
            loader.get_batch_wave(mop_ids, scan_dates)

    @patch('src.data.wave_loader.CDIPWaveLoader')
    def test_cache_functionality(self, mock_cdip_class, mock_cdip_dir, mock_wave_data):
        """Test that cache stores and retrieves WaveData objects."""
        mock_cdip_instance = MagicMock()
        mock_cdip_instance.load_mop.return_value = mock_wave_data
        mock_cdip_class.return_value = mock_cdip_instance

        loader = WaveLoader(mock_cdip_dir, cache_size=2)

        # First call - should load from file
        scan_date = datetime(2023, 12, 15)
        features1, _ = loader.get_wave_for_scan(582, scan_date)

        # Second call - should load from cache
        features2, _ = loader.get_wave_for_scan(582, scan_date)

        # Should be identical
        assert np.array_equal(features1, features2)

        # Check cache has the MOP
        assert 582 in loader._cache

    @patch('src.data.wave_loader.CDIPWaveLoader')
    def test_cache_lru_eviction(self, mock_cdip_class, mock_cdip_dir, mock_wave_data):
        """Test LRU cache eviction."""
        mock_cdip_instance = MagicMock()
        mock_cdip_instance.load_mop.return_value = mock_wave_data
        mock_cdip_class.return_value = mock_cdip_instance

        loader = WaveLoader(mock_cdip_dir, cache_size=2)

        scan_date = datetime(2023, 12, 15)

        # Load 3 MOPs with cache size 2
        loader.get_wave_for_scan(582, scan_date)
        loader.get_wave_for_scan(583, scan_date)
        loader.get_wave_for_scan(584, scan_date)  # Should evict 582

        # Cache should have only 2 items
        assert len(loader._cache) == 2
        # 582 should be evicted
        assert 582 not in loader._cache
        # 583 and 584 should remain
        assert 583 in loader._cache
        assert 584 in loader._cache

    @patch('src.data.wave_loader.CDIPWaveLoader')
    def test_validate_coverage(self, mock_cdip_class, mock_cdip_dir, mock_wave_data):
        """Test coverage validation."""
        mock_cdip_instance = MagicMock()
        mock_cdip_instance.load_mop.return_value = mock_wave_data
        mock_cdip_class.return_value = mock_cdip_instance

        loader = WaveLoader(mock_cdip_dir)

        # All dates within coverage
        mop_ids = [582, 583, 584]
        scan_dates = [datetime(2023, 6, 15)] * 3

        coverage = loader.validate_coverage(mop_ids, scan_dates)

        assert coverage['total'] == 3
        assert coverage['covered'] >= 0  # Depends on mock data dates
        assert 'coverage_pct' in coverage

    @patch('src.data.wave_loader.CDIPWaveLoader')
    def test_summary(self, mock_cdip_class, mock_cdip_dir, mock_wave_data):
        """Test summary method."""
        mock_cdip_instance = MagicMock()
        mock_cdip_instance.load_mop.return_value = mock_wave_data
        mock_cdip_class.return_value = mock_cdip_instance

        loader = WaveLoader(mock_cdip_dir)

        summary = loader.summary()

        # Should have entries for available MOPs
        assert len(summary) == len(loader.available_mops)

        # Check structure
        for mop_id in loader.available_mops:
            assert mop_id in summary
            assert 'n_records' in summary[mop_id]
            assert 'date_range' in summary[mop_id]
            assert 'latitude' in summary[mop_id]


# ============================================================================
# Test WaveDataset
# ============================================================================


class TestWaveDataset:
    """Test WaveDataset integration helper."""

    @patch('src.data.wave_loader.CDIPWaveLoader')
    def test_wave_dataset_init(self, mock_cdip_class, mock_cdip_dir, mock_cube_arrays, mock_wave_data):
        """Test WaveDataset initialization."""
        mock_cdip_instance = MagicMock()
        mock_cdip_instance.load_mop.return_value = mock_wave_data
        mock_cdip_class.return_value = mock_cdip_instance

        loader = WaveLoader(mock_cdip_dir)
        transect_ids, timestamps = mock_cube_arrays

        wave_ds = WaveDataset(loader, transect_ids, timestamps)

        assert wave_ds.loader is loader
        assert np.array_equal(wave_ds.transect_ids, transect_ids)
        assert np.array_equal(wave_ds.timestamps, timestamps)
        assert np.array_equal(wave_ds.mops, transect_ids)

    @patch('src.data.wave_loader.CDIPWaveLoader')
    def test_get_wave_for_sample(self, mock_cdip_class, mock_cdip_dir, mock_cube_arrays, mock_wave_data):
        """Test retrieving wave data for single sample."""
        mock_cdip_instance = MagicMock()
        mock_cdip_instance.load_mop.return_value = mock_wave_data
        mock_cdip_class.return_value = mock_cdip_instance

        loader = WaveLoader(mock_cdip_dir)
        transect_ids, timestamps = mock_cube_arrays

        wave_ds = WaveDataset(loader, transect_ids, timestamps)

        # Get wave data for first transect, third epoch
        features, doy = wave_ds.get_wave_for_sample(transect_idx=0, epoch_idx=2)

        assert features.shape == (360, 4)
        assert doy.shape == (360,)

    @patch('src.data.wave_loader.CDIPWaveLoader')
    def test_get_wave_for_indices(self, mock_cdip_class, mock_cdip_dir, mock_cube_arrays, mock_wave_data):
        """Test batch retrieval with index pairs."""
        mock_cdip_instance = MagicMock()
        mock_cdip_instance.load_mop.return_value = mock_wave_data
        mock_cdip_class.return_value = mock_cdip_instance

        loader = WaveLoader(mock_cdip_dir)
        transect_ids, timestamps = mock_cube_arrays

        wave_ds = WaveDataset(loader, transect_ids, timestamps)

        # Get batch of 3 samples
        indices = [(0, 0), (1, 1), (2, 2)]
        features, doy = wave_ds.get_wave_for_indices(indices)

        assert features.shape == (3, 360, 4)
        assert doy.shape == (3, 360)


# ============================================================================
# Integration Tests (with real CDIP data)
# ============================================================================


@pytest.mark.skipif(
    not Path("data/raw/cdip").exists(),
    reason="CDIP data directory not available",
)
class TestCDIPIntegration:
    """Integration tests with real CDIP data.

    These tests require actual downloaded CDIP NetCDF files.
    They will be skipped if the data directory doesn't exist.
    """

    def test_load_real_cdip_file(self):
        """Test loading a real CDIP NetCDF file."""
        cdip_dir = Path("data/raw/cdip")
        loader = WaveLoader(cdip_dir)

        # Skip if no files available
        if not loader.available_mops:
            pytest.skip("No CDIP data files available")

        # Use first available MOP
        mop_id = loader.available_mops[0]

        # Load wave data
        scan_date = datetime(2023, 6, 15)
        features, doy = loader.get_wave_for_scan(mop_id, scan_date)

        # Verify shapes
        assert features.shape == (360, 4)
        assert doy.shape == (360,)

        # Verify no NaN or inf
        assert not np.isnan(features).any()
        assert not np.isinf(features).any()

    def test_coverage_real_data(self):
        """Test coverage validation with real data."""
        cdip_dir = Path("data/raw/cdip")
        loader = WaveLoader(cdip_dir)

        if not loader.available_mops:
            pytest.skip("No CDIP data files available")

        # Check coverage for available MOPs
        mop_ids = loader.available_mops[:3]  # First 3 MOPs
        scan_dates = [datetime(2023, 6, 15)] * len(mop_ids)

        coverage = loader.validate_coverage(mop_ids, scan_dates)

        assert coverage['total'] == len(mop_ids)
        assert coverage['coverage_pct'] >= 0
        assert coverage['coverage_pct'] <= 100

    def test_summary_real_data(self):
        """Test summary with real data."""
        cdip_dir = Path("data/raw/cdip")
        loader = WaveLoader(cdip_dir)

        if not loader.available_mops:
            pytest.skip("No CDIP data files available")

        summary = loader.summary()

        assert len(summary) == len(loader.available_mops)

        # Check first MOP has valid summary
        first_mop = loader.available_mops[0]
        assert 'n_records' in summary[first_mop]
        assert 'date_range' in summary[first_mop]
        assert summary[first_mop]['n_records'] > 0
