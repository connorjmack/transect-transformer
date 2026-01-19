"""Unit tests for atmospheric data loading and feature computation.

Tests cover:
- AtmosFeatureComputer: Feature computation from raw PRISM data
- AtmosphericLoader: Data loading and alignment to scan dates
- Integration: Full pipeline from raw data to model-ready tensors
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.atmos_features import (
    AtmosFeatureComputer,
    ATMOS_FEATURE_NAMES,
    compute_features_for_beach,
)
from src.data.atmos_loader import (
    AtmosphericLoader,
    AtmosphericDataset,
    get_beach_for_mop,
    BEACH_MOP_RANGES,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_raw_data():
    """Create sample raw PRISM-like data for testing."""
    dates = pd.date_range('2023-01-01', periods=180, freq='D')
    n = len(dates)

    rng = np.random.default_rng(42)

    # Simulate San Diego climate
    doy = dates.dayofyear
    temp_mean = 18 + 5 * np.sin(2 * np.pi * (doy - 200) / 365) + rng.normal(0, 2, n)
    temp_range = 8 + 2 * rng.random(n)

    df = pd.DataFrame({
        'date': dates,
        'precip_mm': rng.exponential(2, n) * (rng.random(n) < 0.3),
        'temp_mean_c': temp_mean,
        'temp_min_c': temp_mean - temp_range / 2,
        'temp_max_c': temp_mean + temp_range / 2,
        'dewpoint_c': temp_mean - temp_range / 2 - 5,
    })

    return df


@pytest.fixture
def feature_computer():
    """Create AtmosFeatureComputer instance."""
    return AtmosFeatureComputer()


@pytest.fixture
def temp_atmos_dir(sample_raw_data):
    """Create temporary directory with sample atmospheric data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Compute features for a test beach
        computer = AtmosFeatureComputer()
        features_df = computer.compute_all_features(sample_raw_data)

        # Save as parquet for 'delmar' beach
        features_df.to_parquet(tmpdir / 'delmar_atmos.parquet', index=False)

        yield tmpdir


@pytest.fixture
def atmos_loader(temp_atmos_dir):
    """Create AtmosphericLoader with test data."""
    return AtmosphericLoader(temp_atmos_dir, lookback_days=30)


# ============================================================================
# AtmosFeatureComputer Tests
# ============================================================================

class TestAtmosFeatureComputerInit:
    """Tests for AtmosFeatureComputer initialization."""

    def test_default_parameters(self):
        computer = AtmosFeatureComputer()
        assert computer.rain_threshold_mm == 1.0
        assert computer.api_decay == 0.9
        assert computer.water_year_start_month == 10

    def test_custom_parameters(self):
        computer = AtmosFeatureComputer(
            rain_threshold_mm=2.5,
            api_decay=0.85,
            water_year_start_month=9,
        )
        assert computer.rain_threshold_mm == 2.5
        assert computer.api_decay == 0.85
        assert computer.water_year_start_month == 9


class TestFeatureComputation:
    """Tests for individual feature computation methods."""

    def test_compute_api(self, feature_computer):
        """Test Antecedent Precipitation Index computation."""
        precip = np.array([10.0, 0.0, 0.0, 5.0, 0.0])
        api = feature_computer.compute_api(precip, k=0.9)

        # API_0 = 10
        # API_1 = 0 + 0.9 * 10 = 9
        # API_2 = 0 + 0.9 * 9 = 8.1
        # API_3 = 5 + 0.9 * 8.1 = 12.29
        # API_4 = 0 + 0.9 * 12.29 = 11.061

        assert api[0] == pytest.approx(10.0)
        assert api[1] == pytest.approx(9.0)
        assert api[2] == pytest.approx(8.1)
        assert api[3] == pytest.approx(12.29)
        assert api[4] == pytest.approx(11.061)

    def test_compute_vpd(self, feature_computer):
        """Test Vapor Pressure Deficit computation."""
        # At T=20°C, dewpoint=15°C, VPD should be positive
        vpd = feature_computer.compute_vpd(20.0, 15.0)
        assert vpd > 0
        assert vpd < 2.0  # Reasonable range for this temp difference

        # When T == dewpoint, VPD should be ~0
        vpd_saturated = feature_computer.compute_vpd(20.0, 20.0)
        assert vpd_saturated == pytest.approx(0.0, abs=0.01)

        # NaN inputs should return NaN
        assert np.isnan(feature_computer.compute_vpd(np.nan, 15.0))
        assert np.isnan(feature_computer.compute_vpd(20.0, np.nan))

    def test_classify_intensity(self, feature_computer):
        """Test precipitation intensity classification."""
        assert feature_computer._classify_intensity(0.0) == 0  # none
        assert feature_computer._classify_intensity(0.5) == 0  # none
        assert feature_computer._classify_intensity(1.0) == 1  # light
        assert feature_computer._classify_intensity(5.0) == 1  # light
        assert feature_computer._classify_intensity(10.0) == 2  # moderate
        assert feature_computer._classify_intensity(20.0) == 2  # moderate
        assert feature_computer._classify_intensity(25.0) == 3  # heavy
        assert feature_computer._classify_intensity(100.0) == 3  # heavy


class TestComputeAllFeatures:
    """Tests for full feature computation pipeline."""

    def test_output_shape(self, feature_computer, sample_raw_data):
        """Test that output has correct shape and columns."""
        features = feature_computer.compute_all_features(sample_raw_data)

        # Should have same number of rows as input
        assert len(features) == len(sample_raw_data)

        # Should have date column plus all feature columns
        assert 'date' in features.columns

        # Check all expected features are present
        for feat in ATMOS_FEATURE_NAMES:
            assert feat in features.columns, f"Missing feature: {feat}"

    def test_no_nan_in_required_features(self, feature_computer, sample_raw_data):
        """Test that computed features don't have unexpected NaN values."""
        features = feature_computer.compute_all_features(sample_raw_data)

        # These features should never be NaN (computed from non-NaN precip)
        no_nan_features = [
            'precip_mm', 'precip_7d', 'precip_30d', 'precip_60d', 'precip_90d',
            'api', 'days_since_rain', 'consecutive_dry_days',
            'rain_day_flag', 'intensity_class',
        ]

        for feat in no_nan_features:
            if feat in features.columns:
                nan_count = features[feat].isna().sum()
                assert nan_count == 0, f"Feature {feat} has {nan_count} NaN values"

    def test_cumulative_features_monotonic(self, feature_computer, sample_raw_data):
        """Test that cumulative features have expected relationships."""
        features = feature_computer.compute_all_features(sample_raw_data)

        # Longer windows should generally have >= values than shorter windows
        # (after initial ramp-up period)
        late_features = features.iloc[90:]  # After 90-day ramp-up

        # 30d >= 7d (when both are computed)
        assert (late_features['precip_30d'] >= late_features['precip_7d'] - 0.01).all()

        # 60d >= 30d
        assert (late_features['precip_60d'] >= late_features['precip_30d'] - 0.01).all()

        # 90d >= 60d
        assert (late_features['precip_90d'] >= late_features['precip_60d'] - 0.01).all()

    def test_binary_flags_are_binary(self, feature_computer, sample_raw_data):
        """Test that binary flags only contain 0 or 1."""
        features = feature_computer.compute_all_features(sample_raw_data)

        binary_features = ['rain_day_flag', 'freeze_flag', 'marginal_freeze_flag']

        for feat in binary_features:
            if feat in features.columns:
                unique_vals = features[feat].dropna().unique()
                assert set(unique_vals).issubset({0.0, 1.0}), \
                    f"Feature {feat} has non-binary values: {unique_vals}"

    def test_intensity_class_range(self, feature_computer, sample_raw_data):
        """Test that intensity class is in valid range."""
        features = feature_computer.compute_all_features(sample_raw_data)

        assert features['intensity_class'].min() >= 0
        assert features['intensity_class'].max() <= 3


class TestWetDryCycles:
    """Tests for wetting/drying cycle computation."""

    def test_cycle_counting(self, feature_computer):
        """Test that wet-dry cycles are counted correctly."""
        # Create controlled precip pattern: wet, wet, dry, wet, dry, dry, wet
        dates = pd.date_range('2023-01-01', periods=7, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'precip_mm': [5.0, 3.0, 0.0, 2.0, 0.0, 0.0, 4.0],
            'temp_mean_c': [15.0] * 7,
            'temp_min_c': [10.0] * 7,
            'temp_max_c': [20.0] * 7,
            'dewpoint_c': [8.0] * 7,
        })

        features = feature_computer.compute_all_features(df)

        # Transitions: wet→dry at index 2, wet→dry at index 4
        # By index 6, we should have 2 cycles in the 7-day window
        assert features['wet_dry_cycles_30d'].iloc[-1] == 2


class TestFreezeTaw:
    """Tests for freeze-thaw cycle computation."""

    def test_freeze_flag(self, feature_computer):
        """Test freeze flag computation."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'precip_mm': [0.0] * 5,
            'temp_mean_c': [5.0, 0.0, -2.0, 3.0, 5.0],
            'temp_min_c': [2.0, -1.0, -5.0, 0.0, 2.0],
            'temp_max_c': [8.0, 3.0, 1.0, 6.0, 8.0],
            'dewpoint_c': [0.0] * 5,
        })

        features = feature_computer.compute_all_features(df)

        # freeze_flag should be 1 where temp_min < 0
        assert features['freeze_flag'].iloc[0] == 0  # temp_min = 2
        assert features['freeze_flag'].iloc[1] == 1  # temp_min = -1
        assert features['freeze_flag'].iloc[2] == 1  # temp_min = -5
        assert features['freeze_flag'].iloc[3] == 0  # temp_min = 0
        assert features['freeze_flag'].iloc[4] == 0  # temp_min = 2


# ============================================================================
# AtmosphericLoader Tests
# ============================================================================

class TestGetBeachForMop:
    """Tests for MOP to beach mapping."""

    def test_valid_mop_mapping(self):
        """Test that valid MOPs map to correct beaches."""
        assert get_beach_for_mop(520) == 'blacks'
        assert get_beach_for_mop(567) == 'blacks'  # Upper boundary of blacks
        assert get_beach_for_mop(568) == 'torrey'  # First of torrey
        assert get_beach_for_mop(600) == 'delmar'
        assert get_beach_for_mop(650) == 'solana'
        assert get_beach_for_mop(700) == 'sanelijo'
        assert get_beach_for_mop(750) == 'encinitas'

    def test_invalid_mop_raises(self):
        """Test that invalid MOPs raise ValueError."""
        with pytest.raises(ValueError):
            get_beach_for_mop(100)  # Too low

        with pytest.raises(ValueError):
            get_beach_for_mop(1000)  # Too high


class TestAtmosphericLoaderInit:
    """Tests for AtmosphericLoader initialization."""

    def test_init_with_valid_dir(self, temp_atmos_dir):
        """Test initialization with valid directory."""
        loader = AtmosphericLoader(temp_atmos_dir)
        assert loader.lookback_days == 90
        assert loader.n_features == 24
        assert 'delmar' in loader.available_beaches

    def test_init_with_missing_dir_raises(self):
        """Test initialization with missing directory raises error."""
        with pytest.raises(FileNotFoundError):
            AtmosphericLoader('/nonexistent/path')

    def test_init_with_custom_lookback(self, temp_atmos_dir):
        """Test initialization with custom lookback days."""
        loader = AtmosphericLoader(temp_atmos_dir, lookback_days=30)
        assert loader.lookback_days == 30


class TestAtmosphericLoaderGetAtmos:
    """Tests for atmospheric data retrieval."""

    def test_get_atmos_for_scan_shape(self, atmos_loader):
        """Test that returned arrays have correct shape."""
        scan_date = datetime(2023, 3, 15)
        features, doy = atmos_loader.get_atmos_for_scan('delmar', scan_date)

        assert features.shape == (30, 24)  # lookback_days=30, n_features=25
        assert doy.shape == (30,)

    def test_get_atmos_for_scan_dtype(self, atmos_loader):
        """Test that returned arrays have correct dtype."""
        scan_date = datetime(2023, 3, 15)
        features, doy = atmos_loader.get_atmos_for_scan('delmar', scan_date)

        assert features.dtype == np.float32
        assert doy.dtype == np.int32

    def test_get_atmos_for_scan_ordinal(self, atmos_loader):
        """Test that ordinal date input works."""
        scan_date = datetime(2023, 3, 15)
        ordinal = scan_date.toordinal()

        features1, doy1 = atmos_loader.get_atmos_for_scan('delmar', scan_date)
        features2, doy2 = atmos_loader.get_atmos_for_scan('delmar', ordinal)

        np.testing.assert_array_equal(features1, features2)
        np.testing.assert_array_equal(doy1, doy2)

    def test_get_atmos_day_of_year_range(self, atmos_loader):
        """Test that day of year values are in valid range."""
        scan_date = datetime(2023, 3, 15)
        features, doy = atmos_loader.get_atmos_for_scan('delmar', scan_date)

        assert doy.min() >= 1
        assert doy.max() <= 366

    def test_get_atmos_for_mop(self, atmos_loader):
        """Test convenience method for MOP-based lookup."""
        scan_date = datetime(2023, 3, 15)

        # MOP 600 is in Del Mar range
        features, doy = atmos_loader.get_atmos_for_mop(600, scan_date)

        assert features.shape == (30, 24)
        assert doy.shape == (30,)


class TestAtmosphericLoaderBatch:
    """Tests for batch loading."""

    def test_get_batch_atmos_shape(self, atmos_loader):
        """Test batch loading returns correct shape."""
        beaches = ['delmar', 'delmar', 'delmar']
        dates = [datetime(2023, 3, 15), datetime(2023, 4, 1), datetime(2023, 4, 15)]

        features, doy = atmos_loader.get_batch_atmos(beaches, dates)

        assert features.shape == (3, 30, 24)
        assert doy.shape == (3, 30)

    def test_get_batch_length_mismatch_raises(self, atmos_loader):
        """Test that mismatched lengths raise error."""
        beaches = ['delmar', 'delmar']
        dates = [datetime(2023, 3, 15)]

        with pytest.raises(ValueError):
            atmos_loader.get_batch_atmos(beaches, dates)


class TestAtmosphericLoaderValidation:
    """Tests for data validation methods."""

    def test_validate_coverage(self, atmos_loader):
        """Test coverage validation."""
        scan_dates = [datetime(2023, 3, 15), datetime(2023, 4, 1)]

        coverage = atmos_loader.validate_coverage('delmar', scan_dates)

        assert 'total' in coverage
        assert 'covered' in coverage
        assert 'partial' in coverage
        assert 'missing' in coverage
        assert 'coverage_pct' in coverage
        assert coverage['total'] == 2

    def test_summary(self, atmos_loader):
        """Test summary method."""
        summary = atmos_loader.summary()

        assert 'delmar' in summary
        assert 'n_records' in summary['delmar']
        assert 'date_range' in summary['delmar']
        assert 'n_features' in summary['delmar']


# ============================================================================
# Integration Tests
# ============================================================================

class TestAtmosphericDataset:
    """Tests for AtmosphericDataset integration helper."""

    def test_dataset_initialization(self, atmos_loader):
        """Test dataset helper initialization."""
        # Create mock transect data
        transect_ids = np.array([600, 605, 610])  # All Del Mar
        timestamps = np.array([
            [datetime(2023, 3, 1).toordinal(), datetime(2023, 3, 15).toordinal()],
            [datetime(2023, 3, 1).toordinal(), datetime(2023, 3, 15).toordinal()],
            [datetime(2023, 3, 1).toordinal(), datetime(2023, 3, 15).toordinal()],
        ])

        dataset = AtmosphericDataset(atmos_loader, transect_ids, timestamps)

        assert len(dataset.beaches) == 3
        assert all(b == 'delmar' for b in dataset.beaches)

    def test_get_atmos_for_sample(self, atmos_loader):
        """Test getting atmos data for a single sample."""
        transect_ids = np.array([600])
        timestamps = np.array([[datetime(2023, 3, 15).toordinal()]])

        dataset = AtmosphericDataset(atmos_loader, transect_ids, timestamps)
        features, doy = dataset.get_atmos_for_sample(0, 0)

        assert features.shape == (30, 24)
        assert doy.shape == (30,)


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_pipeline(self, sample_raw_data):
        """Test full pipeline from raw data to model-ready tensors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Step 1: Compute features
            computer = AtmosFeatureComputer()
            features_df = computer.compute_all_features(sample_raw_data)

            # Step 2: Save to parquet
            features_df.to_parquet(tmpdir / 'delmar_atmos.parquet', index=False)

            # Step 3: Load with AtmosphericLoader
            loader = AtmosphericLoader(tmpdir, lookback_days=30)

            # Step 4: Get data for a scan
            scan_date = datetime(2023, 3, 15)
            features, doy = loader.get_atmos_for_scan('delmar', scan_date)

            # Verify output
            assert features.shape == (30, 24)
            assert doy.shape == (30,)
            assert not np.isnan(features).all()
            assert features.dtype == np.float32


class TestSaveLoadRoundtrip:
    """Test save/load roundtrip for parquet files."""

    def test_parquet_roundtrip(self, sample_raw_data):
        """Test that features survive parquet save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            parquet_path = tmpdir / 'test_atmos.parquet'

            # Compute and save
            computer = AtmosFeatureComputer()
            features_df = computer.compute_all_features(sample_raw_data)
            features_df.to_parquet(parquet_path, index=False)

            # Load and compare
            loaded_df = pd.read_parquet(parquet_path)

            assert len(loaded_df) == len(features_df)
            assert set(loaded_df.columns) == set(features_df.columns)

            # Check numeric columns are close
            for col in ATMOS_FEATURE_NAMES:
                if col in loaded_df.columns:
                    np.testing.assert_array_almost_equal(
                        loaded_df[col].values,
                        features_df[col].values,
                        decimal=5,
                    )
