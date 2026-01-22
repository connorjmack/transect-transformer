"""Tests for transect viewer utilities (cube format)."""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def cube_data():
    """Create synthetic cube format data for testing."""
    np.random.seed(42)

    n_transects = 5
    n_epochs = 3
    n_points = 128
    n_features = 12
    n_meta = 12

    # Create cube arrays
    points = np.random.randn(n_transects, n_epochs, n_points, n_features).astype(np.float32)
    distances = np.tile(
        np.linspace(0, 50, n_points),
        (n_transects, n_epochs, 1)
    ).astype(np.float32)
    metadata = np.random.randn(n_transects, n_epochs, n_meta).astype(np.float32)

    # Set realistic ranges for some features
    points[:, :, :, 0] = distances  # distance_m
    points[:, :, :, 1] = np.random.uniform(0, 30, (n_transects, n_epochs, n_points))  # elevation
    points[:, :, :, 5] = np.random.uniform(0, 1, (n_transects, n_epochs, n_points))  # intensity
    points[:, :, :, 6:9] = np.random.uniform(0, 1, (n_transects, n_epochs, n_points, 3))  # RGB

    # Transect IDs
    transect_ids = np.array(['MOP 100', 'MOP 101', 'MOP 102', 'MOP 103', 'MOP 104'], dtype=object)

    # Epoch info
    epoch_names = np.array(['2018_scan.las', '2019_scan.las', '2020_scan.las'], dtype=object)
    epoch_dates = np.array(['2018-01-01', '2019-01-01', '2020-01-01'], dtype=object)

    # Feature and metadata names
    feature_names = [
        'distance_m', 'elevation_m', 'slope_deg', 'curvature', 'roughness',
        'intensity', 'red', 'green', 'blue', 'classification',
        'return_number', 'num_returns'
    ]
    metadata_names = [
        'cliff_height_m', 'mean_slope_deg', 'max_slope_deg', 'toe_elevation_m',
        'top_elevation_m', 'orientation_deg', 'transect_length_m', 'latitude',
        'longitude', 'transect_id', 'mean_intensity', 'dominant_class'
    ]

    return {
        'points': points,
        'distances': distances,
        'metadata': metadata,
        'transect_ids': transect_ids,
        'epoch_names': epoch_names,
        'epoch_dates': epoch_dates,
        'feature_names': feature_names,
        'metadata_names': metadata_names,
    }


@pytest.fixture
def flat_data():
    """Create synthetic flat format data for comparison testing."""
    np.random.seed(42)

    n_transects = 5
    n_points = 128
    n_features = 12
    n_meta = 12

    points = np.random.randn(n_transects, n_points, n_features).astype(np.float32)
    distances = np.tile(np.linspace(0, 50, n_points), (n_transects, 1)).astype(np.float32)
    metadata = np.random.randn(n_transects, n_meta).astype(np.float32)

    transect_ids = np.array(['MOP 100', 'MOP 101', 'MOP 102', 'MOP 103', 'MOP 104'], dtype=object)

    feature_names = [
        'distance_m', 'elevation_m', 'slope_deg', 'curvature', 'roughness',
        'intensity', 'red', 'green', 'blue', 'classification',
        'return_number', 'num_returns'
    ]
    metadata_names = [
        'cliff_height_m', 'mean_slope_deg', 'max_slope_deg', 'toe_elevation_m',
        'top_elevation_m', 'orientation_deg', 'transect_length_m', 'latitude',
        'longitude', 'transect_id', 'mean_intensity', 'dominant_class'
    ]

    return {
        'points': points,
        'distances': distances,
        'metadata': metadata,
        'transect_ids': transect_ids,
        'feature_names': feature_names,
        'metadata_names': metadata_names,
    }


class TestDataLoader:
    """Test data loader functions."""

    def test_is_cube_format_true(self, cube_data):
        """Test cube format detection for 4D data."""
        from apps.transect_viewer.utils.data_loader import is_cube_format
        assert is_cube_format(cube_data) is True

    def test_is_cube_format_false(self, flat_data):
        """Test cube format detection for 3D data."""
        from apps.transect_viewer.utils.data_loader import is_cube_format
        assert is_cube_format(flat_data) is False

    def test_get_cube_dimensions(self, cube_data):
        """Test getting cube dimensions."""
        from apps.transect_viewer.utils.data_loader import get_cube_dimensions

        dims = get_cube_dimensions(cube_data)

        assert dims['n_transects'] == 5
        assert dims['n_epochs'] == 3
        assert dims['n_points'] == 128
        assert dims['n_features'] == 12

    def test_get_cube_dimensions_flat_fallback(self, flat_data):
        """Test cube dimensions with flat format fallback."""
        from apps.transect_viewer.utils.data_loader import get_cube_dimensions

        dims = get_cube_dimensions(flat_data)

        assert dims['n_transects'] == 5
        assert dims['n_epochs'] == 1  # Fallback for flat format
        assert dims['n_points'] == 128
        assert dims['n_features'] == 12

    def test_get_epoch_dates(self, cube_data):
        """Test getting epoch dates."""
        from apps.transect_viewer.utils.data_loader import get_epoch_dates

        dates = get_epoch_dates(cube_data)

        assert len(dates) == 3
        assert dates[0] == '2018-01-01'
        assert dates[2] == '2020-01-01'

    def test_get_transect_by_id(self, cube_data):
        """Test getting single transect by ID."""
        from apps.transect_viewer.utils.data_loader import get_transect_by_id

        transect = get_transect_by_id(cube_data, 'MOP 101')

        assert 'points' in transect
        assert 'distances' in transect
        assert 'metadata' in transect
        assert transect['transect_id'] == 'MOP 101'

        # Should return all epochs
        assert transect['points'].shape == (3, 128, 12)
        assert transect['distances'].shape == (3, 128)
        assert transect['metadata'].shape == (3, 12)

    def test_get_transect_by_id_single_epoch(self, cube_data):
        """Test getting transect at specific epoch."""
        from apps.transect_viewer.utils.data_loader import get_transect_by_id

        transect = get_transect_by_id(cube_data, 'MOP 101', epoch_idx=1)

        # Should return single epoch
        assert transect['points'].shape == (128, 12)
        assert transect['distances'].shape == (128,)
        assert transect['metadata'].shape == (12,)
        assert transect['epoch_idx'] == 1

    def test_get_transect_by_id_not_found(self, cube_data):
        """Test getting transect with invalid ID raises error."""
        from apps.transect_viewer.utils.data_loader import get_transect_by_id

        with pytest.raises(ValueError, match="not found"):
            get_transect_by_id(cube_data, 'MOP 999')

    def test_get_transect_by_id_epoch_idx_out_of_bounds(self, cube_data):
        """Test getting transect with out-of-bounds epoch_idx raises error."""
        from apps.transect_viewer.utils.data_loader import get_transect_by_id

        # epoch_idx too large (only 3 epochs: 0, 1, 2)
        with pytest.raises(ValueError, match="out of bounds"):
            get_transect_by_id(cube_data, 'MOP 101', epoch_idx=5)

        # epoch_idx exactly at boundary (3 is out of bounds)
        with pytest.raises(ValueError, match="out of bounds"):
            get_transect_by_id(cube_data, 'MOP 101', epoch_idx=3)

    def test_get_transect_by_id_negative_epoch_idx(self, cube_data):
        """Test getting transect with negative epoch_idx for last epoch."""
        from apps.transect_viewer.utils.data_loader import get_transect_by_id

        # Negative index should work like Python list indexing
        transect = get_transect_by_id(cube_data, 'MOP 101', epoch_idx=-1)

        # Should return the last epoch
        assert transect['points'].shape == (128, 12)
        assert transect['epoch_idx'] == 2  # Last of 3 epochs (0, 1, 2)

    def test_get_transect_by_id_negative_epoch_idx_out_of_bounds(self, cube_data):
        """Test getting transect with too-negative epoch_idx raises error."""
        from apps.transect_viewer.utils.data_loader import get_transect_by_id

        # -4 is out of bounds for 3 epochs
        with pytest.raises(ValueError, match="out of bounds"):
            get_transect_by_id(cube_data, 'MOP 101', epoch_idx=-4)

    def test_get_transect_temporal_slice(self, cube_data):
        """Test getting temporal slice for a feature."""
        from apps.transect_viewer.utils.data_loader import get_transect_temporal_slice

        distances, values, dates = get_transect_temporal_slice(
            cube_data, 'MOP 101', 'elevation_m'
        )

        assert distances.shape == (3, 128)  # (T, N)
        assert values.shape == (3, 128)     # (T, N)
        assert len(dates) == 3

    def test_get_transect_temporal_slice_invalid_feature(self, cube_data):
        """Test temporal slice with invalid feature raises error."""
        from apps.transect_viewer.utils.data_loader import get_transect_temporal_slice

        with pytest.raises(ValueError, match="not found"):
            get_transect_temporal_slice(cube_data, 'MOP 101', 'invalid_feature')

    def test_get_all_transect_ids(self, cube_data):
        """Test getting all transect IDs."""
        from apps.transect_viewer.utils.data_loader import get_all_transect_ids

        ids = get_all_transect_ids(cube_data)

        assert len(ids) == 5
        # Should be sorted
        assert ids == sorted(ids)

    def test_get_epoch_slice(self, cube_data):
        """Test getting all transects at single epoch."""
        from apps.transect_viewer.utils.data_loader import get_epoch_slice

        epoch_data = get_epoch_slice(cube_data, epoch_idx=1)

        assert epoch_data['points'].shape == (5, 128, 12)
        assert epoch_data['distances'].shape == (5, 128)
        assert epoch_data['metadata'].shape == (5, 12)
        assert epoch_data['epoch_idx'] == 1

    def test_compute_temporal_change(self, cube_data):
        """Test computing temporal change between epochs."""
        from apps.transect_viewer.utils.data_loader import compute_temporal_change

        change = compute_temporal_change(
            cube_data, 'MOP 101', 'elevation_m',
            epoch1_idx=0, epoch2_idx=2
        )

        assert 'distances' in change
        assert 'difference' in change
        assert 'mean_change' in change
        assert 'max_change' in change
        assert 'min_change' in change
        assert change['epoch1_date'] == '2018-01-01'
        assert change['epoch2_date'] == '2020-01-01'

    def test_check_data_coverage(self, cube_data):
        """Test data coverage checking."""
        from apps.transect_viewer.utils.data_loader import check_data_coverage

        coverage = check_data_coverage(cube_data)

        assert coverage['total_cells'] == 15  # 5 transects x 3 epochs
        assert coverage['coverage_pct'] == 100.0
        assert coverage['full_coverage'] == True

    def test_check_data_coverage_with_missing(self, cube_data):
        """Test data coverage with missing data."""
        from apps.transect_viewer.utils.data_loader import check_data_coverage

        # Add some NaN values
        cube_data['points'][0, 1, 0, 0] = np.nan
        cube_data['points'][2, 2, 0, 0] = np.nan

        coverage = check_data_coverage(cube_data)

        assert coverage['missing_cells'] == 2
        assert coverage['full_coverage'] == False
        assert coverage['coverage_pct'] < 100.0


class TestValidators:
    """Test validation functions."""

    def test_check_nan_values_cube(self, cube_data):
        """Test NaN value checking for cube format."""
        from apps.transect_viewer.utils.validators import check_nan_values

        # No NaN initially
        nan_counts = check_nan_values(cube_data['points'], cube_data['feature_names'])

        for feature, count in nan_counts.items():
            assert count == 0, f"Unexpected NaN in {feature}"

    def test_check_nan_values_with_nans(self, cube_data):
        """Test NaN detection when NaN values present."""
        from apps.transect_viewer.utils.validators import check_nan_values

        # Add NaN values to elevation
        cube_data['points'][:, :, 10, 1] = np.nan  # 5*3 = 15 NaN values

        nan_counts = check_nan_values(cube_data['points'], cube_data['feature_names'])

        assert nan_counts['elevation_m'] == 15

    def test_check_value_ranges(self, cube_data):
        """Test value range checking."""
        from apps.transect_viewer.utils.validators import check_value_ranges

        issues = check_value_ranges(cube_data['points'], cube_data['feature_names'])

        # With random data, we may have range violations
        # Just check that function returns list
        assert isinstance(issues, list)

    def test_check_value_ranges_with_violations(self, cube_data):
        """Test value range checking with explicit violations."""
        from apps.transect_viewer.utils.validators import check_value_ranges

        # Set some values outside expected range
        cube_data['points'][:, :, :, 5] = 2.0  # intensity > 1

        issues = check_value_ranges(cube_data['points'], cube_data['feature_names'])

        # Should find intensity above max
        intensity_issues = [i for i in issues if i['feature'] == 'intensity']
        assert len(intensity_issues) > 0

    def test_validate_dataset(self, cube_data):
        """Test full dataset validation."""
        from apps.transect_viewer.utils.validators import validate_dataset

        report = validate_dataset(cube_data)

        assert 'is_valid' in report
        assert 'is_cube_format' in report
        assert report['is_cube_format'] is True
        assert report['n_transects'] == 5
        assert report['n_epochs'] == 3
        assert report['n_points_per_transect'] == 128

    def test_validate_dataset_flat(self, flat_data):
        """Test dataset validation with flat format."""
        from apps.transect_viewer.utils.validators import validate_dataset

        report = validate_dataset(flat_data)

        assert report['is_cube_format'] is False
        assert report['n_epochs'] == 1

    def test_compute_statistics(self, cube_data):
        """Test computing feature statistics."""
        from apps.transect_viewer.utils.validators import compute_statistics

        stats_df = compute_statistics(cube_data, epoch_idx=-1)

        assert len(stats_df) == 12  # 12 features
        assert 'feature' in stats_df.columns
        assert 'min' in stats_df.columns
        assert 'max' in stats_df.columns
        assert 'mean' in stats_df.columns
        assert 'std' in stats_df.columns

    def test_compute_metadata_statistics(self, cube_data):
        """Test computing metadata statistics."""
        from apps.transect_viewer.utils.validators import compute_metadata_statistics

        stats_df = compute_metadata_statistics(cube_data, epoch_idx=-1)

        assert len(stats_df) == 12  # 12 metadata fields
        assert 'field' in stats_df.columns

    def test_compute_temporal_statistics(self, cube_data):
        """Test computing temporal statistics for a feature."""
        from apps.transect_viewer.utils.validators import compute_temporal_statistics

        stats_df = compute_temporal_statistics(cube_data, 'elevation_m')

        assert len(stats_df) == 3  # 3 epochs
        assert 'epoch' in stats_df.columns
        assert 'epoch_idx' in stats_df.columns
        assert 'n_valid' in stats_df.columns

    def test_compute_temporal_statistics_invalid_feature(self, cube_data):
        """Test temporal statistics with invalid feature raises error."""
        from apps.transect_viewer.utils.validators import compute_temporal_statistics

        with pytest.raises(ValueError, match="not found"):
            compute_temporal_statistics(cube_data, 'invalid_feature')


class TestSaveLoadRoundtrip:
    """Test saving and loading cube data."""

    def test_npz_roundtrip(self, cube_data, tmp_path):
        """Test saving and loading cube data via NPZ."""
        output_path = tmp_path / "test_cube.npz"

        # Save
        save_dict = {}
        for key, value in cube_data.items():
            if isinstance(value, list):
                save_dict[key] = np.array(value, dtype=object)
            else:
                save_dict[key] = value

        np.savez_compressed(output_path, **save_dict)

        # Load using data_loader
        from apps.transect_viewer.utils.data_loader import load_npz

        loaded = load_npz(str(output_path))

        # Verify shapes
        assert loaded['points'].shape == cube_data['points'].shape
        assert loaded['distances'].shape == cube_data['distances'].shape
        assert loaded['metadata'].shape == cube_data['metadata'].shape

        # Verify values
        np.testing.assert_array_almost_equal(loaded['points'], cube_data['points'])

    def test_load_npz_with_string_arrays(self, cube_data, tmp_path):
        """Test loading NPZ with object arrays (string IDs)."""
        output_path = tmp_path / "test_cube_strings.npz"

        # Save with object arrays
        np.savez_compressed(
            output_path,
            points=cube_data['points'],
            transect_ids=cube_data['transect_ids'],
            epoch_dates=cube_data['epoch_dates'],
            feature_names=np.array(cube_data['feature_names'], dtype=object),
        )

        # Load
        from apps.transect_viewer.utils.data_loader import load_npz

        loaded = load_npz(str(output_path))

        # String arrays should be converted to lists
        assert isinstance(loaded['transect_ids'], list)
        assert isinstance(loaded['epoch_dates'], list)
        assert isinstance(loaded['feature_names'], list)


class TestHelpers:
    """Test shared helper functions."""

    def test_safe_date_label_valid_dates(self):
        """Test safe_date_label with valid date strings."""
        from apps.transect_viewer.utils.helpers import safe_date_label

        dates = ['2018-01-01', '2019-06-15', '2020-12-31']

        assert safe_date_label(dates, 0) == '2018-01-01'
        assert safe_date_label(dates, 1) == '2019-06-15'
        assert safe_date_label(dates, 2) == '2020-12-31'

    def test_safe_date_label_with_timestamp(self):
        """Test safe_date_label with full timestamp strings."""
        from apps.transect_viewer.utils.helpers import safe_date_label

        dates = ['2018-01-01T12:30:00', '2019-06-15T08:00:00']

        # Should truncate to first 10 chars (date part only)
        assert safe_date_label(dates, 0) == '2018-01-01'
        assert safe_date_label(dates, 1) == '2019-06-15'

    def test_safe_date_label_out_of_bounds(self):
        """Test safe_date_label with out-of-bounds index."""
        from apps.transect_viewer.utils.helpers import safe_date_label

        dates = ['2018-01-01', '2019-06-15']

        # Should return fallback
        assert safe_date_label(dates, 5) == 'Epoch 5'
        assert safe_date_label(dates, 10, "E") == 'E 10'

    def test_safe_date_label_empty_dates(self):
        """Test safe_date_label with empty or None dates."""
        from apps.transect_viewer.utils.helpers import safe_date_label

        assert safe_date_label(None, 0) == 'Epoch 0'
        assert safe_date_label([], 0) == 'Epoch 0'

    def test_safe_date_label_short_string(self):
        """Test safe_date_label with string shorter than 10 chars."""
        from apps.transect_viewer.utils.helpers import safe_date_label

        dates = ['2018', '2019-06']

        # Should return the short string as-is
        assert safe_date_label(dates, 0) == '2018'
        assert safe_date_label(dates, 1) == '2019-06'

    def test_safe_date_label_none_element(self):
        """Test safe_date_label with None element in dates list."""
        from apps.transect_viewer.utils.helpers import safe_date_label

        dates = ['2018-01-01', None, '2020-12-31']

        assert safe_date_label(dates, 0) == '2018-01-01'
        assert safe_date_label(dates, 1) == 'Epoch 1'  # None element
        assert safe_date_label(dates, 2) == '2020-12-31'

    def test_safe_epoch_option_valid_dates(self):
        """Test safe_epoch_option with valid dates."""
        from apps.transect_viewer.utils.helpers import safe_epoch_option

        dates = ['2018-01-01', '2019-06-15']

        assert safe_epoch_option(dates, 0) == '0: 2018-01-01'
        assert safe_epoch_option(dates, 1) == '1: 2019-06-15'

    def test_safe_epoch_option_no_dates(self):
        """Test safe_epoch_option with no dates."""
        from apps.transect_viewer.utils.helpers import safe_epoch_option

        assert safe_epoch_option(None, 0) == 'Epoch 0'
        assert safe_epoch_option([], 1) == 'Epoch 1'

    def test_safe_epoch_option_out_of_bounds(self):
        """Test safe_epoch_option with out-of-bounds index."""
        from apps.transect_viewer.utils.helpers import safe_epoch_option

        dates = ['2018-01-01']

        assert safe_epoch_option(dates, 5) == 'Epoch 5'

    def test_safe_metadata_value_valid(self):
        """Test safe_metadata_value with valid values."""
        from apps.transect_viewer.utils.helpers import safe_metadata_value

        metadata = np.array([10.5, 45.2, 60.0, np.nan])

        result = safe_metadata_value(metadata, 0, 1, " m")
        assert "10.5" in result and "m" in result

        result = safe_metadata_value(metadata, 1, 0)
        assert "45" in result

    def test_safe_metadata_value_out_of_bounds(self):
        """Test safe_metadata_value with out-of-bounds index."""
        from apps.transect_viewer.utils.helpers import safe_metadata_value

        metadata = np.array([10.5, 45.2])

        assert safe_metadata_value(metadata, 10) == "N/A"

    def test_safe_metadata_value_nan(self):
        """Test safe_metadata_value with NaN value."""
        from apps.transect_viewer.utils.helpers import safe_metadata_value

        metadata = np.array([10.5, np.nan, 60.0])

        assert safe_metadata_value(metadata, 1) == "N/A"

    def test_safe_metadata_format_valid(self):
        """Test safe_metadata_format with valid values."""
        from apps.transect_viewer.utils.helpers import safe_metadata_format

        metadata = np.array([10.567, 45.2, 60.0])

        assert safe_metadata_format(metadata, 0, ".2f") == "10.57"
        assert safe_metadata_format(metadata, 1, ".1f") == "45.2"
        assert safe_metadata_format(metadata, 2, ".0f") == "60"

    def test_safe_metadata_format_out_of_bounds(self):
        """Test safe_metadata_format with out-of-bounds index."""
        from apps.transect_viewer.utils.helpers import safe_metadata_format

        metadata = np.array([10.5])

        assert safe_metadata_format(metadata, 5) == "N/A"

    def test_safe_metadata_format_nan(self):
        """Test safe_metadata_format with NaN value."""
        from apps.transect_viewer.utils.helpers import safe_metadata_format

        metadata = np.array([np.nan])

        assert safe_metadata_format(metadata, 0) == "N/A"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
