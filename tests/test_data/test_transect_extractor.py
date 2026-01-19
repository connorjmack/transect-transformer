"""Tests for transect extraction from point clouds using shapefiles."""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager

from src.data.shapefile_transect_extractor import (
    ShapefileTransectExtractor,
    get_las_bounds,
    bounds_overlap,
)


@pytest.fixture
def extractor():
    """Create a ShapefileTransectExtractor instance for testing."""
    return ShapefileTransectExtractor(n_points=128, buffer_m=1.0, min_points=20)


@pytest.fixture
def synthetic_las_data():
    """Create synthetic LAS data for testing.

    Generates a simple cliff profile along a transect:
    - Points span x=0 to x=50, y=0 to y=10
    - Elevation increases linearly with x (slope)
    """
    np.random.seed(42)

    # Generate points along a profile
    n_points = 1000
    x = np.random.uniform(0, 50, n_points)
    y = np.random.uniform(0, 10, n_points)
    z = x * 0.5 + np.random.normal(0, 0.2, n_points)  # Linear slope with noise

    # Create intensity values
    intensity = np.random.uniform(0.3, 0.9, n_points).astype(np.float32)

    # Create RGB values (normalized 0-1)
    red = np.random.uniform(0.4, 0.6, n_points).astype(np.float32)
    green = np.random.uniform(0.5, 0.7, n_points).astype(np.float32)
    blue = np.random.uniform(0.3, 0.5, n_points).astype(np.float32)

    # Classification (ground=2, vegetation=3, etc)
    classification = np.random.choice([2, 3, 5], n_points).astype(np.float32)

    # Return numbers
    return_number = np.ones(n_points, dtype=np.float32)
    num_returns = np.ones(n_points, dtype=np.float32)

    return {
        'x': x,
        'y': y,
        'z': z,
        'xyz': np.column_stack([x, y, z]),
        'intensity': intensity,
        'red': red,
        'green': green,
        'blue': blue,
        'classification': classification,
        'return_number': return_number,
        'num_returns': num_returns,
    }


@pytest.fixture
def synthetic_transect_line():
    """Create a synthetic transect LineString geometry."""
    try:
        from shapely.geometry import LineString
        # Transect from (0, 5) to (50, 5) - crosses the synthetic point cloud
        return LineString([(0, 5), (50, 5)])
    except ImportError:
        pytest.skip("shapely not available")


@pytest.fixture
def synthetic_gdf():
    """Create a synthetic GeoDataFrame with transect lines."""
    try:
        import geopandas as gpd
        from shapely.geometry import LineString

        # Create 3 parallel transects
        lines = [
            LineString([(0, 3), (50, 3)]),
            LineString([(0, 5), (50, 5)]),
            LineString([(0, 7), (50, 7)]),
        ]

        gdf = gpd.GeoDataFrame({
            'tr_id': [1, 2, 3],
            'geometry': lines
        })

        return gdf
    except ImportError:
        pytest.skip("geopandas not available")


class TestShapefileTransectExtractorInit:
    """Test ShapefileTransectExtractor initialization."""

    def test_default_init(self):
        """Test initialization with default parameters."""
        extractor = ShapefileTransectExtractor()
        assert extractor.n_points == 128
        assert extractor.buffer_m == 1.0
        assert extractor.min_points == 20

    def test_custom_init(self):
        """Test initialization with custom parameters."""
        extractor = ShapefileTransectExtractor(
            n_points=64,
            buffer_m=2.0,
            min_points=10,
        )
        assert extractor.n_points == 64
        assert extractor.buffer_m == 2.0
        assert extractor.min_points == 10

    def test_feature_names(self):
        """Test that feature names are correctly defined."""
        extractor = ShapefileTransectExtractor()
        assert len(extractor.FEATURE_NAMES) == 12
        assert extractor.FEATURE_NAMES[0] == 'distance_m'
        assert extractor.FEATURE_NAMES[1] == 'elevation_m'
        assert extractor.FEATURE_NAMES[5] == 'intensity'
        assert extractor.N_FEATURES == 12

    def test_metadata_names(self):
        """Test that metadata names are correctly defined."""
        extractor = ShapefileTransectExtractor()
        assert len(extractor.METADATA_NAMES) == 12
        assert extractor.METADATA_NAMES[0] == 'cliff_height_m'
        assert extractor.METADATA_NAMES[9] == 'transect_id'
        assert extractor.N_METADATA == 12


class TestTransectDirection:
    """Test transect direction computation."""

    def test_get_transect_direction(self, extractor, synthetic_transect_line):
        """Test getting transect direction from LineString."""
        start, direction, length = extractor.get_transect_direction(synthetic_transect_line)

        # Start should be at (0, 5)
        np.testing.assert_allclose(start, [0, 5], rtol=1e-5)

        # Direction should be unit vector pointing in +x direction
        np.testing.assert_allclose(direction, [1, 0], rtol=1e-5)

        # Length should be 50
        assert length == pytest.approx(50, abs=0.1)


class TestTransectExtraction:
    """Test transect extraction functionality."""

    def test_extract_transect_points(
        self, extractor, synthetic_transect_line, synthetic_las_data
    ):
        """Test extraction of points along a single transect."""
        from scipy.spatial import cKDTree

        tree = cKDTree(synthetic_las_data['xyz'][:, :2])

        result = extractor.extract_transect_points(
            synthetic_transect_line, synthetic_las_data, tree
        )

        assert result is not None
        assert 'indices' in result
        assert 'distances' in result
        assert 'xyz' in result
        assert 'intensity' in result

        # Should have extracted points
        assert len(result['distances']) > extractor.min_points

        # Distances should be monotonically increasing
        assert np.all(np.diff(result['distances']) >= 0)

        # Distances should be within transect length
        assert result['distances'][0] >= 0
        assert result['distances'][-1] <= result['transect_length']

    def test_resample_transect(self, extractor, synthetic_transect_line, synthetic_las_data):
        """Test transect resampling to fixed number of points."""
        from scipy.spatial import cKDTree

        tree = cKDTree(synthetic_las_data['xyz'][:, :2])

        # First extract raw points
        raw_data = extractor.extract_transect_points(
            synthetic_transect_line, synthetic_las_data, tree
        )

        assert raw_data is not None

        # Resample
        resampled = extractor.resample_transect(raw_data)

        assert resampled is not None
        assert 'features' in resampled
        assert 'distances' in resampled
        assert 'metadata' in resampled

        # Check shapes
        assert resampled['features'].shape == (128, 12)
        assert resampled['distances'].shape == (128,)
        assert resampled['metadata'].shape == (12,)

        # Distances should be evenly spaced
        assert np.all(np.diff(resampled['distances']) > 0)


class TestFeatureComputation:
    """Test feature and metadata computation."""

    def test_feature_values_range(self, extractor, synthetic_transect_line, synthetic_las_data):
        """Test that computed features are in expected ranges."""
        from scipy.spatial import cKDTree

        tree = cKDTree(synthetic_las_data['xyz'][:, :2])

        # Extract and resample
        raw_data = extractor.extract_transect_points(
            synthetic_transect_line, synthetic_las_data, tree
        )
        assert raw_data is not None

        resampled = extractor.resample_transect(raw_data)
        assert resampled is not None

        features = resampled['features']

        # Feature 0: Distance (should be positive and increasing)
        assert np.all(features[:, 0] >= 0)
        assert np.all(np.diff(features[:, 0]) > 0)

        # Feature 1: Elevation (should match synthetic data pattern)
        assert features[:, 1].min() >= -5
        assert features[:, 1].max() <= 30

        # Feature 5: Intensity (normalized 0-1)
        assert np.all(features[:, 5] >= 0)
        assert np.all(features[:, 5] <= 1)

        # Features 6-8: RGB (normalized 0-1)
        assert np.all(features[:, 6:9] >= 0)
        assert np.all(features[:, 6:9] <= 1)

        # Feature 9: Classification (discrete values)
        assert np.all(features[:, 9] >= 0)

    def test_metadata_computation(self, extractor, synthetic_transect_line, synthetic_las_data):
        """Test transect metadata computation."""
        from scipy.spatial import cKDTree

        tree = cKDTree(synthetic_las_data['xyz'][:, :2])

        # Extract and resample
        raw_data = extractor.extract_transect_points(
            synthetic_transect_line, synthetic_las_data, tree
        )
        assert raw_data is not None

        resampled = extractor.resample_transect(raw_data)
        assert resampled is not None

        metadata = resampled['metadata']

        # Cliff height (metadata[0]) should be positive
        assert metadata[0] > 0

        # Mean slope (metadata[1]) should be reasonable
        assert metadata[1] >= 0
        assert metadata[1] < 90

        # Max slope (metadata[2]) should be >= mean slope
        assert metadata[2] >= metadata[1]

        # Orientation (metadata[5]) should be in [0, 360)
        assert 0 <= metadata[5] < 360

        # Transect length (metadata[6]) should match line length
        assert metadata[6] == pytest.approx(50, abs=1.0)


class TestSaveLoad:
    """Test saving and loading transects."""

    def test_save_load_npz(self, extractor, tmp_path):
        """Test save/load cycle with NPZ format."""
        # Create dummy transects
        transects = {
            'points': np.random.randn(10, 128, 12).astype(np.float32),
            'distances': np.random.randn(10, 128).astype(np.float32),
            'metadata': np.random.randn(10, 12).astype(np.float32),
            'transect_ids': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64),
            'las_sources': ['file1.las', 'file2.las'] * 5,
            'feature_names': extractor.FEATURE_NAMES,
            'metadata_names': extractor.METADATA_NAMES,
        }

        # Save
        output_path = tmp_path / "test_transects.npz"
        extractor.save_transects(transects, output_path)

        assert output_path.exists()

        # Load
        loaded = extractor.load_transects(output_path)

        # Check all keys present
        assert 'points' in loaded
        assert 'distances' in loaded
        assert 'metadata' in loaded

        # Check shapes match
        assert loaded['points'].shape == transects['points'].shape
        assert loaded['distances'].shape == transects['distances'].shape
        assert loaded['metadata'].shape == transects['metadata'].shape

        # Check values match
        np.testing.assert_allclose(loaded['points'], transects['points'])
        np.testing.assert_allclose(loaded['distances'], transects['distances'])
        np.testing.assert_allclose(loaded['metadata'], transects['metadata'])

    def test_load_nonexistent_file(self, extractor):
        """Test loading from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            extractor.load_transects("nonexistent.npz")


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_insufficient_points_returns_none(self, extractor):
        """Test that insufficient points returns None."""
        from scipy.spatial import cKDTree
        from shapely.geometry import LineString

        # Very sparse points far from transect
        sparse_data = {
            'x': np.array([100, 101, 102]),
            'y': np.array([100, 101, 102]),
            'z': np.array([0, 0, 0]),
            'xyz': np.array([[100, 100, 0], [101, 101, 0], [102, 102, 0]]),
            'intensity': np.zeros(3, dtype=np.float32),
            'red': np.zeros(3, dtype=np.float32),
            'green': np.zeros(3, dtype=np.float32),
            'blue': np.zeros(3, dtype=np.float32),
            'classification': np.zeros(3, dtype=np.float32),
            'return_number': np.ones(3, dtype=np.float32),
            'num_returns': np.ones(3, dtype=np.float32),
        }

        tree = cKDTree(sparse_data['xyz'][:, :2])
        line = LineString([(0, 0), (10, 0)])

        result = extractor.extract_transect_points(line, sparse_data, tree)

        # Should return None for insufficient points
        assert result is None

    def test_resample_with_too_few_points(self, extractor):
        """Test that resampling fails gracefully with too few points."""
        # Raw data with only 1 point
        raw_data = {
            'distances': np.array([5.0]),
            'xyz': np.array([[5, 0, 2.5]]),
            'intensity': np.array([0.5]),
            'red': np.array([0.5]),
            'green': np.array([0.5]),
            'blue': np.array([0.5]),
            'classification': np.array([2.0]),
            'return_number': np.array([1.0]),
            'num_returns': np.array([1.0]),
            'transect_length': 10.0,
            'start': np.array([0, 0]),
            'direction': np.array([1, 0]),
        }

        result = extractor.resample_transect(raw_data)

        # Should return None
        assert result is None


class TestBoundsHelpers:
    """Test bounds checking helper functions."""

    def test_bounds_overlap_true(self):
        """Test that overlapping bounds are detected."""
        bounds1 = (0, 0, 10, 10)
        bounds2 = (5, 5, 15, 15)
        assert bounds_overlap(bounds1, bounds2) is True

    def test_bounds_overlap_false(self):
        """Test that non-overlapping bounds are detected."""
        bounds1 = (0, 0, 10, 10)
        bounds2 = (20, 20, 30, 30)
        assert bounds_overlap(bounds1, bounds2) is False

    def test_bounds_overlap_touching(self):
        """Test that touching bounds are considered overlapping."""
        bounds1 = (0, 0, 10, 10)
        bounds2 = (10, 0, 20, 10)
        assert bounds_overlap(bounds1, bounds2) is True

    def test_bounds_overlap_with_buffer(self):
        """Test that buffer extends overlap detection."""
        bounds1 = (0, 0, 10, 10)
        bounds2 = (15, 0, 25, 10)  # 5m gap
        assert bounds_overlap(bounds1, bounds2, buffer=0) is False
        assert bounds_overlap(bounds1, bounds2, buffer=5) is True
        assert bounds_overlap(bounds1, bounds2, buffer=10) is True

    def test_bounds_overlap_contained(self):
        """Test that contained bounds overlap."""
        bounds1 = (0, 0, 100, 100)
        bounds2 = (25, 25, 75, 75)
        assert bounds_overlap(bounds1, bounds2) is True
        assert bounds_overlap(bounds2, bounds1) is True

    @patch('src.data.shapefile_transect_extractor.laspy')
    def test_get_las_bounds(self, mock_laspy):
        """Test getting bounds from LAS file header."""
        # Mock the laspy.open context manager and header
        mock_header = Mock()
        mock_header.x_min = 100.0
        mock_header.y_min = 200.0
        mock_header.x_max = 150.0
        mock_header.y_max = 250.0

        mock_file = Mock()
        mock_file.header = mock_header
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)

        mock_laspy.open.return_value = mock_file

        bounds = get_las_bounds(Path("test.las"))

        assert bounds is not None
        assert bounds == (100.0, 200.0, 150.0, 250.0)

    @patch('src.data.shapefile_transect_extractor.laspy')
    def test_get_las_bounds_error_returns_none(self, mock_laspy):
        """Test that get_las_bounds returns None on error."""
        mock_laspy.open.side_effect = Exception("File not found")

        bounds = get_las_bounds(Path("nonexistent.las"))

        assert bounds is None


class TestSpatialLoading:
    """Test spatial loading optimizations."""

    @patch('src.data.shapefile_transect_extractor.get_las_bounds')
    @patch('src.data.shapefile_transect_extractor.laspy')
    def test_load_las_file_spatial_filters_points(
        self, mock_laspy, mock_get_bounds, extractor
    ):
        """Test that spatial loading filters points correctly."""
        # Mock bounds check to return overlapping
        mock_get_bounds.return_value = (0, 0, 100, 100)

        # Create synthetic point data - some inside bounds, some outside
        n_points = 1000
        np.random.seed(42)
        x_all = np.random.uniform(-50, 150, n_points)  # Some outside [0, 100]
        y_all = np.random.uniform(-50, 150, n_points)  # Some outside [0, 100]
        z_all = np.random.uniform(0, 30, n_points)

        # Create a mock chunk with the data
        mock_chunk = Mock()
        mock_chunk.x = x_all
        mock_chunk.y = y_all
        mock_chunk.z = z_all
        mock_chunk.intensity = np.random.uniform(0, 65535, n_points).astype(np.uint16)
        mock_chunk.red = np.random.uniform(0, 65535, n_points).astype(np.uint16)
        mock_chunk.green = np.random.uniform(0, 65535, n_points).astype(np.uint16)
        mock_chunk.blue = np.random.uniform(0, 65535, n_points).astype(np.uint16)
        mock_chunk.classification = np.random.choice([2, 3, 5], n_points).astype(np.uint8)
        mock_chunk.return_number = np.ones(n_points, dtype=np.uint8)
        mock_chunk.number_of_returns = np.ones(n_points, dtype=np.uint8)
        mock_chunk.__len__ = Mock(return_value=n_points)

        # Mock the file context manager with chunk iterator
        mock_file = Mock()
        mock_file.chunk_iterator.return_value = iter([mock_chunk])
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=False)

        mock_laspy.open.return_value = mock_file

        # Load with spatial filter
        bounds = (10, 10, 90, 90)  # Smaller than full data range
        result = extractor.load_las_file_spatial(Path("test.las"), bounds)

        assert result is not None

        # All returned points should be within bounds
        assert np.all(result['x'] >= bounds[0])
        assert np.all(result['x'] <= bounds[2])
        assert np.all(result['y'] >= bounds[1])
        assert np.all(result['y'] <= bounds[3])

        # Should have fewer points than total
        assert len(result['x']) < n_points

    @patch('src.data.shapefile_transect_extractor.get_las_bounds')
    def test_load_las_file_spatial_no_overlap_returns_none(
        self, mock_get_bounds, extractor
    ):
        """Test that no overlap returns None without reading points."""
        # Mock bounds check to return non-overlapping
        mock_get_bounds.return_value = (1000, 1000, 2000, 2000)

        # Request bounds that don't overlap with LAS bounds
        bounds = (0, 0, 10, 10)
        result = extractor.load_las_file_spatial(Path("test.las"), bounds)

        assert result is None


class TestFullPipeline:
    """Integration tests for full extraction pipeline."""

    @patch('src.data.shapefile_transect_extractor.get_las_bounds')
    @patch('src.data.shapefile_transect_extractor.laspy')
    def test_extract_from_shapefile_and_las(
        self, mock_laspy, mock_get_bounds, extractor, synthetic_gdf, synthetic_las_data
    ):
        """Test full extraction pipeline with mocked LAS file."""
        # Mock bounds check to return bounds that overlap with synthetic data
        mock_get_bounds.return_value = (0, 0, 60, 15)

        # Mock the laspy.read call (used when use_spatial_filter=False)
        mock_las = Mock()
        mock_las.x = synthetic_las_data['x']
        mock_las.y = synthetic_las_data['y']
        mock_las.z = synthetic_las_data['z']
        mock_las.intensity = (synthetic_las_data['intensity'] * 65535).astype(np.uint16)
        mock_las.red = (synthetic_las_data['red'] * 65535).astype(np.uint16)
        mock_las.green = (synthetic_las_data['green'] * 65535).astype(np.uint16)
        mock_las.blue = (synthetic_las_data['blue'] * 65535).astype(np.uint16)
        mock_las.classification = synthetic_las_data['classification'].astype(np.uint8)
        mock_las.return_number = synthetic_las_data['return_number'].astype(np.uint8)
        mock_las.number_of_returns = synthetic_las_data['num_returns'].astype(np.uint8)

        mock_laspy.read.return_value = mock_las

        # Run extraction with spatial filter disabled (simpler mocking)
        las_files = [Path("fake.las")]
        transects = extractor.extract_from_shapefile_and_las(
            synthetic_gdf, las_files, transect_id_col='tr_id',
            use_spatial_filter=False,
        )

        # Should extract some transects
        assert len(transects['points']) > 0

        # Check output structure
        assert transects['points'].shape[1] == 128  # n_points
        assert transects['points'].shape[2] == 12  # n_features
        assert transects['distances'].shape[1] == 128
        assert transects['metadata'].shape[1] == 12  # n_metadata

        # Check that feature/metadata names are included
        assert 'feature_names' in transects
        assert 'metadata_names' in transects
        assert len(transects['feature_names']) == 12
        assert len(transects['metadata_names']) == 12

        # Check no NaN values
        assert not np.any(np.isnan(transects['points']))
        assert not np.any(np.isnan(transects['distances']))
        assert not np.any(np.isnan(transects['metadata']))

    @patch('src.data.shapefile_transect_extractor.laspy')
    @patch('src.data.shapefile_transect_extractor.get_las_bounds')
    def test_prefilter_skips_non_overlapping_files(
        self, mock_get_bounds, mock_laspy, extractor, synthetic_gdf, synthetic_las_data
    ):
        """Test that LAS files not overlapping transects are skipped."""
        # get_las_bounds is called:
        # 1. During pre-filtering for first file (overlaps)
        # 2. During pre-filtering for second file (doesn't overlap)
        # 3. During extract_single_las for the overlapping file
        mock_get_bounds.side_effect = [
            (0, 0, 60, 15),           # Pre-filter: overlaps with synthetic_gdf
            (1000, 1000, 2000, 2000), # Pre-filter: far away, no overlap
            (0, 0, 60, 15),           # extract_single_las: bounds check again
        ]

        # Mock laspy.read for when we load the overlapping file
        mock_las = Mock()
        mock_las.x = synthetic_las_data['x']
        mock_las.y = synthetic_las_data['y']
        mock_las.z = synthetic_las_data['z']
        mock_las.intensity = (synthetic_las_data['intensity'] * 65535).astype(np.uint16)
        mock_las.red = (synthetic_las_data['red'] * 65535).astype(np.uint16)
        mock_las.green = (synthetic_las_data['green'] * 65535).astype(np.uint16)
        mock_las.blue = (synthetic_las_data['blue'] * 65535).astype(np.uint16)
        mock_las.classification = synthetic_las_data['classification'].astype(np.uint8)
        mock_las.return_number = synthetic_las_data['return_number'].astype(np.uint8)
        mock_las.number_of_returns = synthetic_las_data['num_returns'].astype(np.uint8)
        mock_laspy.read.return_value = mock_las

        las_files = [Path("overlapping.las"), Path("distant.las")]

        # This should filter out the second file before loading
        transects = extractor.extract_from_shapefile_and_las(
            synthetic_gdf, las_files, transect_id_col='tr_id',
            use_spatial_filter=False,
        )

        # Only one file should have been processed (the overlapping one)
        # get_las_bounds called: 2 for pre-filtering + 1 for actual extraction
        assert mock_get_bounds.call_count == 3

        # laspy.read should only be called once (for overlapping.las)
        assert mock_laspy.read.call_count == 1

        # The distant file should have been skipped during pre-filtering
        # Verify by checking the call args
        call_paths = [str(call[0][0]) for call in mock_get_bounds.call_args_list]
        assert 'overlapping.las' in call_paths[0]
        assert 'distant.las' in call_paths[1]
        assert 'overlapping.las' in call_paths[2]  # Called again during extraction


class TestCubeFormat:
    """Test cube format conversion and handling."""

    @pytest.fixture
    def flat_transects(self, extractor):
        """Create synthetic flat transects for testing cube conversion."""
        np.random.seed(42)

        # Simulate 3 transects x 2 epochs
        n_transects_unique = 3
        n_epochs = 2
        n_points = 128
        n_features = 12
        n_meta = 12

        # Create flat arrays (transect-epoch pairs)
        n_total = n_transects_unique * n_epochs
        points = np.random.randn(n_total, n_points, n_features).astype(np.float32)
        distances = np.tile(np.linspace(0, 50, n_points), (n_total, 1)).astype(np.float32)
        metadata = np.random.randn(n_total, n_meta).astype(np.float32)

        # Transect IDs repeat for each epoch
        transect_ids = np.array(['MOP 100', 'MOP 101', 'MOP 102'] * n_epochs, dtype=object)

        # LAS sources alternate (2 files, each seen 3 times)
        las_sources = ['20180101_scan.las', '20180101_scan.las', '20180101_scan.las',
                       '20190101_scan.las', '20190101_scan.las', '20190101_scan.las']

        return {
            'points': points,
            'distances': distances,
            'metadata': metadata,
            'transect_ids': transect_ids,
            'las_sources': las_sources,
            'feature_names': extractor.FEATURE_NAMES,
            'metadata_names': extractor.METADATA_NAMES,
        }

    def test_convert_flat_to_cube(self, flat_transects):
        """Test conversion from flat to cube format."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from scripts.processing.extract_transects import convert_flat_to_cube

        cube = convert_flat_to_cube(flat_transects, n_points=128)

        # Check cube structure
        assert 'points' in cube
        assert 'distances' in cube
        assert 'metadata' in cube
        assert 'timestamps' in cube
        assert 'transect_ids' in cube
        assert 'epoch_names' in cube
        assert 'epoch_dates' in cube

        # Check shapes - should be (n_transects, n_epochs, n_points, n_features)
        assert cube['points'].shape == (3, 2, 128, 12)
        assert cube['distances'].shape == (3, 2, 128)
        assert cube['metadata'].shape == (3, 2, 12)
        assert cube['timestamps'].shape == (3, 2)
        assert len(cube['transect_ids']) == 3
        assert len(cube['epoch_names']) == 2
        assert len(cube['epoch_dates']) == 2

    def test_cube_temporal_ordering(self, flat_transects):
        """Test that epochs are sorted chronologically in cube format."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from scripts.processing.extract_transects import convert_flat_to_cube

        cube = convert_flat_to_cube(flat_transects, n_points=128)

        # Epochs should be sorted by date
        epoch_dates = cube['epoch_dates']
        assert epoch_dates[0] < epoch_dates[1], "Epochs should be chronologically sorted"

    def test_cube_no_duplicate_transects(self, flat_transects):
        """Test that cube has unique transects (no duplicates)."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from scripts.processing.extract_transects import convert_flat_to_cube

        cube = convert_flat_to_cube(flat_transects, n_points=128)

        transect_ids = cube['transect_ids']
        assert len(transect_ids) == len(np.unique(transect_ids)), "Transect IDs should be unique"

    def test_cube_save_load_roundtrip(self, flat_transects, tmp_path):
        """Test saving and loading cube format data."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from scripts.processing.extract_transects import convert_flat_to_cube, save_cube

        cube = convert_flat_to_cube(flat_transects, n_points=128)

        # Save cube
        output_path = tmp_path / "test_cube.npz"
        save_cube(cube, output_path)

        assert output_path.exists()

        # Load and verify
        loaded = np.load(output_path, allow_pickle=True)

        np.testing.assert_array_equal(loaded['points'], cube['points'])
        np.testing.assert_array_equal(loaded['distances'], cube['distances'])
        np.testing.assert_array_equal(loaded['metadata'], cube['metadata'])
        np.testing.assert_array_equal(loaded['transect_ids'], cube['transect_ids'])


class TestSubsetTransects:
    """Test transect subsetting by MOP range."""

    def test_parse_mop_number_string(self):
        """Test parsing MOP number from string formats."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from scripts.processing.subset_transects import parse_mop_number

        assert parse_mop_number("MOP 595") == 595
        assert parse_mop_number("MOP 600_01") == 600
        assert parse_mop_number("MOP600") == 600
        assert parse_mop_number("595") == 595

    def test_parse_mop_number_numeric(self):
        """Test parsing MOP number from numeric types."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from scripts.processing.subset_transects import parse_mop_number

        assert parse_mop_number(595) == 595
        assert parse_mop_number(595.0) == 595

    def test_subset_cube(self, extractor, tmp_path):
        """Test subsetting cube by MOP range."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from scripts.processing.subset_transects import subset_cube

        # Create synthetic cube with known MOP IDs
        n_transects = 10
        n_epochs = 2
        n_points = 128
        n_features = 12

        cube = {
            'points': np.random.randn(n_transects, n_epochs, n_points, n_features).astype(np.float32),
            'distances': np.random.randn(n_transects, n_epochs, n_points).astype(np.float32),
            'metadata': np.random.randn(n_transects, n_epochs, 12).astype(np.float32),
            'transect_ids': np.array([f'MOP {590 + i}' for i in range(n_transects)], dtype=object),
            'epoch_names': np.array(['2018.las', '2019.las'], dtype=object),
            'feature_names': extractor.FEATURE_NAMES,
        }

        # Subset to MOP 595-597 (should get 3 transects: MOP 595, 596, 597)
        subset = subset_cube(cube, mop_min=595, mop_max=597)

        assert subset is not None
        assert len(subset['transect_ids']) == 3
        assert subset['points'].shape[0] == 3
        assert 'MOP 595' in subset['transect_ids']
        assert 'MOP 597' in subset['transect_ids']
        assert 'MOP 590' not in subset['transect_ids']


class TestPathConversion:
    """Test cross-platform path conversion."""

    def test_mac_to_linux_conversion(self):
        """Test converting Mac paths to Linux."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from scripts.processing.extract_transects import convert_path_for_os

        mac_path = "/Volumes/group/lidar/2018/scan.las"
        linux_path = convert_path_for_os(mac_path, target_os='linux')

        assert linux_path == "/project/group/lidar/2018/scan.las"

    def test_linux_to_mac_conversion(self):
        """Test converting Linux paths to Mac."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from scripts.processing.extract_transects import convert_path_for_os

        linux_path = "/project/group/lidar/2018/scan.las"
        mac_path = convert_path_for_os(linux_path, target_os='mac')

        assert mac_path == "/Volumes/group/lidar/2018/scan.las"

    def test_same_os_no_conversion(self):
        """Test that paths are unchanged when already correct OS."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from scripts.processing.extract_transects import convert_path_for_os

        mac_path = "/Volumes/group/lidar/scan.las"
        result = convert_path_for_os(mac_path, target_os='mac')
        assert result == mac_path

        linux_path = "/project/group/lidar/scan.las"
        result = convert_path_for_os(linux_path, target_os='linux')
        assert result == linux_path

    def test_unknown_path_unchanged(self):
        """Test that unknown path formats are unchanged."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from scripts.processing.extract_transects import convert_path_for_os

        local_path = "/home/user/data/scan.las"
        result = convert_path_for_os(local_path, target_os='linux')
        assert result == local_path

    def test_get_current_os(self):
        """Test OS detection returns valid value."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from scripts.processing.extract_transects import get_current_os

        os_name = get_current_os()
        assert os_name in ['mac', 'linux', 'windows']


class TestDateParsing:
    """Test date parsing from LAS filenames."""

    def test_parse_date_yyyymmdd_prefix(self):
        """Test parsing YYYYMMDD at start of filename."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from scripts.processing.extract_transects import parse_date_from_filename
        from datetime import datetime

        date = parse_date_from_filename("20171106_00590_00622_NoWaves.las")
        assert date is not None
        assert date == datetime(2017, 11, 6)

    def test_parse_date_iso_format(self):
        """Test parsing YYYY-MM-DD format."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from scripts.processing.extract_transects import parse_date_from_filename
        from datetime import datetime

        date = parse_date_from_filename("scan_2017-11-06_data.las")
        assert date is not None
        assert date == datetime(2017, 11, 6)

    def test_parse_date_yyyymmdd_anywhere(self):
        """Test parsing YYYYMMDD anywhere in filename."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from scripts.processing.extract_transects import parse_date_from_filename
        from datetime import datetime

        date = parse_date_from_filename("scan_20171106.las")
        assert date is not None
        assert date == datetime(2017, 11, 6)

    def test_parse_date_invalid_returns_none(self):
        """Test that invalid filenames return None."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from scripts.processing.extract_transects import parse_date_from_filename

        date = parse_date_from_filename("no_date_here.las")
        assert date is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
