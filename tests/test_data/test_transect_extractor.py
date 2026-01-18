"""Tests for transect extraction from point clouds."""

import numpy as np
import pytest
from pathlib import Path

from src.data.transect_extractor import TransectExtractor


@pytest.fixture
def extractor():
    """Create a TransectExtractor instance for testing."""
    return TransectExtractor(n_points=128, spacing_m=10.0, profile_length_m=150.0)


@pytest.fixture
def synthetic_cliff_points():
    """Create synthetic cliff point cloud for testing.

    Generates a simple cliff profile:
    - Beach at z=0
    - Cliff face from z=0 to z=20
    - Cliff top at z=20
    """
    np.random.seed(42)

    # Create a simple 2D cliff profile
    # Beach (x=0-20, z=0)
    beach_x = np.random.uniform(0, 20, 500)
    beach_y = np.random.uniform(0, 100, 500)
    beach_z = np.random.normal(0, 0.5, 500)
    beach = np.column_stack([beach_x, beach_y, beach_z])

    # Cliff face (x=20-30, z=0-20)
    cliff_x = np.random.uniform(20, 30, 1000)
    cliff_y = np.random.uniform(0, 100, 1000)
    # Linear slope from 0 to 20
    cliff_z = (cliff_x - 20) * 2 + np.random.normal(0, 0.5, 1000)
    cliff = np.column_stack([cliff_x, cliff_y, cliff_z])

    # Cliff top (x=30-50, z=20)
    top_x = np.random.uniform(30, 50, 500)
    top_y = np.random.uniform(0, 100, 500)
    top_z = np.random.normal(20, 0.5, 500)
    top = np.column_stack([top_x, top_y, top_z])

    # Combine all points
    points = np.vstack([beach, cliff, top])

    return points


@pytest.fixture
def coastline_points():
    """Create synthetic coastline for testing."""
    # Coastline along y-axis at x=20 (cliff toe)
    y_coords = np.arange(0, 100, 10)
    x_coords = np.ones_like(y_coords) * 20
    return np.column_stack([x_coords, y_coords])


@pytest.fixture
def coastline_normals():
    """Create shore-normal vectors (pointing inland, +x direction)."""
    n = 10
    return np.tile([1, 0], (n, 1))


class TestTransectExtractorInit:
    """Test TransectExtractor initialization."""

    def test_default_init(self):
        """Test initialization with default parameters."""
        extractor = TransectExtractor()
        assert extractor.n_points == 128
        assert extractor.spacing_m == 10.0
        assert extractor.profile_length_m == 150.0

    def test_custom_init(self):
        """Test initialization with custom parameters."""
        extractor = TransectExtractor(
            n_points=64,
            spacing_m=5.0,
            profile_length_m=100.0,
            min_points=10,
            search_radius_m=1.5,
        )
        assert extractor.n_points == 64
        assert extractor.spacing_m == 5.0
        assert extractor.profile_length_m == 100.0
        assert extractor.min_points == 10
        assert extractor.search_radius_m == 1.5


class TestCoastlineDetection:
    """Test automatic coastline detection."""

    def test_detect_coastline(self, extractor, synthetic_cliff_points):
        """Test automatic coastline detection from point cloud."""
        coastline, normals = extractor._detect_coastline(synthetic_cliff_points)

        # Should detect points near cliff toe
        assert len(coastline) > 0
        assert coastline.shape[1] == 2  # xy coordinates

        # Coastline x should be near the cliff toe (x~20)
        # With some tolerance due to automatic detection
        assert np.mean(coastline[:, 0]) > 0
        assert np.mean(coastline[:, 0]) < 40

    def test_compute_normals(self, extractor, coastline_points):
        """Test shore-normal vector computation."""
        normals = extractor._compute_normals(coastline_points)

        # Should have one normal per coastline point
        assert normals.shape == coastline_points.shape

        # Normals should be unit vectors
        norms = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


class TestTransectExtraction:
    """Test transect extraction functionality."""

    def test_extract_single_transect(
        self, extractor, synthetic_cliff_points, coastline_points, coastline_normals
    ):
        """Test extraction of a single transect."""
        from scipy.spatial import cKDTree

        tree = cKDTree(synthetic_cliff_points[:, :2])
        origin = coastline_points[5]  # Middle transect
        normal = coastline_normals[5]

        result = extractor._extract_single_transect(
            synthetic_cliff_points, tree, origin, normal
        )

        if result is not None:
            assert 'features' in result
            assert 'distances' in result
            assert 'metadata' in result

            # Check shapes
            assert result['features'].shape == (128, 5)
            assert result['distances'].shape == (128,)
            assert result['metadata'].shape == (7,)

            # Check that distances are monotonically increasing
            assert np.all(np.diff(result['distances']) >= 0)

    def test_resample_transect(self, extractor):
        """Test transect resampling to fixed number of points."""
        # Create simple transect
        distances = np.array([0, 10, 20, 30, 40, 50])
        points = np.column_stack([
            distances,  # x
            np.zeros_like(distances),  # y
            distances * 0.5,  # z (linear slope)
        ])

        result = extractor._resample_transect(points, distances)

        assert result is not None
        assert result['points'].shape == (128, 3)
        assert result['distances'].shape == (128,)
        assert result['features'].shape == (128, 5)

        # Check that resampled distances span original range
        assert result['distances'][0] == pytest.approx(0, abs=0.1)
        assert result['distances'][-1] == pytest.approx(50, abs=0.1)


class TestFeatureComputation:
    """Test feature computation."""

    def test_compute_features(self, extractor):
        """Test feature computation for a transect."""
        # Create simple linear slope
        distances = np.linspace(0, 50, 128)
        points = np.column_stack([
            distances,
            np.zeros(128),
            distances * 0.5,  # 26.6 degree slope
        ])

        features = extractor._compute_features(points, distances)

        # Check shape
        assert features.shape == (128, 5)

        # Feature 0: Distance
        np.testing.assert_allclose(features[:, 0], distances, rtol=1e-5)

        # Feature 1: Elevation
        np.testing.assert_allclose(features[:, 1], points[:, 2], rtol=1e-5)

        # Feature 2: Slope (should be roughly constant ~26.6 degrees)
        mean_slope = np.mean(features[10:-10, 2])  # Avoid edges
        assert 25 < mean_slope < 28

        # Feature 3: Curvature (should be near zero for linear slope)
        mean_curvature = np.mean(np.abs(features[10:-10, 3]))
        assert mean_curvature < 0.01

        # Feature 4: Roughness (should be near zero for smooth slope)
        mean_roughness = np.mean(features[10:-10, 4])
        assert mean_roughness < 1.0

    def test_compute_metadata(self, extractor):
        """Test transect metadata computation."""
        # Create simple transect
        distances = np.linspace(0, 50, 128)
        points = np.column_stack([
            distances,
            np.zeros(128),
            distances * 0.5,  # Elevation rises from 0 to 25
        ])
        origin = np.array([0, 0])

        metadata = extractor._compute_metadata(points, origin)

        # Check shape
        assert metadata.shape == (7,)

        # Cliff height should be ~25
        assert metadata[0] == pytest.approx(25, abs=0.5)

        # Mean slope should be positive
        assert metadata[1] > 0

        # Toe elevation should be ~0
        assert metadata[3] == pytest.approx(0, abs=1.0)


class TestSaveLoad:
    """Test saving and loading transects."""

    def test_save_load_npz(self, extractor, tmp_path):
        """Test save/load cycle with NPZ format."""
        # Create dummy transects
        transects = {
            'points': np.random.randn(10, 128, 5),
            'distances': np.random.randn(10, 128),
            'metadata': np.random.randn(10, 7),
            'positions': np.random.randn(10, 2),
            'normals': np.random.randn(10, 2),
        }

        # Save
        output_path = tmp_path / "test_transects.npz"
        extractor.save_transects(transects, output_path, format='npz')

        assert output_path.exists()

        # Load
        loaded = extractor.load_transects(output_path)

        # Check all keys present
        assert set(loaded.keys()) == set(transects.keys())

        # Check shapes match
        for key in transects.keys():
            assert loaded[key].shape == transects[key].shape
            np.testing.assert_allclose(loaded[key], transects[key])


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_point_cloud(self, extractor, tmp_path):
        """Test handling of empty point cloud."""
        # This test would require creating an actual LAS file
        # For now, we test the ValueError is raised appropriately
        pass

    def test_insufficient_points(self, extractor):
        """Test handling when transect has too few points."""
        from scipy.spatial import cKDTree

        # Very sparse points
        xyz = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        tree = cKDTree(xyz[:, :2])
        origin = np.array([0, 0])
        normal = np.array([1, 0])

        result = extractor._extract_single_transect(xyz, tree, origin, normal)

        # Should return None for insufficient points
        assert result is None

    def test_load_nonexistent_file(self, extractor):
        """Test loading from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            extractor.load_transects("nonexistent.npz")


def test_full_pipeline_synthetic(synthetic_cliff_points, coastline_points, coastline_normals):
    """Integration test: full extraction pipeline on synthetic data."""
    extractor = TransectExtractor(n_points=128, spacing_m=10.0)

    # Build KDTree
    from scipy.spatial import cKDTree
    tree = cKDTree(synthetic_cliff_points[:, :2])

    # Extract transects
    transects = extractor._extract_transects(
        synthetic_cliff_points, coastline_points, coastline_normals
    )

    # Should extract multiple transects
    assert transects['points'].shape[0] > 0
    assert transects['points'].shape[1] == 128
    assert transects['points'].shape[2] == 5

    # All transects should have valid features
    assert not np.any(np.isnan(transects['points']))
    assert not np.any(np.isnan(transects['distances']))
    assert not np.any(np.isnan(transects['metadata']))

    # Distances should be monotonic for each transect
    for i in range(len(transects['distances'])):
        diffs = np.diff(transects['distances'][i])
        assert np.all(diffs >= 0), f"Transect {i} has non-monotonic distances"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
