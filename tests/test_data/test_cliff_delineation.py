"""
Tests for cliff delineation integration.

Tests the feature adapter which transforms transect-transformer features
to CliffDelineaTool format. Model wrapper tests require the actual
CliffDelineaTool package and checkpoint.
"""

import numpy as np
import pytest
from pathlib import Path

from src.data.cliff_delineation.feature_adapter import CliffFeatureAdapter


class TestCliffFeatureAdapter:
    """Tests for CliffFeatureAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with default settings."""
        return CliffFeatureAdapter(n_vert=20)

    @pytest.fixture
    def realistic_transect(self):
        """Create a realistic cliff transect profile."""
        n_points = 128
        distances = np.linspace(0, 100, n_points).astype(np.float32)

        # Create cliff profile: beach -> cliff face -> plateau
        elevation = np.concatenate([
            np.linspace(2, 5, 30),       # Beach (gradual rise)
            np.linspace(5, 35, 58),      # Cliff face (steep)
            np.linspace(35, 38, 40),     # Plateau (gradual)
        ]).astype(np.float32)

        # Create full feature array (12 features)
        features = np.zeros((n_points, 12), dtype=np.float32)
        features[:, 0] = distances       # distance_m
        features[:, 1] = elevation       # elevation_m
        features[:, 2] = np.gradient(elevation, distances) * 45  # slope_deg (approx)
        features[:, 3] = np.gradient(np.gradient(elevation, distances), distances)  # curvature
        features[:, 4] = np.random.rand(n_points) * 0.1  # roughness
        features[:, 5] = np.random.rand(n_points)  # intensity
        features[:, 6:9] = np.random.rand(n_points, 3)  # RGB
        features[:, 9] = 2  # classification (ground)
        features[:, 10] = 1  # return_number
        features[:, 11] = 1  # num_returns

        return features, distances

    def test_transform_output_shape(self, adapter, realistic_transect):
        """Output should be (N, 13)."""
        features, distances = realistic_transect

        output = adapter.transform(features, distances)

        assert output.shape == (128, 13)
        assert output.dtype == np.float32

    def test_normalized_features_in_range(self, adapter, realistic_transect):
        """Normalized features should be in expected ranges."""
        features, distances = realistic_transect

        output = adapter.transform(features, distances)

        # Feature 0: elevation_normalized should be [0, 1]
        assert output[:, 0].min() >= 0
        assert output[:, 0].max() <= 1

        # Feature 1: distance_normalized should be [0, 1]
        assert output[:, 1].min() >= 0
        assert output[:, 1].max() <= 1

        # Feature 10: low_elevation_zone should be binary {0, 1}
        unique_vals = np.unique(output[:, 10])
        assert all(v in [0.0, 1.0] for v in unique_vals)

        # Feature 11: shore_proximity should be (0, 1]
        assert output[:, 11].min() > 0
        assert output[:, 11].max() <= 1

    def test_gradient_and_curvature_normalized(self, adapter, realistic_transect):
        """Gradient and curvature should be std-normalized (mean~0, std~1)."""
        features, distances = realistic_transect

        output = adapter.transform(features, distances)

        # Feature 2: gradient_norm
        gradient_norm = output[:, 2]
        assert abs(gradient_norm.std() - 1.0) < 0.1  # Should be ~1

        # Feature 3: curvature_norm
        curvature_norm = output[:, 3]
        assert abs(curvature_norm.std() - 1.0) < 0.1  # Should be ~1

    def test_slope_features_in_range(self, adapter, realistic_transect):
        """Slope features should be in [0, 1] after /90 normalization."""
        features, distances = realistic_transect

        output = adapter.transform(features, distances)

        # Feature 4: seaward_slope_norm
        assert output[:, 4].min() >= 0
        assert output[:, 4].max() <= 1

        # Feature 5: landward_slope_norm
        assert output[:, 5].min() >= 0
        assert output[:, 5].max() <= 1

    def test_trendline_deviation(self, adapter, realistic_transect):
        """Trendline deviation should sum to approximately zero."""
        features, distances = realistic_transect

        output = adapter.transform(features, distances)

        # Feature 6: trendline_dev_norm
        # For a monotonic profile, deviation should oscillate around 0
        trendline_dev = output[:, 6]
        assert abs(trendline_dev.mean()) < 0.5  # Should be close to 0

    def test_slope_change_symmetric(self, adapter, realistic_transect):
        """Slope change should be in [-1, 1] range."""
        features, distances = realistic_transect

        output = adapter.transform(features, distances)

        # Feature 7: slope_change_norm = (landward - seaward) / 90
        slope_change = output[:, 7]
        assert slope_change.min() >= -1
        assert slope_change.max() <= 1

    def test_relative_elevation_standardized(self, adapter, realistic_transect):
        """Relative elevation should be z-score standardized."""
        features, distances = realistic_transect

        output = adapter.transform(features, distances)

        # Feature 9: rel_elevation (z-score)
        rel_elev = output[:, 9]
        assert abs(rel_elev.mean()) < 0.1  # Mean should be ~0
        assert abs(rel_elev.std() - 1.0) < 0.1  # Std should be ~1

    def test_low_elevation_zone_threshold(self, adapter):
        """Low elevation zone should respect threshold."""
        n_points = 128
        distances = np.linspace(0, 100, n_points).astype(np.float32)

        # Create elevation profile crossing the 15m threshold
        elevation = np.linspace(0, 30, n_points).astype(np.float32)

        features = np.zeros((n_points, 12), dtype=np.float32)
        features[:, 0] = distances
        features[:, 1] = elevation

        output = adapter.transform(features, distances)

        # Feature 10: low_elevation_zone
        low_zone = output[:, 10]

        # Points below 15m should be 1, above should be 0
        threshold_idx = np.searchsorted(elevation, 15.0)
        assert all(low_zone[:threshold_idx] == 1.0)
        assert all(low_zone[threshold_idx:] == 0.0)

    def test_shore_proximity_decay(self, adapter, realistic_transect):
        """Shore proximity should decay exponentially from shore."""
        features, distances = realistic_transect

        output = adapter.transform(features, distances)

        # Feature 11: shore_proximity
        shore_prox = output[:, 11]

        # Should be highest at start (shore), lowest at end
        assert shore_prox[0] > shore_prox[-1]

        # Should be monotonically decreasing (mostly)
        # Allow small numerical variations
        assert shore_prox[0] > 0.9  # Should be ~1 at shore
        assert shore_prox[-1] < 0.01  # Should be ~0 at end

    def test_transform_with_none_distances(self, adapter, realistic_transect):
        """Should use distance_m from features if distances is None."""
        features, distances = realistic_transect

        output1 = adapter.transform(features, distances)
        output2 = adapter.transform(features, None)  # Use features[:, 0]

        np.testing.assert_array_almost_equal(output1, output2)

    def test_batch_transform_3d(self, adapter, realistic_transect):
        """Batch transform should handle (B, N, 12) format."""
        features_single, distances_single = realistic_transect

        # Create batch of 5 transects
        B = 5
        features = np.stack([features_single] * B)
        distances = np.stack([distances_single] * B)

        output = adapter.transform_batch(features, distances)

        assert output.shape == (B, 128, 13)

        # All outputs should be identical since inputs are identical
        for i in range(1, B):
            np.testing.assert_array_almost_equal(output[0], output[i])

    def test_batch_transform_4d_cube(self, adapter, realistic_transect):
        """Batch transform should handle (B, T, N, 12) cube format."""
        features_single, distances_single = realistic_transect

        # Create cube: 3 transects x 4 epochs
        B, T = 3, 4
        features = np.stack([[features_single] * T] * B)
        distances = np.stack([[distances_single] * T] * B)

        output = adapter.transform_batch(features, distances)

        assert output.shape == (B, T, 128, 13)

    def test_batch_transform_handles_nan(self, adapter, realistic_transect):
        """Batch transform should handle NaN (missing epochs)."""
        features_single, distances_single = realistic_transect

        # Create cube with some NaN epochs
        B, T = 2, 3
        features = np.stack([[features_single] * T] * B)
        distances = np.stack([[distances_single] * T] * B)

        # Set one epoch to NaN
        features[0, 1, :, :] = np.nan
        distances[0, 1, :] = np.nan

        output = adapter.transform_batch(features, distances)

        assert output.shape == (B, T, 128, 13)
        # Valid epochs should have valid output
        assert not np.isnan(output[0, 0]).any()
        # NaN epoch should produce NaN output
        assert np.isnan(output[0, 1]).all()

    def test_custom_n_vert(self):
        """Different n_vert should affect slope calculations."""
        adapter_small = CliffFeatureAdapter(n_vert=5)
        adapter_large = CliffFeatureAdapter(n_vert=40)

        n_points = 128
        distances = np.linspace(0, 100, n_points).astype(np.float32)

        # Use a non-linear profile where n_vert will make a difference
        # A cliff profile with varying slopes
        elevation = np.concatenate([
            np.linspace(2, 5, 30),       # Gentle beach
            np.linspace(5, 35, 58),      # Steep cliff
            np.linspace(35, 38, 40),     # Gentle plateau
        ]).astype(np.float32)

        features = np.zeros((n_points, 12), dtype=np.float32)
        features[:, 0] = distances
        features[:, 1] = elevation

        output_small = adapter_small.transform(features, distances)
        output_large = adapter_large.transform(features, distances)

        # At the cliff-plateau transition, slopes should differ
        # Larger window smooths the transition more
        # Check at points near the cliff-plateau boundary (around index 88)
        transition_region = slice(80, 100)
        diff = np.abs(output_small[transition_region, 4] - output_large[transition_region, 4])
        # There should be at least some difference in this region
        assert diff.max() > 0.01, "n_vert should affect slope calculation at transitions"

    def test_custom_low_elevation_threshold(self):
        """Custom low_elevation_threshold should be respected."""
        adapter = CliffFeatureAdapter(low_elevation_threshold=10.0)

        n_points = 100
        distances = np.linspace(0, 50, n_points).astype(np.float32)
        elevation = np.linspace(0, 20, n_points).astype(np.float32)

        features = np.zeros((n_points, 12), dtype=np.float32)
        features[:, 0] = distances
        features[:, 1] = elevation

        output = adapter.transform(features, distances)

        # Feature 10: low_elevation_zone with 10m threshold
        low_zone = output[:, 10]
        threshold_idx = np.searchsorted(elevation, 10.0)

        assert all(low_zone[:threshold_idx] == 1.0)
        assert all(low_zone[threshold_idx:] == 0.0)

    def test_flat_profile(self, adapter):
        """Should handle flat elevation profile without errors."""
        n_points = 128
        distances = np.linspace(0, 100, n_points).astype(np.float32)
        elevation = np.full(n_points, 10.0, dtype=np.float32)

        features = np.zeros((n_points, 12), dtype=np.float32)
        features[:, 0] = distances
        features[:, 1] = elevation

        output = adapter.transform(features, distances)

        assert output.shape == (128, 13)
        assert not np.isnan(output).any()
        assert not np.isinf(output).any()

        # Elevation normalized should be 0 (no range)
        assert all(output[:, 0] == 0)

        # Gradient and curvature should be 0
        assert all(output[:, 2] == 0)
        assert all(output[:, 3] == 0)

    def test_single_point_transect(self, adapter):
        """Should handle edge case of very short transect."""
        features = np.zeros((2, 12), dtype=np.float32)
        features[:, 0] = [0, 1]  # distances
        features[:, 1] = [0, 10]  # elevations

        output = adapter.transform(features, features[:, 0])

        assert output.shape == (2, 13)
        assert not np.isnan(output).any()


class TestCliffDelineationDetector:
    """Tests for detector module functions."""

    def test_get_cliff_metrics_empty(self):
        """Should handle empty results."""
        from src.data.cliff_delineation.detector import get_cliff_metrics

        results = {
            "toe_distances": np.array([-1.0, -1.0]),
            "top_distances": np.array([-1.0, -1.0]),
            "toe_confidences": np.array([0.0, 0.0]),
            "top_confidences": np.array([0.0, 0.0]),
            "has_cliff": np.array([False, False]),
        }

        metrics = get_cliff_metrics(results)

        assert metrics["n_valid"] == 2
        assert metrics["n_detected"] == 0
        assert metrics["detection_rate"] == 0.0

    def test_get_cliff_metrics_with_detections(self):
        """Should compute correct metrics."""
        from src.data.cliff_delineation.detector import get_cliff_metrics

        results = {
            "toe_distances": np.array([10.0, 15.0, -1.0]),
            "top_distances": np.array([50.0, 55.0, -1.0]),
            "toe_confidences": np.array([0.8, 0.9, 0.0]),
            "top_confidences": np.array([0.85, 0.75, 0.0]),
            "has_cliff": np.array([True, True, False]),
        }

        metrics = get_cliff_metrics(results)

        assert metrics["n_valid"] == 3
        assert metrics["n_detected"] == 2
        assert metrics["detection_rate"] == pytest.approx(2/3)
        assert metrics["mean_cliff_width_m"] == pytest.approx(40.0)  # (40 + 40) / 2
        assert metrics["mean_toe_confidence"] == pytest.approx(0.85)
        assert metrics["mean_top_confidence"] == pytest.approx(0.8)

    def test_get_cliff_metrics_cube_format(self):
        """Should handle multi-epoch cube format."""
        from src.data.cliff_delineation.detector import get_cliff_metrics

        # 2 transects x 3 epochs
        results = {
            "toe_distances": np.array([[10.0, 11.0, -1.0], [15.0, -1.0, 16.0]]),
            "top_distances": np.array([[50.0, 51.0, -1.0], [55.0, -1.0, 56.0]]),
            "toe_confidences": np.array([[0.8, 0.7, 0.0], [0.9, 0.0, 0.85]]),
            "top_confidences": np.array([[0.85, 0.75, 0.0], [0.8, 0.0, 0.9]]),
            "has_cliff": np.array([[True, True, False], [True, False, True]]),
        }

        metrics = get_cliff_metrics(results)

        assert metrics["n_valid"] == 6
        assert metrics["n_detected"] == 4
        assert metrics["detection_rate"] == pytest.approx(4/6)
