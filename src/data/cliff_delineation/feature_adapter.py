"""
Feature adapter to transform transect-transformer features to CliffDelineaTool format.

transect-transformer: 12 features per point
    [distance_m, elevation_m, slope_deg, curvature, roughness,
     intensity, red, green, blue, classification, return_number, num_returns]

CliffDelineaTool v2.0: 13 features per point
    [elevation_normalized, distance_normalized, gradient_norm, curvature_norm,
     seaward_slope_norm, landward_slope_norm, trendline_dev_norm, slope_change_norm,
     convexity, rel_elevation, low_elevation_zone, shore_proximity, max_local_slope_norm]
"""

import math
import numpy as np
from typing import Optional


class CliffFeatureAdapter:
    """Transforms transect-transformer point features to CliffDelineaTool format."""

    # Feature indices for transect-transformer format
    TT_DISTANCE_IDX = 0
    TT_ELEVATION_IDX = 1
    TT_SLOPE_IDX = 2
    TT_CURVATURE_IDX = 3
    TT_ROUGHNESS_IDX = 4

    # Output feature indices for CliffDelineaTool
    CDT_ELEV_NORM_IDX = 0
    CDT_DIST_NORM_IDX = 1
    CDT_GRADIENT_NORM_IDX = 2
    CDT_CURV_NORM_IDX = 3
    CDT_SEAWARD_SLOPE_IDX = 4
    CDT_LANDWARD_SLOPE_IDX = 5
    CDT_TRENDLINE_DEV_IDX = 6
    CDT_SLOPE_CHANGE_IDX = 7
    CDT_CONVEXITY_IDX = 8
    CDT_REL_ELEV_IDX = 9
    CDT_LOW_ELEV_ZONE_IDX = 10
    CDT_SHORE_PROX_IDX = 11
    CDT_MAX_LOCAL_SLOPE_IDX = 12

    N_OUTPUT_FEATURES = 13

    def __init__(self, n_vert: int = 20, low_elevation_threshold: float = 15.0):
        """
        Initialize the feature adapter.

        Args:
            n_vert: Window size for local slope calculation. Must match
                   CliffDelineaTool training config (default: 20).
            low_elevation_threshold: Elevation threshold for low_elevation_zone
                                    feature (default: 15.0 meters).
        """
        self.n_vert = n_vert
        self.low_elevation_threshold = low_elevation_threshold

    def transform(
        self,
        features: np.ndarray,
        distances: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Transform transect-transformer features to CliffDelineaTool format.

        Args:
            features: (N, 12) array of point features from transect-transformer.
                     Feature order: [distance_m, elevation_m, slope_deg, curvature,
                                    roughness, intensity, r, g, b, class, ret, numret]
            distances: (N,) array of distances along transect. If None, uses
                      features[:, 0] (distance_m column).

        Returns:
            (N, 13) array of features in CliffDelineaTool format.
        """
        # Extract elevation and distance
        elevation = features[:, self.TT_ELEVATION_IDX].astype(np.float64)

        if distances is not None:
            distance = distances.astype(np.float64)
        else:
            distance = features[:, self.TT_DISTANCE_IDX].astype(np.float64)

        seq_len = len(elevation)
        output = np.zeros((seq_len, self.N_OUTPUT_FEATURES), dtype=np.float32)

        # --- Feature 0: Normalized elevation [0,1] ---
        elev_min, elev_max = elevation.min(), elevation.max()
        elev_range = elev_max - elev_min
        if elev_range > 1e-8:
            output[:, self.CDT_ELEV_NORM_IDX] = (elevation - elev_min) / elev_range
        # else: remains 0

        # --- Feature 1: Normalized distance [0,1] ---
        dist_max = distance.max()
        if dist_max > 1e-8:
            output[:, self.CDT_DIST_NORM_IDX] = distance / dist_max
        # else: remains 0

        # --- Feature 2: Gradient (first derivative), std-normalized ---
        # Use numpy gradient for numerical differentiation
        gradient = np.gradient(elevation, distance)
        grad_std = gradient.std()
        if grad_std > 1e-8:
            output[:, self.CDT_GRADIENT_NORM_IDX] = gradient / grad_std
        # else: remains 0

        # --- Feature 3: Curvature (second derivative), std-normalized ---
        curvature = np.gradient(gradient, distance)
        curv_std = curvature.std()
        if curv_std > 1e-8:
            output[:, self.CDT_CURV_NORM_IDX] = curvature / curv_std
        # else: remains 0

        # --- Features 4-5: Seaward/Landward slopes ---
        seaward_slope = self._compute_local_slope(elevation, distance, "seaward")
        landward_slope = self._compute_local_slope(elevation, distance, "landward")
        output[:, self.CDT_SEAWARD_SLOPE_IDX] = seaward_slope / 90.0
        output[:, self.CDT_LANDWARD_SLOPE_IDX] = landward_slope / 90.0

        # --- Feature 6: Trendline deviation ---
        trendline = np.linspace(elevation[0], elevation[-1], seq_len)
        trendline_dev = elevation - trendline
        if elev_range > 1e-8:
            output[:, self.CDT_TRENDLINE_DEV_IDX] = trendline_dev / elev_range
        # else: remains 0

        # --- Feature 7: Slope change (landward - seaward) ---
        output[:, self.CDT_SLOPE_CHANGE_IDX] = (landward_slope - seaward_slope) / 90.0

        # --- Feature 8: Convexity index (signed curvature formula) ---
        # κ = z'' / (1 + z'^2)^(3/2)
        denominator = np.power(1 + gradient**2, 1.5)
        output[:, self.CDT_CONVEXITY_IDX] = np.where(
            denominator > 1e-8, curvature / denominator, 0.0
        )

        # --- Feature 9: Relative elevation (z-score) ---
        elev_mean = elevation.mean()
        elev_std = elevation.std()
        if elev_std > 1e-8:
            output[:, self.CDT_REL_ELEV_IDX] = (elevation - elev_mean) / elev_std
        # else: remains 0

        # --- Feature 10: Low elevation zone (binary) ---
        output[:, self.CDT_LOW_ELEV_ZONE_IDX] = (
            elevation < self.low_elevation_threshold
        ).astype(np.float32)

        # --- Feature 11: Shore proximity (exponential decay) ---
        # High near shore (distance=0), decays exponentially inland
        dist_norm = output[:, self.CDT_DIST_NORM_IDX]
        output[:, self.CDT_SHORE_PROX_IDX] = np.exp(-5.0 * dist_norm)

        # --- Feature 12: Max local slope (5-point window) ---
        window = 5
        half_w = window // 2
        max_local_slope = np.zeros(seq_len)
        for i in range(seq_len):
            start = max(0, i - half_w)
            end = min(seq_len, i + half_w + 1)
            max_local_slope[i] = np.abs(gradient[start:end]).max()

        slope_min = max_local_slope.min()
        slope_range = max_local_slope.max() - slope_min
        if slope_range > 1e-8:
            output[:, self.CDT_MAX_LOCAL_SLOPE_IDX] = (
                max_local_slope - slope_min
            ) / slope_range
        # else: remains 0

        return output

    def _compute_local_slope(
        self, elevations: np.ndarray, distances: np.ndarray, direction: str
    ) -> np.ndarray:
        """
        Compute local slope to n_vert points in given direction.

        Matches the v1.0 CliffDelineaTool implementation.

        Args:
            elevations: Array of elevation values [seq_len]
            distances: Array of distance values [seq_len]
            direction: 'seaward' or 'landward'

        Returns:
            Array of slope values in degrees [seq_len]
        """
        seq_len = len(elevations)
        slopes = np.zeros(seq_len)

        for i in range(seq_len):
            if direction == "seaward":
                # Average slope to n_vert seaward points (lower indices)
                start_idx = max(0, i - self.n_vert)
                end_idx = i
            else:  # landward
                # Average slope to n_vert landward points (higher indices)
                start_idx = i
                end_idx = min(seq_len, i + self.n_vert)

            if start_idx == end_idx:
                continue

            elev_diff = elevations[end_idx - 1] - elevations[start_idx]
            dist_diff = distances[end_idx - 1] - distances[start_idx]

            if dist_diff > 1e-8:
                angle = math.degrees(math.atan(elev_diff / dist_diff))
                # Clip negative slopes to 0 (matches v1.0 behavior)
                slopes[i] = max(0.0, angle)

        return slopes

    def transform_batch(
        self,
        features: np.ndarray,
        distances: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Transform batch of transects.

        Args:
            features: (B, N, 12) or (B, T, N, 12) array
            distances: (B, N) or (B, T, N) array. If None, uses distance_m from features.

        Returns:
            Transformed features in same shape but with 13 features instead of 12.
        """
        original_shape = features.shape

        if features.ndim == 4:
            # Cube format: (n_transects, T, N, 12) -> process each transect-epoch
            B, T, N, F = features.shape
            output = np.zeros((B, T, N, self.N_OUTPUT_FEATURES), dtype=np.float32)

            for b in range(B):
                for t in range(T):
                    # Skip if NaN (missing epoch)
                    if np.isnan(features[b, t, 0, 0]):
                        output[b, t, :, :] = np.nan
                        continue

                    dist = distances[b, t] if distances is not None else None
                    output[b, t] = self.transform(features[b, t], dist)

            return output

        elif features.ndim == 3:
            # Flat format: (B, N, 12)
            B, N, F = features.shape
            output = np.zeros((B, N, self.N_OUTPUT_FEATURES), dtype=np.float32)

            for b in range(B):
                dist = distances[b] if distances is not None else None
                output[b] = self.transform(features[b], dist)

            return output

        else:
            raise ValueError(
                f"Expected 3D or 4D array, got shape {original_shape}"
            )
