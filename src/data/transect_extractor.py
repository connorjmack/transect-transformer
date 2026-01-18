"""Transect extraction from LiDAR point clouds.

Extracts shore-normal 2D profiles from 3D coastal cliff point clouds.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree

try:
    import laspy
except ImportError:
    laspy = None

from src.utils.logging import get_logger

logger = get_logger(__name__)


class TransectExtractor:
    """Extract and process transects from LiDAR point clouds.

    Extracts shore-normal 2D profiles from 3D point clouds, resamples to
    fixed number of points, and computes geometric features.

    Args:
        n_points: Number of points to resample each transect to (default: 128)
        spacing_m: Alongshore spacing between transects in meters (default: 10)
        profile_length_m: Maximum profile length from toe (default: 150)
        min_points: Minimum points required for valid transect (default: 20)
        search_radius_m: Radius for local neighborhood queries (default: 2.0)

    Example:
        >>> extractor = TransectExtractor(n_points=128, spacing_m=10)
        >>> transects = extractor.extract_from_file("cliff_scan.laz")
        >>> print(f"Extracted {len(transects)} transects")
    """

    def __init__(
        self,
        n_points: int = 128,
        spacing_m: float = 10.0,
        profile_length_m: float = 150.0,
        min_points: int = 20,
        search_radius_m: float = 2.0,
    ):
        if laspy is None:
            raise ImportError(
                "laspy is required for transect extraction. "
                "Install with: pip install laspy[lazrs]"
            )

        self.n_points = n_points
        self.spacing_m = spacing_m
        self.profile_length_m = profile_length_m
        self.min_points = min_points
        self.search_radius_m = search_radius_m

        logger.info(
            f"Initialized TransectExtractor: {n_points} points, "
            f"{spacing_m}m spacing, {profile_length_m}m length"
        )

    def extract_from_file(
        self,
        las_path: Union[str, Path],
        coastline_points: Optional[np.ndarray] = None,
        coastline_normals: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Extract transects from a LAS/LAZ file.

        Args:
            las_path: Path to LAS/LAZ point cloud file
            coastline_points: (N_coast, 2) array of coastline xy positions
                If None, attempts to detect cliff toe automatically
            coastline_normals: (N_coast, 2) array of shore-normal unit vectors
                If None, computed from coastline geometry

        Returns:
            Dictionary containing:
                - 'points': (N_transects, n_points, n_features) transect features
                - 'distances': (N_transects, n_points) distances from toe
                - 'metadata': (N_transects, n_meta) transect-level metadata
                - 'positions': (N_transects, 2) transect origin positions (x, y)
                - 'normals': (N_transects, 2) shore-normal directions

        Raises:
            FileNotFoundError: If LAS file doesn't exist
            ValueError: If point cloud is empty or invalid
        """
        las_path = Path(las_path)
        if not las_path.exists():
            raise FileNotFoundError(f"LAS file not found: {las_path}")

        logger.info(f"Loading point cloud from {las_path}")

        # Load point cloud
        las = laspy.read(las_path)
        xyz = np.vstack([las.x, las.y, las.z]).T

        if len(xyz) == 0:
            raise ValueError(f"Empty point cloud: {las_path}")

        logger.info(f"Loaded {len(xyz):,} points")

        # Extract transects
        if coastline_points is None:
            logger.info("No coastline provided, detecting cliff toe...")
            coastline_points, coastline_normals = self._detect_coastline(xyz)

        if coastline_normals is None:
            logger.info("Computing shore-normal directions...")
            coastline_normals = self._compute_normals(coastline_points)

        # Extract profiles along coastline
        transects_data = self._extract_transects(
            xyz, coastline_points, coastline_normals
        )

        logger.info(f"Extracted {len(transects_data['points'])} transects")

        return transects_data

    def _detect_coastline(
        self, xyz: np.ndarray, quantile: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Automatically detect cliff toe from point cloud.

        Uses elevation quantile to identify approximate cliff base.

        Args:
            xyz: (N, 3) point cloud
            quantile: Elevation quantile for toe detection (default: 0.05)

        Returns:
            Tuple of (coastline_points, coastline_normals)
        """
        # Find low points (cliff toe approximation)
        toe_z = np.quantile(xyz[:, 2], quantile)
        toe_mask = xyz[:, 2] < (toe_z + 2.0)  # Within 2m of toe
        toe_points = xyz[toe_mask]

        # Sort by alongshore position (assume coast is roughly E-W or N-S)
        # Use dominant horizontal direction
        xy_range = np.ptp(toe_points[:, :2], axis=0)
        alongshore_axis = 0 if xy_range[0] > xy_range[1] else 1

        sort_idx = np.argsort(toe_points[:, alongshore_axis])
        sorted_toe = toe_points[sort_idx]

        # Sample points at spacing intervals
        total_length = sorted_toe[-1, alongshore_axis] - sorted_toe[0, alongshore_axis]
        n_samples = max(int(total_length / self.spacing_m), 2)

        indices = np.linspace(0, len(sorted_toe) - 1, n_samples, dtype=int)
        coastline_points = sorted_toe[indices, :2]

        logger.debug(f"Detected coastline with {len(coastline_points)} points")

        # Compute normals (will be computed in _compute_normals)
        return coastline_points, None

    def _compute_normals(self, coastline_points: np.ndarray) -> np.ndarray:
        """Compute shore-normal unit vectors from coastline geometry.

        Args:
            coastline_points: (N, 2) xy positions along coastline

        Returns:
            (N, 2) shore-normal unit vectors (perpendicular to coastline)
        """
        n_points = len(coastline_points)
        normals = np.zeros((n_points, 2))

        for i in range(n_points):
            # Compute tangent vector using neighbors
            if i == 0:
                tangent = coastline_points[i + 1] - coastline_points[i]
            elif i == n_points - 1:
                tangent = coastline_points[i] - coastline_points[i - 1]
            else:
                tangent = coastline_points[i + 1] - coastline_points[i - 1]

            # Normalize tangent
            tangent = tangent / (np.linalg.norm(tangent) + 1e-8)

            # Normal is perpendicular to tangent (rotate 90 degrees)
            # Choose orientation pointing landward (positive y or positive x)
            normal = np.array([-tangent[1], tangent[0]])

            # Ensure consistent orientation (pointing inland/upland)
            # This is a heuristic - may need adjustment based on your data
            normals[i] = normal

        return normals

    def _extract_transects(
        self,
        xyz: np.ndarray,
        coastline_points: np.ndarray,
        coastline_normals: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Extract transects along coastline.

        Args:
            xyz: (N, 3) point cloud
            coastline_points: (M, 2) coastline xy positions
            coastline_normals: (M, 2) shore-normal directions

        Returns:
            Dictionary with transect data
        """
        # Build KDTree for efficient neighbor queries
        tree = cKDTree(xyz[:, :2])

        transects_list = []
        distances_list = []
        metadata_list = []
        positions_list = []
        normals_list = []

        for i, (origin, normal) in enumerate(zip(coastline_points, coastline_normals)):
            # Extract points along this transect
            transect_data = self._extract_single_transect(
                xyz, tree, origin, normal
            )

            if transect_data is not None:
                transects_list.append(transect_data['features'])
                distances_list.append(transect_data['distances'])
                metadata_list.append(transect_data['metadata'])
                positions_list.append(origin)
                normals_list.append(normal)

        if len(transects_list) == 0:
            logger.warning("No valid transects extracted")
            return {
                'points': np.zeros((0, self.n_points, 5)),
                'distances': np.zeros((0, self.n_points)),
                'metadata': np.zeros((0, 7)),
                'positions': np.zeros((0, 2)),
                'normals': np.zeros((0, 2)),
            }

        # Stack into arrays
        return {
            'points': np.stack(transects_list),
            'distances': np.stack(distances_list),
            'metadata': np.stack(metadata_list),
            'positions': np.stack(positions_list),
            'normals': np.stack(normals_list),
        }

    def _extract_single_transect(
        self,
        xyz: np.ndarray,
        tree: cKDTree,
        origin: np.ndarray,
        normal: np.ndarray,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Extract a single shore-normal transect.

        Args:
            xyz: Full point cloud
            tree: KDTree for spatial queries
            origin: (2,) transect origin (cliff toe position)
            normal: (2,) shore-normal unit vector

        Returns:
            Dictionary with 'features', 'distances', 'metadata' or None if invalid
        """
        # Define transect line from origin in normal direction
        max_dist = self.profile_length_m
        end_point = origin + normal * max_dist

        # Find points near the transect line
        # Query points in a corridor along the transect
        query_points = np.linspace(origin, end_point, 100)
        nearby_indices = set()

        for qp in query_points:
            indices = tree.query_ball_point(qp, r=self.search_radius_m)
            nearby_indices.update(indices)

        if len(nearby_indices) < self.min_points:
            return None

        nearby_indices = list(nearby_indices)
        candidate_points = xyz[nearby_indices]

        # Project points onto transect line
        # Distance along transect
        offsets = candidate_points[:, :2] - origin
        along_dist = np.dot(offsets, normal)

        # Distance perpendicular to transect (for filtering)
        perp_dist = np.abs(np.cross(offsets, normal))

        # Keep only points near the transect line and within bounds
        valid_mask = (along_dist >= 0) & (along_dist <= max_dist) & (perp_dist < self.search_radius_m)

        if valid_mask.sum() < self.min_points:
            return None

        valid_points = candidate_points[valid_mask]
        valid_along_dist = along_dist[valid_mask]

        # Sort by distance along transect
        sort_idx = np.argsort(valid_along_dist)
        sorted_points = valid_points[sort_idx]
        sorted_distances = valid_along_dist[sort_idx]

        # Resample to fixed number of points
        resampled_data = self._resample_transect(
            sorted_points, sorted_distances
        )

        if resampled_data is None:
            return None

        # Compute metadata
        metadata = self._compute_metadata(resampled_data['points'], origin)

        return {
            'features': resampled_data['features'],
            'distances': resampled_data['distances'],
            'metadata': metadata,
        }

    def _resample_transect(
        self,
        points: np.ndarray,
        distances: np.ndarray,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Resample transect to fixed number of points.

        Args:
            points: (M, 3) sorted transect points
            distances: (M,) distances along transect

        Returns:
            Dictionary with resampled data or None if resampling fails
        """
        if len(points) < 2:
            return None

        # Target distances (evenly spaced from 0 to max distance)
        target_distances = np.linspace(
            distances[0], distances[-1], self.n_points
        )

        # Interpolate xyz
        try:
            interp_x = interp1d(distances, points[:, 0], kind='linear', fill_value='extrapolate')
            interp_y = interp1d(distances, points[:, 1], kind='linear', fill_value='extrapolate')
            interp_z = interp1d(distances, points[:, 2], kind='linear', fill_value='extrapolate')

            resampled_x = interp_x(target_distances)
            resampled_y = interp_y(target_distances)
            resampled_z = interp_z(target_distances)

            resampled_points = np.stack([resampled_x, resampled_y, resampled_z], axis=1)
        except Exception as e:
            logger.debug(f"Interpolation failed: {e}")
            return None

        # Compute derived features
        features = self._compute_features(resampled_points, target_distances)

        return {
            'points': resampled_points,
            'distances': target_distances,
            'features': features,
        }

    def _compute_features(
        self,
        points: np.ndarray,
        distances: np.ndarray,
    ) -> np.ndarray:
        """Compute per-point features for transect.

        Args:
            points: (N, 3) resampled transect points
            distances: (N,) distances from toe

        Returns:
            (N, 5) features: [distance, elevation, slope, curvature, roughness]
        """
        n = len(points)
        features = np.zeros((n, 5))

        # Feature 0: Distance from toe
        features[:, 0] = distances

        # Feature 1: Elevation
        features[:, 1] = points[:, 2]

        # Feature 2: Slope (degrees)
        slope = np.zeros(n)
        for i in range(n):
            if i == 0:
                dz = points[i + 1, 2] - points[i, 2]
                dx = distances[i + 1] - distances[i]
            elif i == n - 1:
                dz = points[i, 2] - points[i - 1, 2]
                dx = distances[i] - distances[i - 1]
            else:
                dz = points[i + 1, 2] - points[i - 1, 2]
                dx = distances[i + 1] - distances[i - 1]

            slope[i] = np.degrees(np.arctan2(dz, dx + 1e-8))

        features[:, 2] = slope

        # Feature 3: Curvature (second derivative of elevation)
        curvature = np.zeros(n)
        for i in range(1, n - 1):
            # Finite difference approximation
            d2z = points[i + 1, 2] - 2 * points[i, 2] + points[i - 1, 2]
            dx2 = ((distances[i + 1] - distances[i]) ** 2 + 1e-8)
            curvature[i] = d2z / dx2

        features[:, 3] = curvature

        # Feature 4: Roughness (local elevation std)
        roughness = np.zeros(n)
        window = 5  # Use 5-point window
        for i in range(n):
            i_min = max(0, i - window // 2)
            i_max = min(n, i + window // 2 + 1)
            local_z = points[i_min:i_max, 2]

            if len(local_z) > 1:
                # Fit local plane and compute residuals
                local_dist = distances[i_min:i_max]
                if len(np.unique(local_dist)) > 1:
                    coeffs = np.polyfit(local_dist, local_z, 1)
                    fitted = np.polyval(coeffs, local_dist)
                    residuals = local_z - fitted
                    roughness[i] = np.std(residuals)

        features[:, 4] = roughness

        return features

    def _compute_metadata(
        self,
        points: np.ndarray,
        origin: np.ndarray,
    ) -> np.ndarray:
        """Compute transect-level metadata.

        Args:
            points: (N, 3) transect points
            origin: (2,) transect origin position

        Returns:
            (7,) metadata: [cliff_height, mean_slope, max_slope, toe_elevation,
                           orientation, lat, lon]
        """
        metadata = np.zeros(7)

        # Cliff height
        metadata[0] = points[:, 2].max() - points[:, 2].min()

        # Mean and max slope (computed from features)
        dz = np.diff(points[:, 2])
        dx = np.sqrt(np.sum(np.diff(points[:, :2], axis=0) ** 2, axis=1)) + 1e-8
        slopes = np.degrees(np.arctan2(dz, dx))
        metadata[1] = np.mean(np.abs(slopes))
        metadata[2] = np.max(np.abs(slopes))

        # Toe elevation
        metadata[3] = points[0, 2]

        # Orientation (azimuth) - would need coastline normal
        # For now, use 0 as placeholder
        metadata[4] = 0.0

        # Lat/lon (would need coordinate system info)
        # For now, use xy coordinates
        metadata[5] = origin[1]  # y as pseudo-latitude
        metadata[6] = origin[0]  # x as pseudo-longitude

        return metadata

    def save_transects(
        self,
        transects: Dict[str, np.ndarray],
        output_path: Union[str, Path],
        format: str = 'npz',
    ) -> None:
        """Save extracted transects to file.

        Args:
            transects: Dictionary from extract_from_file()
            output_path: Output file path
            format: 'npz' or 'parquet' (default: 'npz')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'npz':
            np.savez_compressed(output_path, **transects)
            logger.info(f"Saved transects to {output_path}")
        else:
            raise NotImplementedError(f"Format '{format}' not yet supported")

    def load_transects(
        self,
        input_path: Union[str, Path],
    ) -> Dict[str, np.ndarray]:
        """Load transects from file.

        Args:
            input_path: Path to saved transects file

        Returns:
            Dictionary with transect data
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Transects file not found: {input_path}")

        data = np.load(input_path)
        transects = {key: data[key] for key in data.keys()}

        logger.info(f"Loaded {len(transects['points'])} transects from {input_path}")

        return transects
