"""Voxelized transect extraction from LiDAR point clouds.

Extracts shore-normal transects using 1D binning (voxelization) along the profile.
More robust to variable point density than interpolation-based methods.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree

try:
    import laspy
except ImportError:
    laspy = None

from src.utils.logging import get_logger

logger = get_logger(__name__)


class TransectVoxelizer:
    """Extract voxelized transects from LiDAR point clouds.

    Bins points along shore-normal profiles into 1D segments and aggregates
    features within each bin. More robust to variable point density than
    interpolation methods.

    Args:
        bin_size_m: Size of bins along transect in meters (default: 1.0)
        corridor_width_m: Width of extraction corridor perpendicular to transect (default: 2.0)
        max_bins: Maximum number of bins (sequence length) (default: 128)
        min_points_per_bin: Minimum points required for valid bin (default: 3)
        profile_length_m: Maximum profile length from origin (default: 150)

    Example:
        >>> voxelizer = TransectVoxelizer(bin_size_m=1.0, max_bins=128)
        >>> transects = voxelizer.extract_from_file(
        ...     "cliff_scan.laz",
        ...     transect_origins=origins,
        ...     transect_normals=normals
        ... )
    """

    def __init__(
        self,
        bin_size_m: float = 1.0,
        corridor_width_m: float = 2.0,
        max_bins: int = 128,
        min_points_per_bin: int = 3,
        profile_length_m: float = 150.0,
    ):
        if laspy is None:
            raise ImportError(
                "laspy is required for transect extraction. "
                "Install with: pip install laspy[lazrs]"
            )

        self.bin_size = bin_size_m
        self.corridor_width = corridor_width_m
        self.max_bins = max_bins
        self.min_points = min_points_per_bin
        self.profile_length = profile_length_m

        logger.info(
            f"Initialized TransectVoxelizer: {bin_size_m}m bins, "
            f"{max_bins} max bins, {corridor_width_m}m corridor"
        )

    def extract_from_file(
        self,
        las_path: Union[str, Path],
        transect_origins: np.ndarray,
        transect_normals: np.ndarray,
        transect_names: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Extract voxelized transects from a LAS/LAZ file.

        Args:
            las_path: Path to LAS/LAZ point cloud file
            transect_origins: (N, 2) or (N, 3) array of transect start points
            transect_normals: (N, 2) array of shore-normal unit vectors
            transect_names: Optional list of transect names

        Returns:
            Dictionary containing:
                - 'bin_features': (N_transects, n_bins, n_features) voxelized features
                - 'bin_centers': (N_transects, n_bins) distance from origin
                - 'bin_mask': (N_transects, n_bins) boolean mask for valid bins
                - 'metadata': (N_transects, n_meta) transect-level metadata
                - 'names': list of transect names

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

        # Ensure origins are 3D
        if transect_origins.shape[1] == 2:
            # Add z=0 for 2D origins
            transect_origins = np.hstack([
                transect_origins,
                np.zeros((len(transect_origins), 1))
            ])

        # Extract voxelized transects
        transects_data = self._extract_transects(
            xyz, transect_origins, transect_normals, transect_names
        )

        logger.info(f"Extracted {len(transects_data['bin_features'])} voxelized transects")

        return transects_data

    def _extract_transects(
        self,
        xyz: np.ndarray,
        origins: np.ndarray,
        normals: np.ndarray,
        names: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Extract voxelized transects for all origins.

        Args:
            xyz: (M, 3) point cloud
            origins: (N, 3) transect start points
            normals: (N, 2) shore-normal unit vectors
            names: Optional list of transect names

        Returns:
            Dictionary with voxelized transect data
        """
        # Build KDTree for efficient spatial queries
        tree = cKDTree(xyz[:, :2])

        features_list = []
        centers_list = []
        mask_list = []
        metadata_list = []

        n_transects = len(origins)
        if names is None:
            names = [f"Transect_{i:04d}" for i in range(n_transects)]

        for i, (origin, normal, name) in enumerate(zip(origins, normals, names)):
            # Extract single voxelized transect
            transect_data = self._extract_single_transect(
                xyz, tree, origin, normal
            )

            if transect_data is not None:
                features_list.append(transect_data['bin_features'])
                centers_list.append(transect_data['bin_centers'])
                mask_list.append(transect_data['bin_mask'])
                metadata_list.append(transect_data['metadata'])
            else:
                logger.warning(f"Failed to extract {name}")
                # Add empty transect
                features_list.append(np.zeros((self.max_bins, 6)))
                centers_list.append(np.linspace(0, self.profile_length, self.max_bins))
                mask_list.append(np.zeros(self.max_bins, dtype=bool))
                metadata_list.append(np.zeros(7))

        return {
            'bin_features': np.stack(features_list),
            'bin_centers': np.stack(centers_list),
            'bin_mask': np.stack(mask_list),
            'metadata': np.stack(metadata_list),
            'names': names,
        }

    def _extract_single_transect(
        self,
        xyz: np.ndarray,
        tree: cKDTree,
        origin: np.ndarray,
        normal: np.ndarray,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Extract a single voxelized transect.

        Args:
            xyz: Full point cloud
            tree: KDTree for spatial queries
            origin: (3,) transect origin
            normal: (2,) shore-normal unit vector

        Returns:
            Dictionary with voxelized data or None if extraction fails
        """
        # Step 1: Extract corridor points
        corridor_points = self._extract_corridor(xyz, tree, origin[:2], normal)

        if len(corridor_points) < self.min_points:
            return None

        # Step 2: Project to transect coordinates
        relative = corridor_points[:, :2] - origin[:2]
        distances = np.dot(relative, normal)
        elevations = corridor_points[:, 2]

        # Filter to valid range
        valid_mask = (distances >= 0) & (distances <= self.profile_length)
        if valid_mask.sum() < self.min_points:
            return None

        corridor_points = corridor_points[valid_mask]
        distances = distances[valid_mask]
        elevations = elevations[valid_mask]

        # Step 3: Determine number of bins
        max_dist = min(distances.max(), self.profile_length)
        n_bins = min(int(max_dist / self.bin_size), self.max_bins)

        if n_bins < 2:
            return None

        # Step 4: Create bins and assign points
        bin_edges = np.linspace(0, max_dist, n_bins + 1)
        bin_indices = np.digitize(distances, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Step 5: Aggregate features per bin
        bin_features = np.zeros((n_bins, 6))
        bin_mask = np.zeros(n_bins, dtype=bool)

        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() >= self.min_points:
                bin_pts = corridor_points[mask]
                bin_elevs = elevations[mask]
                bin_dists = distances[mask]

                bin_features[i] = self._aggregate_bin(bin_pts, bin_elevs, bin_dists)
                bin_mask[i] = True
            # else: features remain zeros, mask remains False

        # Step 6: Pad or truncate to max_bins
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        if n_bins < self.max_bins:
            # Pad
            pad_length = self.max_bins - n_bins
            bin_features = np.vstack([
                bin_features,
                np.zeros((pad_length, 6))
            ])
            bin_mask = np.concatenate([
                bin_mask,
                np.zeros(pad_length, dtype=bool)
            ])
            # Extend bin centers linearly
            last_center = bin_centers[-1] if len(bin_centers) > 0 else 0
            extra_centers = np.linspace(
                last_center + self.bin_size,
                self.profile_length,
                pad_length
            )
            bin_centers = np.concatenate([bin_centers, extra_centers])

        # Compute metadata
        metadata = self._compute_metadata(bin_features, bin_mask, origin)

        return {
            'bin_features': bin_features,
            'bin_centers': bin_centers,
            'bin_mask': bin_mask,
            'metadata': metadata,
        }

    def _extract_corridor(
        self,
        xyz: np.ndarray,
        tree: cKDTree,
        origin: np.ndarray,
        normal: np.ndarray,
    ) -> np.ndarray:
        """Extract points within corridor around transect line.

        Args:
            xyz: Full point cloud
            tree: KDTree for spatial queries
            origin: (2,) transect origin in xy
            normal: (2,) shore-normal unit vector

        Returns:
            (M, 3) points within corridor
        """
        # Sample query points along transect
        n_query = int(self.profile_length / self.bin_size) + 1
        query_distances = np.linspace(0, self.profile_length, n_query)
        query_points = origin[None, :] + query_distances[:, None] * normal[None, :]

        # Find all points within corridor_width of query points
        nearby_indices = set()
        for qp in query_points:
            indices = tree.query_ball_point(qp, r=self.corridor_width)
            nearby_indices.update(indices)

        if len(nearby_indices) == 0:
            return np.zeros((0, 3))

        nearby_indices = list(nearby_indices)
        candidate_points = xyz[nearby_indices]

        # Filter by perpendicular distance to transect line
        relative = candidate_points[:, :2] - origin
        perp_dist = np.abs(np.cross(relative, normal))

        valid_mask = perp_dist < self.corridor_width
        return candidate_points[valid_mask]

    def _aggregate_bin(
        self,
        points: np.ndarray,
        elevations: np.ndarray,
        distances: np.ndarray,
    ) -> np.ndarray:
        """Aggregate point features within a bin.

        Args:
            points: (M, 3) points in bin
            elevations: (M,) elevations
            distances: (M,) distances along transect

        Returns:
            (6,) feature vector:
                [mean_elevation, roughness, height_range, slope, curvature, point_density]
        """
        features = np.zeros(6)

        # Feature 0: Mean elevation
        features[0] = np.mean(elevations)

        # Feature 1: Roughness (elevation std)
        features[1] = np.std(elevations) if len(elevations) > 1 else 0.0

        # Feature 2: Height range
        features[2] = np.max(elevations) - np.min(elevations)

        # Feature 3: Slope (mean gradient)
        if len(distances) > 1 and np.ptp(distances) > 1e-6:
            # Fit line to elevation vs distance
            coeffs = np.polyfit(distances, elevations, 1)
            slope_rad = np.arctan(coeffs[0])
            features[3] = np.degrees(slope_rad)
        else:
            features[3] = 0.0

        # Feature 4: Curvature (approximated from elevation variance)
        # For proper curvature, need neighboring bins - approximate here
        if len(elevations) > 2:
            # Use second moment of elevation deviations
            elev_centered = elevations - np.mean(elevations)
            features[4] = np.mean(elev_centered ** 2) / (self.bin_size ** 2 + 1e-8)
        else:
            features[4] = 0.0

        # Feature 5: Point density (normalized)
        # Number of points per cubic meter (bin_size × corridor_width × height_range)
        volume = self.bin_size * self.corridor_width * (features[2] + 0.1)
        features[5] = len(points) / volume

        return features

    def _compute_metadata(
        self,
        bin_features: np.ndarray,
        bin_mask: np.ndarray,
        origin: np.ndarray,
    ) -> np.ndarray:
        """Compute transect-level metadata.

        Args:
            bin_features: (n_bins, 6) voxelized features
            bin_mask: (n_bins,) valid bin mask
            origin: (3,) transect origin

        Returns:
            (7,) metadata: [cliff_height, mean_slope, max_slope, toe_elevation,
                           orientation, lat, lon]
        """
        metadata = np.zeros(7)

        # Only use valid bins
        valid_features = bin_features[bin_mask] if bin_mask.any() else bin_features[:1]

        if len(valid_features) == 0:
            return metadata

        # Cliff height (range of mean elevations)
        mean_elevations = valid_features[:, 0]
        metadata[0] = np.max(mean_elevations) - np.min(mean_elevations)

        # Mean and max slope
        slopes = valid_features[:, 3]
        metadata[1] = np.mean(np.abs(slopes))
        metadata[2] = np.max(np.abs(slopes))

        # Toe elevation (first valid bin)
        metadata[3] = valid_features[0, 0] if len(valid_features) > 0 else 0.0

        # Orientation (placeholder - would need normal vector azimuth)
        metadata[4] = 0.0

        # Lat/lon (use origin xy as pseudo-coordinates)
        metadata[5] = origin[1]  # y
        metadata[6] = origin[0]  # x

        return metadata

    def save_transects(
        self,
        transects: Dict[str, np.ndarray],
        output_path: Union[str, Path],
    ) -> None:
        """Save voxelized transects to file.

        Args:
            transects: Dictionary from extract_from_file()
            output_path: Output .npz file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'bin_features': transects['bin_features'],
            'bin_centers': transects['bin_centers'],
            'bin_mask': transects['bin_mask'],
            'metadata': transects['metadata'],
            'names': np.array(transects['names']),
        }

        np.savez_compressed(output_path, **save_dict)
        logger.info(f"Saved voxelized transects to {output_path}")
