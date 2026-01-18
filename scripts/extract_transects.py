#!/usr/bin/env python3
"""Extract transects from LiDAR point clouds using predefined transect lines.

This script processes LAS/LAZ files and extracts point data along transect lines
defined in a shapefile. For each transect, points within a buffer distance are
extracted, projected onto the transect line, and resampled to a fixed number of points.

The output includes all available LAS attributes (elevation, intensity, RGB, classification)
along with computed geometric features (slope, curvature, roughness).

Usage:
    python scripts/extract_transects.py \
        --transects data/mops/transects_10m/transect_lines.shp \
        --las-dir data/testing/ \
        --output data/processed/transects.npz \
        --buffer 1.0 \
        --n-points 128
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import geopandas as gpd
    from shapely.geometry import LineString, Point
    from shapely.ops import transform
    import pyproj
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install geopandas shapely pyproj")
    sys.exit(1)

try:
    import laspy
except ImportError:
    print("laspy is required. Install with: pip install laspy[lazrs]")
    sys.exit(1)

from src.utils.logging import setup_logger

logger = setup_logger(__name__, level="INFO")


class ShapefileTransectExtractor:
    """Extract transect data from LAS files using predefined transect lines from a shapefile.

    This extractor takes transect lines defined in a shapefile and extracts point cloud
    data along each transect from LAS files. Points within a buffer distance of each
    transect are collected, projected onto the transect line, and resampled to a
    fixed number of points.

    Attributes for each point include:
        - Point features: distance_m, elevation_m, slope_deg, curvature, roughness
        - LAS attributes: intensity, red, green, blue, classification, return_number

    Args:
        n_points: Number of points to resample each transect to (default: 128)
        buffer_m: Buffer distance around transect line in meters (default: 1.0)
        min_points: Minimum points required for valid transect (default: 20)
    """

    # Feature indices for the output arrays
    FEATURE_NAMES = [
        'distance_m',      # 0: Distance from transect start (toe)
        'elevation_m',     # 1: Elevation (z coordinate)
        'slope_deg',       # 2: Local slope in degrees
        'curvature',       # 3: Profile curvature (1/m)
        'roughness',       # 4: Local surface roughness
        'intensity',       # 5: LAS intensity value (normalized 0-1)
        'red',             # 6: Red channel (normalized 0-1)
        'green',           # 7: Green channel (normalized 0-1)
        'blue',            # 8: Blue channel (normalized 0-1)
        'classification',  # 9: LAS classification code
        'return_number',   # 10: Return number
        'num_returns',     # 11: Number of returns
    ]
    N_FEATURES = len(FEATURE_NAMES)

    # Metadata indices
    METADATA_NAMES = [
        'cliff_height_m',   # 0: Max elevation - min elevation
        'mean_slope_deg',   # 1: Mean absolute slope
        'max_slope_deg',    # 2: Maximum absolute slope
        'toe_elevation_m',  # 3: Elevation at transect start
        'top_elevation_m',  # 4: Elevation at transect end
        'orientation_deg',  # 5: Transect azimuth (degrees from north)
        'transect_length_m',# 6: Actual transect length
        'latitude',         # 7: Start point latitude (or Y coordinate)
        'longitude',        # 8: Start point longitude (or X coordinate)
        'transect_id',      # 9: Original transect ID from shapefile
        'mean_intensity',   # 10: Mean intensity along transect
        'dominant_class',   # 11: Most common classification
    ]
    N_METADATA = len(METADATA_NAMES)

    def __init__(
        self,
        n_points: int = 128,
        buffer_m: float = 1.0,
        min_points: int = 20,
    ):
        self.n_points = n_points
        self.buffer_m = buffer_m
        self.min_points = min_points

        logger.info(
            f"Initialized ShapefileTransectExtractor: {n_points} points, "
            f"{buffer_m}m buffer, min {min_points} points"
        )

    def load_transect_lines(
        self,
        shapefile_path: Path,
    ) -> gpd.GeoDataFrame:
        """Load transect lines from shapefile.

        Args:
            shapefile_path: Path to shapefile containing transect LineStrings

        Returns:
            GeoDataFrame with transect geometries
        """
        gdf = gpd.read_file(shapefile_path)
        logger.info(f"Loaded {len(gdf)} transect lines from {shapefile_path}")
        logger.info(f"CRS: {gdf.crs}")
        logger.info(f"Bounds: {gdf.total_bounds}")
        return gdf

    def load_las_file(
        self,
        las_path: Path,
    ) -> Dict[str, np.ndarray]:
        """Load LAS file and extract all relevant attributes.

        Args:
            las_path: Path to LAS/LAZ file

        Returns:
            Dictionary with point cloud data arrays
        """
        logger.info(f"Loading LAS file: {las_path}")
        las = laspy.read(las_path)

        data = {
            'x': np.array(las.x),
            'y': np.array(las.y),
            'z': np.array(las.z),
            'xyz': np.column_stack([las.x, las.y, las.z]),
        }

        # Extract optional attributes with fallbacks
        if hasattr(las, 'intensity'):
            # Normalize intensity to 0-1 range
            intensity = np.array(las.intensity).astype(np.float32)
            max_intensity = intensity.max() if intensity.max() > 0 else 1
            data['intensity'] = intensity / max_intensity
        else:
            data['intensity'] = np.zeros(len(las.x), dtype=np.float32)

        # RGB colors (normalize from 16-bit to 0-1)
        if hasattr(las, 'red'):
            data['red'] = np.array(las.red).astype(np.float32) / 65535.0
            data['green'] = np.array(las.green).astype(np.float32) / 65535.0
            data['blue'] = np.array(las.blue).astype(np.float32) / 65535.0
        else:
            data['red'] = np.zeros(len(las.x), dtype=np.float32)
            data['green'] = np.zeros(len(las.x), dtype=np.float32)
            data['blue'] = np.zeros(len(las.x), dtype=np.float32)

        # Classification
        if hasattr(las, 'classification'):
            data['classification'] = np.array(las.classification).astype(np.float32)
        else:
            data['classification'] = np.zeros(len(las.x), dtype=np.float32)

        # Return information
        if hasattr(las, 'return_number'):
            data['return_number'] = np.array(las.return_number).astype(np.float32)
        else:
            data['return_number'] = np.ones(len(las.x), dtype=np.float32)

        if hasattr(las, 'number_of_returns'):
            data['num_returns'] = np.array(las.number_of_returns).astype(np.float32)
        else:
            data['num_returns'] = np.ones(len(las.x), dtype=np.float32)

        logger.info(f"Loaded {len(las.x):,} points")
        logger.info(f"  X range: {data['x'].min():.2f} - {data['x'].max():.2f}")
        logger.info(f"  Y range: {data['y'].min():.2f} - {data['y'].max():.2f}")
        logger.info(f"  Z range: {data['z'].min():.2f} - {data['z'].max():.2f}")

        return data

    def get_transect_direction(self, line: LineString) -> Tuple[np.ndarray, np.ndarray, float]:
        """Get transect start point, direction vector, and length.

        Args:
            line: Shapely LineString geometry

        Returns:
            Tuple of (start_point, direction_unit_vector, length)
        """
        coords = np.array(line.coords)
        start = coords[0, :2]  # First point (x, y)
        end = coords[-1, :2]   # Last point (x, y)

        direction = end - start
        length = np.linalg.norm(direction)
        direction_unit = direction / (length + 1e-8)

        return start, direction_unit, length

    def extract_transect_points(
        self,
        line: LineString,
        las_data: Dict[str, np.ndarray],
        tree: cKDTree,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Extract and project points onto a single transect line.

        Args:
            line: Transect LineString geometry
            las_data: Dictionary with point cloud data
            tree: KDTree for spatial queries (built on XY coordinates)

        Returns:
            Dictionary with extracted point data, or None if insufficient points
        """
        start, direction, length = self.get_transect_direction(line)

        # Query points along the transect using multiple sample points
        n_samples = max(int(length / (self.buffer_m / 2)), 10)
        sample_distances = np.linspace(0, length, n_samples)
        sample_points = start + np.outer(sample_distances, direction)

        # Find all points within buffer of any sample point
        nearby_indices = set()
        for sp in sample_points:
            indices = tree.query_ball_point(sp, r=self.buffer_m)
            nearby_indices.update(indices)

        if len(nearby_indices) < self.min_points:
            return None

        nearby_indices = np.array(list(nearby_indices))

        # Get candidate points
        candidate_xy = las_data['xyz'][nearby_indices, :2]

        # Project points onto transect line
        offsets = candidate_xy - start
        along_dist = np.dot(offsets, direction)  # Distance along transect
        # Perpendicular distance: |offset x direction| for 2D vectors
        # cross product of 2D vectors gives scalar: a[0]*b[1] - a[1]*b[0]
        perp_dist = np.abs(offsets[:, 0] * direction[1] - offsets[:, 1] * direction[0])

        # Filter: within buffer and within transect length
        valid_mask = (
            (along_dist >= 0) &
            (along_dist <= length) &
            (perp_dist <= self.buffer_m)
        )

        if valid_mask.sum() < self.min_points:
            return None

        # Get valid point indices
        valid_indices = nearby_indices[valid_mask]
        valid_along_dist = along_dist[valid_mask]

        # Sort by distance along transect
        sort_idx = np.argsort(valid_along_dist)
        sorted_indices = valid_indices[sort_idx]
        sorted_distances = valid_along_dist[sort_idx]

        # Collect all attributes for sorted points
        result = {
            'indices': sorted_indices,
            'distances': sorted_distances,
            'xyz': las_data['xyz'][sorted_indices],
            'intensity': las_data['intensity'][sorted_indices],
            'red': las_data['red'][sorted_indices],
            'green': las_data['green'][sorted_indices],
            'blue': las_data['blue'][sorted_indices],
            'classification': las_data['classification'][sorted_indices],
            'return_number': las_data['return_number'][sorted_indices],
            'num_returns': las_data['num_returns'][sorted_indices],
            'transect_length': length,
            'start': start,
            'direction': direction,
        }

        return result

    def resample_transect(
        self,
        raw_data: Dict[str, np.ndarray],
    ) -> Optional[Dict[str, np.ndarray]]:
        """Resample transect to fixed number of points.

        Args:
            raw_data: Dictionary from extract_transect_points

        Returns:
            Dictionary with resampled features and metadata, or None if failed
        """
        distances = raw_data['distances']
        xyz = raw_data['xyz']

        if len(distances) < 2:
            return None

        # Target distances (evenly spaced)
        target_distances = np.linspace(distances[0], distances[-1], self.n_points)

        # Interpolate all attributes
        try:
            # XYZ interpolation
            interp_x = interp1d(distances, xyz[:, 0], kind='linear', fill_value='extrapolate')
            interp_y = interp1d(distances, xyz[:, 1], kind='linear', fill_value='extrapolate')
            interp_z = interp1d(distances, xyz[:, 2], kind='linear', fill_value='extrapolate')

            resampled_z = interp_z(target_distances)

            # Intensity interpolation
            interp_intensity = interp1d(
                distances, raw_data['intensity'],
                kind='linear', fill_value='extrapolate'
            )
            resampled_intensity = interp_intensity(target_distances)

            # RGB interpolation
            interp_r = interp1d(distances, raw_data['red'], kind='linear', fill_value='extrapolate')
            interp_g = interp1d(distances, raw_data['green'], kind='linear', fill_value='extrapolate')
            interp_b = interp1d(distances, raw_data['blue'], kind='linear', fill_value='extrapolate')

            resampled_r = interp_r(target_distances)
            resampled_g = interp_g(target_distances)
            resampled_b = interp_b(target_distances)

            # Classification - use nearest neighbor
            interp_class = interp1d(
                distances, raw_data['classification'],
                kind='nearest', fill_value='extrapolate'
            )
            resampled_class = interp_class(target_distances)

            # Return info - use nearest neighbor
            interp_ret = interp1d(
                distances, raw_data['return_number'],
                kind='nearest', fill_value='extrapolate'
            )
            resampled_return = interp_ret(target_distances)

            interp_numret = interp1d(
                distances, raw_data['num_returns'],
                kind='nearest', fill_value='extrapolate'
            )
            resampled_numret = interp_numret(target_distances)

        except Exception as e:
            logger.debug(f"Interpolation failed: {e}")
            return None

        # Compute derived features
        n = self.n_points

        # Slope (degrees)
        slope = np.zeros(n)
        for i in range(n):
            if i == 0:
                dz = resampled_z[i + 1] - resampled_z[i]
                dd = target_distances[i + 1] - target_distances[i]
            elif i == n - 1:
                dz = resampled_z[i] - resampled_z[i - 1]
                dd = target_distances[i] - target_distances[i - 1]
            else:
                dz = resampled_z[i + 1] - resampled_z[i - 1]
                dd = target_distances[i + 1] - target_distances[i - 1]
            slope[i] = np.degrees(np.arctan2(dz, dd + 1e-8))

        # Curvature (second derivative)
        curvature = np.zeros(n)
        for i in range(1, n - 1):
            d2z = resampled_z[i + 1] - 2 * resampled_z[i] + resampled_z[i - 1]
            dd2 = ((target_distances[i + 1] - target_distances[i]) ** 2) + 1e-8
            curvature[i] = d2z / dd2

        # Roughness (local elevation std)
        roughness = np.zeros(n)
        window = 5
        for i in range(n):
            i_min = max(0, i - window // 2)
            i_max = min(n, i + window // 2 + 1)
            local_z = resampled_z[i_min:i_max]
            local_d = target_distances[i_min:i_max]

            if len(local_z) > 1 and len(np.unique(local_d)) > 1:
                coeffs = np.polyfit(local_d, local_z, 1)
                fitted = np.polyval(coeffs, local_d)
                residuals = local_z - fitted
                roughness[i] = np.std(residuals)

        # Assemble feature array [n_points, n_features]
        features = np.zeros((self.n_points, self.N_FEATURES), dtype=np.float32)
        features[:, 0] = target_distances        # distance_m
        features[:, 1] = resampled_z             # elevation_m
        features[:, 2] = slope                   # slope_deg
        features[:, 3] = curvature               # curvature
        features[:, 4] = roughness               # roughness
        features[:, 5] = resampled_intensity     # intensity
        features[:, 6] = resampled_r             # red
        features[:, 7] = resampled_g             # green
        features[:, 8] = resampled_b             # blue
        features[:, 9] = resampled_class         # classification
        features[:, 10] = resampled_return       # return_number
        features[:, 11] = resampled_numret       # num_returns

        # Compute metadata
        metadata = np.zeros(self.N_METADATA, dtype=np.float32)
        metadata[0] = resampled_z.max() - resampled_z.min()  # cliff_height_m
        metadata[1] = np.mean(np.abs(slope))                  # mean_slope_deg
        metadata[2] = np.max(np.abs(slope))                   # max_slope_deg
        metadata[3] = resampled_z[0]                          # toe_elevation_m
        metadata[4] = resampled_z[-1]                         # top_elevation_m

        # Orientation (azimuth from north)
        direction = raw_data['direction']
        azimuth = np.degrees(np.arctan2(direction[0], direction[1]))  # E from N
        if azimuth < 0:
            azimuth += 360
        metadata[5] = azimuth                                 # orientation_deg

        metadata[6] = raw_data['transect_length']             # transect_length_m
        metadata[7] = raw_data['start'][1]                    # latitude (Y)
        metadata[8] = raw_data['start'][0]                    # longitude (X)
        # transect_id (9) will be set by caller
        metadata[10] = np.mean(resampled_intensity)           # mean_intensity

        # Dominant classification
        unique, counts = np.unique(resampled_class, return_counts=True)
        metadata[11] = unique[np.argmax(counts)]              # dominant_class

        return {
            'features': features,
            'distances': target_distances.astype(np.float32),
            'metadata': metadata,
        }

    def extract_from_shapefile_and_las(
        self,
        transect_gdf: gpd.GeoDataFrame,
        las_files: List[Path],
        transect_id_col: str = 'tr_id',
    ) -> Dict[str, np.ndarray]:
        """Extract transects from multiple LAS files using shapefile transect lines.

        Args:
            transect_gdf: GeoDataFrame with transect LineString geometries
            las_files: List of paths to LAS/LAZ files
            transect_id_col: Column name for transect IDs

        Returns:
            Dictionary containing:
                - 'points': (N_transects, n_points, n_features) features
                - 'distances': (N_transects, n_points) distances along transect
                - 'metadata': (N_transects, n_meta) transect-level metadata
                - 'transect_ids': (N_transects,) original transect IDs
                - 'las_sources': list of source LAS file names per transect
        """
        all_features = []
        all_distances = []
        all_metadata = []
        all_transect_ids = []
        all_las_sources = []

        for las_path in las_files:
            logger.info(f"\nProcessing {las_path.name}...")

            # Load LAS data
            las_data = self.load_las_file(las_path)

            # Build KDTree on XY coordinates
            tree = cKDTree(las_data['xyz'][:, :2])

            # Get LAS bounds
            las_minx, las_maxx = las_data['x'].min(), las_data['x'].max()
            las_miny, las_maxy = las_data['y'].min(), las_data['y'].max()

            # Filter transects that potentially intersect LAS bounds
            # (with some buffer for transects that start outside but cross into the data)
            buffer = 50  # meters
            transect_bounds = transect_gdf.bounds
            potential_mask = (
                (transect_bounds['minx'] <= las_maxx + buffer) &
                (transect_bounds['maxx'] >= las_minx - buffer) &
                (transect_bounds['miny'] <= las_maxy + buffer) &
                (transect_bounds['maxy'] >= las_miny - buffer)
            )

            candidate_transects = transect_gdf[potential_mask]
            logger.info(f"  {len(candidate_transects)} transects potentially overlap LAS bounds")

            extracted_count = 0
            skipped_count = 0

            for idx, row in candidate_transects.iterrows():
                transect_id = row[transect_id_col] if transect_id_col in row.index else idx
                line = row.geometry

                # Extract raw points
                raw_data = self.extract_transect_points(line, las_data, tree)

                if raw_data is None:
                    skipped_count += 1
                    continue

                # Resample to fixed points
                resampled = self.resample_transect(raw_data)

                if resampled is None:
                    skipped_count += 1
                    continue

                # Set transect ID in metadata
                resampled['metadata'][9] = float(transect_id)

                all_features.append(resampled['features'])
                all_distances.append(resampled['distances'])
                all_metadata.append(resampled['metadata'])
                all_transect_ids.append(transect_id)
                all_las_sources.append(las_path.name)

                extracted_count += 1

            logger.info(f"  Extracted: {extracted_count}, Skipped: {skipped_count}")

        if len(all_features) == 0:
            logger.warning("No transects extracted!")
            return {
                'points': np.zeros((0, self.n_points, self.N_FEATURES), dtype=np.float32),
                'distances': np.zeros((0, self.n_points), dtype=np.float32),
                'metadata': np.zeros((0, self.N_METADATA), dtype=np.float32),
                'transect_ids': np.array([], dtype=np.int64),
                'las_sources': [],
                'feature_names': self.FEATURE_NAMES,
                'metadata_names': self.METADATA_NAMES,
            }

        return {
            'points': np.stack(all_features),
            'distances': np.stack(all_distances),
            'metadata': np.stack(all_metadata),
            'transect_ids': np.array(all_transect_ids, dtype=np.int64),
            'las_sources': all_las_sources,
            'feature_names': self.FEATURE_NAMES,
            'metadata_names': self.METADATA_NAMES,
        }

    def save_transects(
        self,
        transects: Dict[str, np.ndarray],
        output_path: Path,
    ) -> None:
        """Save extracted transects to NPZ file.

        Args:
            transects: Dictionary from extract_from_shapefile_and_las
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert lists to arrays for saving
        save_dict = {}
        for key, value in transects.items():
            if isinstance(value, list):
                # Convert string lists to object arrays
                save_dict[key] = np.array(value, dtype=object)
            else:
                save_dict[key] = value

        np.savez_compressed(output_path, **save_dict)
        logger.info(f"Saved {len(transects['points'])} transects to {output_path}")


def visualize_transects(transects: Dict, output_path: Path, n_samples: int = 10):
    """Create visualization of extracted transects.

    Args:
        transects: Dictionary from extractor
        output_path: Path to save visualization
        n_samples: Number of sample transects to plot
    """
    import matplotlib.pyplot as plt

    n_transects = len(transects['points'])
    n_plot = min(n_samples, n_transects)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Sample random transects
    sample_idx = np.random.choice(n_transects, n_plot, replace=False)
    sample_idx = np.sort(sample_idx)

    # Plot 1: Elevation profiles
    ax = axes[0, 0]
    for i in sample_idx:
        distances = transects['distances'][i]
        elevations = transects['points'][i, :, 1]  # Feature 1 is elevation
        ax.plot(distances, elevations, alpha=0.7, linewidth=1)
    ax.set_xlabel("Distance from Start (m)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title(f"Elevation Profiles ({n_plot} samples)")
    ax.grid(True, alpha=0.3)

    # Plot 2: Slope profiles
    ax = axes[0, 1]
    for i in sample_idx:
        distances = transects['distances'][i]
        slopes = transects['points'][i, :, 2]  # Feature 2 is slope
        ax.plot(distances, slopes, alpha=0.7, linewidth=1)
    ax.set_xlabel("Distance from Start (m)")
    ax.set_ylabel("Slope (degrees)")
    ax.set_title(f"Slope Profiles ({n_plot} samples)")
    ax.grid(True, alpha=0.3)

    # Plot 3: Intensity profiles
    ax = axes[1, 0]
    for i in sample_idx:
        distances = transects['distances'][i]
        intensity = transects['points'][i, :, 5]  # Feature 5 is intensity
        ax.plot(distances, intensity, alpha=0.7, linewidth=1)
    ax.set_xlabel("Distance from Start (m)")
    ax.set_ylabel("Intensity (normalized)")
    ax.set_title(f"Intensity Profiles ({n_plot} samples)")
    ax.grid(True, alpha=0.3)

    # Plot 4: Cliff height distribution
    ax = axes[1, 1]
    cliff_heights = transects['metadata'][:, 0]
    ax.hist(cliff_heights, bins=30, alpha=0.7, edgecolor='black')
    ax.set_xlabel("Cliff Height (m)")
    ax.set_ylabel("Count")
    ax.set_title(f"Cliff Height Distribution (N={n_transects})")
    ax.grid(True, alpha=0.3)

    # Plot 5: Mean slope distribution
    ax = axes[2, 0]
    mean_slopes = transects['metadata'][:, 1]
    ax.hist(mean_slopes, bins=30, alpha=0.7, edgecolor='black')
    ax.set_xlabel("Mean Slope (degrees)")
    ax.set_ylabel("Count")
    ax.set_title("Mean Slope Distribution")
    ax.grid(True, alpha=0.3)

    # Plot 6: Spatial distribution of transects
    ax = axes[2, 1]
    x_coords = transects['metadata'][:, 8]  # longitude/X
    y_coords = transects['metadata'][:, 7]  # latitude/Y
    colors = transects['metadata'][:, 0]    # cliff height for color
    sc = ax.scatter(x_coords, y_coords, c=colors, cmap='viridis', s=5, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='Cliff Height (m)')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Transect Locations")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization to {output_path}")


def main():
    """Main extraction script."""
    parser = argparse.ArgumentParser(
        description="Extract transects from LAS files using predefined transect lines"
    )

    parser.add_argument(
        "--transects",
        type=Path,
        required=True,
        help="Path to shapefile with transect LineStrings",
    )

    parser.add_argument(
        "--las-dir",
        type=Path,
        default=None,
        help="Directory containing LAS/LAZ files",
    )

    parser.add_argument(
        "--las-files",
        type=Path,
        nargs='+',
        default=None,
        help="Specific LAS/LAZ file(s) to process",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output NPZ file path",
    )

    parser.add_argument(
        "--buffer",
        type=float,
        default=1.0,
        help="Buffer distance around transect line in meters (default: 1.0)",
    )

    parser.add_argument(
        "--n-points",
        type=int,
        default=128,
        help="Number of points per transect (default: 128)",
    )

    parser.add_argument(
        "--min-points",
        type=int,
        default=20,
        help="Minimum raw points required for valid transect (default: 20)",
    )

    parser.add_argument(
        "--transect-id-col",
        type=str,
        default="tr_id",
        help="Column name for transect IDs in shapefile (default: tr_id)",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization of extracted transects",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*.las",
        help="Glob pattern for LAS files when using --las-dir (default: *.las)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.transects.exists():
        logger.error(f"Transect shapefile not found: {args.transects}")
        return 1

    # Collect LAS files
    las_files = []
    if args.las_files:
        las_files.extend(args.las_files)
    if args.las_dir:
        las_files.extend(sorted(args.las_dir.glob(args.pattern)))

    if not las_files:
        logger.error("No LAS files specified. Use --las-dir or --las-files")
        return 1

    # Check all files exist
    for f in las_files:
        if not f.exists():
            logger.error(f"LAS file not found: {f}")
            return 1

    logger.info(f"Found {len(las_files)} LAS file(s) to process")

    # Initialize extractor
    extractor = ShapefileTransectExtractor(
        n_points=args.n_points,
        buffer_m=args.buffer,
        min_points=args.min_points,
    )

    # Load transect lines
    transect_gdf = extractor.load_transect_lines(args.transects)

    # Extract transects
    try:
        transects = extractor.extract_from_shapefile_and_las(
            transect_gdf,
            las_files,
            transect_id_col=args.transect_id_col,
        )
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Check results
    n_extracted = len(transects['points'])
    if n_extracted == 0:
        logger.error("No transects extracted! Check that LAS files overlap with transect lines.")
        return 1

    logger.info(f"\n{'='*60}")
    logger.info(f"EXTRACTION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  Total transects extracted: {n_extracted}")
    logger.info(f"  Points per transect: {args.n_points}")
    logger.info(f"  Features per point: {extractor.N_FEATURES}")
    logger.info(f"  Metadata fields: {extractor.N_METADATA}")

    # Print feature/metadata names
    logger.info(f"\n  Feature names: {transects['feature_names']}")
    logger.info(f"  Metadata names: {transects['metadata_names']}")

    # Statistics
    cliff_heights = transects['metadata'][:, 0]
    logger.info(f"\n  Cliff height range: {cliff_heights.min():.1f} - {cliff_heights.max():.1f} m")
    logger.info(f"  Mean cliff height: {cliff_heights.mean():.1f} m")

    mean_slopes = transects['metadata'][:, 1]
    logger.info(f"  Mean slope range: {mean_slopes.min():.1f} - {mean_slopes.max():.1f} degrees")

    # Save
    extractor.save_transects(transects, args.output)

    # Visualize if requested
    if args.visualize:
        viz_path = args.output.parent / f"{args.output.stem}_viz.png"
        try:
            visualize_transects(transects, viz_path)
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")

    logger.info(f"\nExtraction complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
