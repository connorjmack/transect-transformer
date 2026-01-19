"""Transect extraction from LiDAR point clouds using predefined transect lines.

This module extracts point data along transect lines defined in a shapefile.
For each transect, points within a buffer distance are extracted, projected
onto the transect line, and resampled to a fixed number of points.

The output includes all available LAS attributes (elevation, intensity, RGB,
classification) along with computed geometric features (slope, curvature, roughness).

Performance optimizations (v2.0):
    - Header-only bounds checking to skip non-overlapping LAS files
    - Chunked spatial reading to load only relevant points
    - Reduced memory copies through numpy views

Example:
    >>> from src.data.shapefile_transect_extractor import ShapefileTransectExtractor
    >>> extractor = ShapefileTransectExtractor(n_points=128, buffer_m=1.0)
    >>> transect_gdf = extractor.load_transect_lines("transects.shp")
    >>> transects = extractor.extract_from_shapefile_and_las(transect_gdf, las_files)
    >>> extractor.save_transects(transects, "output.npz")
"""

import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None

try:
    import geopandas as gpd
    from shapely.geometry import LineString
except ImportError:
    gpd = None
    LineString = None

try:
    import laspy
except ImportError:
    laspy = None

from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_las_bounds(las_path: Path) -> Optional[Tuple[float, float, float, float]]:
    """Read LAS file header to get spatial bounds without loading points.

    This is a fast operation that only reads the file header, not the point data.
    Use this to pre-filter LAS files before expensive full reads.

    Args:
        las_path: Path to LAS/LAZ file

    Returns:
        Tuple of (x_min, y_min, x_max, y_max) or None if file cannot be read
    """
    if laspy is None:
        raise ImportError("laspy is required. Install with: pip install laspy[lazrs]")

    try:
        with laspy.open(las_path) as f:
            header = f.header
            return (
                float(header.x_min),
                float(header.y_min),
                float(header.x_max),
                float(header.y_max),
            )
    except Exception as e:
        logger.warning(f"Could not read bounds from {las_path}: {e}")
        return None


def bounds_overlap(
    bounds1: Tuple[float, float, float, float],
    bounds2: Tuple[float, float, float, float],
    buffer: float = 0.0,
) -> bool:
    """Check if two bounding boxes overlap.

    Args:
        bounds1: (x_min, y_min, x_max, y_max) for first box
        bounds2: (x_min, y_min, x_max, y_max) for second box
        buffer: Additional buffer to add around bounds1

    Returns:
        True if boxes overlap (with buffer)
    """
    x1_min, y1_min, x1_max, y1_max = bounds1
    x2_min, y2_min, x2_max, y2_max = bounds2

    return not (
        x1_max + buffer < x2_min or
        x1_min - buffer > x2_max or
        y1_max + buffer < y2_min or
        y1_min - buffer > y2_max
    )


def _extract_single_las_worker(
    las_path: Path,
    transect_gdf: "gpd.GeoDataFrame",
    n_points: int,
    buffer_m: float,
    min_points: int,
    transect_id_col: str,
    use_spatial_filter: bool = True,
) -> Dict:
    """Worker function for parallel LAS processing.

    This is a module-level function so it can be pickled for multiprocessing.
    """
    # Create extractor instance in worker process
    extractor = ShapefileTransectExtractor(
        n_points=n_points,
        buffer_m=buffer_m,
        min_points=min_points,
    )
    return extractor.extract_single_las(
        las_path,
        transect_gdf,
        transect_id_col,
        show_progress=False,
        use_spatial_filter=use_spatial_filter,
    )


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

    Example:
        >>> extractor = ShapefileTransectExtractor(n_points=128, buffer_m=1.0)
        >>> gdf = extractor.load_transect_lines("transects.shp")
        >>> transects = extractor.extract_from_shapefile_and_las(gdf, [Path("scan.las")])
        >>> print(f"Extracted {len(transects['points'])} transects")
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
        if gpd is None:
            raise ImportError(
                "geopandas is required for shapefile transect extraction. "
                "Install with: pip install geopandas shapely pyproj"
            )
        if laspy is None:
            raise ImportError(
                "laspy is required for transect extraction. "
                "Install with: pip install laspy[lazrs]"
            )

        self.n_points = n_points
        self.buffer_m = buffer_m
        self.min_points = min_points

        logger.info(
            f"Initialized ShapefileTransectExtractor: {n_points} points, "
            f"{buffer_m}m buffer, min {min_points} points"
        )

    def load_transect_lines(
        self,
        shapefile_path: Union[str, Path],
    ) -> "gpd.GeoDataFrame":
        """Load transect lines from shapefile.

        Args:
            shapefile_path: Path to shapefile containing transect LineStrings

        Returns:
            GeoDataFrame with transect geometries
        """
        shapefile_path = Path(shapefile_path)
        gdf = gpd.read_file(shapefile_path)
        logger.info(f"Loaded {len(gdf)} transect lines from {shapefile_path}")
        logger.info(f"CRS: {gdf.crs}")
        logger.info(f"Bounds: {gdf.total_bounds}")
        return gdf

    def load_las_file(
        self,
        las_path: Union[str, Path],
    ) -> Dict[str, np.ndarray]:
        """Load LAS file and extract all relevant attributes.

        Optimized to minimize memory copies by using views where possible.

        Args:
            las_path: Path to LAS/LAZ file

        Returns:
            Dictionary with point cloud data arrays
        """
        las_path = Path(las_path)
        logger.info(f"Loading LAS file: {las_path}")
        las = laspy.read(las_path)

        # Create single XYZ array and use views for x, y, z (reduces copies)
        xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)

        data = {
            'xyz': xyz,
            'x': xyz[:, 0],  # View, not copy
            'y': xyz[:, 1],  # View, not copy
            'z': xyz[:, 2],  # View, not copy
        }

        n_points = len(xyz)

        # Extract optional attributes with fallbacks
        if hasattr(las, 'intensity'):
            # Normalize intensity to 0-1 range (single copy)
            intensity = np.asarray(las.intensity, dtype=np.float32)
            max_intensity = intensity.max() if intensity.max() > 0 else 1
            intensity /= max_intensity  # In-place division
            data['intensity'] = intensity
        else:
            data['intensity'] = np.zeros(n_points, dtype=np.float32)

        # RGB colors (normalize from 16-bit to 0-1)
        if hasattr(las, 'red'):
            data['red'] = np.asarray(las.red, dtype=np.float32) / 65535.0
            data['green'] = np.asarray(las.green, dtype=np.float32) / 65535.0
            data['blue'] = np.asarray(las.blue, dtype=np.float32) / 65535.0
        else:
            data['red'] = np.zeros(n_points, dtype=np.float32)
            data['green'] = np.zeros(n_points, dtype=np.float32)
            data['blue'] = np.zeros(n_points, dtype=np.float32)

        # Classification
        if hasattr(las, 'classification'):
            data['classification'] = np.asarray(las.classification, dtype=np.float32)
        else:
            data['classification'] = np.zeros(n_points, dtype=np.float32)

        # Return information
        if hasattr(las, 'return_number'):
            data['return_number'] = np.asarray(las.return_number, dtype=np.float32)
        else:
            data['return_number'] = np.ones(n_points, dtype=np.float32)

        if hasattr(las, 'number_of_returns'):
            data['num_returns'] = np.asarray(las.number_of_returns, dtype=np.float32)
        else:
            data['num_returns'] = np.ones(n_points, dtype=np.float32)

        logger.info(f"Loaded {n_points:,} points")
        logger.info(f"  X range: {xyz[:, 0].min():.2f} - {xyz[:, 0].max():.2f}")
        logger.info(f"  Y range: {xyz[:, 1].min():.2f} - {xyz[:, 1].max():.2f}")
        logger.info(f"  Z range: {xyz[:, 2].min():.2f} - {xyz[:, 2].max():.2f}")

        return data

    def load_las_file_spatial(
        self,
        las_path: Union[str, Path],
        bounds: Tuple[float, float, float, float],
        chunk_size: int = 500_000,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Load LAS file with spatial filtering - only loads points within bounds.

        This is much faster than load_las_file() when you only need a small
        spatial subset of a large file. If the file is COPC format, uses fast
        spatial index queries (10-100x faster). Otherwise reads in chunks.

        Args:
            las_path: Path to LAS/LAZ file
            bounds: (x_min, y_min, x_max, y_max) bounding box to filter points
            chunk_size: Number of points to read per chunk (default: 500K)

        Returns:
            Dictionary with point cloud data arrays, or None if no points in bounds
        """
        las_path = Path(las_path)
        x_min, y_min, x_max, y_max = bounds

        # First check header bounds for quick rejection
        las_bounds = get_las_bounds(las_path)
        if las_bounds is None:
            return None

        if not bounds_overlap(bounds, las_bounds, buffer=self.buffer_m):
            logger.debug(f"Skipping {las_path.name}: no overlap with bounds")
            return None

        # Try COPC fast path if file is COPC format
        if '.copc.' in las_path.name.lower():
            copc_result = self._load_copc_spatial(las_path, bounds)
            if copc_result is not None:
                return copc_result
            # Fall through to chunked loading if COPC fails

        logger.info(f"Loading LAS file (chunked spatial filter): {las_path}")

        # Collect filtered points from chunks
        xyz_chunks = []
        intensity_chunks = []
        red_chunks = []
        green_chunks = []
        blue_chunks = []
        classification_chunks = []
        return_number_chunks = []
        num_returns_chunks = []

        total_read = 0
        total_kept = 0

        with laspy.open(las_path) as f:
            for chunk in f.chunk_iterator(chunk_size):
                total_read += len(chunk)

                # Get coordinates
                x = np.asarray(chunk.x)
                y = np.asarray(chunk.y)

                # Spatial filter
                mask = (
                    (x >= x_min) & (x <= x_max) &
                    (y >= y_min) & (y <= y_max)
                )

                if not mask.any():
                    continue

                kept = mask.sum()
                total_kept += kept

                # Extract filtered points
                z = np.asarray(chunk.z)
                xyz_chunks.append(np.column_stack([x[mask], y[mask], z[mask]]))

                # Extract attributes with fallbacks
                if hasattr(chunk, 'intensity'):
                    intensity_chunks.append(np.asarray(chunk.intensity, dtype=np.float32)[mask])
                else:
                    intensity_chunks.append(np.zeros(kept, dtype=np.float32))

                if hasattr(chunk, 'red'):
                    red_chunks.append(np.asarray(chunk.red, dtype=np.float32)[mask])
                    green_chunks.append(np.asarray(chunk.green, dtype=np.float32)[mask])
                    blue_chunks.append(np.asarray(chunk.blue, dtype=np.float32)[mask])
                else:
                    red_chunks.append(np.zeros(kept, dtype=np.float32))
                    green_chunks.append(np.zeros(kept, dtype=np.float32))
                    blue_chunks.append(np.zeros(kept, dtype=np.float32))

                if hasattr(chunk, 'classification'):
                    classification_chunks.append(np.asarray(chunk.classification, dtype=np.float32)[mask])
                else:
                    classification_chunks.append(np.zeros(kept, dtype=np.float32))

                if hasattr(chunk, 'return_number'):
                    return_number_chunks.append(np.asarray(chunk.return_number, dtype=np.float32)[mask])
                else:
                    return_number_chunks.append(np.ones(kept, dtype=np.float32))

                if hasattr(chunk, 'number_of_returns'):
                    num_returns_chunks.append(np.asarray(chunk.number_of_returns, dtype=np.float32)[mask])
                else:
                    num_returns_chunks.append(np.ones(kept, dtype=np.float32))

        if total_kept == 0:
            logger.debug(f"No points within bounds in {las_path.name}")
            return None

        # Concatenate all chunks
        xyz = np.vstack(xyz_chunks).astype(np.float64)

        # Normalize intensity
        intensity = np.concatenate(intensity_chunks)
        max_intensity = intensity.max() if intensity.max() > 0 else 1
        intensity /= max_intensity

        # Normalize RGB from 16-bit
        red = np.concatenate(red_chunks) / 65535.0
        green = np.concatenate(green_chunks) / 65535.0
        blue = np.concatenate(blue_chunks) / 65535.0

        data = {
            'xyz': xyz,
            'x': xyz[:, 0],
            'y': xyz[:, 1],
            'z': xyz[:, 2],
            'intensity': intensity,
            'red': red,
            'green': green,
            'blue': blue,
            'classification': np.concatenate(classification_chunks),
            'return_number': np.concatenate(return_number_chunks),
            'num_returns': np.concatenate(num_returns_chunks),
        }

        reduction_pct = 100 * (1 - total_kept / total_read) if total_read > 0 else 0
        logger.info(f"Loaded {total_kept:,} of {total_read:,} points ({reduction_pct:.1f}% filtered out)")
        logger.info(f"  X range: {xyz[:, 0].min():.2f} - {xyz[:, 0].max():.2f}")
        logger.info(f"  Y range: {xyz[:, 1].min():.2f} - {xyz[:, 1].max():.2f}")
        logger.info(f"  Z range: {xyz[:, 2].min():.2f} - {xyz[:, 2].max():.2f}")

        return data

    def _load_copc_spatial(
        self,
        las_path: Path,
        bounds: Tuple[float, float, float, float],
    ) -> Optional[Dict[str, np.ndarray]]:
        """Load points within bounds using COPC spatial index (10-100x faster).

        COPC (Cloud Optimized Point Cloud) files contain spatial indices that
        allow direct retrieval of points within a bounding box without scanning
        the entire file.

        Args:
            las_path: Path to COPC-enabled LAS/LAZ file
            bounds: (x_min, y_min, x_max, y_max) bounding box

        Returns:
            Dictionary with point data, or None if query fails/no points
        """
        try:
            from laspy import CopcReader
        except ImportError:
            logger.debug("CopcReader not available, falling back to chunked loading")
            return None

        x_min, y_min, x_max, y_max = bounds
        logger.info(f"Loading COPC file (fast spatial query): {las_path}")

        try:
            reader = CopcReader.open(str(las_path))

            # Query points within 2D bounds (use extreme z range)
            points = reader.query(
                mins=np.array([x_min, y_min, -1e10]),
                maxs=np.array([x_max, y_max, 1e10])
            )
            reader.close()

            if len(points) == 0:
                logger.debug(f"COPC query returned no points for {las_path.name}")
                return None

            # Convert to standard dict format
            xyz = np.column_stack([points.x, points.y, points.z]).astype(np.float64)
            n_points = len(xyz)

            # Normalize intensity
            if hasattr(points, 'intensity'):
                intensity = np.asarray(points.intensity, dtype=np.float32)
                max_int = intensity.max() if intensity.max() > 0 else 1
                intensity /= max_int
            else:
                intensity = np.zeros(n_points, dtype=np.float32)

            # Normalize RGB from 16-bit
            if hasattr(points, 'red'):
                red = np.asarray(points.red, dtype=np.float32) / 65535.0
                green = np.asarray(points.green, dtype=np.float32) / 65535.0
                blue = np.asarray(points.blue, dtype=np.float32) / 65535.0
            else:
                red = np.zeros(n_points, dtype=np.float32)
                green = np.zeros(n_points, dtype=np.float32)
                blue = np.zeros(n_points, dtype=np.float32)

            # Classification
            if hasattr(points, 'classification'):
                classification = np.asarray(points.classification, dtype=np.float32)
            else:
                classification = np.zeros(n_points, dtype=np.float32)

            # Return info
            if hasattr(points, 'return_number'):
                return_number = np.asarray(points.return_number, dtype=np.float32)
            else:
                return_number = np.ones(n_points, dtype=np.float32)

            if hasattr(points, 'number_of_returns'):
                num_returns = np.asarray(points.number_of_returns, dtype=np.float32)
            else:
                num_returns = np.ones(n_points, dtype=np.float32)

            data = {
                'xyz': xyz,
                'x': xyz[:, 0],
                'y': xyz[:, 1],
                'z': xyz[:, 2],
                'intensity': intensity,
                'red': red,
                'green': green,
                'blue': blue,
                'classification': classification,
                'return_number': return_number,
                'num_returns': num_returns,
            }

            logger.info(f"COPC query returned {n_points:,} points (fast path)")
            logger.info(f"  X range: {xyz[:, 0].min():.2f} - {xyz[:, 0].max():.2f}")
            logger.info(f"  Y range: {xyz[:, 1].min():.2f} - {xyz[:, 1].max():.2f}")
            logger.info(f"  Z range: {xyz[:, 2].min():.2f} - {xyz[:, 2].max():.2f}")

            return data

        except Exception as e:
            logger.warning(f"COPC query failed for {las_path}: {e}, falling back to chunked")
            return None

    def get_transect_direction(
        self, line: "LineString"
    ) -> Tuple[np.ndarray, np.ndarray, float]:
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
        line: "LineString",
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

        # Uniform step size (target_distances are linearly spaced)
        dx = (target_distances[-1] - target_distances[0]) / (n - 1) if n > 1 else 1.0
        dx = max(dx, 0.01)  # Minimum 1cm step to avoid numerical issues

        # Smooth elevation before derivative computation to reduce noise
        # Use a simple moving average (Gaussian-like smoothing)
        smooth_window = 5
        z_smooth = np.convolve(
            resampled_z,
            np.ones(smooth_window) / smooth_window,
            mode='same'
        )
        # Handle edge effects by keeping original values at boundaries
        half_win = smooth_window // 2
        z_smooth[:half_win] = resampled_z[:half_win]
        z_smooth[-half_win:] = resampled_z[-half_win:]

        # First derivative (slope) using central differences on smoothed data
        dz_dx = np.zeros(n)
        dz_dx[0] = (z_smooth[1] - z_smooth[0]) / dx
        dz_dx[-1] = (z_smooth[-1] - z_smooth[-2]) / dx
        dz_dx[1:-1] = (z_smooth[2:] - z_smooth[:-2]) / (2 * dx)

        # Slope in degrees
        slope = np.degrees(np.arctan(dz_dx))

        # Second derivative using central differences
        d2z_dx2 = np.zeros(n)
        d2z_dx2[1:-1] = (z_smooth[2:] - 2 * z_smooth[1:-1] + z_smooth[:-2]) / (dx ** 2)
        # Extrapolate edges
        d2z_dx2[0] = d2z_dx2[1]
        d2z_dx2[-1] = d2z_dx2[-2]

        # Profile curvature: κ = z'' / (1 + z'^2)^(3/2)
        # This is the proper curvature formula that accounts for slope
        curvature = d2z_dx2 / np.power(1 + dz_dx ** 2, 1.5)

        # Clip to physically reasonable bounds (±10 1/m is already extreme for cliffs)
        # Values beyond this are numerical artifacts from noise/interpolation
        curvature = np.clip(curvature, -10.0, 10.0)

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

    def extract_single_las(
        self,
        las_path: Path,
        transect_gdf: "gpd.GeoDataFrame",
        transect_id_col: str = 'tr_id',
        show_progress: bool = False,
        use_spatial_filter: bool = True,
    ) -> Dict[str, any]:
        """Extract transects from a single LAS file.

        Args:
            las_path: Path to LAS/LAZ file
            transect_gdf: GeoDataFrame with transect LineString geometries
            transect_id_col: Column name for transect IDs
            show_progress: Whether to show progress bar for transects
            use_spatial_filter: If True, use chunked spatial loading (faster for
                large files). If False, load entire file (better for small files).

        Returns:
            Dictionary with extracted data and stats
        """
        las_path = Path(las_path)

        # First, check LAS bounds from header (very fast - no point loading)
        las_bounds = get_las_bounds(las_path)
        if las_bounds is None:
            logger.warning(f"Could not read header from {las_path}")
            return {
                'features': [],
                'distances': [],
                'metadata': [],
                'transect_ids': [],
                'las_source': las_path.name,
                'extracted': 0,
                'skipped': 0,
                'candidates': 0,
            }

        las_minx, las_miny, las_maxx, las_maxy = las_bounds

        # Filter transects that potentially intersect LAS bounds (before loading any points)
        buffer = 50  # meters
        transect_bounds = transect_gdf.bounds
        potential_mask = (
            (transect_bounds['minx'] <= las_maxx + buffer) &
            (transect_bounds['maxx'] >= las_minx - buffer) &
            (transect_bounds['miny'] <= las_maxy + buffer) &
            (transect_bounds['maxy'] >= las_miny - buffer)
        )

        candidate_transects = transect_gdf[potential_mask]

        if len(candidate_transects) == 0:
            logger.debug(f"No transects overlap with {las_path.name}")
            return {
                'features': [],
                'distances': [],
                'metadata': [],
                'transect_ids': [],
                'las_source': las_path.name,
                'extracted': 0,
                'skipped': 0,
                'candidates': 0,
            }

        # Calculate bounds of candidate transects (with buffer for spatial loading)
        transect_total_bounds = candidate_transects.total_bounds  # (minx, miny, maxx, maxy)
        load_bounds = (
            transect_total_bounds[0] - self.buffer_m - 10,  # x_min with padding
            transect_total_bounds[1] - self.buffer_m - 10,  # y_min with padding
            transect_total_bounds[2] + self.buffer_m + 10,  # x_max with padding
            transect_total_bounds[3] + self.buffer_m + 10,  # y_max with padding
        )

        # Load LAS data (use spatial filter for efficiency)
        if use_spatial_filter:
            las_data = self.load_las_file_spatial(las_path, load_bounds)
            if las_data is None:
                logger.debug(f"No points in bounds from {las_path.name}")
                return {
                    'features': [],
                    'distances': [],
                    'metadata': [],
                    'transect_ids': [],
                    'las_source': las_path.name,
                    'extracted': 0,
                    'skipped': 0,
                    'candidates': len(candidate_transects),
                }
        else:
            las_data = self.load_las_file(las_path)

        # Build KDTree on XY coordinates
        tree = cKDTree(las_data['xyz'][:, :2])

        features = []
        distances = []
        metadata = []
        transect_ids = []

        extracted_count = 0
        skipped_count = 0

        # Optionally wrap with progress bar
        iterator = candidate_transects.iterrows()
        if show_progress and HAS_TQDM:
            iterator = tqdm(
                list(iterator),
                desc=f"  {las_path.name[:30]}",
                unit="transect",
                leave=False,
            )

        for idx, row in iterator:
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
            try:
                resampled['metadata'][9] = float(transect_id)
            except (ValueError, TypeError):
                resampled['metadata'][9] = float(idx)

            features.append(resampled['features'])
            distances.append(resampled['distances'])
            metadata.append(resampled['metadata'])
            transect_ids.append(transect_id)

            extracted_count += 1

        return {
            'features': features,
            'distances': distances,
            'metadata': metadata,
            'transect_ids': transect_ids,
            'las_source': las_path.name,
            'extracted': extracted_count,
            'skipped': skipped_count,
            'candidates': len(candidate_transects),
        }

    def extract_from_shapefile_and_las(
        self,
        transect_gdf: "gpd.GeoDataFrame",
        las_files: List[Path],
        transect_id_col: str = 'tr_id',
        show_progress: bool = True,
        n_workers: int = 1,
        use_spatial_filter: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Extract transects from multiple LAS files using shapefile transect lines.

        Includes performance optimizations:
        - Pre-filters LAS files by bounds before loading (reads only headers)
        - Uses chunked spatial loading to read only relevant points
        - Reduces memory copies through numpy views

        Args:
            transect_gdf: GeoDataFrame with transect LineString geometries
            las_files: List of paths to LAS/LAZ files
            transect_id_col: Column name for transect IDs
            show_progress: Whether to show progress bars
            n_workers: Number of parallel workers (1 = sequential)
            use_spatial_filter: If True, use chunked spatial loading (recommended
                for large files). Set to False for small files or debugging.

        Returns:
            Dictionary containing:
                - 'points': (N_transects, n_points, n_features) features
                - 'distances': (N_transects, n_points) distances along transect
                - 'metadata': (N_transects, n_meta) transect-level metadata
                - 'transect_ids': (N_transects,) original transect IDs
                - 'las_sources': list of source LAS file names per transect
        """
        # Pre-filter LAS files by bounds (Quick Win #2 - header-only check)
        transect_total_bounds = transect_gdf.total_bounds  # (minx, miny, maxx, maxy)
        buffer = self.buffer_m + 50  # Extra buffer for safety

        valid_las_files = []
        skipped_files = 0

        logger.info(f"Pre-filtering {len(las_files)} LAS files by bounds...")
        for las_path in las_files:
            las_bounds = get_las_bounds(las_path)
            if las_bounds is None:
                logger.warning(f"Could not read bounds from {las_path}, including in processing")
                valid_las_files.append(las_path)
                continue

            if bounds_overlap(
                (transect_total_bounds[0], transect_total_bounds[1],
                 transect_total_bounds[2], transect_total_bounds[3]),
                las_bounds,
                buffer=buffer
            ):
                valid_las_files.append(las_path)
            else:
                skipped_files += 1
                logger.debug(f"Skipping {las_path.name}: no overlap with transects")

        if skipped_files > 0:
            logger.info(f"Skipped {skipped_files} LAS files with no transect overlap")
        logger.info(f"Processing {len(valid_las_files)} LAS files with potential overlap")

        if len(valid_las_files) == 0:
            logger.warning("No LAS files overlap with transects!")
            return {
                'points': np.zeros((0, self.n_points, self.N_FEATURES), dtype=np.float32),
                'distances': np.zeros((0, self.n_points), dtype=np.float32),
                'metadata': np.zeros((0, self.N_METADATA), dtype=np.float32),
                'transect_ids': np.array([], dtype=object),
                'las_sources': [],
                'feature_names': self.FEATURE_NAMES,
                'metadata_names': self.METADATA_NAMES,
            }

        all_features = []
        all_distances = []
        all_metadata = []
        all_transect_ids = []
        all_las_sources = []

        total_extracted = 0
        total_skipped = 0

        if n_workers > 1:
            # Parallel processing
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import multiprocessing as mp

            # Use spawn method for compatibility
            ctx = mp.get_context('spawn')

            logger.info(f"Processing {len(valid_las_files)} LAS files with {n_workers} workers...")

            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
                # Submit all jobs
                future_to_path = {
                    executor.submit(
                        _extract_single_las_worker,
                        las_path,
                        transect_gdf,
                        self.n_points,
                        self.buffer_m,
                        self.min_points,
                        transect_id_col,
                        use_spatial_filter,
                    ): las_path
                    for las_path in valid_las_files
                }

                # Process results with progress bar
                if show_progress and HAS_TQDM:
                    futures_iter = tqdm(
                        as_completed(future_to_path),
                        total=len(valid_las_files),
                        desc="Processing LAS files",
                        unit="file",
                    )
                else:
                    futures_iter = as_completed(future_to_path)

                for future in futures_iter:
                    las_path = future_to_path[future]
                    try:
                        result = future.result()
                        all_features.extend(result['features'])
                        all_distances.extend(result['distances'])
                        all_metadata.extend(result['metadata'])
                        all_transect_ids.extend(result['transect_ids'])
                        all_las_sources.extend([result['las_source']] * len(result['features']))
                        total_extracted += result['extracted']
                        total_skipped += result['skipped']
                    except Exception as e:
                        logger.error(f"Error processing {las_path}: {e}")
        else:
            # Sequential processing with progress
            if show_progress and HAS_TQDM:
                las_iter = tqdm(valid_las_files, desc="Processing LAS files", unit="file")
            else:
                las_iter = valid_las_files

            for las_path in las_iter:
                result = self.extract_single_las(
                    las_path,
                    transect_gdf,
                    transect_id_col,
                    show_progress=False,  # Don't show nested progress
                    use_spatial_filter=use_spatial_filter,
                )

                all_features.extend(result['features'])
                all_distances.extend(result['distances'])
                all_metadata.extend(result['metadata'])
                all_transect_ids.extend(result['transect_ids'])
                all_las_sources.extend([result['las_source']] * len(result['features']))
                total_extracted += result['extracted']
                total_skipped += result['skipped']

                # Explicit memory cleanup after each file to prevent OOM
                del result
                gc.collect()

                if not show_progress or not HAS_TQDM:
                    logger.info(
                        f"  {result['las_source']}: {result['extracted']} extracted, "
                        f"{result['skipped']} skipped (of {result['candidates']} candidates)"
                    )

        logger.info(f"Total: {total_extracted} transect-epochs extracted, {total_skipped} skipped")

        if len(all_features) == 0:
            logger.warning("No transects extracted!")
            return {
                'points': np.zeros((0, self.n_points, self.N_FEATURES), dtype=np.float32),
                'distances': np.zeros((0, self.n_points), dtype=np.float32),
                'metadata': np.zeros((0, self.N_METADATA), dtype=np.float32),
                'transect_ids': np.array([], dtype=object),
                'las_sources': [],
                'feature_names': self.FEATURE_NAMES,
                'metadata_names': self.METADATA_NAMES,
            }

        return {
            'points': np.stack(all_features),
            'distances': np.stack(all_distances),
            'metadata': np.stack(all_metadata),
            'transect_ids': np.array(all_transect_ids, dtype=object),
            'las_sources': all_las_sources,
            'feature_names': self.FEATURE_NAMES,
            'metadata_names': self.METADATA_NAMES,
        }

    def save_transects(
        self,
        transects: Dict[str, np.ndarray],
        output_path: Union[str, Path],
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

    def load_transects(
        self,
        input_path: Union[str, Path],
    ) -> Dict[str, np.ndarray]:
        """Load transects from NPZ file.

        Args:
            input_path: Path to saved transects file

        Returns:
            Dictionary with transect data
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Transects file not found: {input_path}")

        data = np.load(input_path, allow_pickle=True)
        transects = {key: data[key] for key in data.keys()}

        logger.info(f"Loaded {len(transects['points'])} transects from {input_path}")

        return transects
