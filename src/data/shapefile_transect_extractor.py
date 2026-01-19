"""Transect extraction from LiDAR point clouds using predefined transect lines.

This module extracts point data along transect lines defined in a shapefile.
For each transect, points within a buffer distance are extracted, projected
onto the transect line, and resampled to a fixed number of points.

The output includes all available LAS attributes (elevation, intensity, RGB,
classification) along with computed geometric features (slope, curvature, roughness).

Example:
    >>> from src.data.shapefile_transect_extractor import ShapefileTransectExtractor
    >>> extractor = ShapefileTransectExtractor(n_points=128, buffer_m=1.0)
    >>> transect_gdf = extractor.load_transect_lines("transects.shp")
    >>> transects = extractor.extract_from_shapefile_and_las(transect_gdf, las_files)
    >>> extractor.save_transects(transects, "output.npz")
"""

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


def _extract_single_las_worker(
    las_path: Path,
    transect_gdf: "gpd.GeoDataFrame",
    n_points: int,
    buffer_m: float,
    min_points: int,
    transect_id_col: str,
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

        Args:
            las_path: Path to LAS/LAZ file

        Returns:
            Dictionary with point cloud data arrays
        """
        las_path = Path(las_path)
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

    def extract_single_las(
        self,
        las_path: Path,
        transect_gdf: "gpd.GeoDataFrame",
        transect_id_col: str = 'tr_id',
        show_progress: bool = False,
    ) -> Dict[str, any]:
        """Extract transects from a single LAS file.

        Args:
            las_path: Path to LAS/LAZ file
            transect_gdf: GeoDataFrame with transect LineString geometries
            transect_id_col: Column name for transect IDs
            show_progress: Whether to show progress bar for transects

        Returns:
            Dictionary with extracted data and stats
        """
        las_path = Path(las_path)

        # Load LAS data
        las_data = self.load_las_file(las_path)

        # Build KDTree on XY coordinates
        tree = cKDTree(las_data['xyz'][:, :2])

        # Get LAS bounds
        las_minx, las_maxx = las_data['x'].min(), las_data['x'].max()
        las_miny, las_maxy = las_data['y'].min(), las_data['y'].max()

        # Filter transects that potentially intersect LAS bounds
        buffer = 50  # meters
        transect_bounds = transect_gdf.bounds
        potential_mask = (
            (transect_bounds['minx'] <= las_maxx + buffer) &
            (transect_bounds['maxx'] >= las_minx - buffer) &
            (transect_bounds['miny'] <= las_maxy + buffer) &
            (transect_bounds['maxy'] >= las_miny - buffer)
        )

        candidate_transects = transect_gdf[potential_mask]

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
    ) -> Dict[str, np.ndarray]:
        """Extract transects from multiple LAS files using shapefile transect lines.

        Args:
            transect_gdf: GeoDataFrame with transect LineString geometries
            las_files: List of paths to LAS/LAZ files
            transect_id_col: Column name for transect IDs
            show_progress: Whether to show progress bars
            n_workers: Number of parallel workers (1 = sequential)

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

        total_extracted = 0
        total_skipped = 0

        if n_workers > 1:
            # Parallel processing
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import multiprocessing as mp

            # Use spawn method for compatibility
            ctx = mp.get_context('spawn')

            logger.info(f"Processing {len(las_files)} LAS files with {n_workers} workers...")

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
                    ): las_path
                    for las_path in las_files
                }

                # Process results with progress bar
                if show_progress and HAS_TQDM:
                    futures_iter = tqdm(
                        as_completed(future_to_path),
                        total=len(las_files),
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
                las_iter = tqdm(las_files, desc="Processing LAS files", unit="file")
            else:
                las_iter = las_files

            for las_path in las_iter:
                result = self.extract_single_las(
                    las_path,
                    transect_gdf,
                    transect_id_col,
                    show_progress=False,  # Don't show nested progress
                )

                all_features.extend(result['features'])
                all_distances.extend(result['distances'])
                all_metadata.extend(result['metadata'])
                all_transect_ids.extend(result['transect_ids'])
                all_las_sources.extend([result['las_source']] * len(result['features']))
                total_extracted += result['extracted']
                total_skipped += result['skipped']

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
