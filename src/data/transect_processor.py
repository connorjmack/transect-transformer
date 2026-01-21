"""Transect post-processing: crop to cliff window, resample, and clean features.

This module processes raw extracted transects (256 points, 12 features) into
cliff-centered training data (128 points, 7 features) suitable for the
CliffCast susceptibility model.

Processing steps:
    1. Load raw transects and cliff delineation results
    2. For each transect-epoch with valid cliff detection:
        a. Define window: [toe_distance - 10m] to [top_distance + 5m]
        b. Crop points within window
        c. Resample to 128 points
        d. Drop RGB and return features (keep 7 features)
        e. Recompute slope/curvature on resampled points
    3. Save processed transects with updated metadata

Usage:
    >>> from src.data.transect_processor import TransectProcessor
    >>> processor = TransectProcessor(n_output_points=128, toe_buffer_m=10.0, top_buffer_m=5.0)
    >>> # Combined file (cliff data embedded):
    >>> processor.process(
    ...     raw_npz_path="data/raw/transects/transects_with_cliffs.npz",
    ...     output_path="data/processed/transects/transects_processed.npz"
    ... )
    >>> # Separate sidecar file:
    >>> processor.process(
    ...     raw_npz_path="data/raw/transects/transects.npz",
    ...     output_path="data/processed/transects/transects_processed.npz",
    ...     cliff_npz_path="data/raw/transects/transects.cliff.npz"
    ... )
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from scipy.interpolate import interp1d

from src.utils.logging import get_logger

logger = get_logger(__name__)


# Feature indices in raw extraction (12 features)
RAW_FEATURE_NAMES = [
    'distance_m',      # 0
    'elevation_m',     # 1
    'slope_deg',       # 2
    'curvature',       # 3
    'roughness',       # 4
    'intensity',       # 5
    'red',             # 6  - DROP
    'green',           # 7  - DROP
    'blue',            # 8  - DROP
    'classification',  # 9
    'return_number',   # 10 - DROP
    'num_returns',     # 11 - DROP
]

# Feature indices to keep (7 features after dropping RGB and return info)
KEEP_FEATURE_INDICES = [0, 1, 2, 3, 4, 5, 9]  # distance, elev, slope, curv, rough, intensity, class
KEEP_FEATURE_NAMES = [
    'distance_m',
    'elevation_m',
    'slope_deg',
    'curvature',
    'roughness',
    'intensity',
    'classification',
]

# Output feature names (7 features, with M3C2 added later as 8th)
OUTPUT_FEATURE_NAMES = KEEP_FEATURE_NAMES.copy()


class TransectProcessor:
    """Post-extraction processor: crop to cliff window, resample, clean features.

    Transforms raw extracted transects into cliff-centered training data by:
    - Cropping to a window around the detected cliff (toe - buffer to top + buffer)
    - Resampling to a fixed number of points
    - Dropping uninformative features (RGB, return number/count)
    - Recomputing derived features (slope, curvature) on resampled data

    Args:
        n_output_points: Number of points in processed output (default: 128)
        toe_buffer_m: Distance seaward of toe to include (default: 10.0)
        top_buffer_m: Distance landward of top to include (default: 5.0)
        min_cliff_width_m: Minimum cliff width to process (default: 5.0)
        fallback_to_full: If True, use full transect when cliff not detected (default: True)
        reference_epoch: How to determine the reference window for each transect.
            - 'first': Use the first epoch with valid cliff detection as reference
                       for ALL epochs (ensures temporal consistency)
            - 'per_epoch': Use each epoch's own cliff detection (legacy behavior)
            Default is 'first' for temporal consistency.

    Example:
        >>> processor = TransectProcessor(n_output_points=128, toe_buffer_m=10.0, top_buffer_m=5.0)
        >>> result = processor.process(
        ...     raw_npz_path="data/raw/transects/transects_with_cliffs.npz",
        ...     output_path="data/processed/transects/transects_processed.npz"
        ... )
    """

    def __init__(
        self,
        n_output_points: int = 128,
        toe_buffer_m: float = 10.0,
        top_buffer_m: float = 5.0,
        min_cliff_width_m: float = 5.0,
        fallback_to_full: bool = True,
        reference_epoch: str = 'first',
    ):
        if reference_epoch not in ('first', 'per_epoch'):
            raise ValueError(f"reference_epoch must be 'first' or 'per_epoch', got '{reference_epoch}'")

        self.n_output_points = n_output_points
        self.toe_buffer_m = toe_buffer_m
        self.top_buffer_m = top_buffer_m
        self.min_cliff_width_m = min_cliff_width_m
        self.fallback_to_full = fallback_to_full
        self.reference_epoch = reference_epoch

        logger.debug(
            f"TransectProcessor initialized: {n_output_points} points, "
            f"window=[toe-{toe_buffer_m}m, top+{top_buffer_m}m], "
            f"reference_epoch={reference_epoch}"
        )

    def process(
        self,
        raw_npz_path: Union[str, Path],
        output_path: Union[str, Path],
        cliff_npz_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, any]:
        """Process raw transects using cliff delineation results.

        Args:
            raw_npz_path: Path to raw extracted transects NPZ (may contain cliff data)
            output_path: Path to save processed transects
            cliff_npz_path: Optional path to cliff delineation sidecar NPZ.
                           If None, cliff data is read from raw_npz_path.

        Returns:
            Dictionary with processing statistics
        """
        raw_npz_path = Path(raw_npz_path)
        output_path = Path(output_path)

        # Load data
        logger.info(f"Loading raw transects from {raw_npz_path}")
        raw_data = np.load(raw_npz_path, allow_pickle=True)

        # Load cliff data from sidecar or from combined file
        if cliff_npz_path is not None:
            cliff_npz_path = Path(cliff_npz_path)
            logger.info(f"Loading cliff delineation from {cliff_npz_path}")
            cliff_data = np.load(cliff_npz_path, allow_pickle=True)
        elif 'toe_distances' in raw_data:
            logger.info("Using cliff delineation from combined file")
            cliff_data = raw_data
        else:
            raise ValueError(
                f"No cliff data found. Either provide cliff_npz_path or use a "
                f"combined file with cliff arrays (toe_distances, top_distances, etc.)"
            )

        # Get dimensions
        points = raw_data['points']
        distances = raw_data['distances']
        metadata = raw_data['metadata']

        # Handle both cube (4D) and flat (3D) formats
        if points.ndim == 4:
            n_transects, n_epochs, n_raw_points, n_raw_features = points.shape
            is_cube = True
            logger.info(f"Cube format: {n_transects} transects × {n_epochs} epochs × {n_raw_points} points × {n_raw_features} features")
        elif points.ndim == 3:
            n_transects, n_raw_points, n_raw_features = points.shape
            n_epochs = 1
            is_cube = False
            # Add epoch dimension for uniform processing
            points = points[:, np.newaxis, :, :]
            distances = distances[:, np.newaxis, :]
            metadata = metadata[:, np.newaxis, :]
            logger.info(f"Flat format: {n_transects} transects × {n_raw_points} points × {n_raw_features} features")
        else:
            raise ValueError(f"Unexpected points shape: {points.shape}")

        # Get cliff delineation arrays
        toe_distances = cliff_data['toe_distances']
        top_distances = cliff_data['top_distances']
        has_cliff = cliff_data['has_cliff']
        toe_confidences = cliff_data['toe_confidences']
        top_confidences = cliff_data['top_confidences']

        # Handle flat format for cliff data too
        if toe_distances.ndim == 1:
            toe_distances = toe_distances[:, np.newaxis]
            top_distances = top_distances[:, np.newaxis]
            has_cliff = has_cliff[:, np.newaxis]
            toe_confidences = toe_confidences[:, np.newaxis]
            top_confidences = top_confidences[:, np.newaxis]

        # Initialize output arrays
        n_output_features = len(KEEP_FEATURE_INDICES)
        out_points = np.full(
            (n_transects, n_epochs, self.n_output_points, n_output_features),
            np.nan, dtype=np.float32
        )
        out_distances = np.full(
            (n_transects, n_epochs, self.n_output_points),
            np.nan, dtype=np.float32
        )
        out_metadata = np.full(
            (n_transects, n_epochs, metadata.shape[-1]),
            np.nan, dtype=np.float32
        )

        # Window metadata arrays
        window_start_m = np.full((n_transects, n_epochs), np.nan, dtype=np.float32)
        window_end_m = np.full((n_transects, n_epochs), np.nan, dtype=np.float32)
        toe_distance_m = np.full((n_transects, n_epochs), np.nan, dtype=np.float32)
        top_distance_m = np.full((n_transects, n_epochs), np.nan, dtype=np.float32)
        # Relative positions within the processed transect (0 = window start)
        toe_relative_m = np.full((n_transects, n_epochs), np.nan, dtype=np.float32)
        top_relative_m = np.full((n_transects, n_epochs), np.nan, dtype=np.float32)
        delineation_confidence = np.full((n_transects, n_epochs), np.nan, dtype=np.float32)
        used_fallback = np.zeros((n_transects, n_epochs), dtype=bool)

        # Processing statistics
        stats = {
            'n_processed': 0,
            'n_with_cliff': 0,
            'n_fallback': 0,
            'n_skipped': 0,
            'n_invalid_input': 0,
        }

        # Process each transect-epoch
        total = n_transects * n_epochs
        logger.info(f"Processing {total} transect-epochs...")

        for t_idx in range(n_transects):
            for e_idx in range(n_epochs):
                # Check for valid input data
                if np.isnan(points[t_idx, e_idx, 0, 0]):
                    stats['n_invalid_input'] += 1
                    continue

                # Get raw points and distances for this transect-epoch
                raw_pts = points[t_idx, e_idx]  # (n_raw_points, n_raw_features)
                raw_dist = distances[t_idx, e_idx]  # (n_raw_points,)
                raw_meta = metadata[t_idx, e_idx]  # (n_meta,)

                # Determine window based on cliff detection
                if has_cliff[t_idx, e_idx]:
                    toe_dist = toe_distances[t_idx, e_idx]
                    top_dist = top_distances[t_idx, e_idx]
                    cliff_width = top_dist - toe_dist

                    # Validate cliff width
                    if cliff_width < self.min_cliff_width_m:
                        if self.fallback_to_full:
                            # Use full transect
                            win_start = raw_dist[0]
                            win_end = raw_dist[-1]
                            used_fallback[t_idx, e_idx] = True
                            stats['n_fallback'] += 1
                        else:
                            stats['n_skipped'] += 1
                            continue
                    else:
                        # Use cliff-centered window
                        win_start = max(raw_dist[0], toe_dist - self.toe_buffer_m)
                        win_end = min(raw_dist[-1], top_dist + self.top_buffer_m)
                        stats['n_with_cliff'] += 1

                    toe_distance_m[t_idx, e_idx] = toe_dist
                    top_distance_m[t_idx, e_idx] = top_dist
                    # Relative positions (will be computed after win_start is finalized)
                    delineation_confidence[t_idx, e_idx] = (
                        toe_confidences[t_idx, e_idx] + top_confidences[t_idx, e_idx]
                    ) / 2.0
                else:
                    # No cliff detected
                    if self.fallback_to_full:
                        win_start = raw_dist[0]
                        win_end = raw_dist[-1]
                        used_fallback[t_idx, e_idx] = True
                        delineation_confidence[t_idx, e_idx] = 0.0
                        stats['n_fallback'] += 1
                    else:
                        stats['n_skipped'] += 1
                        continue

                # Store window bounds
                window_start_m[t_idx, e_idx] = win_start
                window_end_m[t_idx, e_idx] = win_end

                # Compute relative toe/top positions (relative to window start)
                if has_cliff[t_idx, e_idx] and not used_fallback[t_idx, e_idx]:
                    toe_relative_m[t_idx, e_idx] = toe_distances[t_idx, e_idx] - win_start
                    top_relative_m[t_idx, e_idx] = top_distances[t_idx, e_idx] - win_start

                # Process this transect-epoch
                processed = self._process_single(
                    raw_pts, raw_dist, win_start, win_end
                )

                if processed is None:
                    stats['n_skipped'] += 1
                    continue

                # Store results
                out_points[t_idx, e_idx] = processed['points']
                out_distances[t_idx, e_idx] = processed['distances']
                out_metadata[t_idx, e_idx] = raw_meta  # Keep original metadata

                stats['n_processed'] += 1

        # Remove epoch dimension if input was flat
        if not is_cube:
            out_points = out_points[:, 0, :, :]
            out_distances = out_distances[:, 0, :]
            out_metadata = out_metadata[:, 0, :]
            window_start_m = window_start_m[:, 0]
            window_end_m = window_end_m[:, 0]
            toe_distance_m = toe_distance_m[:, 0]
            top_distance_m = top_distance_m[:, 0]
            toe_relative_m = toe_relative_m[:, 0]
            top_relative_m = top_relative_m[:, 0]
            delineation_confidence = delineation_confidence[:, 0]
            used_fallback = used_fallback[:, 0]

        # Prepare output dictionary
        output = {
            'points': out_points,
            'distances': out_distances,
            'metadata': out_metadata,

            # Window information (absolute coordinates)
            'window_start_m': window_start_m,
            'window_end_m': window_end_m,
            'toe_distance_m': toe_distance_m,
            'top_distance_m': top_distance_m,

            # Cliff positions relative to window start (use these to locate toe/top in processed transect)
            'toe_relative_m': toe_relative_m,
            'top_relative_m': top_relative_m,

            'delineation_confidence': delineation_confidence,
            'used_fallback': used_fallback,

            # Feature info
            'feature_names': np.array(OUTPUT_FEATURE_NAMES, dtype=object),

            # Copy relevant arrays from raw data
            'transect_ids': raw_data['transect_ids'],
        }

        # Copy additional metadata if present
        for key in ['mop_ids', 'epoch_dates', 'epoch_files', 'epoch_names',
                    'survey_date', 'timestamps', 'coverage_mask', 'beach_slices',
                    'metadata_names']:
            if key in raw_data:
                output[key] = raw_data[key]

        # Add processing metadata
        output['processing_config'] = np.array({
            'n_output_points': self.n_output_points,
            'toe_buffer_m': self.toe_buffer_m,
            'top_buffer_m': self.top_buffer_m,
            'min_cliff_width_m': self.min_cliff_width_m,
            'fallback_to_full': self.fallback_to_full,
            'raw_feature_count': n_raw_features,
            'output_feature_count': n_output_features,
        })

        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **output)

        # Log summary
        logger.info(f"Processing complete:")
        logger.info(f"  Processed: {stats['n_processed']}")
        logger.info(f"  With cliff: {stats['n_with_cliff']}")
        logger.info(f"  Fallback (full transect): {stats['n_fallback']}")
        logger.info(f"  Skipped: {stats['n_skipped']}")
        logger.info(f"  Invalid input: {stats['n_invalid_input']}")
        logger.info(f"  Output: {output_path}")

        return stats

    def _process_single(
        self,
        raw_points: np.ndarray,
        raw_distances: np.ndarray,
        window_start: float,
        window_end: float,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Process a single transect-epoch.

        Args:
            raw_points: (n_raw_points, n_raw_features) raw feature array
            raw_distances: (n_raw_points,) distances along transect
            window_start: Start of window in meters
            window_end: End of window in meters

        Returns:
            Dictionary with processed 'points' and 'distances', or None if failed
        """
        # Find points within window
        mask = (raw_distances >= window_start) & (raw_distances <= window_end)
        n_in_window = mask.sum()

        if n_in_window < 2:
            return None

        # Get windowed data
        win_dist = raw_distances[mask]
        win_pts = raw_points[mask]

        # Create target distances (relative to window start, evenly spaced)
        target_dist = np.linspace(0, win_dist[-1] - win_dist[0], self.n_output_points)

        # Normalize source distances to start at 0
        src_dist = win_dist - win_dist[0]

        # Interpolate each feature
        try:
            out_points = np.zeros((self.n_output_points, len(KEEP_FEATURE_INDICES)), dtype=np.float32)

            for out_idx, raw_idx in enumerate(KEEP_FEATURE_INDICES):
                if raw_idx == 0:
                    # distance_m: use target distances (relative to window)
                    out_points[:, out_idx] = target_dist
                elif raw_idx in [9]:  # classification: nearest neighbor
                    interp_func = interp1d(
                        src_dist, win_pts[:, raw_idx],
                        kind='nearest', fill_value='extrapolate'
                    )
                    out_points[:, out_idx] = interp_func(target_dist)
                else:
                    # Linear interpolation for continuous features
                    interp_func = interp1d(
                        src_dist, win_pts[:, raw_idx],
                        kind='linear', fill_value='extrapolate'
                    )
                    out_points[:, out_idx] = interp_func(target_dist)

            # Recompute slope and curvature from resampled elevation
            elevation = out_points[:, 1]  # elevation_m is index 1
            slope, curvature = self._compute_derivatives(target_dist, elevation)
            out_points[:, 2] = slope       # slope_deg is index 2
            out_points[:, 3] = curvature   # curvature is index 3

        except Exception as e:
            logger.debug(f"Interpolation failed: {e}")
            return None

        return {
            'points': out_points,
            'distances': target_dist.astype(np.float32),
        }

    def _compute_derivatives(
        self,
        distances: np.ndarray,
        elevation: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute slope and curvature from elevation profile.

        Args:
            distances: (n_points,) distance along transect
            elevation: (n_points,) elevation values

        Returns:
            Tuple of (slope_deg, curvature) arrays
        """
        n = len(distances)

        # Compute step size
        dx = (distances[-1] - distances[0]) / (n - 1) if n > 1 else 1.0
        dx = max(dx, 0.01)  # Minimum 1cm step

        # Smooth elevation before derivatives
        smooth_window = 5
        z_smooth = np.convolve(
            elevation,
            np.ones(smooth_window) / smooth_window,
            mode='same'
        )
        # Handle edge effects
        half_win = smooth_window // 2
        z_smooth[:half_win] = elevation[:half_win]
        z_smooth[-half_win:] = elevation[-half_win:]

        # First derivative (slope) using central differences
        dz_dx = np.zeros(n)
        dz_dx[0] = (z_smooth[1] - z_smooth[0]) / dx
        dz_dx[-1] = (z_smooth[-1] - z_smooth[-2]) / dx
        dz_dx[1:-1] = (z_smooth[2:] - z_smooth[:-2]) / (2 * dx)

        # Slope in degrees
        slope = np.degrees(np.arctan(dz_dx))

        # Second derivative
        d2z_dx2 = np.zeros(n)
        d2z_dx2[1:-1] = (z_smooth[2:] - 2 * z_smooth[1:-1] + z_smooth[:-2]) / (dx ** 2)
        d2z_dx2[0] = d2z_dx2[1]
        d2z_dx2[-1] = d2z_dx2[-2]

        # Profile curvature
        curvature = d2z_dx2 / np.power(1 + dz_dx ** 2, 1.5)
        curvature = np.clip(curvature, -10.0, 10.0)

        return slope.astype(np.float32), curvature.astype(np.float32)


def process_directory(
    raw_dir: Union[str, Path],
    output_dir: Union[str, Path],
    n_output_points: int = 128,
    toe_buffer_m: float = 10.0,
    top_buffer_m: float = 5.0,
    min_cliff_width_m: float = 5.0,
    fallback_to_full: bool = True,
    pattern: str = "*.npz",
    skip_existing: bool = False,
) -> Dict[str, any]:
    """Process all transect files in a directory.

    Expects raw transect files (*.npz) with corresponding cliff delineation
    sidecar files (*.cliff.npz) in the same directory.

    Args:
        raw_dir: Directory containing raw transect NPZ files
        output_dir: Directory to save processed files
        n_output_points: Points per processed transect
        toe_buffer_m: Distance seaward of toe to include
        top_buffer_m: Distance landward of top to include
        min_cliff_width_m: Minimum cliff width to process
        fallback_to_full: Use full transect when cliff not detected
        pattern: Glob pattern for raw NPZ files
        skip_existing: Skip files that already exist in output_dir

    Returns:
        Dictionary with overall processing statistics
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all raw NPZ files (exclude cliff sidecar files)
    raw_files = [f for f in sorted(raw_dir.glob(pattern)) if '.cliff.' not in f.name]

    if not raw_files:
        logger.warning(f"No raw transect files found in {raw_dir}")
        return {'files_processed': 0}

    logger.info(f"Found {len(raw_files)} raw transect files to process")

    # Initialize processor
    processor = TransectProcessor(
        n_output_points=n_output_points,
        toe_buffer_m=toe_buffer_m,
        top_buffer_m=top_buffer_m,
        min_cliff_width_m=min_cliff_width_m,
        fallback_to_full=fallback_to_full,
    )

    # Aggregate statistics
    overall_stats = {
        'files_processed': 0,
        'files_skipped': 0,
        'files_failed': 0,
        'total_processed': 0,
        'total_with_cliff': 0,
        'total_fallback': 0,
    }

    for raw_path in raw_files:
        # Check for cliff data: first try combined file, then sidecar
        cliff_path = None
        try:
            with np.load(raw_path, allow_pickle=True) as data:
                has_embedded_cliff = 'toe_distances' in data
        except Exception:
            has_embedded_cliff = False

        if not has_embedded_cliff:
            # Look for sidecar file
            cliff_path = raw_path.with_suffix('.cliff.npz')
            if not cliff_path.exists():
                cliff_path = raw_path.parent / f"{raw_path.stem}.cliff.npz"

            if not cliff_path.exists():
                logger.warning(f"No cliff delineation found for {raw_path.name}, skipping")
                overall_stats['files_skipped'] += 1
                continue

        # Determine output path
        output_path = output_dir / raw_path.name

        if skip_existing and output_path.exists():
            logger.debug(f"Skipping existing: {output_path.name}")
            overall_stats['files_skipped'] += 1
            continue

        # Process
        try:
            logger.info(f"Processing {raw_path.name}...")
            stats = processor.process(raw_path, output_path, cliff_npz_path=cliff_path)

            overall_stats['files_processed'] += 1
            overall_stats['total_processed'] += stats['n_processed']
            overall_stats['total_with_cliff'] += stats['n_with_cliff']
            overall_stats['total_fallback'] += stats['n_fallback']

        except Exception as e:
            logger.error(f"Failed to process {raw_path.name}: {e}")
            overall_stats['files_failed'] += 1

    # Summary
    logger.info("=" * 60)
    logger.info("DIRECTORY PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Files processed: {overall_stats['files_processed']}")
    logger.info(f"Files skipped: {overall_stats['files_skipped']}")
    logger.info(f"Files failed: {overall_stats['files_failed']}")
    logger.info(f"Total transect-epochs: {overall_stats['total_processed']}")
    logger.info(f"  With cliff detection: {overall_stats['total_with_cliff']}")
    logger.info(f"  Fallback to full: {overall_stats['total_fallback']}")

    return overall_stats
