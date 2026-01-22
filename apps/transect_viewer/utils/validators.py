"""Data validation utilities for CUBE FORMAT transect data.

Cube format: (n_transects, T, N, 12) where T = temporal epochs
"""

from typing import Any

import numpy as np
import pandas as pd

from apps.transect_viewer.utils.helpers import safe_date_label


# Expected value ranges for each feature
FEATURE_RANGES = {
    'distance_m': (0, 500),
    'elevation_m': (-50, 500),
    'slope_deg': (-90, 90),
    'curvature': (-10, 10),
    'roughness': (0, 10),
    'intensity': (0, 1),
    'red': (0, 1),
    'green': (0, 1),
    'blue': (0, 1),
    'classification': (0, 31),
    'return_number': (0, 15),
    'num_returns': (1, 15),
}

# Expected value ranges for metadata
METADATA_RANGES = {
    'cliff_height_m': (0, 200),
    'mean_slope_deg': (0, 90),
    'max_slope_deg': (0, 90),
    'toe_elevation_m': (-20, 100),
    'top_elevation_m': (-20, 500),
    'orientation_deg': (0, 360),
    'transect_length_m': (0, 1000),
    'latitude': (-90, 90),  # Note: may be UTM Y coordinate
    'longitude': (-180, 180),  # Note: may be UTM X coordinate
    'transect_id': (0, 100000),
    'mean_intensity': (0, 1),
    'dominant_class': (0, 31),
}


def check_nan_values(points: np.ndarray, feature_names: list[str]) -> dict[str, int]:
    """
    Check for NaN values per feature (cube format).

    Args:
        points: (n_transects, T, N, 12) array or (n_transects, N, 12) flat array
        feature_names: List of feature names

    Returns:
        Dictionary mapping feature names to NaN counts
    """
    nan_counts = {}
    n_features = points.shape[-1]

    for i, name in enumerate(feature_names[:n_features]):
        if points.ndim == 4:
            feature_data = points[:, :, :, i]
        else:
            feature_data = points[:, :, i]
        nan_counts[name] = int(np.isnan(feature_data).sum())

    return nan_counts


def check_value_ranges(
    points: np.ndarray,
    feature_names: list[str]
) -> list[dict[str, Any]]:
    """
    Check that feature values are within expected ranges (cube format).

    Args:
        points: (n_transects, T, N, 12) array or (n_transects, N, 12) flat array
        feature_names: List of feature names

    Returns:
        List of issues found
    """
    issues = []
    n_features = points.shape[-1]

    for i, name in enumerate(feature_names[:n_features]):
        if name not in FEATURE_RANGES:
            continue

        expected_min, expected_max = FEATURE_RANGES[name]

        if points.ndim == 4:
            feature_data = points[:, :, :, i].flatten()
        else:
            feature_data = points[:, :, i].flatten()

        # Remove NaN for comparison
        valid_data = feature_data[~np.isnan(feature_data)]
        if len(valid_data) == 0:
            continue

        # Check below minimum
        below_min = valid_data < expected_min
        if below_min.any():
            issues.append({
                'feature': name,
                'issue': 'below_min',
                'count': int(below_min.sum()),
                'min_val': float(valid_data.min()),
                'expected_min': expected_min,
            })

        # Check above maximum
        above_max = valid_data > expected_max
        if above_max.any():
            issues.append({
                'feature': name,
                'issue': 'above_max',
                'count': int(above_max.sum()),
                'max_val': float(valid_data.max()),
                'expected_max': expected_max,
            })

    return issues


def check_metadata_ranges(
    metadata: np.ndarray,
    metadata_names: list[str]
) -> list[dict[str, Any]]:
    """
    Check that metadata values are within expected ranges (cube format).

    Args:
        metadata: (n_transects, T, 12) array or (n_transects, 12) flat array
        metadata_names: List of metadata field names

    Returns:
        List of issues found
    """
    issues = []
    n_meta = metadata.shape[-1]

    for i, name in enumerate(metadata_names[:n_meta]):
        if name not in METADATA_RANGES:
            continue

        expected_min, expected_max = METADATA_RANGES[name]

        if metadata.ndim == 3:
            values = metadata[:, :, i].flatten()
        else:
            values = metadata[:, i]

        # Remove NaN for comparison
        valid_values = values[~np.isnan(values)]
        if len(valid_values) == 0:
            continue

        # Skip coordinate checks for UTM (large values expected)
        if name in ['latitude', 'longitude']:
            if valid_values.min() < -180 or valid_values.max() > 180:
                continue

        below_min = valid_values < expected_min
        if below_min.any():
            issues.append({
                'field': name,
                'issue': 'below_min',
                'count': int(below_min.sum()),
                'min_val': float(valid_values.min()),
                'expected_min': expected_min,
            })

        above_max = valid_values > expected_max
        if above_max.any():
            issues.append({
                'field': name,
                'issue': 'above_max',
                'count': int(above_max.sum()),
                'max_val': float(valid_values.max()),
                'expected_max': expected_max,
            })

    return issues


def validate_dataset(data: dict[str, Any]) -> dict[str, Any]:
    """
    Full validation report for a cube format dataset.

    Args:
        data: Dataset from load_npz (cube format)

    Returns:
        Validation report
    """
    points = data['points']
    metadata = data['metadata']
    feature_names = data.get('feature_names', [])
    metadata_names = data.get('metadata_names', [])

    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()
    if isinstance(metadata_names, np.ndarray):
        metadata_names = metadata_names.tolist()

    # Detect cube vs flat format
    is_cube = points.ndim == 4

    if is_cube:
        n_transects, n_epochs, n_points, n_features = points.shape
        _, _, n_meta = metadata.shape
    else:
        n_transects, n_points, n_features = points.shape
        n_epochs = 1
        n_meta = metadata.shape[1]

    report = {
        'is_valid': True,
        'is_cube_format': is_cube,
        'n_transects': n_transects,
        'n_epochs': n_epochs,
        'n_points_per_transect': n_points,
        'n_features': n_features,
        'n_metadata_fields': n_meta,
        'nan_counts': {},
        'feature_issues': [],
        'metadata_issues': [],
        'warnings': [],
    }

    # Check data coverage (cube format specific)
    if is_cube:
        coverage_matrix = ~np.isnan(points[:, :, 0, 0])
        missing_count = (~coverage_matrix).sum()
        total_cells = n_transects * n_epochs
        if missing_count > 0:
            missing_pct = 100 * missing_count / total_cells
            report['warnings'].append(
                f"Missing data: {missing_count}/{total_cells} ({missing_pct:.1f}%) transect-epoch pairs"
            )
        report['coverage_pct'] = 100 * coverage_matrix.sum() / total_cells

    # Check NaN values
    if feature_names:
        report['nan_counts'] = check_nan_values(points, feature_names)
        total_nan = sum(report['nan_counts'].values())
        if total_nan > 0:
            report['warnings'].append(f"Found {total_nan} NaN values in features")

    # Check feature ranges
    if feature_names:
        report['feature_issues'] = check_value_ranges(points, feature_names)
        if report['feature_issues']:
            report['warnings'].append(f"Found {len(report['feature_issues'])} feature range issues")

    # Check metadata ranges
    if metadata_names:
        report['metadata_issues'] = check_metadata_ranges(metadata, metadata_names)
        if report['metadata_issues']:
            report['warnings'].append(f"Found {len(report['metadata_issues'])} metadata range issues")

    # Check for expected shapes
    if n_points != 128:
        report['warnings'].append(f"Unexpected points per transect: {n_points} (expected 128)")

    if n_features != 12:
        report['warnings'].append(f"Unexpected feature count: {n_features} (expected 12)")

    if n_meta != 12:
        report['warnings'].append(f"Unexpected metadata count: {n_meta} (expected 12)")

    # Mark as invalid if critical issues
    if report['warnings']:
        if any('NaN' in w for w in report['warnings']):
            report['is_valid'] = False

    return report


def compute_statistics(data: dict[str, Any], epoch_idx: int = -1) -> pd.DataFrame:
    """
    Compute summary statistics for all features at a specific epoch.

    Args:
        data: Dataset from load_npz (cube format)
        epoch_idx: Epoch index to compute stats for (default: -1 = latest)

    Returns:
        DataFrame with statistics for each feature
    """
    points = data['points']
    feature_names = data.get('feature_names', [])

    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()

    # Handle cube vs flat format
    if points.ndim == 4:
        n_epochs = points.shape[1]
        if epoch_idx < 0:
            epoch_idx = n_epochs + epoch_idx
        points_slice = points[:, epoch_idx]  # (n_transects, N, 12)
    else:
        points_slice = points

    if not feature_names:
        feature_names = [f'feature_{i}' for i in range(points_slice.shape[2])]

    stats = []
    for i, name in enumerate(feature_names):
        values = points_slice[:, :, i].flatten()
        valid_values = values[~np.isnan(values)]

        stats.append({
            'feature': name,
            'min': float(valid_values.min()) if len(valid_values) > 0 else np.nan,
            'max': float(valid_values.max()) if len(valid_values) > 0 else np.nan,
            'mean': float(valid_values.mean()) if len(valid_values) > 0 else np.nan,
            'std': float(valid_values.std()) if len(valid_values) > 0 else np.nan,
            'median': float(np.median(valid_values)) if len(valid_values) > 0 else np.nan,
            'nan_count': int(np.isnan(values).sum()),
        })

    return pd.DataFrame(stats)


def compute_metadata_statistics(data: dict[str, Any], epoch_idx: int = -1) -> pd.DataFrame:
    """
    Compute summary statistics for all metadata fields at a specific epoch.

    Args:
        data: Dataset from load_npz (cube format)
        epoch_idx: Epoch index to compute stats for (default: -1 = latest)

    Returns:
        DataFrame with statistics for each metadata field
    """
    metadata = data['metadata']
    metadata_names = data.get('metadata_names', [])

    if isinstance(metadata_names, np.ndarray):
        metadata_names = metadata_names.tolist()

    # Handle cube vs flat format
    if metadata.ndim == 3:
        n_epochs = metadata.shape[1]
        if epoch_idx < 0:
            epoch_idx = n_epochs + epoch_idx
        metadata_slice = metadata[:, epoch_idx]  # (n_transects, 12)
    else:
        metadata_slice = metadata

    if not metadata_names:
        metadata_names = [f'meta_{i}' for i in range(metadata_slice.shape[1])]

    stats = []
    for i, name in enumerate(metadata_names):
        values = metadata_slice[:, i]
        valid_values = values[~np.isnan(values)]

        stats.append({
            'field': name,
            'min': float(valid_values.min()) if len(valid_values) > 0 else np.nan,
            'max': float(valid_values.max()) if len(valid_values) > 0 else np.nan,
            'mean': float(valid_values.mean()) if len(valid_values) > 0 else np.nan,
            'std': float(valid_values.std()) if len(valid_values) > 0 else np.nan,
            'median': float(np.median(valid_values)) if len(valid_values) > 0 else np.nan,
        })

    return pd.DataFrame(stats)


def compute_temporal_statistics(data: dict[str, Any], feature_name: str) -> pd.DataFrame:
    """
    Compute statistics for a feature across all epochs.

    Args:
        data: Dataset from load_npz (cube format)
        feature_name: Feature to analyze

    Returns:
        DataFrame with per-epoch statistics
    """
    points = data['points']
    feature_names = data.get('feature_names', [])
    epoch_dates = data.get('epoch_dates', [])

    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()
    if isinstance(epoch_dates, np.ndarray):
        epoch_dates = epoch_dates.tolist()

    if feature_name not in feature_names:
        raise ValueError(f"Feature '{feature_name}' not found")

    feature_idx = feature_names.index(feature_name)
    n_epochs = points.shape[1]

    stats = []
    for t in range(n_epochs):
        values = points[:, t, :, feature_idx].flatten()
        valid_values = values[~np.isnan(values)]

        epoch_label = safe_date_label(epoch_dates, t)

        stats.append({
            'epoch': epoch_label,
            'epoch_idx': t,
            'min': float(valid_values.min()) if len(valid_values) > 0 else np.nan,
            'max': float(valid_values.max()) if len(valid_values) > 0 else np.nan,
            'mean': float(valid_values.mean()) if len(valid_values) > 0 else np.nan,
            'std': float(valid_values.std()) if len(valid_values) > 0 else np.nan,
            'n_valid': len(valid_values),
        })

    return pd.DataFrame(stats)
