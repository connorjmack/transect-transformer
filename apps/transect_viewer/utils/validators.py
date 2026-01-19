"""Data validation utilities for transect data."""

from typing import Any

import numpy as np
import pandas as pd


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
    Check for NaN values per feature.

    Args:
        points: (N, 128, 12) array of transect points
        feature_names: List of 12 feature names

    Returns:
        Dictionary mapping feature names to NaN counts
    """
    nan_counts = {}
    for i, name in enumerate(feature_names):
        feature_data = points[:, :, i]
        nan_counts[name] = int(np.isnan(feature_data).sum())
    return nan_counts


def check_value_ranges(
    points: np.ndarray,
    feature_names: list[str]
) -> list[dict[str, Any]]:
    """
    Check that feature values are within expected ranges.

    Args:
        points: (N, 128, 12) array of transect points
        feature_names: List of 12 feature names

    Returns:
        List of issues found, each with keys:
        - feature: feature name
        - issue: 'below_min' or 'above_max'
        - count: number of violations
        - min_val/max_val: actual extreme value
        - expected_min/expected_max: expected range
    """
    issues = []

    for i, name in enumerate(feature_names):
        if name not in FEATURE_RANGES:
            continue

        expected_min, expected_max = FEATURE_RANGES[name]
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
    Check that metadata values are within expected ranges.

    Args:
        metadata: (N, 12) array of transect metadata
        metadata_names: List of 12 metadata field names

    Returns:
        List of issues found (same format as check_value_ranges)
    """
    issues = []

    for i, name in enumerate(metadata_names):
        if name not in METADATA_RANGES:
            continue

        expected_min, expected_max = METADATA_RANGES[name]
        values = metadata[:, i]

        # Remove NaN for comparison
        valid_values = values[~np.isnan(values)]
        if len(valid_values) == 0:
            continue

        # Skip coordinate checks for UTM (large values expected)
        if name in ['latitude', 'longitude']:
            # Check if values look like UTM (> 180 or < -180)
            if valid_values.min() < -180 or valid_values.max() > 180:
                continue  # Skip range check for UTM coordinates

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
    Full validation report for a dataset.

    Args:
        data: Dataset from load_npz

    Returns:
        Validation report with keys:
        - is_valid: bool
        - n_transects: int
        - n_points_per_transect: int
        - n_features: int
        - nan_counts: dict of NaN counts per feature
        - feature_issues: list of range issues
        - metadata_issues: list of metadata range issues
        - warnings: list of warning messages
    """
    points = data['points']
    metadata = data['metadata']
    feature_names = data.get('feature_names', [])
    metadata_names = data.get('metadata_names', [])

    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()
    if isinstance(metadata_names, np.ndarray):
        metadata_names = metadata_names.tolist()

    report = {
        'is_valid': True,
        'n_transects': points.shape[0],
        'n_points_per_transect': points.shape[1],
        'n_features': points.shape[2],
        'n_metadata_fields': metadata.shape[1],
        'nan_counts': {},
        'feature_issues': [],
        'metadata_issues': [],
        'warnings': [],
    }

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
    if points.shape[1] != 128:
        report['warnings'].append(f"Unexpected points per transect: {points.shape[1]} (expected 128)")

    if points.shape[2] != 12:
        report['warnings'].append(f"Unexpected feature count: {points.shape[2]} (expected 12)")

    if metadata.shape[1] != 12:
        report['warnings'].append(f"Unexpected metadata count: {metadata.shape[1]} (expected 12)")

    # Mark as invalid if critical issues
    if report['warnings']:
        # Only mark invalid for critical issues (NaN or shape problems)
        if any('NaN' in w for w in report['warnings']):
            report['is_valid'] = False

    return report


def compute_statistics(data: dict[str, Any]) -> pd.DataFrame:
    """
    Compute summary statistics for all features.

    Args:
        data: Dataset from load_npz

    Returns:
        DataFrame with statistics for each feature
    """
    points = data['points']
    feature_names = data.get('feature_names', [])

    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()

    if not feature_names:
        feature_names = [f'feature_{i}' for i in range(points.shape[2])]

    stats = []
    for i, name in enumerate(feature_names):
        values = points[:, :, i].flatten()
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


def compute_metadata_statistics(data: dict[str, Any]) -> pd.DataFrame:
    """
    Compute summary statistics for all metadata fields.

    Args:
        data: Dataset from load_npz

    Returns:
        DataFrame with statistics for each metadata field
    """
    metadata = data['metadata']
    metadata_names = data.get('metadata_names', [])

    if isinstance(metadata_names, np.ndarray):
        metadata_names = metadata_names.tolist()

    if not metadata_names:
        metadata_names = [f'meta_{i}' for i in range(metadata.shape[1])]

    stats = []
    for i, name in enumerate(metadata_names):
        values = metadata[:, i]
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
