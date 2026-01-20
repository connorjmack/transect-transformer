"""Data loading utilities with Streamlit caching for CUBE FORMAT data.

Cube format arrays:
- points: (n_transects, T, N, 12) - point features across all epochs
- distances: (n_transects, T, N) - distance along transect
- metadata: (n_transects, T, 12) - per-epoch metadata
- timestamps: (n_transects, T) - scan dates as ordinal days
- transect_ids: (n_transects,) - unique transect IDs
- epoch_names: (T,) - LAS filenames for each epoch
- epoch_dates: (T,) - ISO date strings for each epoch
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import streamlit as st


@st.cache_data
def load_npz(file_path: str) -> dict[str, Any]:
    """
    Load NPZ file with Streamlit caching (cube format).

    Args:
        file_path: Path to NPZ file

    Returns:
        Dictionary with numpy arrays in cube format
    """
    data = np.load(file_path, allow_pickle=True)

    # Convert to regular dict and handle object arrays
    result = {}
    for key in data.keys():
        arr = data[key]
        if arr.dtype == object:
            # Convert object arrays to lists for string data
            result[key] = arr.tolist() if arr.ndim == 1 else arr
        else:
            result[key] = arr

    return result


def load_npz_from_upload(uploaded_file) -> dict[str, Any]:
    """
    Load NPZ file from Streamlit file uploader (cube format).

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        Dictionary with numpy arrays in cube format
    """
    data = np.load(uploaded_file, allow_pickle=True)

    result = {}
    for key in data.keys():
        arr = data[key]
        if arr.dtype == object:
            result[key] = arr.tolist() if arr.ndim == 1 else arr
        else:
            result[key] = arr

    return result


def is_cube_format(data: dict[str, Any]) -> bool:
    """Check if data is in cube format (4D points array)."""
    points = data.get('points')
    if points is None:
        return False
    return points.ndim == 4


def get_cube_dimensions(data: dict[str, Any]) -> dict[str, int]:
    """
    Get dimensions of cube format data.

    Returns:
        Dictionary with n_transects, n_epochs, n_points, n_features
    """
    points = data['points']
    if points.ndim == 4:
        n_transects, n_epochs, n_points, n_features = points.shape
    else:
        # Flat format fallback
        n_transects, n_points, n_features = points.shape
        n_epochs = 1

    return {
        'n_transects': n_transects,
        'n_epochs': n_epochs,
        'n_points': n_points,
        'n_features': n_features,
    }


def get_epoch_dates(data: dict[str, Any]) -> list[str]:
    """Get list of epoch date strings."""
    epoch_dates = data.get('epoch_dates', [])
    if isinstance(epoch_dates, np.ndarray):
        epoch_dates = epoch_dates.tolist()
    return epoch_dates


def get_epoch_names(data: dict[str, Any]) -> list[str]:
    """Get list of epoch filenames."""
    epoch_names = data.get('epoch_names', [])
    if isinstance(epoch_names, np.ndarray):
        epoch_names = epoch_names.tolist()
    return epoch_names


def get_transect_by_id(
    data: dict[str, Any],
    transect_id: int,
    epoch_idx: Optional[int] = None
) -> dict[str, Any]:
    """
    Extract single transect data by ID.

    Args:
        data: Full dataset in cube format
        transect_id: Transect ID to extract
        epoch_idx: Optional epoch index. If None, returns all epochs.

    Returns:
        Dictionary with transect data:
        - If epoch_idx is None: full temporal data (T, N, 12)
        - If epoch_idx is int: single epoch data (N, 12)
    """
    transect_ids = data['transect_ids']
    if isinstance(transect_ids, list):
        transect_ids = np.array(transect_ids)

    idx = np.where(transect_ids == transect_id)[0]
    if len(idx) == 0:
        raise ValueError(f"Transect ID {transect_id} not found")
    idx = idx[0]

    points = data['points']
    distances = data['distances']
    metadata = data['metadata']

    if epoch_idx is not None:
        # Single epoch
        return {
            'points': points[idx, epoch_idx],  # (N, 12)
            'distances': distances[idx, epoch_idx],  # (N,)
            'metadata': metadata[idx, epoch_idx],  # (12,)
            'transect_id': transect_id,
            'epoch_idx': epoch_idx,
            'epoch_date': get_epoch_dates(data)[epoch_idx] if get_epoch_dates(data) else None,
            'feature_names': data.get('feature_names', []),
            'metadata_names': data.get('metadata_names', []),
        }
    else:
        # All epochs
        return {
            'points': points[idx],  # (T, N, 12)
            'distances': distances[idx],  # (T, N)
            'metadata': metadata[idx],  # (T, 12)
            'transect_id': transect_id,
            'epoch_dates': get_epoch_dates(data),
            'epoch_names': get_epoch_names(data),
            'feature_names': data.get('feature_names', []),
            'metadata_names': data.get('metadata_names', []),
        }


def get_transect_temporal_slice(
    data: dict[str, Any],
    transect_id: int,
    feature_name: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Get temporal evolution of a feature for a transect.

    Args:
        data: Full dataset in cube format
        transect_id: Transect ID
        feature_name: Name of feature to extract

    Returns:
        Tuple of (distances, values, epoch_dates)
        - distances: (T, N) array
        - values: (T, N) array of feature values
        - epoch_dates: list of date strings
    """
    transect = get_transect_by_id(data, transect_id)

    feature_names = transect.get('feature_names', [])
    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()

    if feature_name not in feature_names:
        raise ValueError(f"Feature '{feature_name}' not found")

    feature_idx = feature_names.index(feature_name)

    return (
        transect['distances'],  # (T, N)
        transect['points'][:, :, feature_idx],  # (T, N)
        transect['epoch_dates'],
    )


def get_all_transect_ids(data: dict[str, Any]) -> list[int]:
    """Get sorted list of all transect IDs in dataset."""
    transect_ids = data['transect_ids']
    if isinstance(transect_ids, list):
        return sorted(transect_ids)
    return sorted(transect_ids.tolist())


def get_epoch_slice(
    data: dict[str, Any],
    epoch_idx: int
) -> dict[str, Any]:
    """
    Get all transects for a single epoch.

    Args:
        data: Full dataset in cube format
        epoch_idx: Epoch index

    Returns:
        Dictionary with single-epoch data:
        - points: (n_transects, N, 12)
        - distances: (n_transects, N)
        - metadata: (n_transects, 12)
    """
    return {
        'points': data['points'][:, epoch_idx],
        'distances': data['distances'][:, epoch_idx],
        'metadata': data['metadata'][:, epoch_idx],
        'transect_ids': data['transect_ids'],
        'epoch_idx': epoch_idx,
        'epoch_date': get_epoch_dates(data)[epoch_idx] if get_epoch_dates(data) else None,
        'epoch_name': get_epoch_names(data)[epoch_idx] if get_epoch_names(data) else None,
        'feature_names': data.get('feature_names', []),
        'metadata_names': data.get('metadata_names', []),
    }


def compute_temporal_change(
    data: dict[str, Any],
    transect_id: int,
    feature_name: str,
    epoch1_idx: int = 0,
    epoch2_idx: int = -1
) -> dict[str, Any]:
    """
    Compute change in a feature between two epochs.

    Args:
        data: Full dataset in cube format
        transect_id: Transect ID
        feature_name: Feature to compare
        epoch1_idx: First epoch index (default: 0, earliest)
        epoch2_idx: Second epoch index (default: -1, latest)

    Returns:
        Dictionary with change data
    """
    transect = get_transect_by_id(data, transect_id)

    feature_names = transect.get('feature_names', [])
    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()

    feature_idx = feature_names.index(feature_name)

    n_epochs = transect['points'].shape[0]
    if epoch2_idx < 0:
        epoch2_idx = n_epochs + epoch2_idx

    values1 = transect['points'][epoch1_idx, :, feature_idx]
    values2 = transect['points'][epoch2_idx, :, feature_idx]
    distances = transect['distances'][epoch2_idx]

    difference = values2 - values1

    epoch_dates = transect.get('epoch_dates', [])

    return {
        'distances': distances,
        'values1': values1,
        'values2': values2,
        'difference': difference,
        'epoch1_date': epoch_dates[epoch1_idx] if epoch_dates else f"Epoch {epoch1_idx}",
        'epoch2_date': epoch_dates[epoch2_idx] if epoch_dates else f"Epoch {epoch2_idx}",
        'mean_change': float(np.nanmean(difference)),
        'max_change': float(np.nanmax(difference)),
        'min_change': float(np.nanmin(difference)),
        'std_change': float(np.nanstd(difference)),
    }


def has_cliff_data(data: dict[str, Any]) -> bool:
    """Check if cliff detection data is available in the dataset."""
    # Check for raw format (from extraction with cliff detection)
    if 'toe_distances' in data and 'top_distances' in data:
        return True
    # Check for processed format (from transect_processor)
    if 'toe_relative_m' in data and 'top_relative_m' in data:
        return True
    return False


def get_cliff_positions(
    data: dict[str, Any],
    transect_idx: int,
    epoch_idx: int
) -> dict[str, Any] | None:
    """
    Get cliff toe and top positions for a specific transect-epoch.

    Args:
        data: Full dataset (must contain cliff detection arrays)
        transect_idx: Index of transect in the data arrays
        epoch_idx: Epoch index

    Returns:
        Dictionary with toe/top info, or None if no cliff detected
    """
    if not has_cliff_data(data):
        return None

    # Check for processed format first (toe_relative_m, top_relative_m)
    if 'toe_relative_m' in data:
        return _get_cliff_positions_processed(data, transect_idx, epoch_idx)

    # Original raw format (toe_distances, top_distances, has_cliff)
    return _get_cliff_positions_raw(data, transect_idx, epoch_idx)


def _get_cliff_positions_raw(
    data: dict[str, Any],
    transect_idx: int,
    epoch_idx: int
) -> dict[str, Any] | None:
    """Get cliff positions from raw extraction format."""
    has_cliff = data['has_cliff']
    toe_distances = data['toe_distances']
    top_distances = data['top_distances']
    toe_indices = data.get('toe_indices')
    top_indices = data.get('top_indices')
    toe_confidences = data.get('toe_confidences')
    top_confidences = data.get('top_confidences')

    # Handle both cube (n_transects, n_epochs) and flat (n_transects,) formats
    if has_cliff.ndim == 2:
        if not has_cliff[transect_idx, epoch_idx]:
            return None
        return {
            'has_cliff': True,
            'toe_distance': float(toe_distances[transect_idx, epoch_idx]),
            'top_distance': float(top_distances[transect_idx, epoch_idx]),
            'toe_idx': int(toe_indices[transect_idx, epoch_idx]) if toe_indices is not None else None,
            'top_idx': int(top_indices[transect_idx, epoch_idx]) if top_indices is not None else None,
            'toe_confidence': float(toe_confidences[transect_idx, epoch_idx]) if toe_confidences is not None else None,
            'top_confidence': float(top_confidences[transect_idx, epoch_idx]) if top_confidences is not None else None,
        }
    else:
        # Flat format (single epoch)
        if not has_cliff[transect_idx]:
            return None
        return {
            'has_cliff': True,
            'toe_distance': float(toe_distances[transect_idx]),
            'top_distance': float(top_distances[transect_idx]),
            'toe_idx': int(toe_indices[transect_idx]) if toe_indices is not None else None,
            'top_idx': int(top_indices[transect_idx]) if top_indices is not None else None,
            'toe_confidence': float(toe_confidences[transect_idx]) if toe_confidences is not None else None,
            'top_confidence': float(top_confidences[transect_idx]) if top_confidences is not None else None,
        }


def _get_cliff_positions_processed(
    data: dict[str, Any],
    transect_idx: int,
    epoch_idx: int
) -> dict[str, Any] | None:
    """Get cliff positions from processed format (transect_processor output)."""
    toe_relative = data['toe_relative_m']
    top_relative = data['top_relative_m']
    distances = data['distances']
    delineation_conf = data.get('delineation_confidence')
    used_fallback = data.get('used_fallback')

    # Handle both cube (n_transects, n_epochs) and flat (n_transects,) formats
    if toe_relative.ndim == 2:
        toe_dist = toe_relative[transect_idx, epoch_idx]
        top_dist = top_relative[transect_idx, epoch_idx]
        trans_distances = distances[transect_idx, epoch_idx]
        confidence = delineation_conf[transect_idx, epoch_idx] if delineation_conf is not None else None
        is_fallback = used_fallback[transect_idx, epoch_idx] if used_fallback is not None else False
    else:
        toe_dist = toe_relative[transect_idx]
        top_dist = top_relative[transect_idx]
        trans_distances = distances[transect_idx]
        confidence = delineation_conf[transect_idx] if delineation_conf is not None else None
        is_fallback = used_fallback[transect_idx] if used_fallback is not None else False

    # Check if cliff data is valid (not NaN and not fallback)
    if np.isnan(toe_dist) or np.isnan(top_dist):
        return None

    # Skip fallback transects (no cliff detected, used full transect)
    if is_fallback:
        return None

    # Find indices closest to toe/top distances
    toe_idx = int(np.argmin(np.abs(trans_distances - toe_dist)))
    top_idx = int(np.argmin(np.abs(trans_distances - top_dist)))

    return {
        'has_cliff': True,
        'toe_distance': float(toe_dist),
        'top_distance': float(top_dist),
        'toe_idx': toe_idx,
        'top_idx': top_idx,
        'toe_confidence': float(confidence) if confidence is not None else None,
        'top_confidence': float(confidence) if confidence is not None else None,
    }


def get_cliff_positions_by_id(
    data: dict[str, Any],
    transect_id: int,
    epoch_idx: int
) -> dict[str, Any] | None:
    """
    Get cliff toe and top positions by transect ID.

    Args:
        data: Full dataset
        transect_id: Transect ID to look up
        epoch_idx: Epoch index

    Returns:
        Dictionary with toe/top info, or None if no cliff detected
    """
    if not has_cliff_data(data):
        return None

    transect_ids = data['transect_ids']
    if isinstance(transect_ids, list):
        transect_ids = np.array(transect_ids)

    idx = np.where(transect_ids == transect_id)[0]
    if len(idx) == 0:
        return None
    transect_idx = idx[0]

    return get_cliff_positions(data, transect_idx, epoch_idx)


def get_cliff_elevation_at_position(
    data: dict[str, Any],
    transect_idx: int,
    epoch_idx: int,
    cliff_pos: dict[str, Any]
) -> dict[str, float] | None:
    """
    Get elevation values at cliff toe and top positions.

    Args:
        data: Full dataset
        transect_idx: Transect index
        epoch_idx: Epoch index
        cliff_pos: Output from get_cliff_positions()

    Returns:
        Dictionary with toe_elevation and top_elevation
    """
    if cliff_pos is None:
        return None

    points = data['points']
    toe_idx = cliff_pos.get('toe_idx')
    top_idx = cliff_pos.get('top_idx')

    if toe_idx is None or top_idx is None:
        return None

    # Get elevation (feature index 1)
    if points.ndim == 4:
        toe_elev = float(points[transect_idx, epoch_idx, toe_idx, 1])
        top_elev = float(points[transect_idx, epoch_idx, top_idx, 1])
    else:
        toe_elev = float(points[transect_idx, toe_idx, 1])
        top_elev = float(points[transect_idx, top_idx, 1])

    return {
        'toe_elevation': toe_elev,
        'top_elevation': top_elev,
    }


def check_data_coverage(data: dict[str, Any]) -> dict[str, Any]:
    """
    Check data coverage across transects and epochs.

    Returns:
        Dictionary with coverage statistics
    """
    points = data['points']
    n_transects, n_epochs, n_points, _ = points.shape

    # Check for NaN at first point of each transect-epoch
    coverage_matrix = ~np.isnan(points[:, :, 0, 0])  # (n_transects, n_epochs)

    total_cells = n_transects * n_epochs
    present_cells = coverage_matrix.sum()
    missing_cells = total_cells - present_cells

    # Coverage per epoch
    coverage_per_epoch = coverage_matrix.sum(axis=0)  # (n_epochs,)

    # Coverage per transect
    coverage_per_transect = coverage_matrix.sum(axis=1)  # (n_transects,)

    return {
        'total_cells': total_cells,
        'present_cells': int(present_cells),
        'missing_cells': int(missing_cells),
        'coverage_pct': 100 * present_cells / total_cells,
        'coverage_matrix': coverage_matrix,
        'coverage_per_epoch': coverage_per_epoch,
        'coverage_per_transect': coverage_per_transect,
        'full_coverage': missing_cells == 0,
    }
