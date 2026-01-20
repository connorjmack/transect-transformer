"""Data loading utilities for the transect labeler.

Reuses and extends functionality from transect_viewer.
"""

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


def get_transect_ids(data: dict[str, Any]) -> list:
    """Get list of transect IDs."""
    transect_ids = data.get('transect_ids', [])
    if isinstance(transect_ids, np.ndarray):
        transect_ids = transect_ids.tolist()
    return transect_ids


def get_feature_names(data: dict[str, Any]) -> list[str]:
    """Get list of feature names."""
    feature_names = data.get('feature_names', [])
    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()
    return feature_names


def get_transect_pair(
    data: dict[str, Any],
    transect_idx: int,
    pair_idx: int
) -> dict[str, Any]:
    """
    Extract data for a transect epoch pair.

    Args:
        data: Full dataset in cube format
        transect_idx: Index of transect (0-based)
        pair_idx: Index of epoch pair (0-based, pair_idx and pair_idx+1)

    Returns:
        Dictionary with:
        - epoch1: data for earlier epoch
        - epoch2: data for later epoch
        - transect_id: transect identifier
        - epoch1_date, epoch2_date: date strings
    """
    points = data['points']
    distances = data['distances']
    metadata = data['metadata']
    epoch_dates = get_epoch_dates(data)
    transect_ids = get_transect_ids(data)
    feature_names = get_feature_names(data)

    epoch1_idx = pair_idx
    epoch2_idx = pair_idx + 1

    return {
        'epoch1': {
            'points': points[transect_idx, epoch1_idx],  # (N, 12)
            'distances': distances[transect_idx, epoch1_idx],  # (N,)
            'metadata': metadata[transect_idx, epoch1_idx],  # (12,)
        },
        'epoch2': {
            'points': points[transect_idx, epoch2_idx],  # (N, 12)
            'distances': distances[transect_idx, epoch2_idx],  # (N,)
            'metadata': metadata[transect_idx, epoch2_idx],  # (12,)
        },
        'transect_idx': transect_idx,
        'transect_id': transect_ids[transect_idx] if transect_ids else transect_idx,
        'epoch1_idx': epoch1_idx,
        'epoch2_idx': epoch2_idx,
        'epoch1_date': epoch_dates[epoch1_idx][:10] if epoch_dates else f"Epoch {epoch1_idx}",
        'epoch2_date': epoch_dates[epoch2_idx][:10] if epoch_dates else f"Epoch {epoch2_idx}",
        'feature_names': feature_names,
    }


def compute_pair_change(
    pair_data: dict[str, Any],
    feature_idx: int = 1
) -> dict[str, Any]:
    """
    Compute change statistics for an epoch pair.

    Args:
        pair_data: Output from get_transect_pair
        feature_idx: Index of feature to compute change for (default 1 = elevation)

    Returns:
        Dictionary with change statistics
    """
    val1 = pair_data['epoch1']['points'][:, feature_idx]
    val2 = pair_data['epoch2']['points'][:, feature_idx]
    dist2 = pair_data['epoch2']['distances']

    # Interpolate val1 to val2's distance grid if needed
    dist1 = pair_data['epoch1']['distances']
    if not np.allclose(dist1, dist2):
        val1_interp = np.interp(dist2, dist1, val1)
    else:
        val1_interp = val1

    difference = val2 - val1_interp

    # Handle NaN values
    valid_mask = ~np.isnan(difference)
    valid_diff = difference[valid_mask]

    if len(valid_diff) == 0:
        return {
            'difference': difference,
            'distances': dist2,
            'mean_change': 0.0,
            'max_gain': 0.0,
            'max_loss': 0.0,
            'std_change': 0.0,
        }

    return {
        'difference': difference,
        'distances': dist2,
        'mean_change': float(np.mean(valid_diff)),
        'max_gain': float(np.max(valid_diff)),
        'max_loss': float(np.min(valid_diff)),
        'std_change': float(np.std(valid_diff)),
    }
