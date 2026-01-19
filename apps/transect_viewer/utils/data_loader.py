"""Data loading utilities with Streamlit caching."""

from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

from apps.transect_viewer.utils.date_parser import infer_epoch_date


@st.cache_data
def load_npz(file_path: str) -> dict[str, Any]:
    """
    Load NPZ file with Streamlit caching.

    Args:
        file_path: Path to NPZ file

    Returns:
        Dictionary with numpy arrays:
        - points: (N, 128, 12) transect point features
        - distances: (N, 128) distances along transect
        - metadata: (N, 12) transect-level metadata
        - transect_ids: (N,) unique transect IDs
        - las_sources: (N,) source LAS filenames
        - feature_names: list of 12 feature names
        - metadata_names: list of 12 metadata names
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
    Load NPZ file from Streamlit file uploader.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        Dictionary with numpy arrays (same structure as load_npz)
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


@st.cache_data
def load_multiple_epochs(file_paths: list[str]) -> dict[str, dict[str, Any]]:
    """
    Load multiple NPZ files, keyed by extracted date.

    Args:
        file_paths: List of paths to NPZ files

    Returns:
        Dictionary mapping date strings to data dicts
    """
    epochs = {}

    for path in file_paths:
        data = load_npz(path)

        # Extract date from las_sources
        date = infer_epoch_date(data.get('las_sources', []))
        if date:
            date_key = date.strftime('%Y-%m-%d')
        else:
            # Fallback to filename
            date_key = Path(path).stem

        epochs[date_key] = data

    return epochs


def get_transect_by_id(data: dict[str, Any], transect_id: int) -> dict[str, Any]:
    """
    Extract single transect data by ID.

    Args:
        data: Full dataset from load_npz
        transect_id: Transect ID to extract

    Returns:
        Dictionary with single transect data:
        - points: (128, 12) features
        - distances: (128,) distances
        - metadata: (12,) metadata
        - transect_id: int
        - las_source: str
    """
    # Find index for this transect ID
    transect_ids = data['transect_ids']
    if isinstance(transect_ids, list):
        transect_ids = np.array(transect_ids)

    idx = np.where(transect_ids == transect_id)[0]
    if len(idx) == 0:
        raise ValueError(f"Transect ID {transect_id} not found")
    idx = idx[0]

    las_sources = data['las_sources']

    return {
        'points': data['points'][idx],
        'distances': data['distances'][idx],
        'metadata': data['metadata'][idx],
        'transect_id': transect_id,
        'las_source': las_sources[idx] if isinstance(las_sources, list) else las_sources[idx],
        'feature_names': data.get('feature_names', []),
        'metadata_names': data.get('metadata_names', []),
    }


def get_feature_by_name(
    data: dict[str, Any],
    transect_idx: int,
    feature_name: str
) -> np.ndarray:
    """
    Get a specific feature array for a transect.

    Args:
        data: Full dataset from load_npz
        transect_idx: Index of transect (not ID)
        feature_name: Name of feature to extract

    Returns:
        (128,) array of feature values
    """
    feature_names = data.get('feature_names', [])
    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()

    if feature_name not in feature_names:
        raise ValueError(f"Feature '{feature_name}' not found. Available: {feature_names}")

    feature_idx = feature_names.index(feature_name)
    return data['points'][transect_idx, :, feature_idx]


def get_all_transect_ids(data: dict[str, Any]) -> list[int]:
    """Get sorted list of all transect IDs in dataset."""
    transect_ids = data['transect_ids']
    if isinstance(transect_ids, list):
        return sorted(transect_ids)
    return sorted(transect_ids.tolist())


def get_common_transect_ids(epochs: dict[str, dict[str, Any]]) -> list[int]:
    """
    Find transect IDs present in all epochs.

    Args:
        epochs: Dictionary of epoch data from load_multiple_epochs

    Returns:
        Sorted list of transect IDs present in all epochs
    """
    if not epochs:
        return []

    # Get sets of IDs from each epoch
    id_sets = []
    for epoch_data in epochs.values():
        ids = epoch_data['transect_ids']
        if isinstance(ids, list):
            id_sets.append(set(ids))
        else:
            id_sets.append(set(ids.tolist()))

    # Find intersection
    common_ids = id_sets[0]
    for id_set in id_sets[1:]:
        common_ids = common_ids.intersection(id_set)

    return sorted(list(common_ids))
