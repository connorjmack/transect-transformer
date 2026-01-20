"""Labels file management: create, load, save, validate."""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import hashlib

import numpy as np


def get_default_labels_path(data_path: str) -> Path:
    """Generate default labels file path from data file path."""
    data_path = Path(data_path)
    stem = data_path.stem
    return data_path.parent / f"{stem}_labels.npz"


def compute_file_hash(file_path: str) -> str:
    """Compute MD5 hash of file for integrity checking."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def create_empty_labels(
    data: dict[str, Any],
    data_path: str,
    labeler_name: str = "",
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Create empty labels array and metadata for a data file.

    Args:
        data: Loaded transect data dict
        data_path: Path to source data file
        labeler_name: Name of labeler creating the file

    Returns:
        Tuple of (labels_array, metadata_dict)
    """
    n_transects = data['points'].shape[0]
    n_epochs = data['points'].shape[1]
    n_pairs = n_epochs - 1

    # Initialize all as unlabeled (-1)
    labels = np.full((n_transects, n_pairs), -1, dtype=np.int8)

    # Build metadata
    metadata = {
        'transect_ids': np.array(data['transect_ids']),
        'epoch_dates': np.array(data.get('epoch_dates', [])),
        'source_file': Path(data_path).name,
        'source_hash': compute_file_hash(data_path),
        'class_names': np.array([
            'Stable', 'Beach erosion', 'Toe erosion',
            'Small rockfall', 'Large failure'
        ]),
        'class_ids': np.array([0, 1, 2, 3, 4]),
        'labeler_name': labeler_name,
        'created_at': datetime.now().isoformat(),
        'modified_at': datetime.now().isoformat(),
        'session_log': np.array([{
            'timestamp': datetime.now().isoformat(),
            'labeler': labeler_name,
            'action': 'created',
        }], dtype=object),
    }

    return labels, metadata


def load_labels(
    labels_path: str,
    data: Optional[dict[str, Any]] = None,
    data_path: Optional[str] = None,
) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    """
    Load labels from NPZ file with validation.

    Args:
        labels_path: Path to labels NPZ file
        data: Optional loaded data dict for validation
        data_path: Optional path to data file for hash validation

    Returns:
        Tuple of (labels_array, metadata_dict, warnings_list)
    """
    labels_data = np.load(labels_path, allow_pickle=True)
    warnings = []

    labels = labels_data['labels']

    # Extract metadata
    metadata = {
        'transect_ids': labels_data['transect_ids'],
        'epoch_dates': labels_data['epoch_dates'],
        'source_file': str(labels_data.get('source_file', '')),
        'source_hash': str(labels_data.get('source_hash', '')),
        'class_names': labels_data.get('class_names', np.array([])),
        'class_ids': labels_data.get('class_ids', np.array([])),
        'labeler_name': str(labels_data.get('labeler_name', '')),
        'created_at': str(labels_data.get('created_at', '')),
        'modified_at': str(labels_data.get('modified_at', '')),
        'session_log': labels_data.get('session_log', np.array([], dtype=object)),
    }

    # Validation against data if provided
    if data is not None:
        # Check transect count matches
        n_transects_data = data['points'].shape[0]
        n_transects_labels = labels.shape[0]
        if n_transects_labels != n_transects_data:
            warnings.append(f"Transect count mismatch: data={n_transects_data}, labels={n_transects_labels}")

        # Check epoch count matches
        data_epochs = data['points'].shape[1]
        label_pairs = labels.shape[1]
        if label_pairs != data_epochs - 1:
            warnings.append(f"Epoch mismatch: data has {data_epochs} epochs but labels have {label_pairs} pairs")

    # Hash validation if path provided
    if data_path is not None and metadata['source_hash']:
        try:
            current_hash = compute_file_hash(data_path)
            if current_hash != metadata['source_hash']:
                warnings.append("Data file has changed since labels were created!")
        except FileNotFoundError:
            warnings.append("Could not verify data file hash - file not found")

    return labels, metadata, warnings


def save_labels(
    labels_path: str,
    labels: np.ndarray,
    metadata: dict[str, Any],
    labeler_name: str = "",
) -> None:
    """
    Save labels to NPZ file with updated metadata.

    Args:
        labels_path: Path to save labels file
        labels: Labels array (n_transects, n_pairs)
        metadata: Metadata dictionary
        labeler_name: Current labeler name (for session log)
    """
    # Update modification time
    metadata['modified_at'] = datetime.now().isoformat()

    # Add session log entry
    session_log = list(metadata.get('session_log', []))
    n_labeled = int(np.sum(labels != -1))
    session_log.append({
        'timestamp': datetime.now().isoformat(),
        'labeler': labeler_name,
        'action': 'saved',
        'n_labeled': n_labeled,
    })
    metadata['session_log'] = np.array(session_log, dtype=object)

    # Compute statistics
    label_counts = np.array([int(np.sum(labels == i)) for i in range(5)])

    # Save
    np.savez(
        labels_path,
        labels=labels,
        transect_ids=metadata['transect_ids'],
        epoch_dates=metadata['epoch_dates'],
        source_file=metadata['source_file'],
        source_hash=metadata['source_hash'],
        class_names=metadata['class_names'],
        class_ids=metadata['class_ids'],
        labeler_name=metadata['labeler_name'],
        created_at=metadata['created_at'],
        modified_at=metadata['modified_at'],
        session_log=metadata['session_log'],
        n_labeled=n_labeled,
        n_unlabeled=int(np.sum(labels == -1)),
        label_counts=label_counts,
    )


def validate_labels_data_compatibility(
    labels: np.ndarray,
    data: dict[str, Any],
) -> list[str]:
    """
    Validate that labels are compatible with data.

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    n_transects_data = data['points'].shape[0]
    n_epochs_data = data['points'].shape[1]
    n_transects_labels, n_pairs_labels = labels.shape

    if n_transects_labels != n_transects_data:
        errors.append(f"Transect count mismatch: data={n_transects_data}, labels={n_transects_labels}")

    expected_pairs = n_epochs_data - 1
    if n_pairs_labels != expected_pairs:
        errors.append(f"Pair count mismatch: expected={expected_pairs}, labels={n_pairs_labels}")

    return errors
