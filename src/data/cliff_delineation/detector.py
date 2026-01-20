"""
Cliff edge detection for transect NPZ files.

Processes extracted transects and detects cliff toe/top positions using
the CliffDelineaTool v2.0 CNN-BiLSTM model. Results are stored in a
sidecar file (*.cliff.npz) to preserve original NPZ integrity.

Usage:
    >>> from src.data.cliff_delineation import detect_cliff_edges
    >>> results = detect_cliff_edges(
    ...     npz_path="data/processed/delmar.npz",
    ...     checkpoint_path="/path/to/best_model.pth"
    ... )
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union

from src.utils.logging import get_logger
from .feature_adapter import CliffFeatureAdapter
from .model_wrapper import CliffDelineationModel

logger = get_logger(__name__)

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None


def detect_cliff_edges(
    npz_path: Union[str, Path],
    checkpoint_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    confidence_threshold: float = 0.5,
    n_vert: int = 20,
    device: str = "auto",
    show_progress: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Run cliff edge detection on NPZ transect file.

    Args:
        npz_path: Path to transect-transformer NPZ file (cube or flat format).
        checkpoint_path: Path to CliffDelineaTool checkpoint.
        output_path: Where to save results. If None, uses npz_path.replace('.npz', '.cliff.npz').
        confidence_threshold: Confidence threshold for detection.
        n_vert: Local slope window size (must match CliffDelineaTool training).
        device: 'cuda', 'cpu', or 'auto'.
        show_progress: Show progress bar.

    Returns:
        Dictionary with detection results.
    """
    npz_path = Path(npz_path)

    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    logger.info(f"Loading transect data from {npz_path}")

    # Load NPZ
    data = np.load(npz_path, allow_pickle=True)
    points = data["points"]
    distances = data["distances"]

    # Check format (cube vs flat)
    if points.ndim == 4:
        n_transects, n_epochs, n_points, n_features = points.shape
        is_cube = True
        logger.info(
            f"Cube format: {n_transects} transects x {n_epochs} epochs x {n_points} points"
        )
    elif points.ndim == 3:
        n_transects, n_points, n_features = points.shape
        n_epochs = 1
        is_cube = False
        logger.info(f"Flat format: {n_transects} transects x {n_points} points")
        # Add epoch dimension for uniform processing
        points = points[:, np.newaxis, :, :]
        distances = distances[:, np.newaxis, :]
    else:
        raise ValueError(f"Unexpected points shape: {points.shape}")

    # Initialize components
    logger.info("Initializing feature adapter and model...")
    adapter = CliffFeatureAdapter(n_vert=n_vert)
    model = CliffDelineationModel(
        checkpoint_path=checkpoint_path,
        device=device,
        confidence_threshold=confidence_threshold,
    )

    # Prepare output arrays
    toe_distances = np.full((n_transects, n_epochs), -1.0, dtype=np.float32)
    top_distances = np.full((n_transects, n_epochs), -1.0, dtype=np.float32)
    toe_indices = np.full((n_transects, n_epochs), -1, dtype=np.int32)
    top_indices = np.full((n_transects, n_epochs), -1, dtype=np.int32)
    toe_confidences = np.zeros((n_transects, n_epochs), dtype=np.float32)
    top_confidences = np.zeros((n_transects, n_epochs), dtype=np.float32)
    has_cliff = np.zeros((n_transects, n_epochs), dtype=bool)

    # Process each transect-epoch
    total = n_transects * n_epochs
    logger.info(f"Processing {total} transect-epochs...")

    iterator = range(total)
    if show_progress and HAS_TQDM:
        iterator = tqdm(iterator, desc="Detecting cliff edges", unit="transect")

    n_valid = 0
    n_detected = 0

    for flat_idx in iterator:
        t_idx = flat_idx // n_epochs
        e_idx = flat_idx % n_epochs

        # Check for NaN (missing data)
        if np.isnan(points[t_idx, e_idx, 0, 0]):
            continue

        n_valid += 1

        # Transform features to CliffDelineaTool format
        cliff_features = adapter.transform(
            points[t_idx, e_idx], distances[t_idx, e_idx]
        )

        # Run detection
        result = model.predict(
            cliff_features, distances[t_idx, e_idx], confidence_threshold
        )

        # Store results
        toe_distances[t_idx, e_idx] = result["toe_distance"]
        top_distances[t_idx, e_idx] = result["top_distance"]
        toe_indices[t_idx, e_idx] = result["toe_idx"]
        top_indices[t_idx, e_idx] = result["top_idx"]
        toe_confidences[t_idx, e_idx] = result["toe_confidence"]
        top_confidences[t_idx, e_idx] = result["top_confidence"]
        has_cliff[t_idx, e_idx] = result["has_cliff"]

        if result["has_cliff"]:
            n_detected += 1

    # Remove epoch dim if original was flat
    if not is_cube:
        toe_distances = toe_distances[:, 0]
        top_distances = top_distances[:, 0]
        toe_indices = toe_indices[:, 0]
        top_indices = top_indices[:, 0]
        toe_confidences = toe_confidences[:, 0]
        top_confidences = top_confidences[:, 0]
        has_cliff = has_cliff[:, 0]

    # Prepare results dict
    results = {
        "toe_distances": toe_distances,
        "top_distances": top_distances,
        "toe_indices": toe_indices,
        "top_indices": top_indices,
        "toe_confidences": toe_confidences,
        "top_confidences": top_confidences,
        "has_cliff": has_cliff,
        "source_npz": str(npz_path.name),
        "model_checkpoint": str(Path(checkpoint_path).name),
        "confidence_threshold": np.float32(confidence_threshold),
        "n_vert": np.int32(n_vert),
    }

    # Copy transect_ids for alignment
    if "transect_ids" in data:
        results["transect_ids"] = data["transect_ids"]
    if "mop_ids" in data:
        results["mop_ids"] = data["mop_ids"]

    # Copy epoch info if cube format
    if is_cube:
        if "epoch_dates" in data:
            results["epoch_dates"] = data["epoch_dates"]
        if "epoch_names" in data:
            results["epoch_names"] = data["epoch_names"]
        if "las_sources" in data:
            results["las_sources"] = data["las_sources"]

    # Save results
    if output_path is None:
        output_path = npz_path.with_suffix(".cliff.npz")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **results)

    # Report summary
    detection_rate = (n_detected / n_valid * 100) if n_valid > 0 else 0
    logger.info(
        f"Cliff detection complete: {n_detected}/{n_valid} transect-epochs "
        f"with detected cliffs ({detection_rate:.1f}%)"
    )
    logger.info(f"Results saved to: {output_path}")

    return results


def load_cliff_results(
    cliff_npz_path: Union[str, Path],
) -> Dict[str, np.ndarray]:
    """
    Load cliff detection results from sidecar file.

    Args:
        cliff_npz_path: Path to *.cliff.npz file.

    Returns:
        Dictionary with detection results.
    """
    cliff_npz_path = Path(cliff_npz_path)

    if not cliff_npz_path.exists():
        raise FileNotFoundError(f"Cliff results not found: {cliff_npz_path}")

    data = np.load(cliff_npz_path, allow_pickle=True)
    return {key: data[key] for key in data.keys()}


def get_cliff_metrics(results: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Compute summary metrics from cliff detection results.

    Args:
        results: Dictionary from detect_cliff_edges or load_cliff_results.

    Returns:
        Dictionary with metrics:
            - detection_rate: Fraction of transects with detected cliffs
            - mean_cliff_height: Mean (top - toe) elevation difference
            - mean_toe_confidence: Mean confidence score for toe predictions
            - mean_top_confidence: Mean confidence score for top predictions
    """
    has_cliff = results["has_cliff"]
    toe_distances = results["toe_distances"]
    top_distances = results["top_distances"]
    toe_confidences = results["toe_confidences"]
    top_confidences = results["top_confidences"]

    # Flatten if multi-epoch
    if has_cliff.ndim > 1:
        has_cliff = has_cliff.flatten()
        toe_distances = toe_distances.flatten()
        top_distances = top_distances.flatten()
        toe_confidences = toe_confidences.flatten()
        top_confidences = top_confidences.flatten()

    # Filter valid entries (not NaN)
    valid_mask = ~np.isnan(toe_distances)
    has_cliff = has_cliff[valid_mask]
    toe_distances = toe_distances[valid_mask]
    top_distances = top_distances[valid_mask]
    toe_confidences = toe_confidences[valid_mask]
    top_confidences = top_confidences[valid_mask]

    n_valid = len(has_cliff)
    n_detected = has_cliff.sum()

    metrics = {
        "n_valid": n_valid,
        "n_detected": n_detected,
        "detection_rate": n_detected / n_valid if n_valid > 0 else 0.0,
    }

    if n_detected > 0:
        detected_mask = has_cliff
        cliff_widths = (
            top_distances[detected_mask] - toe_distances[detected_mask]
        )
        metrics["mean_cliff_width_m"] = float(cliff_widths.mean())
        metrics["std_cliff_width_m"] = float(cliff_widths.std())
        metrics["min_cliff_width_m"] = float(cliff_widths.min())
        metrics["max_cliff_width_m"] = float(cliff_widths.max())
        metrics["mean_toe_confidence"] = float(toe_confidences[detected_mask].mean())
        metrics["mean_top_confidence"] = float(top_confidences[detected_mask].mean())
    else:
        metrics["mean_cliff_width_m"] = 0.0
        metrics["std_cliff_width_m"] = 0.0
        metrics["min_cliff_width_m"] = 0.0
        metrics["max_cliff_width_m"] = 0.0
        metrics["mean_toe_confidence"] = 0.0
        metrics["mean_top_confidence"] = 0.0

    return metrics
