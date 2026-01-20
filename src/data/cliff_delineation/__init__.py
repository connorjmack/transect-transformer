"""
Cliff edge detection using CliffDelineaTool v2.0.

This module integrates the CNN-BiLSTM cliff delineation model to detect
cliff toe and cliff top positions in transect data.

Prerequisites:
    Install CliffDelineaTool as an editable package:
    ```
    pip install -e /path/to/CliffDelineaTool_2.0/v2
    ```

Usage:
    >>> from src.data.cliff_delineation import detect_cliff_edges
    >>> results = detect_cliff_edges(
    ...     npz_path="data/processed/delmar.npz",
    ...     checkpoint_path="/path/to/best_model.pth"
    ... )

    Or via CLI:
    ```
    python scripts/processing/detect_cliff_edges.py \\
        --input data/processed/delmar.npz \\
        --checkpoint /path/to/best_model.pth
    ```
"""

from .feature_adapter import CliffFeatureAdapter
from .model_wrapper import CliffDelineationModel
from .detector import detect_cliff_edges, load_cliff_results

__all__ = [
    "CliffFeatureAdapter",
    "CliffDelineationModel",
    "detect_cliff_edges",
    "load_cliff_results",
]
