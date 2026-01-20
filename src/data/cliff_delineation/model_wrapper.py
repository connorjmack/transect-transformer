"""
Wrapper for CliffDelineaTool v2.0 CNN-BiLSTM model inference.

Requires CliffDelineaTool to be installed:
    pip install -e /path/to/CliffDelineaTool_2.0/v2
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F

from src.utils.logging import get_logger

logger = get_logger(__name__)


class CliffDelineationModel:
    """Wrapper for CliffDelineaTool v2.0 model inference."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: str = "auto",
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize model wrapper.

        Args:
            checkpoint_path: Path to best_model.pth checkpoint.
            device: 'cuda', 'cpu', or 'auto' (auto-detect).
            confidence_threshold: Minimum confidence to accept prediction.
        """
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.confidence_threshold = confidence_threshold

        # Load checkpoint and model
        self._load_model()

    def _load_model(self) -> None:
        """Load model from checkpoint."""
        try:
            from cliff_dl.models.cnn_lstm import CNN_BiLSTM_CliffDetector
        except ImportError as e:
            raise ImportError(
                "CliffDelineaTool (cliff_dl) not installed. "
                "Install with: pip install -e /path/to/CliffDelineaTool_2.0/v2"
            ) from e

        logger.info(f"Loading checkpoint from {self.checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )

        # Extract config from checkpoint
        if "config" in checkpoint:
            self.config = checkpoint["config"]
        else:
            # Use default config if not in checkpoint
            self.config = self._get_default_config()

        # Create model with config
        model_cfg = self.config.get("model", {})
        features_cfg = self.config.get("features", {})

        self.model = CNN_BiLSTM_CliffDetector(
            input_dim=features_cfg.get("input_dim", 13),
            cnn_channels=model_cfg.get("cnn", {}).get("channels", [64, 128, 64]),
            cnn_kernel_sizes=model_cfg.get("cnn", {}).get("kernel_sizes", [5, 5, 3]),
            cnn_dropout=model_cfg.get("cnn", {}).get("dropout", 0.2),
            lstm_hidden_size=model_cfg.get("lstm", {}).get("hidden_size", 128),
            lstm_num_layers=model_cfg.get("lstm", {}).get("num_layers", 2),
            lstm_dropout=model_cfg.get("lstm", {}).get("dropout", 0.3),
            attention_heads=model_cfg.get("attention", {}).get("num_heads", 8),
            attention_dim=model_cfg.get("attention", {}).get("dim", 256),
            attention_dropout=model_cfg.get("attention", {}).get("dropout", 0.1),
        )

        # Load weights
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Try loading directly if checkpoint is just state dict
            self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Override confidence threshold from config if available
        postprocess_cfg = self.config.get("postprocess", {})
        if "confidence_threshold" in postprocess_cfg:
            self.confidence_threshold = postprocess_cfg["confidence_threshold"]

        logger.info(
            f"Model loaded: input_dim={features_cfg.get('input_dim', 13)}, "
            f"device={self.device}, confidence_threshold={self.confidence_threshold}"
        )

    def _get_default_config(self) -> dict:
        """Return default model configuration."""
        return {
            "model": {
                "cnn": {
                    "channels": [64, 128, 64],
                    "kernel_sizes": [5, 5, 3],
                    "dropout": 0.2,
                },
                "lstm": {
                    "hidden_size": 128,
                    "num_layers": 2,
                    "dropout": 0.3,
                },
                "attention": {
                    "num_heads": 8,
                    "dim": 256,
                    "dropout": 0.1,
                },
            },
            "features": {
                "input_dim": 13,
                "n_vert": 20,
            },
            "postprocess": {
                "confidence_threshold": 0.5,
            },
        }

    @torch.no_grad()
    def predict(
        self,
        features: np.ndarray,
        distances: np.ndarray,
        confidence_threshold: Optional[float] = None,
    ) -> Dict[str, Union[float, int, bool, np.ndarray]]:
        """
        Run inference on a single transect.

        Args:
            features: (N, 13) array in CliffDelineaTool format.
            distances: (N,) array of distances along transect.
            confidence_threshold: Override default threshold.

        Returns:
            Dictionary with:
                - toe_distance: float (or -1 if no cliff detected)
                - top_distance: float (or -1 if no cliff detected)
                - toe_idx: int (point index, or -1)
                - top_idx: int (point index, or -1)
                - toe_confidence: float [0,1]
                - top_confidence: float [0,1]
                - has_cliff: bool
                - segmentation_probs: (N, 3) array [background, toe, top]
        """
        threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else self.confidence_threshold
        )

        # Convert to tensors
        features_t = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        lengths = torch.LongTensor([len(features)])

        # Run model
        outputs = self.model.forward(features_t, mask=None, lengths=lengths)

        # Get segmentation probabilities
        seg_probs = F.softmax(outputs["segmentation"][0], dim=-1).cpu().numpy()
        confidence = outputs["confidence"][0, 0].cpu().item()

        has_cliff = confidence > threshold

        if not has_cliff:
            return {
                "toe_distance": -1.0,
                "top_distance": -1.0,
                "toe_idx": -1,
                "top_idx": -1,
                "toe_confidence": 0.0,
                "top_confidence": 0.0,
                "has_cliff": False,
                "segmentation_probs": seg_probs,
            }

        # Find argmax for toe (channel 1) and top (channel 2)
        toe_probs = seg_probs[:, 1]
        top_probs = seg_probs[:, 2]

        toe_idx = int(np.argmax(toe_probs))
        top_idx = int(np.argmax(top_probs))

        toe_conf = float(toe_probs[toe_idx])
        top_conf = float(top_probs[top_idx])

        # Enforce constraint: top must be landward of toe
        if distances[top_idx] <= distances[toe_idx]:
            return {
                "toe_distance": -1.0,
                "top_distance": -1.0,
                "toe_idx": -1,
                "top_idx": -1,
                "toe_confidence": toe_conf,
                "top_confidence": top_conf,
                "has_cliff": False,
                "segmentation_probs": seg_probs,
            }

        return {
            "toe_distance": float(distances[toe_idx]),
            "top_distance": float(distances[top_idx]),
            "toe_idx": toe_idx,
            "top_idx": top_idx,
            "toe_confidence": toe_conf,
            "top_confidence": top_conf,
            "has_cliff": True,
            "segmentation_probs": seg_probs,
        }

    def predict_batch(
        self,
        features: np.ndarray,
        distances: np.ndarray,
        confidence_threshold: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on batch of transects.

        Args:
            features: (B, N, 13) array in CliffDelineaTool format.
            distances: (B, N) array of distances.
            confidence_threshold: Override default threshold.

        Returns:
            Dictionary with arrays for batch:
                - toe_distances: (B,)
                - top_distances: (B,)
                - toe_indices: (B,)
                - top_indices: (B,)
                - toe_confidences: (B,)
                - top_confidences: (B,)
                - has_cliff: (B,) boolean
        """
        B = features.shape[0]

        results = {
            "toe_distances": np.full(B, -1.0, dtype=np.float32),
            "top_distances": np.full(B, -1.0, dtype=np.float32),
            "toe_indices": np.full(B, -1, dtype=np.int32),
            "top_indices": np.full(B, -1, dtype=np.int32),
            "toe_confidences": np.zeros(B, dtype=np.float32),
            "top_confidences": np.zeros(B, dtype=np.float32),
            "has_cliff": np.zeros(B, dtype=bool),
        }

        for i in range(B):
            pred = self.predict(features[i], distances[i], confidence_threshold)
            results["toe_distances"][i] = pred["toe_distance"]
            results["top_distances"][i] = pred["top_distance"]
            results["toe_indices"][i] = pred["toe_idx"]
            results["top_indices"][i] = pred["top_idx"]
            results["toe_confidences"][i] = pred["toe_confidence"]
            results["top_confidences"][i] = pred["top_confidence"]
            results["has_cliff"][i] = pred["has_cliff"]

        return results
