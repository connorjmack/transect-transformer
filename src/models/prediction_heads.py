"""
Multi-task prediction heads for cliff erosion forecasting.

This module implements four prediction heads:
1. Risk Index: Continuous [0,1] risk score
2. Collapse Probability: Multi-horizon binary classification (1wk, 1mo, 3mo, 1yr)
3. Expected Retreat: Continuous positive value (m/yr)
4. Failure Mode: Multi-class classification (5 modes)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictionHeads(nn.Module):
    """
    Multi-task prediction heads for cliff erosion forecasting.

    All heads operate on the pooled representation from the fusion module.
    Heads can be selectively enabled/disabled for phased training.

    Args:
        d_model: Hidden dimension from fusion module (default 256)
        enable_risk: Enable risk index head (default True)
        enable_retreat: Enable expected retreat head (default True)
        enable_collapse: Enable collapse probability head (default True)
        enable_failure_mode: Enable failure mode head (default True)
        n_collapse_horizons: Number of collapse time horizons (default 4)
        n_failure_modes: Number of failure mode classes (default 5)
        dropout: Dropout rate (default 0.1)
    """

    def __init__(
        self,
        d_model: int = 256,
        enable_risk: bool = True,
        enable_retreat: bool = True,
        enable_collapse: bool = True,
        enable_failure_mode: bool = True,
        n_collapse_horizons: int = 4,
        n_failure_modes: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.enable_risk = enable_risk
        self.enable_retreat = enable_retreat
        self.enable_collapse = enable_collapse
        self.enable_failure_mode = enable_failure_mode
        self.n_collapse_horizons = n_collapse_horizons
        self.n_failure_modes = n_failure_modes

        # Risk Index Head
        if enable_risk:
            self.risk_head = RiskIndexHead(d_model, dropout)

        # Expected Retreat Head
        if enable_retreat:
            self.retreat_head = ExpectedRetreatHead(d_model, dropout)

        # Collapse Probability Head
        if enable_collapse:
            self.collapse_head = CollapseProbabilityHead(
                d_model, n_collapse_horizons, dropout
            )

        # Failure Mode Head
        if enable_failure_mode:
            self.failure_mode_head = FailureModeHead(d_model, n_failure_modes, dropout)

    def forward(
        self, pooled: torch.Tensor  # (B, d_model)
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all enabled prediction heads.

        Args:
            pooled: Pooled representation from fusion module (B, d_model)

        Returns:
            Dictionary containing predictions from enabled heads:
                - risk_index: (B,) - risk scores in [0, 1]
                - retreat_m: (B,) - expected retreat in m/yr (positive)
                - p_collapse: (B, n_horizons) - collapse probabilities [0, 1]
                - failure_mode_logits: (B, n_modes) - logits for failure mode classes
        """
        outputs = {}

        if self.enable_risk:
            outputs['risk_index'] = self.risk_head(pooled)

        if self.enable_retreat:
            outputs['retreat_m'] = self.retreat_head(pooled)

        if self.enable_collapse:
            outputs['p_collapse'] = self.collapse_head(pooled)

        if self.enable_failure_mode:
            outputs['failure_mode_logits'] = self.failure_mode_head(pooled)

        return outputs


class RiskIndexHead(nn.Module):
    """
    Risk Index prediction head.

    Predicts continuous risk score in [0, 1] using sigmoid activation.
    Risk index is computed from retreat and cliff height, representing
    overall erosion risk at the site.

    Architecture:
        Input (B, d_model)
        -> Linear(d_model, d_model)
        -> GELU
        -> Dropout
        -> Linear(d_model, 1)
        -> Sigmoid
        Output: (B,)
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Pooled embeddings (B, d_model)

        Returns:
            Risk index (B,) in range [0, 1]
        """
        return self.net(x).squeeze(-1)  # (B, 1) -> (B,)


class ExpectedRetreatHead(nn.Module):
    """
    Expected Retreat prediction head.

    Predicts continuous retreat distance in m/yr using softplus activation
    to ensure positive values. Retreat is a fundamental physical quantity
    representing cliff recession rate.

    Architecture:
        Input (B, d_model)
        -> Linear(d_model, d_model)
        -> GELU
        -> Dropout
        -> Linear(d_model, 1)
        -> Softplus
        Output: (B,)
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Pooled embeddings (B, d_model)

        Returns:
            Expected retreat (B,) in m/yr (positive)
        """
        logits = self.net(x).squeeze(-1)  # (B, 1) -> (B,)
        # Softplus ensures positive values: softplus(x) = log(1 + exp(x))
        return F.softplus(logits)


class CollapseProbabilityHead(nn.Module):
    """
    Collapse Probability prediction head.

    Predicts multi-label binary probabilities for collapse at multiple
    time horizons: 1 week, 1 month, 3 months, 1 year. Each horizon is
    an independent binary prediction (using sigmoid, not softmax).

    Architecture:
        Input (B, d_model)
        -> Linear(d_model, d_model)
        -> GELU
        -> Dropout
        -> Linear(d_model, n_horizons)
        -> Sigmoid
        Output: (B, n_horizons)
    """

    def __init__(self, d_model: int, n_horizons: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_horizons = n_horizons
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_horizons),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Pooled embeddings (B, d_model)

        Returns:
            Collapse probabilities (B, n_horizons)
            Each value in [0, 1] representing P(collapse in horizon_i)
        """
        return self.net(x)  # (B, n_horizons)


class FailureModeHead(nn.Module):
    """
    Failure Mode prediction head.

    Predicts multi-class failure mode classification:
        0: stable (no failure)
        1: topple (rotational failure at cliff top)
        2: planar (sliding failure along bedding plane)
        3: rotational (circular failure surface)
        4: rockfall (detachment of rock blocks)

    Returns logits (not probabilities) - softmax applied during loss computation.

    Architecture:
        Input (B, d_model)
        -> Linear(d_model, d_model)
        -> GELU
        -> Dropout
        -> Linear(d_model, n_modes)
        Output: (B, n_modes) logits
    """

    def __init__(self, d_model: int, n_modes: int = 5, dropout: float = 0.1):
        super().__init__()
        self.n_modes = n_modes
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_modes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Pooled embeddings (B, d_model)

        Returns:
            Failure mode logits (B, n_modes)
            Use torch.argmax(logits, dim=-1) for predicted class
            Use F.cross_entropy(logits, targets) for loss
        """
        return self.net(x)  # (B, n_modes) - logits, not probabilities
