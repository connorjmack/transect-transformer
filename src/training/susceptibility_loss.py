"""
Susceptibility loss for 5-class erosion mode classification.

Implements weighted cross-entropy with label smoothing to handle:
- Class imbalance (most samples are stable)
- Asymmetric importance (large failures are most critical)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SusceptibilityLoss(nn.Module):
    """
    Weighted cross-entropy loss for susceptibility classification.

    Class weights reflect importance hierarchy:
        - Class 0 (stable): 0.3 - Low weight for weak negative evidence
        - Class 1 (beach): 1.0 - Baseline weight
        - Class 2 (toe): 2.0 - Higher for cliff-related erosion
        - Class 3 (rockfall): 2.0 - Higher for cliff failures
        - Class 4 (large failure): 5.0 - Highest for safety-critical events

    Args:
        class_weights: Optional tensor of class weights. If None, uses defaults.
        label_smoothing: Amount of label smoothing (default 0.1)
        reduction: Loss reduction method ('mean', 'sum', 'none')
    """

    # Default class weights from model plan
    DEFAULT_WEIGHTS = [0.3, 1.0, 2.0, 2.0, 5.0]

    # Risk weights for deriving risk score from probabilities
    RISK_WEIGHTS = [0.0, 0.1, 0.4, 0.6, 1.0]

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.reduction = reduction

        # Register class weights as buffer (moves with model to device)
        if class_weights is None:
            class_weights = torch.tensor(self.DEFAULT_WEIGHTS)
        self.register_buffer('class_weights', class_weights)

        # Register risk weights for risk score derivation
        self.register_buffer('risk_weights', torch.tensor(self.RISK_WEIGHTS))

    def forward(
        self,
        logits: torch.Tensor,  # (B, 5)
        targets: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """
        Compute weighted cross-entropy loss with label smoothing.

        Args:
            logits: Model output logits (B, 5)
            targets: Ground truth class indices (B,)

        Returns:
            Scalar loss value (if reduction='mean' or 'sum') or
            Per-sample losses (B,) if reduction='none'
        """
        return F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
        )

    def compute_risk_score(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Derive risk score from class probabilities.

        Risk score is weighted sum of class probabilities:
            risk = sum(probs * [0.0, 0.1, 0.4, 0.6, 1.0])

        Args:
            logits: Model output logits (B, 5)

        Returns:
            Risk scores (B,) in range [0, 1]
        """
        probs = F.softmax(logits, dim=-1)  # (B, 5)
        risk = (probs * self.risk_weights).sum(dim=-1)  # (B,)
        return risk


class FocalSusceptibilityLoss(nn.Module):
    """
    Focal loss variant for susceptibility classification.

    Focal loss down-weights easy examples, focusing training on hard cases.
    Useful when class imbalance is severe.

    Loss = -alpha * (1 - p)^gamma * log(p)

    Args:
        class_weights: Optional tensor of class weights.
        gamma: Focusing parameter (default 2.0)
        reduction: Loss reduction method
    """

    DEFAULT_WEIGHTS = [0.3, 1.0, 2.0, 2.0, 5.0]

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if class_weights is None:
            class_weights = torch.tensor(self.DEFAULT_WEIGHTS)
        self.register_buffer('class_weights', class_weights)

    def forward(
        self,
        logits: torch.Tensor,  # (B, 5)
        targets: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Model output logits (B, 5)
            targets: Ground truth class indices (B,)

        Returns:
            Scalar loss value
        """
        ce_loss = F.cross_entropy(
            logits, targets, weight=self.class_weights, reduction='none'
        )  # (B,)

        # Get probabilities for correct class
        probs = F.softmax(logits, dim=-1)  # (B, 5)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma  # (B,)

        # Apply focal weight
        loss = focal_weight * ce_loss  # (B,)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
