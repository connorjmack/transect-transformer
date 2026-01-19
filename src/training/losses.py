"""
Loss functions for CliffCast multi-task learning.

Combines multiple objectives with configurable weights:
- Risk Index: Smooth L1 loss
- Expected Retreat: Smooth L1 loss
- Collapse Probability: Binary cross-entropy per horizon
- Failure Mode: Cross-entropy (only on non-stable samples)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CliffCastLoss(nn.Module):
    """
    Combined loss function for CliffCast multi-task learning.

    Weights reflect task importance:
    - Risk and Retreat: weight=1.0 (fundamental predictions)
    - Collapse: weight=2.0 (safety-critical, needs higher weight)
    - Failure Mode: weight=0.5 (fewer labels, lower weight)

    Args:
        weight_risk: Weight for risk index loss (default 1.0)
        weight_retreat: Weight for expected retreat loss (default 1.0)
        weight_collapse: Weight for collapse probability loss (default 2.0)
        weight_failure_mode: Weight for failure mode loss (default 0.5)
        reduction: Loss reduction method ('mean', 'sum', 'none') (default 'mean')
    """

    def __init__(
        self,
        weight_risk: float = 1.0,
        weight_retreat: float = 1.0,
        weight_collapse: float = 2.0,
        weight_failure_mode: float = 0.5,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.weight_risk = weight_risk
        self.weight_retreat = weight_retreat
        self.weight_collapse = weight_collapse
        self.weight_failure_mode = weight_failure_mode
        self.reduction = reduction

        # Validate reduction
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction: {reduction}. Must be 'mean', 'sum', or 'none'")

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss from predictions and targets.

        Args:
            predictions: Dictionary from model forward pass containing:
                - risk_index: (B,) predicted risk in [0,1]
                - retreat_m: (B,) predicted retreat (positive)
                - p_collapse: (B, 4) collapse probabilities
                - failure_mode_logits: (B, 5) failure mode logits
            targets: Dictionary of ground truth containing:
                - risk_index: (B,) target risk in [0,1]
                - retreat_m: (B,) target retreat (m/yr)
                - collapse_labels: (B, 4) binary labels per horizon
                - failure_mode: (B,) class indices [0-4]

        Returns:
            Dictionary containing:
                - loss: Combined weighted loss
                - loss_risk: Risk index loss component
                - loss_retreat: Retreat loss component
                - loss_collapse: Collapse loss component
                - loss_failure_mode: Failure mode loss component
        """
        losses = {}
        total_loss = 0.0

        # 1. Risk Index Loss (Smooth L1)
        if 'risk_index' in predictions and 'risk_index' in targets:
            loss_risk = F.smooth_l1_loss(
                predictions['risk_index'],
                targets['risk_index'],
                reduction=self.reduction,
            )
            losses['loss_risk'] = loss_risk
            total_loss += self.weight_risk * loss_risk

        # 2. Expected Retreat Loss (Smooth L1)
        if 'retreat_m' in predictions and 'retreat_m' in targets:
            loss_retreat = F.smooth_l1_loss(
                predictions['retreat_m'],
                targets['retreat_m'],
                reduction=self.reduction,
            )
            losses['loss_retreat'] = loss_retreat
            total_loss += self.weight_retreat * loss_retreat

        # 3. Collapse Probability Loss (Binary cross-entropy per horizon)
        if 'p_collapse' in predictions and 'collapse_labels' in targets:
            loss_collapse = F.binary_cross_entropy(
                predictions['p_collapse'],
                targets['collapse_labels'].float(),
                reduction=self.reduction,
            )
            losses['loss_collapse'] = loss_collapse
            total_loss += self.weight_collapse * loss_collapse

        # 4. Failure Mode Loss (Cross-entropy, only on non-stable samples)
        if 'failure_mode_logits' in predictions and 'failure_mode' in targets:
            failure_mode = targets['failure_mode']

            # Only compute loss on samples where failure occurred (mode > 0)
            failure_mask = failure_mode > 0

            if failure_mask.any():
                # Filter to only failed samples
                pred_logits = predictions['failure_mode_logits'][failure_mask]
                target_modes = failure_mode[failure_mask]

                # Cross-entropy loss
                loss_failure_mode = F.cross_entropy(
                    pred_logits,
                    target_modes,
                    reduction=self.reduction if self.reduction != 'none' else 'mean',
                )

                # If reduction='none', we filtered samples, so take mean
                if self.reduction == 'none':
                    # Expand back to full batch size with zeros for stable samples
                    full_loss = torch.zeros(
                        failure_mode.size(0),
                        device=failure_mode.device,
                        dtype=loss_failure_mode.dtype,
                    )
                    full_loss[failure_mask] = loss_failure_mode
                    loss_failure_mode = full_loss
            else:
                # No failures in batch, loss is zero
                if self.reduction == 'none':
                    loss_failure_mode = torch.zeros(
                        failure_mode.size(0),
                        device=failure_mode.device,
                    )
                else:
                    loss_failure_mode = torch.tensor(
                        0.0, device=failure_mode.device, requires_grad=True
                    )

            losses['loss_failure_mode'] = loss_failure_mode
            total_loss += self.weight_failure_mode * loss_failure_mode

        losses['loss'] = total_loss
        return losses


class RiskIndexLoss(nn.Module):
    """
    Individual loss for risk index prediction.

    Uses Smooth L1 loss (Huber loss) which is less sensitive to outliers
    than MSE and provides smooth gradients near zero.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: (B,) predicted risk in [0,1]
            targets: (B,) target risk in [0,1]

        Returns:
            Scalar loss value
        """
        return F.smooth_l1_loss(predictions, targets, reduction=self.reduction)


class ExpectedRetreatLoss(nn.Module):
    """
    Individual loss for expected retreat prediction.

    Uses Smooth L1 loss suitable for continuous positive values.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: (B,) predicted retreat (m/yr)
            targets: (B,) target retreat (m/yr)

        Returns:
            Scalar loss value
        """
        return F.smooth_l1_loss(predictions, targets, reduction=self.reduction)


class CollapseProbabilityLoss(nn.Module):
    """
    Individual loss for collapse probability prediction.

    Uses binary cross-entropy for multi-label classification across
    multiple time horizons. Each horizon is treated independently.
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: (B, n_horizons) probabilities in [0,1]
            targets: (B, n_horizons) binary labels

        Returns:
            Scalar loss value
        """
        return F.binary_cross_entropy(
            predictions, targets.float(), reduction=self.reduction
        )


class FailureModeLoss(nn.Module):
    """
    Individual loss for failure mode classification.

    Uses cross-entropy loss for multi-class classification.
    Optionally filters to only samples where failure occurred.
    """

    def __init__(self, only_failures: bool = True, reduction: str = 'mean'):
        super().__init__()
        self.only_failures = only_failures
        self.reduction = reduction

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: (B, n_modes) logits for failure modes
            targets: (B,) class indices [0-4]
                0 = stable (no failure)
                1-4 = failure modes

        Returns:
            Scalar loss value
        """
        if self.only_failures:
            # Only compute loss on samples where failure occurred
            failure_mask = targets > 0

            if not failure_mask.any():
                # No failures in batch
                return torch.tensor(0.0, device=targets.device, requires_grad=True)

            predictions = predictions[failure_mask]
            targets = targets[failure_mask]

        return F.cross_entropy(predictions, targets, reduction=self.reduction)
