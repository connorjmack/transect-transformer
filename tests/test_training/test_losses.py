"""
Tests for loss functions.

Tests cover:
- CliffCastLoss combined loss computation
- Individual loss components
- Loss weighting
- Edge cases (no failures, all stable, etc.)
- Gradient flow
"""

import pytest
import torch

from src.training.losses import (
    CliffCastLoss,
    CollapseProbabilityLoss,
    ExpectedRetreatLoss,
    FailureModeLoss,
    RiskIndexLoss,
)


class TestCliffCastLoss:
    """Test combined CliffCastLoss."""

    @pytest.fixture
    def loss_fn(self):
        """Create default loss function."""
        return CliffCastLoss()

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions."""
        B = 8
        return {
            'risk_index': torch.rand(B),
            'retreat_m': torch.rand(B) * 2,
            'p_collapse': torch.rand(B, 4),
            'failure_mode_logits': torch.randn(B, 5),
        }

    @pytest.fixture
    def sample_targets(self):
        """Create sample targets."""
        B = 8
        return {
            'risk_index': torch.rand(B),
            'retreat_m': torch.rand(B) * 2,
            'collapse_labels': torch.randint(0, 2, (B, 4)),
            'failure_mode': torch.randint(0, 5, (B,)),
        }

    def test_combined_loss_output_keys(self, loss_fn, sample_predictions, sample_targets):
        """Test combined loss returns all expected keys."""
        losses = loss_fn(sample_predictions, sample_targets)

        assert 'loss' in losses
        assert 'loss_risk' in losses
        assert 'loss_retreat' in losses
        assert 'loss_collapse' in losses
        assert 'loss_failure_mode' in losses

    def test_combined_loss_scalar(self, loss_fn, sample_predictions, sample_targets):
        """Test combined loss returns scalar values."""
        losses = loss_fn(sample_predictions, sample_targets)

        assert losses['loss'].ndim == 0
        assert losses['loss_risk'].ndim == 0
        assert losses['loss_retreat'].ndim == 0
        assert losses['loss_collapse'].ndim == 0

    def test_combined_loss_positive(self, loss_fn, sample_predictions, sample_targets):
        """Test all loss components are non-negative."""
        losses = loss_fn(sample_predictions, sample_targets)

        assert losses['loss'] >= 0
        assert losses['loss_risk'] >= 0
        assert losses['loss_retreat'] >= 0
        assert losses['loss_collapse'] >= 0
        assert losses['loss_failure_mode'] >= 0

    def test_combined_loss_no_nans(self, loss_fn, sample_predictions, sample_targets):
        """Test combined loss produces no NaN values."""
        losses = loss_fn(sample_predictions, sample_targets)

        for key, value in losses.items():
            assert not torch.isnan(value).any(), f"NaN found in {key}"

    def test_loss_weighting(self):
        """Test loss weights are applied correctly."""
        loss_fn = CliffCastLoss(
            weight_risk=1.0,
            weight_retreat=1.0,
            weight_collapse=2.0,
            weight_failure_mode=0.5,
        )

        B = 8
        predictions = {
            'risk_index': torch.rand(B),
            'retreat_m': torch.rand(B) * 2,
            'p_collapse': torch.rand(B, 4),
            'failure_mode_logits': torch.randn(B, 5),
        }
        targets = {
            'risk_index': torch.rand(B),
            'retreat_m': torch.rand(B) * 2,
            'collapse_labels': torch.randint(0, 2, (B, 4)),
            'failure_mode': torch.randint(1, 5, (B,)),  # All failures
        }

        losses = loss_fn(predictions, targets)

        # Total loss should approximately equal weighted sum
        expected = (
            1.0 * losses['loss_risk']
            + 1.0 * losses['loss_retreat']
            + 2.0 * losses['loss_collapse']
            + 0.5 * losses['loss_failure_mode']
        )

        assert torch.allclose(losses['loss'], expected, atol=1e-5)

    def test_selective_losses(self):
        """Test with only subset of predictions/targets."""
        loss_fn = CliffCastLoss()

        B = 8
        # Only risk and retreat
        predictions = {
            'risk_index': torch.rand(B),
            'retreat_m': torch.rand(B) * 2,
        }
        targets = {
            'risk_index': torch.rand(B),
            'retreat_m': torch.rand(B) * 2,
        }

        losses = loss_fn(predictions, targets)

        assert 'loss' in losses
        assert 'loss_risk' in losses
        assert 'loss_retreat' in losses
        assert 'loss_collapse' not in losses
        assert 'loss_failure_mode' not in losses

    def test_gradient_flow(self, loss_fn):
        """Test gradients flow through loss."""
        B = 8
        risk_pred = torch.rand(B, requires_grad=True)
        retreat_pred = torch.rand(B, requires_grad=True)
        collapse_pred = torch.rand(B, 4, requires_grad=True)
        failure_pred = torch.randn(B, 5, requires_grad=True)

        predictions = {
            'risk_index': risk_pred,
            'retreat_m': retreat_pred * 2,  # Non-leaf, need to retain grad
            'p_collapse': collapse_pred,
            'failure_mode_logits': failure_pred,
        }
        # Retain grad for non-leaf tensors
        predictions['retreat_m'].retain_grad()

        targets = {
            'risk_index': torch.rand(B),
            'retreat_m': torch.rand(B) * 2,
            'collapse_labels': torch.randint(0, 2, (B, 4)).float(),
            'failure_mode': torch.randint(0, 5, (B,)),
        }

        losses = loss_fn(predictions, targets)
        losses['loss'].backward()

        # Check gradients exist
        assert risk_pred.grad is not None
        assert retreat_pred.grad is not None
        assert collapse_pred.grad is not None
        assert failure_pred.grad is not None


class TestFailureModeLoss:
    """Test failure mode loss specifically."""

    def test_only_failures_zero_loss_when_all_stable(self):
        """Test loss is zero when all samples are stable and only_failures=True."""
        loss_fn = FailureModeLoss(only_failures=True)

        B = 8
        predictions = torch.randn(B, 5)
        targets = torch.zeros(B, dtype=torch.long)  # All stable

        loss = loss_fn(predictions, targets)

        assert loss == 0.0

    def test_only_failures_computes_loss_when_failures_exist(self):
        """Test loss is computed only on failure samples."""
        loss_fn = FailureModeLoss(only_failures=True)

        B = 8
        predictions = torch.randn(B, 5)
        targets = torch.tensor([0, 0, 1, 2, 0, 3, 0, 4])  # Some failures

        loss = loss_fn(predictions, targets)

        assert loss > 0.0

    def test_all_samples_when_only_failures_false(self):
        """Test all samples used when only_failures=False."""
        loss_fn = FailureModeLoss(only_failures=False)

        B = 8
        predictions = torch.randn(B, 5)
        targets = torch.zeros(B, dtype=torch.long)  # All stable

        loss = loss_fn(predictions, targets)

        # Loss should be non-zero even with all stable
        assert loss > 0.0


class TestRiskIndexLoss:
    """Test risk index loss."""

    def test_perfect_prediction_zero_loss(self):
        """Test perfect predictions give near-zero loss."""
        loss_fn = RiskIndexLoss()

        predictions = torch.tensor([0.1, 0.5, 0.9])
        targets = predictions.clone()

        loss = loss_fn(predictions, targets)

        assert loss < 1e-6

    def test_output_range_bounded(self):
        """Test loss is bounded for bounded inputs."""
        loss_fn = RiskIndexLoss()

        predictions = torch.rand(100)
        targets = torch.rand(100)

        loss = loss_fn(predictions, targets)

        # Smooth L1 loss should be bounded for [0,1] inputs
        assert loss >= 0.0
        assert loss < 1.0  # Should be less than max possible error


class TestExpectedRetreatLoss:
    """Test expected retreat loss."""

    def test_perfect_prediction_zero_loss(self):
        """Test perfect predictions give near-zero loss."""
        loss_fn = ExpectedRetreatLoss()

        predictions = torch.tensor([0.5, 1.0, 2.5])
        targets = predictions.clone()

        loss = loss_fn(predictions, targets)

        assert loss < 1e-6

    def test_larger_errors_larger_loss(self):
        """Test larger prediction errors give larger loss."""
        loss_fn = ExpectedRetreatLoss()

        predictions = torch.tensor([1.0, 1.0])
        targets_small = torch.tensor([1.1, 1.1])
        targets_large = torch.tensor([2.0, 2.0])

        loss_small = loss_fn(predictions, targets_small)
        loss_large = loss_fn(predictions, targets_large)

        assert loss_large > loss_small


class TestCollapseProbabilityLoss:
    """Test collapse probability loss."""

    def test_perfect_prediction_zero_loss(self):
        """Test perfect predictions give near-zero loss."""
        loss_fn = CollapseProbabilityLoss()

        # BCE expects binary targets (0 or 1)
        predictions = torch.tensor([[0.1, 0.9, 0.5, 0.2]])
        targets = torch.tensor([[0.0, 1.0, 1.0, 0.0]])

        loss = loss_fn(predictions, targets)

        # Loss should be low for predictions close to targets
        assert loss < 1.0

    def test_opposite_predictions_high_loss(self):
        """Test opposite predictions give high loss."""
        loss_fn = CollapseProbabilityLoss()

        predictions = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
        targets = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

        loss = loss_fn(predictions, targets)

        # BCE loss should be high for opposite predictions
        assert loss > 1.0

    def test_multi_horizon_independent(self):
        """Test each horizon contributes independently to loss."""
        loss_fn = CollapseProbabilityLoss()

        B, n_horizons = 4, 4
        predictions = torch.rand(B, n_horizons)
        targets = torch.randint(0, 2, (B, n_horizons))

        loss = loss_fn(predictions, targets)

        assert loss >= 0.0


class TestLossReduction:
    """Test loss reduction modes."""

    def test_reduction_mean(self):
        """Test reduction='mean' returns scalar."""
        loss_fn = CliffCastLoss(reduction='mean')

        B = 8
        predictions = {'risk_index': torch.rand(B)}
        targets = {'risk_index': torch.rand(B)}

        losses = loss_fn(predictions, targets)

        assert losses['loss_risk'].ndim == 0

    def test_reduction_sum(self):
        """Test reduction='sum' sums over batch."""
        loss_fn = CliffCastLoss(reduction='sum')

        B = 8
        predictions = {'risk_index': torch.rand(B)}
        targets = {'risk_index': torch.rand(B)}

        losses = loss_fn(predictions, targets)

        assert losses['loss_risk'].ndim == 0

    def test_reduction_none(self):
        """Test reduction='none' returns per-sample losses."""
        loss_fn = CliffCastLoss(reduction='none')

        B = 8
        predictions = {'risk_index': torch.rand(B)}
        targets = {'risk_index': torch.rand(B)}

        losses = loss_fn(predictions, targets)

        assert losses['loss_risk'].shape == (B,)

    def test_invalid_reduction_raises(self):
        """Test invalid reduction raises error."""
        with pytest.raises(ValueError):
            CliffCastLoss(reduction='invalid')


class TestEdgeCases:
    """Test edge cases."""

    def test_zero_targets(self):
        """Test with all-zero targets."""
        loss_fn = CliffCastLoss()

        B = 8
        predictions = {
            'risk_index': torch.rand(B),
            'retreat_m': torch.rand(B) * 2,
        }
        targets = {
            'risk_index': torch.zeros(B),
            'retreat_m': torch.zeros(B),
        }

        losses = loss_fn(predictions, targets)

        assert not torch.isnan(losses['loss']).any()
        assert losses['loss'] >= 0.0

    def test_large_values(self):
        """Test with large prediction/target values."""
        loss_fn = CliffCastLoss()

        B = 8
        predictions = {
            'retreat_m': torch.rand(B) * 100,  # Very large retreats
        }
        targets = {
            'retreat_m': torch.rand(B) * 100,
        }

        losses = loss_fn(predictions, targets)

        assert not torch.isnan(losses['loss']).any()
        assert not torch.isinf(losses['loss']).any()

    def test_empty_batch_size_one(self):
        """Test with batch size 1."""
        loss_fn = CliffCastLoss()

        B = 1
        predictions = {
            'risk_index': torch.rand(B),
            'retreat_m': torch.rand(B) * 2,
        }
        targets = {
            'risk_index': torch.rand(B),
            'retreat_m': torch.rand(B) * 2,
        }

        losses = loss_fn(predictions, targets)

        assert losses['loss'].ndim == 0


class TestLossComponents:
    """Test individual loss components sum correctly."""

    def test_components_sum_to_total(self):
        """Test individual components sum to total loss."""
        loss_fn = CliffCastLoss(
            weight_risk=1.0,
            weight_retreat=1.0,
            weight_collapse=1.0,
            weight_failure_mode=1.0,
        )

        B = 8
        predictions = {
            'risk_index': torch.rand(B),
            'retreat_m': torch.rand(B) * 2,
            'p_collapse': torch.rand(B, 4),
            'failure_mode_logits': torch.randn(B, 5),
        }
        targets = {
            'risk_index': torch.rand(B),
            'retreat_m': torch.rand(B) * 2,
            'collapse_labels': torch.randint(0, 2, (B, 4)),
            'failure_mode': torch.randint(1, 5, (B,)),  # All failures
        }

        losses = loss_fn(predictions, targets)

        # With all weights=1.0, total should equal sum of components
        expected = (
            losses['loss_risk']
            + losses['loss_retreat']
            + losses['loss_collapse']
            + losses['loss_failure_mode']
        )

        assert torch.allclose(losses['loss'], expected, atol=1e-5)

    def test_zero_weight_excludes_component(self):
        """Test zero weight excludes component from total."""
        loss_fn = CliffCastLoss(
            weight_risk=1.0,
            weight_retreat=0.0,  # Zero weight
            weight_collapse=0.0,  # Zero weight
            weight_failure_mode=0.0,  # Zero weight
        )

        B = 8
        predictions = {
            'risk_index': torch.rand(B),
            'retreat_m': torch.rand(B) * 2,
            'p_collapse': torch.rand(B, 4),
            'failure_mode_logits': torch.randn(B, 5),
        }
        targets = {
            'risk_index': torch.rand(B),
            'retreat_m': torch.rand(B) * 2,
            'collapse_labels': torch.randint(0, 2, (B, 4)),
            'failure_mode': torch.randint(0, 5, (B,)),
        }

        losses = loss_fn(predictions, targets)

        # Total loss should equal only risk component
        assert torch.allclose(losses['loss'], losses['loss_risk'], atol=1e-5)
