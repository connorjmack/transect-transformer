"""
Tests for PredictionHeads module.

Tests cover:
- Initialization with selective head enabling
- Individual head functionality
- Output value ranges and constraints
- Shape validation
- Edge cases
"""

import pytest
import torch

from src.models.prediction_heads import (
    CollapseProbabilityHead,
    ExpectedRetreatHead,
    FailureModeHead,
    PredictionHeads,
    RiskIndexHead,
)


class TestPredictionHeadsInitialization:
    """Test PredictionHeads initialization with selective enabling."""

    def test_all_heads_enabled_by_default(self):
        """Test all heads are enabled by default."""
        heads = PredictionHeads()

        assert heads.enable_risk is True
        assert heads.enable_retreat is True
        assert heads.enable_collapse is True
        assert heads.enable_failure_mode is True
        assert hasattr(heads, 'risk_head')
        assert hasattr(heads, 'retreat_head')
        assert hasattr(heads, 'collapse_head')
        assert hasattr(heads, 'failure_mode_head')

    def test_risk_only(self):
        """Test initialization with only risk head enabled."""
        heads = PredictionHeads(
            enable_risk=True,
            enable_retreat=False,
            enable_collapse=False,
            enable_failure_mode=False,
        )

        assert hasattr(heads, 'risk_head')
        assert not hasattr(heads, 'retreat_head')
        assert not hasattr(heads, 'collapse_head')
        assert not hasattr(heads, 'failure_mode_head')

    def test_selective_heads(self):
        """Test initialization with selective heads enabled."""
        heads = PredictionHeads(
            enable_risk=True,
            enable_retreat=True,
            enable_collapse=False,
            enable_failure_mode=False,
        )

        assert hasattr(heads, 'risk_head')
        assert hasattr(heads, 'retreat_head')
        assert not hasattr(heads, 'collapse_head')
        assert not hasattr(heads, 'failure_mode_head')

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        heads = PredictionHeads(
            d_model=128,
            n_collapse_horizons=3,
            n_failure_modes=4,
            dropout=0.2,
        )

        assert heads.d_model == 128
        assert heads.n_collapse_horizons == 3
        assert heads.n_failure_modes == 4


class TestForwardPass:
    """Test forward pass with various configurations."""

    @pytest.fixture
    def heads_all(self):
        """Create heads with all enabled."""
        return PredictionHeads(d_model=64, dropout=0.0)

    @pytest.fixture
    def sample_pooled(self):
        """Create sample pooled embeddings."""
        B, d_model = 8, 64
        return torch.randn(B, d_model)

    def test_forward_all_heads_enabled(self, heads_all, sample_pooled):
        """Test forward pass with all heads enabled."""
        outputs = heads_all(sample_pooled)

        assert 'risk_index' in outputs
        assert 'retreat_m' in outputs
        assert 'p_collapse' in outputs
        assert 'failure_mode_logits' in outputs

    def test_forward_shapes(self, heads_all, sample_pooled):
        """Test output shapes are correct."""
        B = 8
        outputs = heads_all(sample_pooled)

        assert outputs['risk_index'].shape == (B,)
        assert outputs['retreat_m'].shape == (B,)
        assert outputs['p_collapse'].shape == (B, 4)  # 4 horizons
        assert outputs['failure_mode_logits'].shape == (B, 5)  # 5 modes

    def test_forward_no_nans(self, heads_all, sample_pooled):
        """Test forward pass produces no NaN values."""
        outputs = heads_all(sample_pooled)

        assert not torch.isnan(outputs['risk_index']).any()
        assert not torch.isnan(outputs['retreat_m']).any()
        assert not torch.isnan(outputs['p_collapse']).any()
        assert not torch.isnan(outputs['failure_mode_logits']).any()

    def test_forward_selective_heads(self, sample_pooled):
        """Test forward pass with selective heads."""
        heads = PredictionHeads(
            d_model=64,
            enable_risk=True,
            enable_retreat=False,
            enable_collapse=True,
            enable_failure_mode=False,
        )

        outputs = heads(sample_pooled)

        assert 'risk_index' in outputs
        assert 'retreat_m' not in outputs
        assert 'p_collapse' in outputs
        assert 'failure_mode_logits' not in outputs

    def test_forward_batch_size_one(self, heads_all):
        """Test forward pass with batch size 1."""
        pooled = torch.randn(1, 64)
        outputs = heads_all(pooled)

        assert outputs['risk_index'].shape == (1,)
        assert outputs['retreat_m'].shape == (1,)


class TestRiskIndexHead:
    """Test RiskIndexHead functionality."""

    @pytest.fixture
    def head(self):
        """Create risk index head."""
        return RiskIndexHead(d_model=64, dropout=0.0)

    def test_output_shape(self, head):
        """Test output shape is correct."""
        B, d_model = 8, 64
        x = torch.randn(B, d_model)

        output = head(x)

        assert output.shape == (B,)

    def test_output_range(self, head):
        """Test output is in [0, 1] range."""
        B, d_model = 8, 64
        x = torch.randn(B, d_model)

        output = head(x)

        assert (output >= 0).all()
        assert (output <= 1).all()

    def test_output_no_nans(self, head):
        """Test output has no NaN values."""
        B, d_model = 8, 64
        x = torch.randn(B, d_model)

        output = head(x)

        assert not torch.isnan(output).any()

    def test_different_inputs_produce_different_outputs(self, head):
        """Test different inputs produce different risk scores."""
        x1 = torch.randn(4, 64)
        x2 = torch.randn(4, 64)

        output1 = head(x1)
        output2 = head(x2)

        assert not torch.allclose(output1, output2)


class TestExpectedRetreatHead:
    """Test ExpectedRetreatHead functionality."""

    @pytest.fixture
    def head(self):
        """Create expected retreat head."""
        return ExpectedRetreatHead(d_model=64, dropout=0.0)

    def test_output_shape(self, head):
        """Test output shape is correct."""
        B, d_model = 8, 64
        x = torch.randn(B, d_model)

        output = head(x)

        assert output.shape == (B,)

    def test_output_positive(self, head):
        """Test output is always positive (softplus)."""
        B, d_model = 8, 64
        x = torch.randn(B, d_model) * 10  # Large range

        output = head(x)

        assert (output > 0).all()

    def test_output_no_nans(self, head):
        """Test output has no NaN values."""
        B, d_model = 8, 64
        x = torch.randn(B, d_model)

        output = head(x)

        assert not torch.isnan(output).any()

    def test_output_reasonable_range(self, head):
        """Test output is in reasonable physical range."""
        B, d_model = 100, 64
        x = torch.randn(B, d_model)

        output = head(x)

        # Most retreat values should be < 10 m/yr (extreme values possible but rare)
        assert output.mean() < 10.0


class TestCollapseProbabilityHead:
    """Test CollapseProbabilityHead functionality."""

    @pytest.fixture
    def head(self):
        """Create collapse probability head."""
        return CollapseProbabilityHead(d_model=64, n_horizons=4, dropout=0.0)

    def test_output_shape(self, head):
        """Test output shape is correct."""
        B, d_model = 8, 64
        x = torch.randn(B, d_model)

        output = head(x)

        assert output.shape == (B, 4)  # 4 horizons

    def test_output_range(self, head):
        """Test all outputs are in [0, 1] range."""
        B, d_model = 8, 64
        x = torch.randn(B, d_model)

        output = head(x)

        assert (output >= 0).all()
        assert (output <= 1).all()

    def test_output_no_nans(self, head):
        """Test output has no NaN values."""
        B, d_model = 8, 64
        x = torch.randn(B, d_model)

        output = head(x)

        assert not torch.isnan(output).any()

    def test_custom_horizons(self):
        """Test with custom number of horizons."""
        head = CollapseProbabilityHead(d_model=64, n_horizons=6)

        x = torch.randn(4, 64)
        output = head(x)

        assert output.shape == (4, 6)

    def test_probabilistic_interpretation(self, head):
        """Test that outputs can be interpreted as probabilities."""
        B, d_model = 100, 64
        x = torch.randn(B, d_model)

        output = head(x)

        # Probabilities should be in [0, 1]
        assert (output >= 0).all() and (output <= 1).all()

        # Should have reasonable distribution (not all 0 or all 1)
        assert output.mean() > 0.01
        assert output.mean() < 0.99


class TestFailureModeHead:
    """Test FailureModeHead functionality."""

    @pytest.fixture
    def head(self):
        """Create failure mode head."""
        return FailureModeHead(d_model=64, n_modes=5, dropout=0.0)

    def test_output_shape(self, head):
        """Test output shape is correct."""
        B, d_model = 8, 64
        x = torch.randn(B, d_model)

        output = head(x)

        assert output.shape == (B, 5)  # 5 modes

    def test_output_no_nans(self, head):
        """Test output has no NaN values."""
        B, d_model = 8, 64
        x = torch.randn(B, d_model)

        output = head(x)

        assert not torch.isnan(output).any()

    def test_custom_modes(self):
        """Test with custom number of failure modes."""
        head = FailureModeHead(d_model=64, n_modes=3)

        x = torch.randn(4, 64)
        output = head(x)

        assert output.shape == (4, 3)

    def test_logits_not_probabilities(self, head):
        """Test output is logits (not probabilities)."""
        B, d_model = 8, 64
        x = torch.randn(B, d_model)

        logits = head(x)

        # Logits can be any real number (not constrained to [0, 1])
        # Just check they're finite
        assert torch.isfinite(logits).all()

    def test_predicted_class(self, head):
        """Test extracting predicted class from logits."""
        B, d_model = 8, 64
        x = torch.randn(B, d_model)

        logits = head(x)
        predicted = torch.argmax(logits, dim=-1)

        assert predicted.shape == (B,)
        assert (predicted >= 0).all()
        assert (predicted < 5).all()  # 5 modes


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_input(self):
        """Test with all-zero input."""
        heads = PredictionHeads(d_model=64, dropout=0.0)
        pooled = torch.zeros(4, 64)

        outputs = heads(pooled)

        # Should not crash, outputs should be valid
        assert not torch.isnan(outputs['risk_index']).any()
        assert not torch.isnan(outputs['retreat_m']).any()
        assert (outputs['retreat_m'] > 0).all()  # Still positive

    def test_large_input(self):
        """Test with very large input values."""
        heads = PredictionHeads(d_model=64, dropout=0.0)
        pooled = torch.randn(4, 64) * 100

        outputs = heads(pooled)

        assert not torch.isnan(outputs['risk_index']).any()
        assert not torch.isnan(outputs['retreat_m']).any()
        assert not torch.isinf(outputs['retreat_m']).any()

    def test_eval_mode(self):
        """Test in eval mode."""
        heads = PredictionHeads(d_model=64, dropout=0.0)
        heads.eval()

        pooled = torch.randn(4, 64)

        with torch.no_grad():
            outputs = heads(pooled)

        assert outputs['risk_index'].shape == (4,)

    def test_gradient_flow(self):
        """Test gradients flow through all heads."""
        heads = PredictionHeads(d_model=64, dropout=0.0)
        heads.train()

        pooled = torch.randn(4, 64, requires_grad=True)

        outputs = heads(pooled)

        # Compute loss as sum of all outputs
        loss = outputs['risk_index'].sum()
        loss += outputs['retreat_m'].sum()
        loss += outputs['p_collapse'].sum()
        loss += outputs['failure_mode_logits'].sum()

        loss.backward()

        assert pooled.grad is not None
        assert not torch.isnan(pooled.grad).any()

    def test_batch_size_large(self):
        """Test with large batch size."""
        heads = PredictionHeads(d_model=64, dropout=0.0)
        pooled = torch.randn(128, 64)

        outputs = heads(pooled)

        assert outputs['risk_index'].shape == (128,)
        assert outputs['retreat_m'].shape == (128,)


class TestModelSize:
    """Test model size scales appropriately."""

    def test_parameter_count_all_heads(self):
        """Test parameter count with all heads enabled."""
        heads = PredictionHeads(d_model=256)

        param_count = sum(p.numel() for p in heads.parameters())

        # Should be in hundreds of thousands
        assert 100_000 < param_count < 2_000_000

    def test_fewer_params_with_fewer_heads(self):
        """Test fewer heads means fewer parameters."""
        all_heads = PredictionHeads(d_model=256)
        one_head = PredictionHeads(
            d_model=256,
            enable_risk=True,
            enable_retreat=False,
            enable_collapse=False,
            enable_failure_mode=False,
        )

        all_params = sum(p.numel() for p in all_heads.parameters())
        one_params = sum(p.numel() for p in one_head.parameters())

        assert one_params < all_params

    def test_parameter_count_scales_with_d_model(self):
        """Test parameter count scales with d_model."""
        small = PredictionHeads(d_model=64)
        large = PredictionHeads(d_model=256)

        small_params = sum(p.numel() for p in small.parameters())
        large_params = sum(p.numel() for p in large.parameters())

        assert large_params > small_params
