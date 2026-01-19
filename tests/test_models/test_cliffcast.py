"""
Tests for CliffCast main model.

Tests cover:
- Initialization and parameter validation
- End-to-end forward pass
- Selective head enabling (phased training)
- Attention weight extraction
- Integration of all components
- Edge cases and error handling
"""

import pytest
import torch

from src.models.cliffcast import CliffCast


class TestCliffCastInitialization:
    """Test CliffCast initialization."""

    def test_default_initialization(self):
        """Test model initializes with default parameters."""
        model = CliffCast()

        assert model.d_model == 256
        assert hasattr(model, 'transect_encoder')
        assert hasattr(model, 'wave_encoder')
        assert hasattr(model, 'atmos_encoder')
        assert hasattr(model, 'fusion')
        assert hasattr(model, 'heads')

    def test_custom_initialization(self):
        """Test model initializes with custom parameters."""
        model = CliffCast(
            d_model=128,
            n_heads=4,
            n_layers_spatial=1,
            n_layers_temporal=1,
            n_layers_env=2,
            n_layers_fusion=1,
            dropout=0.2,
        )

        assert model.d_model == 128

    def test_selective_heads_initialization(self):
        """Test model with selective heads enabled."""
        model = CliffCast(
            enable_risk=True,
            enable_retreat=False,
            enable_collapse=False,
            enable_failure_mode=False,
        )

        # Check that only risk head is enabled
        assert model.heads.enable_risk is True
        assert model.heads.enable_retreat is False
        assert model.heads.enable_collapse is False
        assert model.heads.enable_failure_mode is False


class TestForwardPass:
    """Test end-to-end forward pass."""

    @pytest.fixture
    def model(self):
        """Create small CliffCast model for testing."""
        return CliffCast(
            d_model=64,
            n_heads=4,
            n_layers_spatial=1,
            n_layers_temporal=1,
            n_layers_env=1,
            n_layers_fusion=1,
            dropout=0.0,
        )

    @pytest.fixture
    def sample_batch(self):
        """Create sample batch data."""
        B, T, N = 4, 3, 128
        T_w, T_a = 360, 90

        return {
            # Transect inputs
            'point_features': torch.randn(B, T, N, 12),
            'metadata': torch.randn(B, T, 12),
            'distances': torch.linspace(0, 50, N).unsqueeze(0).unsqueeze(0).expand(B, T, N),
            # Wave inputs
            'wave_features': torch.randn(B, T_w, 4),
            'wave_doy': torch.randint(0, 366, (B, T_w)),
            # Atmospheric inputs
            'atmos_features': torch.randn(B, T_a, 24),
            'atmos_doy': torch.randint(0, 366, (B, T_a)),
        }

    def test_forward_output_keys(self, model, sample_batch):
        """Test forward pass returns expected output keys."""
        outputs = model(**sample_batch)

        # All heads enabled by default
        assert 'risk_index' in outputs
        assert 'retreat_m' in outputs
        assert 'p_collapse' in outputs
        assert 'failure_mode_logits' in outputs

    def test_forward_output_shapes(self, model, sample_batch):
        """Test forward pass returns correct output shapes."""
        B = 4
        outputs = model(**sample_batch)

        assert outputs['risk_index'].shape == (B,)
        assert outputs['retreat_m'].shape == (B,)
        assert outputs['p_collapse'].shape == (B, 4)  # 4 horizons
        assert outputs['failure_mode_logits'].shape == (B, 5)  # 5 modes

    def test_forward_no_nans(self, model, sample_batch):
        """Test forward pass produces no NaN values."""
        outputs = model(**sample_batch)

        assert not torch.isnan(outputs['risk_index']).any()
        assert not torch.isnan(outputs['retreat_m']).any()
        assert not torch.isnan(outputs['p_collapse']).any()
        assert not torch.isnan(outputs['failure_mode_logits']).any()

    def test_forward_output_ranges(self, model, sample_batch):
        """Test output values are in expected ranges."""
        outputs = model(**sample_batch)

        # Risk index in [0, 1]
        assert (outputs['risk_index'] >= 0).all()
        assert (outputs['risk_index'] <= 1).all()

        # Retreat positive
        assert (outputs['retreat_m'] > 0).all()

        # Collapse probabilities in [0, 1]
        assert (outputs['p_collapse'] >= 0).all()
        assert (outputs['p_collapse'] <= 1).all()

    def test_forward_batch_size_one(self, model):
        """Test forward pass with batch size 1."""
        B, T, N = 1, 3, 128
        T_w, T_a = 360, 90

        inputs = {
            'point_features': torch.randn(B, T, N, 12),
            'metadata': torch.randn(B, T, 12),
            'distances': torch.linspace(0, 50, N).unsqueeze(0).unsqueeze(0).expand(B, T, N),
            'wave_features': torch.randn(B, T_w, 4),
            'wave_doy': torch.randint(0, 366, (B, T_w)),
            'atmos_features': torch.randn(B, T_a, 24),
            'atmos_doy': torch.randint(0, 366, (B, T_a)),
        }

        outputs = model(**inputs)

        assert outputs['risk_index'].shape == (1,)


class TestSelectiveHeads:
    """Test model with selective heads enabled."""

    def test_risk_only(self):
        """Test model with only risk head enabled."""
        model = CliffCast(
            d_model=64,
            enable_risk=True,
            enable_retreat=False,
            enable_collapse=False,
            enable_failure_mode=False,
        )

        B, T, N, T_w, T_a = 4, 3, 128, 360, 90

        inputs = {
            'point_features': torch.randn(B, T, N, 12),
            'metadata': torch.randn(B, T, 12),
            'distances': torch.linspace(0, 50, N).unsqueeze(0).unsqueeze(0).expand(B, T, N),
            'wave_features': torch.randn(B, T_w, 4),
            'atmos_features': torch.randn(B, T_a, 24),
        }

        outputs = model(**inputs)

        assert 'risk_index' in outputs
        assert 'retreat_m' not in outputs
        assert 'p_collapse' not in outputs
        assert 'failure_mode_logits' not in outputs

    def test_risk_and_retreat(self):
        """Test model with risk and retreat heads enabled."""
        model = CliffCast(
            d_model=64,
            enable_risk=True,
            enable_retreat=True,
            enable_collapse=False,
            enable_failure_mode=False,
        )

        B, T, N, T_w, T_a = 4, 3, 128, 360, 90

        inputs = {
            'point_features': torch.randn(B, T, N, 12),
            'metadata': torch.randn(B, T, 12),
            'distances': torch.linspace(0, 50, N).unsqueeze(0).unsqueeze(0).expand(B, T, N),
            'wave_features': torch.randn(B, T_w, 4),
            'atmos_features': torch.randn(B, T_a, 24),
        }

        outputs = model(**inputs)

        assert 'risk_index' in outputs
        assert 'retreat_m' in outputs
        assert 'p_collapse' not in outputs
        assert 'failure_mode_logits' not in outputs


class TestAttentionExtraction:
    """Test attention weight extraction."""

    @pytest.fixture
    def model(self):
        """Create model for attention testing."""
        return CliffCast(
            d_model=64,
            n_heads=4,
            n_layers_spatial=1,
            n_layers_temporal=1,
            n_layers_env=1,
            n_layers_fusion=1,
            dropout=0.0,
        )

    @pytest.fixture
    def sample_batch(self):
        """Create sample batch."""
        B, T, N = 2, 3, 128
        T_w, T_a = 360, 90

        return {
            'point_features': torch.randn(B, T, N, 12),
            'metadata': torch.randn(B, T, 12),
            'distances': torch.linspace(0, 50, N).unsqueeze(0).unsqueeze(0).expand(B, T, N),
            'wave_features': torch.randn(B, T_w, 4),
            'atmos_features': torch.randn(B, T_a, 24),
        }

    def test_spatial_attention_extraction(self, model, sample_batch):
        """Test spatial attention weights can be extracted."""
        outputs = model(**sample_batch, return_spatial_attention=True)

        assert 'spatial_attention' in outputs
        # Note: Currently returns None (placeholder)
        # When implemented, shape should be (B, n_heads, T, N, N)

    def test_temporal_attention_extraction(self, model, sample_batch):
        """Test temporal attention weights can be extracted."""
        outputs = model(**sample_batch, return_temporal_attention=True)

        assert 'temporal_attention' in outputs
        # Note: Currently returns None (placeholder)
        # When implemented, shape should be (B, n_heads, T, T)

    def test_env_attention_extraction(self, model, sample_batch):
        """Test environmental attention weights can be extracted."""
        outputs = model(**sample_batch, return_env_attention=True)

        assert 'env_attention' in outputs
        assert outputs['env_attention'] is not None

        B, T = 2, 3
        T_env = 360 + 90  # wave + atmos
        n_heads = 4

        assert outputs['env_attention'].shape == (B, n_heads, T, T_env)

    def test_get_attention_weights_method(self, model, sample_batch):
        """Test convenience method for extracting all attention weights."""
        outputs = model.get_attention_weights(**sample_batch)

        assert 'spatial_attention' in outputs
        assert 'temporal_attention' in outputs
        assert 'env_attention' in outputs


class TestPaddingMask:
    """Test padding mask functionality."""

    @pytest.fixture
    def model(self):
        """Create model for padding mask testing."""
        return CliffCast(
            d_model=64,
            n_heads=4,
            n_layers_spatial=1,
            n_layers_temporal=1,
            n_layers_env=1,
            n_layers_fusion=1,
            dropout=0.0,
        )

    def test_wave_padding_mask(self, model):
        """Test with wave padding mask."""
        B, T, N, T_w, T_a = 4, 3, 128, 360, 90

        wave_padding_mask = torch.zeros(B, T_w, dtype=torch.bool)
        wave_padding_mask[:, 300:] = True  # Last 60 timesteps are padding

        inputs = {
            'point_features': torch.randn(B, T, N, 12),
            'metadata': torch.randn(B, T, 12),
            'distances': torch.linspace(0, 50, N).unsqueeze(0).unsqueeze(0).expand(B, T, N),
            'wave_features': torch.randn(B, T_w, 4),
            'wave_padding_mask': wave_padding_mask,
            'atmos_features': torch.randn(B, T_a, 24),
        }

        outputs = model(**inputs)

        assert not torch.isnan(outputs['risk_index']).any()

    def test_atmos_padding_mask(self, model):
        """Test with atmospheric padding mask."""
        B, T, N, T_w, T_a = 4, 3, 128, 360, 90

        atmos_padding_mask = torch.zeros(B, T_a, dtype=torch.bool)
        atmos_padding_mask[:, 70:] = True  # Last 20 timesteps are padding

        inputs = {
            'point_features': torch.randn(B, T, N, 12),
            'metadata': torch.randn(B, T, 12),
            'distances': torch.linspace(0, 50, N).unsqueeze(0).unsqueeze(0).expand(B, T, N),
            'wave_features': torch.randn(B, T_w, 4),
            'atmos_features': torch.randn(B, T_a, 24),
            'atmos_padding_mask': atmos_padding_mask,
        }

        outputs = model(**inputs)

        assert not torch.isnan(outputs['risk_index']).any()

    def test_both_padding_masks(self, model):
        """Test with both padding masks."""
        B, T, N, T_w, T_a = 4, 3, 128, 360, 90

        wave_padding_mask = torch.zeros(B, T_w, dtype=torch.bool)
        wave_padding_mask[:, 300:] = True

        atmos_padding_mask = torch.zeros(B, T_a, dtype=torch.bool)
        atmos_padding_mask[:, 70:] = True

        inputs = {
            'point_features': torch.randn(B, T, N, 12),
            'metadata': torch.randn(B, T, 12),
            'distances': torch.linspace(0, 50, N).unsqueeze(0).unsqueeze(0).expand(B, T, N),
            'wave_features': torch.randn(B, T_w, 4),
            'wave_padding_mask': wave_padding_mask,
            'atmos_features': torch.randn(B, T_a, 24),
            'atmos_padding_mask': atmos_padding_mask,
        }

        outputs = model(**inputs)

        assert not torch.isnan(outputs['risk_index']).any()


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def model(self):
        """Create model for edge case testing."""
        return CliffCast(
            d_model=64,
            n_heads=4,
            n_layers_spatial=1,
            n_layers_temporal=1,
            n_layers_env=1,
            n_layers_fusion=1,
            dropout=0.0,
        )

    def test_single_lidar_epoch(self, model):
        """Test with single LiDAR epoch (T=1)."""
        B, T, N, T_w, T_a = 4, 1, 128, 360, 90

        inputs = {
            'point_features': torch.randn(B, T, N, 12),
            'metadata': torch.randn(B, T, 12),
            'distances': torch.linspace(0, 50, N).unsqueeze(0).unsqueeze(0).expand(B, T, N),
            'wave_features': torch.randn(B, T_w, 4),
            'atmos_features': torch.randn(B, T_a, 24),
        }

        outputs = model(**inputs)

        assert outputs['risk_index'].shape == (B,)

    def test_many_lidar_epochs(self, model):
        """Test with many LiDAR epochs."""
        B, T, N, T_w, T_a = 4, 10, 128, 360, 90

        inputs = {
            'point_features': torch.randn(B, T, N, 12),
            'metadata': torch.randn(B, T, 12),
            'distances': torch.linspace(0, 50, N).unsqueeze(0).unsqueeze(0).expand(B, T, N),
            'wave_features': torch.randn(B, T_w, 4),
            'atmos_features': torch.randn(B, T_a, 24),
        }

        outputs = model(**inputs)

        assert outputs['risk_index'].shape == (B,)

    def test_eval_mode(self, model):
        """Test model in eval mode."""
        model.eval()

        B, T, N, T_w, T_a = 4, 3, 128, 360, 90

        inputs = {
            'point_features': torch.randn(B, T, N, 12),
            'metadata': torch.randn(B, T, 12),
            'distances': torch.linspace(0, 50, N).unsqueeze(0).unsqueeze(0).expand(B, T, N),
            'wave_features': torch.randn(B, T_w, 4),
            'atmos_features': torch.randn(B, T_a, 24),
        }

        with torch.no_grad():
            outputs = model(**inputs)

        assert outputs['risk_index'].shape == (B,)

    def test_gradient_flow(self, model):
        """Test gradients flow through entire model."""
        model.train()

        B, T, N, T_w, T_a = 2, 3, 128, 360, 90

        point_features = torch.randn(B, T, N, 12, requires_grad=True)
        metadata = torch.randn(B, T, 12, requires_grad=True)
        distances = torch.linspace(0, 50, N).unsqueeze(0).unsqueeze(0).expand(B, T, N)
        wave_features = torch.randn(B, T_w, 4, requires_grad=True)
        atmos_features = torch.randn(B, T_a, 24, requires_grad=True)

        outputs = model(
            point_features=point_features,
            metadata=metadata,
            distances=distances,
            wave_features=wave_features,
            atmos_features=atmos_features,
        )

        # Compute loss as sum of all outputs
        loss = outputs['risk_index'].sum()
        loss += outputs['retreat_m'].sum()
        loss += outputs['p_collapse'].sum()
        loss += outputs['failure_mode_logits'].sum()

        loss.backward()

        # Check gradients exist for inputs
        assert point_features.grad is not None
        assert metadata.grad is not None
        assert wave_features.grad is not None
        assert atmos_features.grad is not None

        # Check no NaNs in gradients
        assert not torch.isnan(point_features.grad).any()
        assert not torch.isnan(metadata.grad).any()
        assert not torch.isnan(wave_features.grad).any()
        assert not torch.isnan(atmos_features.grad).any()


class TestModelSize:
    """Test model size and parameter counts."""

    def test_parameter_count_full_model(self):
        """Test full model has reasonable parameter count."""
        model = CliffCast()

        param_count = sum(p.numel() for p in model.parameters())

        # Should be in low millions (e.g., 2-10M parameters)
        assert 1_000_000 < param_count < 20_000_000

    def test_smaller_model_fewer_parameters(self):
        """Test smaller model has fewer parameters."""
        small = CliffCast(d_model=64, n_layers_spatial=1, n_layers_temporal=1)
        large = CliffCast(d_model=256, n_layers_spatial=2, n_layers_temporal=2)

        small_params = sum(p.numel() for p in small.parameters())
        large_params = sum(p.numel() for p in large.parameters())

        assert small_params < large_params

    def test_fewer_heads_fewer_parameters(self):
        """Test model with fewer heads has fewer parameters."""
        all_heads = CliffCast(d_model=128)
        one_head = CliffCast(
            d_model=128,
            enable_risk=True,
            enable_retreat=False,
            enable_collapse=False,
            enable_failure_mode=False,
        )

        all_params = sum(p.numel() for p in all_heads.parameters())
        one_params = sum(p.numel() for p in one_head.parameters())

        assert one_params < all_params


class TestIntegration:
    """Test integration of all components."""

    def test_all_components_work_together(self):
        """Test that all model components work together seamlessly."""
        model = CliffCast(
            d_model=64,
            n_heads=4,
            n_layers_spatial=1,
            n_layers_temporal=1,
            n_layers_env=1,
            n_layers_fusion=1,
        )

        B, T, N, T_w, T_a = 4, 3, 128, 360, 90

        inputs = {
            'point_features': torch.randn(B, T, N, 12),
            'metadata': torch.randn(B, T, 12),
            'distances': torch.linspace(0, 50, N).unsqueeze(0).unsqueeze(0).expand(B, T, N),
            'timestamps': torch.arange(T).unsqueeze(0).expand(B, T),
            'wave_features': torch.randn(B, T_w, 4),
            'wave_doy': torch.randint(0, 366, (B, T_w)),
            'wave_timestamps': torch.arange(T_w).unsqueeze(0).expand(B, T_w),
            'atmos_features': torch.randn(B, T_a, 24),
            'atmos_doy': torch.randint(0, 366, (B, T_a)),
            'atmos_timestamps': torch.arange(T_a).unsqueeze(0).expand(B, T_a),
        }

        outputs = model(**inputs)

        # Verify all outputs are produced
        assert 'risk_index' in outputs
        assert 'retreat_m' in outputs
        assert 'p_collapse' in outputs
        assert 'failure_mode_logits' in outputs

        # Verify all outputs have correct shapes
        assert outputs['risk_index'].shape == (B,)
        assert outputs['retreat_m'].shape == (B,)
        assert outputs['p_collapse'].shape == (B, 4)
        assert outputs['failure_mode_logits'].shape == (B, 5)

        # Verify no NaNs
        for key, value in outputs.items():
            if key.endswith('_attention'):
                continue
            assert not torch.isnan(value).any(), f"NaN found in {key}"

    def test_deterministic_with_seed(self):
        """Test model produces deterministic outputs with same seed."""
        torch.manual_seed(42)
        model1 = CliffCast(d_model=64, dropout=0.0)

        torch.manual_seed(42)
        model2 = CliffCast(d_model=64, dropout=0.0)

        B, T, N, T_w, T_a = 2, 3, 128, 100, 50

        torch.manual_seed(123)
        inputs = {
            'point_features': torch.randn(B, T, N, 12),
            'metadata': torch.randn(B, T, 12),
            'distances': torch.linspace(0, 50, N).unsqueeze(0).unsqueeze(0).expand(B, T, N),
            'wave_features': torch.randn(B, T_w, 4),
            'atmos_features': torch.randn(B, T_a, 24),
        }

        model1.eval()
        model2.eval()

        with torch.no_grad():
            outputs1 = model1(**inputs)
            outputs2 = model2(**inputs)

        # Same initialization should produce same outputs
        assert torch.allclose(outputs1['risk_index'], outputs2['risk_index'], atol=1e-6)
