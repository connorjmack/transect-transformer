"""
Tests for CrossAttentionFusion module.

Tests cover:
- Initialization and parameter validation
- Forward pass with various input configurations
- Attention weight extraction
- Padding mask handling
- Shape validation
- Edge cases
"""

import pytest
import torch

from src.models.fusion import CrossAttentionFusion, CrossAttentionLayer


class TestCrossAttentionFusionInitialization:
    """Test CrossAttentionFusion initialization."""

    def test_default_initialization(self):
        """Test fusion module initializes with default parameters."""
        fusion = CrossAttentionFusion()

        assert fusion.d_model == 256
        assert fusion.n_heads == 8
        assert fusion.n_layers == 2

    def test_custom_initialization(self):
        """Test fusion module initializes with custom parameters."""
        fusion = CrossAttentionFusion(
            d_model=128,
            n_heads=4,
            n_layers=3,
            dropout=0.2,
        )

        assert fusion.d_model == 128
        assert fusion.n_heads == 4
        assert fusion.n_layers == 3
        assert len(fusion.layers) == 3

    def test_layers_created(self):
        """Test that cross-attention layers are created."""
        fusion = CrossAttentionFusion(n_layers=3)

        assert len(fusion.layers) == 3
        assert all(isinstance(layer, CrossAttentionLayer) for layer in fusion.layers)


class TestForwardPass:
    """Test forward pass with various configurations."""

    @pytest.fixture
    def fusion(self):
        """Create small fusion module for testing."""
        return CrossAttentionFusion(
            d_model=64,
            n_heads=4,
            n_layers=2,
            dropout=0.0,
        )

    @pytest.fixture
    def sample_inputs(self):
        """Create sample input data."""
        B = 4
        T_cliff = 5  # 5 LiDAR epochs
        T_env = 450  # 360 wave + 90 atmos timesteps
        d_model = 64

        cliff_embeddings = torch.randn(B, T_cliff, d_model)
        env_embeddings = torch.randn(B, T_env, d_model)

        return {
            'cliff_embeddings': cliff_embeddings,
            'env_embeddings': env_embeddings,
        }

    def test_forward_output_shapes(self, fusion, sample_inputs):
        """Test forward pass returns correct output shapes."""
        outputs = fusion(**sample_inputs)

        B, T_cliff = 4, 5
        d_model = 64

        assert 'fused' in outputs
        assert 'pooled' in outputs

        assert outputs['fused'].shape == (B, T_cliff, d_model)
        assert outputs['pooled'].shape == (B, d_model)

    def test_forward_no_nans(self, fusion, sample_inputs):
        """Test forward pass produces no NaN values."""
        outputs = fusion(**sample_inputs)

        assert not torch.isnan(outputs['fused']).any()
        assert not torch.isnan(outputs['pooled']).any()

    def test_forward_with_attention_return(self, fusion, sample_inputs):
        """Test forward pass with attention weight return."""
        outputs = fusion(**sample_inputs, return_attention=True)

        B, T_cliff, T_env = 4, 5, 450
        n_heads = 4

        assert 'attention' in outputs
        assert outputs['attention'] is not None
        assert outputs['attention'].shape == (B, n_heads, T_cliff, T_env)

    def test_forward_without_attention_return(self, fusion, sample_inputs):
        """Test forward pass without attention weight return."""
        outputs = fusion(**sample_inputs, return_attention=False)

        assert 'attention' not in outputs

    def test_forward_with_padding_mask(self, fusion):
        """Test forward pass with environmental padding mask."""
        B, T_cliff, T_env, d_model = 4, 5, 450, 64

        cliff_embeddings = torch.randn(B, T_cliff, d_model)
        env_embeddings = torch.randn(B, T_env, d_model)

        # Last 100 timesteps are padding
        env_padding_mask = torch.zeros(B, T_env, dtype=torch.bool)
        env_padding_mask[:, 350:] = True

        outputs = fusion(
            cliff_embeddings=cliff_embeddings,
            env_embeddings=env_embeddings,
            env_padding_mask=env_padding_mask,
        )

        assert outputs['fused'].shape == (B, T_cliff, d_model)
        assert outputs['pooled'].shape == (B, d_model)
        assert not torch.isnan(outputs['pooled']).any()

    def test_forward_single_cliff_timestep(self, fusion):
        """Test forward pass with single cliff timestep."""
        B, T_cliff, T_env, d_model = 4, 1, 450, 64

        cliff_embeddings = torch.randn(B, T_cliff, d_model)
        env_embeddings = torch.randn(B, T_env, d_model)

        outputs = fusion(cliff_embeddings, env_embeddings)

        assert outputs['fused'].shape == (B, T_cliff, d_model)
        assert outputs['pooled'].shape == (B, d_model)

    def test_forward_batch_size_one(self, fusion):
        """Test forward pass with batch size 1."""
        B, T_cliff, T_env, d_model = 1, 5, 450, 64

        cliff_embeddings = torch.randn(B, T_cliff, d_model)
        env_embeddings = torch.randn(B, T_env, d_model)

        outputs = fusion(cliff_embeddings, env_embeddings)

        assert outputs['pooled'].shape == (1, d_model)


class TestAttentionWeights:
    """Test attention weight extraction."""

    @pytest.fixture
    def fusion(self):
        """Create fusion module for attention testing."""
        return CrossAttentionFusion(
            d_model=64,
            n_heads=4,
            n_layers=2,
            dropout=0.0,
        )

    def test_attention_weights_shape(self, fusion):
        """Test attention weights have correct shape."""
        B, T_cliff, T_env, d_model = 4, 5, 450, 64

        cliff_embeddings = torch.randn(B, T_cliff, d_model)
        env_embeddings = torch.randn(B, T_env, d_model)

        outputs = fusion(cliff_embeddings, env_embeddings, return_attention=True)

        n_heads = 4
        assert outputs['attention'].shape == (B, n_heads, T_cliff, T_env)

    def test_attention_weights_sum_to_one(self, fusion):
        """Test attention weights sum to 1 over environmental dimension."""
        B, T_cliff, T_env, d_model = 2, 3, 100, 64

        cliff_embeddings = torch.randn(B, T_cliff, d_model)
        env_embeddings = torch.randn(B, T_env, d_model)

        outputs = fusion(cliff_embeddings, env_embeddings, return_attention=True)

        # Sum over environmental dimension (last dim)
        attn_sum = outputs['attention'].sum(dim=-1)

        # Should sum to approximately 1
        assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5)

    def test_attention_weights_non_negative(self, fusion):
        """Test attention weights are non-negative."""
        B, T_cliff, T_env, d_model = 2, 3, 100, 64

        cliff_embeddings = torch.randn(B, T_cliff, d_model)
        env_embeddings = torch.randn(B, T_env, d_model)

        outputs = fusion(cliff_embeddings, env_embeddings, return_attention=True)

        assert (outputs['attention'] >= 0).all()


class TestPaddingMask:
    """Test padding mask functionality."""

    @pytest.fixture
    def fusion(self):
        """Create fusion module for padding mask testing."""
        return CrossAttentionFusion(
            d_model=64,
            n_heads=4,
            n_layers=2,
            dropout=0.0,
        )

    def test_padding_mask_basic(self, fusion):
        """Test basic padding mask functionality."""
        B, T_cliff, T_env, d_model = 4, 5, 200, 64

        cliff_embeddings = torch.randn(B, T_cliff, d_model)
        env_embeddings = torch.randn(B, T_env, d_model)

        # Last 50 timesteps are padding
        env_padding_mask = torch.zeros(B, T_env, dtype=torch.bool)
        env_padding_mask[:, 150:] = True

        outputs = fusion(cliff_embeddings, env_embeddings, env_padding_mask)

        assert outputs['pooled'].shape == (B, d_model)
        assert not torch.isnan(outputs['pooled']).any()

    def test_padding_mask_per_sample_different(self, fusion):
        """Test padding mask with different lengths per sample."""
        B, T_cliff, T_env, d_model = 4, 5, 200, 64

        cliff_embeddings = torch.randn(B, T_cliff, d_model)
        env_embeddings = torch.randn(B, T_env, d_model)

        # Different padding for each sample
        env_padding_mask = torch.zeros(B, T_env, dtype=torch.bool)
        env_padding_mask[0, 180:] = True
        env_padding_mask[1, 160:] = True
        env_padding_mask[2, 140:] = True
        env_padding_mask[3, 120:] = True

        outputs = fusion(cliff_embeddings, env_embeddings, env_padding_mask)

        assert outputs['pooled'].shape == (B, d_model)
        assert not torch.isnan(outputs['pooled']).any()

    def test_attention_respects_padding_mask(self, fusion):
        """Test that attention weights are zero for padded positions."""
        B, T_cliff, T_env, d_model = 2, 3, 100, 64

        cliff_embeddings = torch.randn(B, T_cliff, d_model)
        env_embeddings = torch.randn(B, T_env, d_model)

        # Last 20 timesteps are padding
        env_padding_mask = torch.zeros(B, T_env, dtype=torch.bool)
        env_padding_mask[:, 80:] = True

        outputs = fusion(
            cliff_embeddings, env_embeddings, env_padding_mask, return_attention=True
        )

        # Attention weights for padded positions should be zero
        # (or very close to zero due to softmax)
        padded_attention = outputs['attention'][:, :, :, 80:]
        assert torch.allclose(padded_attention, torch.zeros_like(padded_attention), atol=1e-6)


class TestCrossAttentionLayer:
    """Test individual CrossAttentionLayer."""

    @pytest.fixture
    def layer(self):
        """Create cross-attention layer for testing."""
        return CrossAttentionLayer(d_model=64, n_heads=4, dropout=0.0)

    def test_layer_forward_shapes(self, layer):
        """Test layer forward pass returns correct shapes."""
        B, T_query, T_kv, d_model = 4, 5, 100, 64

        query = torch.randn(B, T_query, d_model)
        key_value = torch.randn(B, T_kv, d_model)

        output, attn_weights = layer(query, key_value, return_attention=True)

        assert output.shape == (B, T_query, d_model)
        assert attn_weights.shape == (B, 4, T_query, T_kv)  # 4 heads

    def test_layer_residual_connection(self, layer):
        """Test residual connection is applied."""
        B, T_query, T_kv, d_model = 2, 3, 50, 64

        query = torch.randn(B, T_query, d_model)
        key_value = torch.randn(B, T_kv, d_model)

        output, _ = layer(query, key_value)

        # Output should be different from input (attention applied)
        assert not torch.allclose(output, query)

        # But shape should be preserved
        assert output.shape == query.shape


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def fusion(self):
        """Create fusion module for edge case testing."""
        return CrossAttentionFusion(
            d_model=64,
            n_heads=4,
            n_layers=2,
            dropout=0.0,
        )

    def test_zero_embeddings(self, fusion):
        """Test fusion with all-zero embeddings."""
        B, T_cliff, T_env, d_model = 4, 5, 100, 64

        cliff_embeddings = torch.zeros(B, T_cliff, d_model)
        env_embeddings = torch.zeros(B, T_env, d_model)

        outputs = fusion(cliff_embeddings, env_embeddings)

        assert not torch.isnan(outputs['pooled']).any()

    def test_large_environmental_sequence(self, fusion):
        """Test fusion with very long environmental sequence."""
        B, T_cliff, T_env, d_model = 2, 5, 1000, 64

        cliff_embeddings = torch.randn(B, T_cliff, d_model)
        env_embeddings = torch.randn(B, T_env, d_model)

        outputs = fusion(cliff_embeddings, env_embeddings)

        assert outputs['fused'].shape == (B, T_cliff, d_model)
        assert outputs['pooled'].shape == (B, d_model)

    def test_eval_mode(self, fusion):
        """Test fusion in eval mode."""
        fusion.eval()

        B, T_cliff, T_env, d_model = 4, 5, 100, 64

        cliff_embeddings = torch.randn(B, T_cliff, d_model)
        env_embeddings = torch.randn(B, T_env, d_model)

        with torch.no_grad():
            outputs = fusion(cliff_embeddings, env_embeddings)

        assert outputs['pooled'].shape == (B, d_model)
        assert not torch.isnan(outputs['pooled']).any()

    def test_gradient_flow(self, fusion):
        """Test gradients flow through fusion module."""
        fusion.train()

        B, T_cliff, T_env, d_model = 4, 5, 100, 64

        cliff_embeddings = torch.randn(B, T_cliff, d_model, requires_grad=True)
        env_embeddings = torch.randn(B, T_env, d_model, requires_grad=True)

        outputs = fusion(cliff_embeddings, env_embeddings)
        loss = outputs['pooled'].sum()
        loss.backward()

        # Check gradients exist
        assert cliff_embeddings.grad is not None
        assert env_embeddings.grad is not None
        assert not torch.isnan(cliff_embeddings.grad).any()
        assert not torch.isnan(env_embeddings.grad).any()


class TestModelSize:
    """Test model scales appropriately with parameters."""

    def test_parameter_count_increases_with_layers(self):
        """Test more layers means more parameters."""
        small = CrossAttentionFusion(d_model=64, n_layers=1)
        large = CrossAttentionFusion(d_model=64, n_layers=3)

        small_params = sum(p.numel() for p in small.parameters())
        large_params = sum(p.numel() for p in large.parameters())

        assert large_params > small_params

    def test_parameter_count_increases_with_d_model(self):
        """Test larger d_model means more parameters."""
        small = CrossAttentionFusion(d_model=64, n_layers=2)
        large = CrossAttentionFusion(d_model=256, n_layers=2)

        small_params = sum(p.numel() for p in small.parameters())
        large_params = sum(p.numel() for p in large.parameters())

        assert large_params > small_params

    def test_parameter_count_reasonable(self):
        """Test default fusion module has reasonable parameter count."""
        fusion = CrossAttentionFusion()

        param_count = sum(p.numel() for p in fusion.parameters())

        # Should be in hundreds of thousands to low millions
        assert 100_000 < param_count < 10_000_000
