"""
Tests for SpatioTemporalTransectEncoder.

Tests cover:
- Initialization and parameter validation
- Distance-based positional encoding
- Spatial attention (within timesteps)
- Temporal attention (across timesteps)
- Shape validation throughout pipeline
- Edge cases and error handling
"""

import pytest
import torch

from src.models.transect_encoder import SpatioTemporalTransectEncoder


class TestInitialization:
    """Test encoder initialization and parameter validation."""

    def test_default_initialization(self):
        """Test encoder initializes with default parameters."""
        encoder = SpatioTemporalTransectEncoder()

        assert encoder.d_model == 256
        assert encoder.n_heads == 8
        assert encoder.n_layers_spatial == 2
        assert encoder.n_layers_temporal == 2

    def test_custom_initialization(self):
        """Test encoder initializes with custom parameters."""
        encoder = SpatioTemporalTransectEncoder(
            d_model=128,
            n_heads=4,
            n_layers_spatial=3,
            n_layers_temporal=2,
            dropout=0.2,
            max_timesteps=10,
            n_point_features=5,
            n_metadata_features=7,
        )

        assert encoder.d_model == 128
        assert encoder.n_heads == 4
        assert encoder.n_layers_spatial == 3
        assert encoder.n_layers_temporal == 2

    def test_embedders_created(self):
        """Test that linear embedders are created with correct dimensions."""
        encoder = SpatioTemporalTransectEncoder(
            d_model=128,
            n_point_features=12,
            n_metadata_features=12,
        )

        assert encoder.point_embed.in_features == 12
        assert encoder.point_embed.out_features == 128
        assert encoder.metadata_embed.in_features == 12
        assert encoder.metadata_embed.out_features == 128

    def test_cls_token_shape(self):
        """Test CLS token has correct shape."""
        encoder = SpatioTemporalTransectEncoder(d_model=256)
        assert encoder.cls_token.shape == (1, 1, 256)


class TestDistancePositionalEncoding:
    """Test distance-based sinusoidal positional encoding."""

    def test_encoding_shape(self):
        """Test positional encoding returns correct shape."""
        encoder = SpatioTemporalTransectEncoder(d_model=256)

        B, T, N = 4, 5, 128
        distances = torch.randn(B, T, N).abs() * 50  # 0-50m range

        pe = encoder.distance_positional_encoding(distances, d_model=256)

        assert pe.shape == (B, T, N, 256)

    def test_encoding_deterministic(self):
        """Test positional encoding is deterministic for same distances."""
        encoder = SpatioTemporalTransectEncoder(d_model=256)

        distances = torch.tensor([[[0.0, 10.0, 20.0, 30.0]]])  # (1, 1, 4)

        pe1 = encoder.distance_positional_encoding(distances, d_model=256)
        pe2 = encoder.distance_positional_encoding(distances, d_model=256)

        assert torch.allclose(pe1, pe2)

    def test_encoding_different_for_different_distances(self):
        """Test that different distances produce different encodings."""
        encoder = SpatioTemporalTransectEncoder(d_model=256)

        distances1 = torch.tensor([[[0.0, 10.0, 20.0]]])
        distances2 = torch.tensor([[[5.0, 15.0, 25.0]]])

        pe1 = encoder.distance_positional_encoding(distances1, d_model=256)
        pe2 = encoder.distance_positional_encoding(distances2, d_model=256)

        assert not torch.allclose(pe1, pe2)

    def test_encoding_no_nans(self):
        """Test positional encoding produces no NaN values."""
        encoder = SpatioTemporalTransectEncoder(d_model=256)

        B, T, N = 2, 3, 128
        distances = torch.randn(B, T, N).abs() * 50

        pe = encoder.distance_positional_encoding(distances, d_model=256)

        assert not torch.isnan(pe).any()


class TestForwardPass:
    """Test forward pass with various input configurations."""

    @pytest.fixture
    def encoder(self):
        """Create small encoder for testing."""
        return SpatioTemporalTransectEncoder(
            d_model=64,
            n_heads=4,
            n_layers_spatial=1,
            n_layers_temporal=1,
            dropout=0.0,
            max_timesteps=10,
        )

    @pytest.fixture
    def sample_batch(self):
        """Create sample batch data."""
        B, T, N = 2, 3, 128
        point_features = torch.randn(B, T, N, 12)
        metadata = torch.randn(B, T, 12)
        distances = torch.linspace(0, 50, N).unsqueeze(0).unsqueeze(0).expand(B, T, N)

        return {
            'point_features': point_features,
            'metadata': metadata,
            'distances': distances,
        }

    def test_forward_output_shapes(self, encoder, sample_batch):
        """Test forward pass returns correct output shapes."""
        outputs = encoder(**sample_batch)

        B, T, N = 2, 3, 128
        d_model = 64

        assert 'embeddings' in outputs
        assert 'temporal_embeddings' in outputs
        assert 'pooled' in outputs

        assert outputs['embeddings'].shape == (B, T, N, d_model)
        assert outputs['temporal_embeddings'].shape == (B, T, d_model)
        assert outputs['pooled'].shape == (B, d_model)

    def test_forward_no_nans(self, encoder, sample_batch):
        """Test forward pass produces no NaN values."""
        outputs = encoder(**sample_batch)

        assert not torch.isnan(outputs['embeddings']).any()
        assert not torch.isnan(outputs['temporal_embeddings']).any()
        assert not torch.isnan(outputs['pooled']).any()

    def test_forward_with_timestamps(self, encoder, sample_batch):
        """Test forward pass with explicit timestamps."""
        B, T = 2, 3
        timestamps = torch.tensor([[0, 1, 2], [0, 2, 4]])  # Non-sequential

        outputs = encoder(**sample_batch, timestamps=timestamps)

        assert outputs['pooled'].shape == (B, 64)

    def test_forward_with_attention_flags(self, encoder, sample_batch):
        """Test forward pass with attention return flags."""
        outputs = encoder(
            **sample_batch,
            return_spatial_attention=True,
            return_temporal_attention=True,
        )

        assert 'spatial_attention' in outputs
        assert 'temporal_attention' in outputs
        # Note: Currently returns None (placeholder)
        assert outputs['spatial_attention'] is None
        assert outputs['temporal_attention'] is None

    def test_forward_single_timestep(self, encoder):
        """Test forward pass with single timestep (T=1)."""
        B, T, N = 2, 1, 128
        point_features = torch.randn(B, T, N, 12)
        metadata = torch.randn(B, T, 12)
        distances = torch.linspace(0, 50, N).unsqueeze(0).unsqueeze(0).expand(B, T, N)

        outputs = encoder(point_features, metadata, distances)

        assert outputs['embeddings'].shape == (B, T, N, 64)
        assert outputs['temporal_embeddings'].shape == (B, T, 64)
        assert outputs['pooled'].shape == (B, 64)

    def test_forward_many_timesteps(self, encoder):
        """Test forward pass with many timesteps."""
        B, T, N = 2, 10, 128
        point_features = torch.randn(B, T, N, 12)
        metadata = torch.randn(B, T, 12)
        distances = torch.linspace(0, 50, N).unsqueeze(0).unsqueeze(0).expand(B, T, N)

        outputs = encoder(point_features, metadata, distances)

        assert outputs['embeddings'].shape == (B, T, N, 64)
        assert outputs['temporal_embeddings'].shape == (B, T, 64)
        assert outputs['pooled'].shape == (B, 64)

    def test_forward_batch_size_one(self, encoder):
        """Test forward pass with batch size 1."""
        B, T, N = 1, 3, 128
        point_features = torch.randn(B, T, N, 12)
        metadata = torch.randn(B, T, 12)
        distances = torch.linspace(0, 50, N).unsqueeze(0).unsqueeze(0).expand(B, T, N)

        outputs = encoder(point_features, metadata, distances)

        assert outputs['pooled'].shape == (1, 64)


class TestSpatialAttention:
    """Test spatial attention mechanism within timesteps."""

    @pytest.fixture
    def encoder(self):
        """Create encoder with multiple spatial layers."""
        return SpatioTemporalTransectEncoder(
            d_model=64,
            n_heads=4,
            n_layers_spatial=2,
            n_layers_temporal=1,
            dropout=0.0,
        )

    def test_spatial_attention_processes_all_timesteps(self, encoder):
        """Test spatial attention is applied to all timesteps."""
        B, T, N = 2, 5, 128
        point_features = torch.randn(B, T, N, 12)
        metadata = torch.randn(B, T, 12)
        distances = torch.linspace(0, 50, N).unsqueeze(0).unsqueeze(0).expand(B, T, N)

        outputs = encoder(point_features, metadata, distances)

        # Embeddings should have full spatial resolution
        assert outputs['embeddings'].shape == (B, T, N, 64)

    def test_spatial_attention_aggregates_to_temporal(self, encoder):
        """Test spatial features are aggregated for temporal attention."""
        B, T, N = 2, 5, 128
        point_features = torch.randn(B, T, N, 12)
        metadata = torch.randn(B, T, 12)
        distances = torch.linspace(0, 50, N).unsqueeze(0).unsqueeze(0).expand(B, T, N)

        outputs = encoder(point_features, metadata, distances)

        # Temporal embeddings should be pooled over spatial dimension
        assert outputs['temporal_embeddings'].shape == (B, T, 64)


class TestTemporalAttention:
    """Test temporal attention mechanism across timesteps."""

    @pytest.fixture
    def encoder(self):
        """Create encoder with multiple temporal layers."""
        return SpatioTemporalTransectEncoder(
            d_model=64,
            n_heads=4,
            n_layers_spatial=1,
            n_layers_temporal=2,
            dropout=0.0,
        )

    def test_temporal_attention_with_cls_token(self, encoder):
        """Test CLS token is used for pooled representation."""
        B, T, N = 2, 5, 128
        point_features = torch.randn(B, T, N, 12)
        metadata = torch.randn(B, T, 12)
        distances = torch.linspace(0, 50, N).unsqueeze(0).unsqueeze(0).expand(B, T, N)

        outputs = encoder(point_features, metadata, distances)

        # Pooled representation should come from CLS token
        assert outputs['pooled'].shape == (B, 64)

    def test_temporal_pos_encoding_applied(self, encoder):
        """Test temporal positional encoding is applied."""
        B, T, N = 2, 5, 128
        point_features = torch.randn(B, T, N, 12)
        metadata = torch.randn(B, T, 12)
        distances = torch.linspace(0, 50, N).unsqueeze(0).unsqueeze(0).expand(B, T, N)

        # Run without timestamps (uses sequential)
        outputs1 = encoder(point_features, metadata, distances)

        # Run with explicit timestamps (same sequential)
        timestamps = torch.arange(T).unsqueeze(0).expand(B, T)
        outputs2 = encoder(point_features, metadata, distances, timestamps=timestamps)

        # Should produce same results
        assert torch.allclose(outputs1['pooled'], outputs2['pooled'], atol=1e-6)


class TestMetadataEmbedding:
    """Test metadata embedding and broadcasting."""

    @pytest.fixture
    def encoder(self):
        """Create encoder for metadata testing."""
        return SpatioTemporalTransectEncoder(
            d_model=64,
            n_heads=4,
            n_layers_spatial=1,
            n_layers_temporal=1,
            dropout=0.0,
        )

    def test_metadata_broadcast_to_all_points(self, encoder):
        """Test metadata is broadcast to all spatial points."""
        B, T, N = 2, 3, 128

        # Create constant metadata for easy verification
        point_features = torch.randn(B, T, N, 12)
        metadata = torch.ones(B, T, 12)
        distances = torch.linspace(0, 50, N).unsqueeze(0).unsqueeze(0).expand(B, T, N)

        outputs = encoder(point_features, metadata, distances)

        # All outputs should be valid (no NaNs from broadcasting issues)
        assert not torch.isnan(outputs['embeddings']).any()

    def test_different_metadata_produces_different_embeddings(self, encoder):
        """Test that different metadata produces different embeddings."""
        B, T, N = 2, 3, 128
        point_features = torch.randn(B, T, N, 12)
        distances = torch.linspace(0, 50, N).unsqueeze(0).unsqueeze(0).expand(B, T, N)

        metadata1 = torch.ones(B, T, 12)
        metadata2 = torch.ones(B, T, 12) * 2

        outputs1 = encoder(point_features, metadata1, distances)
        outputs2 = encoder(point_features, metadata2, distances)

        # Different metadata should produce different embeddings
        assert not torch.allclose(outputs1['pooled'], outputs2['pooled'])


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def encoder(self):
        """Create encoder for edge case testing."""
        return SpatioTemporalTransectEncoder(
            d_model=64,
            n_heads=4,
            n_layers_spatial=1,
            n_layers_temporal=1,
            dropout=0.0,
            max_timesteps=10,
        )

    def test_zero_distances(self, encoder):
        """Test encoder handles zero distances."""
        B, T, N = 2, 3, 128
        point_features = torch.randn(B, T, N, 12)
        metadata = torch.randn(B, T, 12)
        distances = torch.zeros(B, T, N)

        outputs = encoder(point_features, metadata, distances)

        assert not torch.isnan(outputs['pooled']).any()

    def test_large_distances(self, encoder):
        """Test encoder handles large distances."""
        B, T, N = 2, 3, 128
        point_features = torch.randn(B, T, N, 12)
        metadata = torch.randn(B, T, 12)
        distances = torch.ones(B, T, N) * 1000  # 1km distances

        outputs = encoder(point_features, metadata, distances)

        assert not torch.isnan(outputs['pooled']).any()

    def test_negative_distances_not_expected(self, encoder):
        """Test encoder behavior with negative distances (not physical)."""
        B, T, N = 2, 3, 128
        point_features = torch.randn(B, T, N, 12)
        metadata = torch.randn(B, T, 12)
        distances = torch.randn(B, T, N)  # Can be negative

        # Should not crash, but results may be unexpected
        outputs = encoder(point_features, metadata, distances)

        assert outputs['pooled'].shape == (B, 64)

    def test_eval_mode(self, encoder):
        """Test encoder in eval mode."""
        encoder.eval()

        B, T, N = 2, 3, 128
        point_features = torch.randn(B, T, N, 12)
        metadata = torch.randn(B, T, 12)
        distances = torch.linspace(0, 50, N).unsqueeze(0).unsqueeze(0).expand(B, T, N)

        with torch.no_grad():
            outputs = encoder(point_features, metadata, distances)

        assert outputs['pooled'].shape == (B, 64)
        assert not torch.isnan(outputs['pooled']).any()

    def test_gradient_flow(self, encoder):
        """Test gradients flow through encoder."""
        encoder.train()

        B, T, N = 2, 3, 128
        point_features = torch.randn(B, T, N, 12, requires_grad=True)
        metadata = torch.randn(B, T, 12, requires_grad=True)
        distances = torch.linspace(0, 50, N).unsqueeze(0).unsqueeze(0).expand(B, T, N)

        outputs = encoder(point_features, metadata, distances)
        loss = outputs['pooled'].sum()
        loss.backward()

        # Check gradients exist
        assert point_features.grad is not None
        assert metadata.grad is not None
        assert not torch.isnan(point_features.grad).any()
        assert not torch.isnan(metadata.grad).any()


class TestModelSize:
    """Test model scales appropriately with parameters."""

    def test_parameter_count_increases_with_model_size(self):
        """Test larger models have more parameters."""
        small = SpatioTemporalTransectEncoder(d_model=64, n_layers_spatial=1, n_layers_temporal=1)
        large = SpatioTemporalTransectEncoder(d_model=256, n_layers_spatial=2, n_layers_temporal=2)

        small_params = sum(p.numel() for p in small.parameters())
        large_params = sum(p.numel() for p in large.parameters())

        assert large_params > small_params

    def test_parameter_count_reasonable(self):
        """Test default model has reasonable parameter count."""
        encoder = SpatioTemporalTransectEncoder()

        param_count = sum(p.numel() for p in encoder.parameters())

        # Should be in millions but not too large
        assert 1_000_000 < param_count < 20_000_000
