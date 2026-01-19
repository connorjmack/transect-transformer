"""
Tests for EnvironmentalEncoder, WaveEncoder, and AtmosphericEncoder.

Tests cover:
- Initialization and parameter validation
- Temporal positional encoding
- Day-of-year seasonality embedding
- Padding mask handling
- Shape validation throughout pipeline
- Edge cases for wave and atmospheric data
"""

import pytest
import torch

from src.models.environmental_encoder import (
    AtmosphericEncoder,
    EnvironmentalEncoder,
    WaveEncoder,
)


class TestEnvironmentalEncoderInitialization:
    """Test EnvironmentalEncoder initialization."""

    def test_default_initialization(self):
        """Test encoder initializes with default parameters."""
        encoder = EnvironmentalEncoder(n_features=4)

        assert encoder.d_model == 256
        assert encoder.n_heads == 8
        assert encoder.n_layers == 3
        assert encoder.use_seasonality is True

    def test_custom_initialization(self):
        """Test encoder initializes with custom parameters."""
        encoder = EnvironmentalEncoder(
            n_features=24,
            d_model=128,
            n_heads=4,
            n_layers=2,
            dropout=0.2,
            max_timesteps=256,
            use_seasonality=False,
        )

        assert encoder.d_model == 128
        assert encoder.n_heads == 4
        assert encoder.n_layers == 2
        assert encoder.use_seasonality is False

    def test_embedder_dimensions(self):
        """Test feature embedder has correct dimensions."""
        encoder = EnvironmentalEncoder(n_features=4, d_model=128)

        assert encoder.feature_embed.in_features == 4
        assert encoder.feature_embed.out_features == 128

    def test_seasonality_embedding_created_when_enabled(self):
        """Test day-of-year embedding is created when seasonality is enabled."""
        encoder = EnvironmentalEncoder(n_features=4, use_seasonality=True)

        assert hasattr(encoder, 'doy_embed')
        assert encoder.doy_embed.num_embeddings == 367  # 366 days + 0

    def test_seasonality_embedding_not_created_when_disabled(self):
        """Test day-of-year embedding is not created when seasonality is disabled."""
        encoder = EnvironmentalEncoder(n_features=4, use_seasonality=False)

        assert not hasattr(encoder, 'doy_embed')


class TestWaveEncoderInitialization:
    """Test WaveEncoder initialization and defaults."""

    def test_wave_encoder_defaults(self):
        """Test WaveEncoder initializes with wave-specific defaults."""
        encoder = WaveEncoder()

        assert encoder.d_model == 256
        assert encoder.n_heads == 8
        assert encoder.n_layers == 3
        assert encoder.use_seasonality is True

        # Wave-specific: 4 features
        assert encoder.feature_embed.in_features == 4

    def test_wave_encoder_custom_params(self):
        """Test WaveEncoder with custom parameters."""
        encoder = WaveEncoder(d_model=128, n_heads=4, n_layers=2, dropout=0.2)

        assert encoder.d_model == 128
        assert encoder.n_heads == 4
        assert encoder.n_layers == 2


class TestAtmosphericEncoderInitialization:
    """Test AtmosphericEncoder initialization and defaults."""

    def test_atmospheric_encoder_defaults(self):
        """Test AtmosphericEncoder initializes with atmospheric-specific defaults."""
        encoder = AtmosphericEncoder()

        assert encoder.d_model == 256
        assert encoder.n_heads == 8
        assert encoder.n_layers == 3
        assert encoder.use_seasonality is True

        # Atmospheric-specific: 24 features
        assert encoder.feature_embed.in_features == 24

    def test_atmospheric_encoder_custom_params(self):
        """Test AtmosphericEncoder with custom parameters."""
        encoder = AtmosphericEncoder(d_model=128, n_heads=4, n_layers=2, dropout=0.2)

        assert encoder.d_model == 128
        assert encoder.n_heads == 4
        assert encoder.n_layers == 2


class TestForwardPass:
    """Test forward pass with various configurations."""

    @pytest.fixture
    def encoder(self):
        """Create small encoder for testing."""
        return EnvironmentalEncoder(
            n_features=4,
            d_model=64,
            n_heads=4,
            n_layers=2,
            dropout=0.0,
            use_seasonality=True,
        )

    @pytest.fixture
    def sample_batch(self):
        """Create sample batch data."""
        B, T = 4, 360  # 4 transects, 360 timesteps (90 days @ 6hr)
        features = torch.randn(B, T, 4)
        day_of_year = torch.randint(0, 366, (B, T))

        return {
            'features': features,
            'day_of_year': day_of_year,
        }

    def test_forward_output_shapes(self, encoder, sample_batch):
        """Test forward pass returns correct output shapes."""
        outputs = encoder(**sample_batch)

        B, T = 4, 360
        d_model = 64

        assert 'embeddings' in outputs
        assert 'pooled' in outputs

        assert outputs['embeddings'].shape == (B, T, d_model)
        assert outputs['pooled'].shape == (B, d_model)

    def test_forward_no_nans(self, encoder, sample_batch):
        """Test forward pass produces no NaN values."""
        outputs = encoder(**sample_batch)

        assert not torch.isnan(outputs['embeddings']).any()
        assert not torch.isnan(outputs['pooled']).any()

    def test_forward_without_seasonality(self, sample_batch):
        """Test forward pass without seasonality embedding."""
        encoder = EnvironmentalEncoder(
            n_features=4,
            d_model=64,
            n_heads=4,
            n_layers=2,
            use_seasonality=False,
        )

        # Don't pass day_of_year
        outputs = encoder(features=sample_batch['features'])

        assert outputs['embeddings'].shape == (4, 360, 64)
        assert outputs['pooled'].shape == (4, 64)

    def test_forward_with_seasonality_but_no_doy(self, encoder, sample_batch):
        """Test forward pass with seasonality enabled but no DOY provided."""
        # Should skip seasonality embedding gracefully
        outputs = encoder(features=sample_batch['features'])

        assert outputs['embeddings'].shape == (4, 360, 64)
        assert outputs['pooled'].shape == (4, 64)

    def test_forward_with_timestamps(self, encoder, sample_batch):
        """Test forward pass with explicit timestamps."""
        B, T = 4, 360
        timestamps = torch.arange(T).unsqueeze(0).expand(B, T)

        outputs = encoder(**sample_batch, timestamps=timestamps)

        assert outputs['pooled'].shape == (B, 64)

    def test_forward_with_padding_mask(self, encoder):
        """Test forward pass with padding mask."""
        B, T = 4, 360
        features = torch.randn(B, T, 4)
        day_of_year = torch.randint(0, 366, (B, T))

        # Create padding mask (last 100 timesteps are padding for some samples)
        padding_mask = torch.zeros(B, T, dtype=torch.bool)
        padding_mask[0, 260:] = True  # Sample 0 has padding
        padding_mask[1, 300:] = True  # Sample 1 has padding

        outputs = encoder(features, day_of_year, padding_mask=padding_mask)

        assert outputs['embeddings'].shape == (B, T, 64)
        assert outputs['pooled'].shape == (B, 64)
        assert not torch.isnan(outputs['pooled']).any()

    def test_forward_single_timestep(self, encoder):
        """Test forward pass with single timestep."""
        B, T = 4, 1
        features = torch.randn(B, T, 4)
        day_of_year = torch.randint(0, 366, (B, T))

        outputs = encoder(features, day_of_year)

        assert outputs['embeddings'].shape == (B, T, 64)
        assert outputs['pooled'].shape == (B, 64)

    def test_forward_batch_size_one(self, encoder):
        """Test forward pass with batch size 1."""
        B, T = 1, 360
        features = torch.randn(B, T, 4)
        day_of_year = torch.randint(0, 366, (B, T))

        outputs = encoder(features, day_of_year)

        assert outputs['pooled'].shape == (1, 64)


class TestWaveEncoder:
    """Test WaveEncoder with realistic wave data."""

    @pytest.fixture
    def encoder(self):
        """Create wave encoder."""
        return WaveEncoder(d_model=64, n_heads=4, n_layers=2)

    def test_wave_data_shape(self, encoder):
        """Test wave encoder with realistic wave data dimensions."""
        B = 4
        T_w = 360  # 90 days @ 6hr intervals

        # Wave features: [hs, tp, dp, power]
        wave_features = torch.randn(B, T_w, 4).abs()  # All positive
        day_of_year = torch.randint(0, 366, (B, T_w))

        outputs = encoder(wave_features, day_of_year)

        assert outputs['embeddings'].shape == (B, T_w, 64)
        assert outputs['pooled'].shape == (B, 64)

    def test_wave_data_no_nans(self, encoder):
        """Test wave encoder produces no NaNs with realistic data."""
        B, T_w = 4, 360

        # Simulate realistic wave data
        hs = torch.rand(B, T_w) * 5  # 0-5m significant wave height
        tp = torch.rand(B, T_w) * 15 + 5  # 5-20s peak period
        dp = torch.rand(B, T_w) * 360  # 0-360 degrees
        power = torch.rand(B, T_w) * 100  # 0-100 kW/m

        wave_features = torch.stack([hs, tp, dp, power], dim=-1)
        day_of_year = torch.randint(0, 366, (B, T_w))

        outputs = encoder(wave_features, day_of_year)

        assert not torch.isnan(outputs['pooled']).any()


class TestAtmosphericEncoder:
    """Test AtmosphericEncoder with realistic atmospheric data."""

    @pytest.fixture
    def encoder(self):
        """Create atmospheric encoder."""
        return AtmosphericEncoder(d_model=64, n_heads=4, n_layers=2)

    def test_atmospheric_data_shape(self, encoder):
        """Test atmospheric encoder with realistic data dimensions."""
        B = 4
        T_p = 90  # 90 days @ daily intervals

        # Atmospheric features: 24 features
        atmos_features = torch.randn(B, T_p, 24)
        day_of_year = torch.randint(0, 366, (B, T_p))

        outputs = encoder(atmos_features, day_of_year)

        assert outputs['embeddings'].shape == (B, T_p, 64)
        assert outputs['pooled'].shape == (B, 64)

    def test_atmospheric_data_no_nans(self, encoder):
        """Test atmospheric encoder produces no NaNs with realistic data."""
        B, T_p = 4, 90

        # Simulate realistic atmospheric data (24 features)
        atmos_features = torch.randn(B, T_p, 24)
        day_of_year = torch.randint(0, 366, (B, T_p))

        outputs = encoder(atmos_features, day_of_year)

        assert not torch.isnan(outputs['pooled']).any()


class TestSeasonalityEmbedding:
    """Test day-of-year seasonality embedding."""

    @pytest.fixture
    def encoder(self):
        """Create encoder with seasonality."""
        return EnvironmentalEncoder(
            n_features=4,
            d_model=64,
            n_heads=4,
            n_layers=1,
            dropout=0.0,  # No dropout for deterministic tests
            use_seasonality=True,
        )

    def test_doy_embedding_deterministic(self, encoder):
        """Test same day-of-year produces same embedding."""
        B, T = 2, 10
        features = torch.randn(B, T, 4)
        doy = torch.ones(B, T, dtype=torch.long) * 100  # All day 100

        outputs1 = encoder(features, doy)
        outputs2 = encoder(features, doy)

        assert torch.allclose(outputs1['pooled'], outputs2['pooled'])

    def test_different_doy_produces_different_embeddings(self, encoder):
        """Test different day-of-year produces different embeddings."""
        B, T = 2, 10
        features = torch.randn(B, T, 4)

        doy1 = torch.ones(B, T, dtype=torch.long) * 1  # Day 1
        doy2 = torch.ones(B, T, dtype=torch.long) * 200  # Day 200

        outputs1 = encoder(features, doy1)
        outputs2 = encoder(features, doy2)

        assert not torch.allclose(outputs1['pooled'], outputs2['pooled'])

    def test_doy_clamping(self, encoder):
        """Test day-of-year values are clamped to valid range."""
        B, T = 2, 10
        features = torch.randn(B, T, 4)

        # Test out-of-range values (should be clamped to [0, 366])
        doy_invalid = torch.tensor([[400, -10, 500, 0, 366, 367, 100, 50, 200, 300]])
        doy_invalid = doy_invalid.expand(B, T)

        # Should not crash
        outputs = encoder(features, doy_invalid)

        assert not torch.isnan(outputs['pooled']).any()


class TestPaddingMask:
    """Test padding mask functionality."""

    @pytest.fixture
    def encoder(self):
        """Create encoder for padding mask testing."""
        return EnvironmentalEncoder(
            n_features=4,
            d_model=64,
            n_heads=4,
            n_layers=2,
            dropout=0.0,  # No dropout for deterministic tests
        )

    def test_padding_mask_basic(self, encoder):
        """Test basic padding mask functionality."""
        B, T = 4, 100
        features = torch.randn(B, T, 4)

        # Last 20 timesteps are padding
        padding_mask = torch.zeros(B, T, dtype=torch.bool)
        padding_mask[:, 80:] = True

        outputs = encoder(features, padding_mask=padding_mask)

        assert outputs['pooled'].shape == (B, 64)
        assert not torch.isnan(outputs['pooled']).any()

    def test_padding_mask_per_sample_different(self, encoder):
        """Test padding mask with different lengths per sample."""
        B, T = 4, 100
        features = torch.randn(B, T, 4)

        # Different padding for each sample
        padding_mask = torch.zeros(B, T, dtype=torch.bool)
        padding_mask[0, 90:] = True  # Sample 0: 90 valid timesteps
        padding_mask[1, 80:] = True  # Sample 1: 80 valid timesteps
        padding_mask[2, 70:] = True  # Sample 2: 70 valid timesteps
        padding_mask[3, 60:] = True  # Sample 3: 60 valid timesteps

        outputs = encoder(features, padding_mask=padding_mask)

        assert outputs['pooled'].shape == (B, 64)
        assert not torch.isnan(outputs['pooled']).any()

    def test_padding_mask_all_padded(self, encoder):
        """Test padding mask with all timesteps padded (edge case)."""
        B, T = 4, 100
        features = torch.randn(B, T, 4)

        # All timesteps are padding (not realistic, but should not crash)
        padding_mask = torch.ones(B, T, dtype=torch.bool)

        outputs = encoder(features, padding_mask=padding_mask)

        # Should still produce output (will be from padding, but shouldn't crash)
        assert outputs['pooled'].shape == (B, 64)

    def test_padding_mask_none_vs_all_false(self, encoder):
        """Test padding_mask=None is equivalent to all False."""
        B, T = 4, 100
        features = torch.randn(B, T, 4)

        # No mask
        outputs1 = encoder(features, padding_mask=None)

        # All False mask (no padding)
        padding_mask = torch.zeros(B, T, dtype=torch.bool)
        outputs2 = encoder(features, padding_mask=padding_mask)

        # Should produce same results
        assert torch.allclose(outputs1['pooled'], outputs2['pooled'], atol=1e-6)


class TestTemporalPositionalEncoding:
    """Test temporal positional encoding."""

    @pytest.fixture
    def encoder(self):
        """Create encoder for positional encoding testing."""
        return EnvironmentalEncoder(
            n_features=4,
            d_model=64,
            n_heads=4,
            n_layers=1,
        )

    def test_sequential_timestamps(self, encoder):
        """Test encoder with sequential timestamps."""
        B, T = 2, 50
        features = torch.randn(B, T, 4)

        # Sequential timestamps
        timestamps = torch.arange(T).unsqueeze(0).expand(B, T)

        outputs = encoder(features, timestamps=timestamps)

        assert outputs['pooled'].shape == (B, 64)

    def test_non_sequential_timestamps(self, encoder):
        """Test encoder with non-sequential timestamps."""
        B, T = 2, 50
        features = torch.randn(B, T, 4)

        # Non-sequential timestamps (e.g., sparse sampling)
        timestamps = torch.arange(0, T * 2, 2).unsqueeze(0).expand(B, T)

        outputs = encoder(features, timestamps=timestamps)

        assert outputs['pooled'].shape == (B, 64)


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def encoder(self):
        """Create encoder for edge case testing."""
        return EnvironmentalEncoder(
            n_features=4,
            d_model=64,
            n_heads=4,
            n_layers=2,
        )

    def test_zero_features(self, encoder):
        """Test encoder with all-zero features."""
        B, T = 4, 100
        features = torch.zeros(B, T, 4)

        outputs = encoder(features)

        assert not torch.isnan(outputs['pooled']).any()

    def test_large_feature_values(self, encoder):
        """Test encoder with large feature values."""
        B, T = 4, 100
        features = torch.randn(B, T, 4) * 1000

        outputs = encoder(features)

        assert not torch.isnan(outputs['pooled']).any()

    def test_eval_mode(self, encoder):
        """Test encoder in eval mode."""
        encoder.eval()

        B, T = 4, 100
        features = torch.randn(B, T, 4)

        with torch.no_grad():
            outputs = encoder(features)

        assert outputs['pooled'].shape == (B, 64)
        assert not torch.isnan(outputs['pooled']).any()

    def test_gradient_flow(self, encoder):
        """Test gradients flow through encoder."""
        encoder.train()

        B, T = 4, 100
        features = torch.randn(B, T, 4, requires_grad=True)

        outputs = encoder(features)
        loss = outputs['pooled'].sum()
        loss.backward()

        assert features.grad is not None
        assert not torch.isnan(features.grad).any()

    def test_very_long_sequence(self, encoder):
        """Test encoder with very long sequence."""
        B, T = 2, 500
        features = torch.randn(B, T, 4)

        outputs = encoder(features)

        assert outputs['embeddings'].shape == (B, T, 64)
        assert outputs['pooled'].shape == (B, 64)


class TestModelSize:
    """Test model scales appropriately with parameters."""

    def test_parameter_count_increases_with_model_size(self):
        """Test larger models have more parameters."""
        small = EnvironmentalEncoder(n_features=4, d_model=64, n_layers=1)
        large = EnvironmentalEncoder(n_features=4, d_model=256, n_layers=3)

        small_params = sum(p.numel() for p in small.parameters())
        large_params = sum(p.numel() for p in large.parameters())

        assert large_params > small_params

    def test_parameter_count_reasonable(self):
        """Test default encoder has reasonable parameter count."""
        encoder = EnvironmentalEncoder(n_features=4)

        param_count = sum(p.numel() for p in encoder.parameters())

        # Should be in hundreds of thousands to low millions
        assert 100_000 < param_count < 10_000_000
