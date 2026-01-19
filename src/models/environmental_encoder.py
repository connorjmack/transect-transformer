"""
Environmental time-series encoder for wave and atmospheric forcing data.

This module provides a shared transformer architecture for encoding environmental
time series (wave conditions, precipitation, temperature, etc.) that drive coastal
cliff erosion processes.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


class EnvironmentalEncoder(nn.Module):
    """
    Transformer encoder for environmental time series data.

    Shared architecture used for both wave and atmospheric data. Processes
    time series with learned temporal positional encoding and day-of-year
    seasonality embedding.

    Wave data example:
        - Input: (B, 360, 4) for 90 days @ 6hr intervals
        - Features: [hs, tp, dp, power]

    Atmospheric data example:
        - Input: (B, 90, 24) for 90 days @ daily intervals
        - Features: precip_mm, temp, cumulative precip, freeze-thaw cycles, etc.

    Args:
        n_features: Number of input features (4 for wave, 24 for atmospheric)
        d_model: Hidden dimension (default 256)
        n_heads: Number of attention heads (default 8)
        n_layers: Number of transformer layers (default 3)
        dropout: Dropout rate (default 0.1)
        max_timesteps: Maximum sequence length (default 512)
            Should be >= max(T_w, T_p) for your data
        use_seasonality: Whether to add day-of-year embedding (default True)
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
        max_timesteps: int = 512,
        use_seasonality: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.use_seasonality = use_seasonality

        # Feature embedder
        self.feature_embed = nn.Linear(n_features, d_model)

        # Temporal positional encoding (learned)
        self.temporal_pos_embed = nn.Embedding(max_timesteps, d_model)

        # Day-of-year seasonality embedding (optional)
        if use_seasonality:
            # 366 days to handle leap years
            self.doy_embed = nn.Embedding(367, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for training stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # Output normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        features: torch.Tensor,  # (B, T, n_features)
        day_of_year: Optional[torch.Tensor] = None,  # (B, T)
        timestamps: Optional[torch.Tensor] = None,  # (B, T) - position indices
        padding_mask: Optional[torch.Tensor] = None,  # (B, T) - True for padding
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through environmental encoder.

        Args:
            features: Time series features (B, T, n_features)
                For wave: [hs, tp, dp, power]
                For atmospheric: [precip_mm, temp, ..., 24 features total]
            day_of_year: Day-of-year for seasonality (B, T)
                Values in range [0, 366]. If None and use_seasonality=True,
                seasonality embedding is skipped.
            timestamps: Temporal position indices (B, T)
                If None, uses sequential indices [0, 1, 2, ..., T-1]
            padding_mask: Mask for padded positions (B, T)
                True for positions that should be masked (padding)
                None means no masking (all positions valid)

        Returns:
            Dictionary containing:
                - embeddings: Encoded time series (B, T, d_model)
                - pooled: Mean-pooled representation (B, d_model)
                    Pooling respects padding mask if provided
        """
        B, T, _ = features.shape

        # Embed features: (B, T, n_features) -> (B, T, d_model)
        x = self.feature_embed(features)

        # Add temporal positional encoding
        if timestamps is None:
            timestamps = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)

        temporal_pe = self.temporal_pos_embed(timestamps)  # (B, T, d_model)
        x = x + temporal_pe

        # Add day-of-year seasonality encoding
        if self.use_seasonality and day_of_year is not None:
            # Clamp to valid range [0, 366] and convert to long
            doy_clamped = torch.clamp(day_of_year, 0, 366).long()
            seasonality_embed = self.doy_embed(doy_clamped)  # (B, T, d_model)
            x = x + seasonality_embed

        # Apply transformer encoder
        # PyTorch transformer expects padding_mask as (B, T) with True for padding
        x = self.encoder(x, src_key_padding_mask=padding_mask)  # (B, T, d_model)

        # Output normalization
        x = self.norm(x)

        # Compute pooled representation (mean over time)
        if padding_mask is not None:
            # Compute masked mean (exclude padded positions)
            # Create mask: 1 for valid positions, 0 for padding
            mask = ~padding_mask  # (B, T)
            mask_expanded = mask.unsqueeze(-1).float()  # (B, T, 1)

            # Weighted sum and normalization
            pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            # Simple mean pooling
            pooled = x.mean(dim=1)  # (B, d_model)

        outputs = {
            'embeddings': x,  # (B, T, d_model)
            'pooled': pooled,  # (B, d_model)
        }

        return outputs


class WaveEncoder(EnvironmentalEncoder):
    """
    Specialized encoder for wave time series.

    Wrapper around EnvironmentalEncoder with wave-specific defaults.
    Wave data format:
        - T_w = 360 timesteps (90 days @ 6hr intervals)
        - 4 features: [hs, tp, dp, power]

    Args:
        d_model: Hidden dimension (default 256)
        n_heads: Number of attention heads (default 8)
        n_layers: Number of transformer layers (default 3)
        dropout: Dropout rate (default 0.1)
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__(
            n_features=4,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            max_timesteps=512,  # Supports up to ~128 days @ 6hr
            use_seasonality=True,
        )


class AtmosphericEncoder(EnvironmentalEncoder):
    """
    Specialized encoder for atmospheric time series.

    Wrapper around EnvironmentalEncoder with atmospheric-specific defaults.
    Atmospheric data format:
        - T_p = 90 timesteps (90 days @ daily intervals)
        - 24 features: precip, temp, cumulative precip, freeze-thaw, etc.

    Args:
        d_model: Hidden dimension (default 256)
        n_heads: Number of attention heads (default 8)
        n_layers: Number of transformer layers (default 3)
        dropout: Dropout rate (default 0.1)
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__(
            n_features=24,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            max_timesteps=512,  # Supports up to ~1.4 years daily
            use_seasonality=True,
        )
