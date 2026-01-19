"""
Spatio-temporal transformer encoder for multi-temporal cliff transect data.

This module implements hierarchical attention over cliff geometry:
1. Spatial attention within each LiDAR epoch (over N=128 points)
2. Temporal attention across T epochs (learning cliff evolution patterns)

The encoder uses distance-based sinusoidal positional encoding for spatial
dimensions and learned positional encoding for temporal dimensions.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat


class SpatioTemporalTransectEncoder(nn.Module):
    """
    Hierarchical attention encoder for multi-temporal cliff geometry.

    Architecture:
    1. Embed point features (12D) and metadata (12D)
    2. Apply distance-based sinusoidal positional encoding (spatial)
    3. Apply learned temporal positional encoding
    4. Spatial attention within each timestep (T independent attention ops)
    5. Temporal attention across timesteps
    6. Return fused embeddings + pooled representation via CLS token

    Args:
        d_model: Hidden dimension (default 256)
        n_heads: Number of attention heads (default 8)
        n_layers_spatial: Number of spatial transformer layers (default 2)
        n_layers_temporal: Number of temporal transformer layers (default 2)
        dropout: Dropout rate (default 0.1)
        max_timesteps: Maximum number of LiDAR epochs (default 20)
        n_point_features: Number of point-level features (default 12)
        n_metadata_features: Number of transect-level metadata fields (default 12)
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers_spatial: int = 2,
        n_layers_temporal: int = 2,
        dropout: float = 0.1,
        max_timesteps: int = 20,
        n_point_features: int = 12,
        n_metadata_features: int = 12,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers_spatial = n_layers_spatial
        self.n_layers_temporal = n_layers_temporal

        # Feature embedders
        self.point_embed = nn.Linear(n_point_features, d_model)
        self.metadata_embed = nn.Linear(n_metadata_features, d_model)

        # Temporal positional encoding (learned)
        self.temporal_pos_embed = nn.Embedding(max_timesteps, d_model)

        # Spatial transformer (within each timestep)
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for training stability
        )
        self.spatial_encoder = nn.TransformerEncoder(
            spatial_layer,
            num_layers=n_layers_spatial
        )

        # Temporal transformer (across timesteps)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for training stability
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=n_layers_temporal
        )

        # CLS token for pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Layer norms
        self.norm = nn.LayerNorm(d_model)

    def distance_positional_encoding(
        self,
        distances: torch.Tensor,  # (B, T, N)
        d_model: int
    ) -> torch.Tensor:
        """
        Sinusoidal positional encoding based on distance from cliff toe.

        Uses distance in meters rather than sequential indices, allowing the
        model to understand actual spatial relationships along the cliff profile.

        Args:
            distances: Distance values in meters (B, T, N)
            d_model: Embedding dimension

        Returns:
            Positional encodings (B, T, N, d_model)
        """
        B, T, N = distances.shape

        # Normalize distances to reasonable range for encoding
        # Typical transects are 0-50m from toe
        normalized_dist = distances / 50.0  # (B, T, N)

        # Create encoding
        pe = torch.zeros(B, T, N, d_model, device=distances.device)

        # Frequency bands
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=distances.device).float() *
            (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices, cos to odd indices
        pe[..., 0::2] = torch.sin(
            normalized_dist.unsqueeze(-1) * div_term
        )
        pe[..., 1::2] = torch.cos(
            normalized_dist.unsqueeze(-1) * div_term
        )

        return pe

    def forward(
        self,
        point_features: torch.Tensor,  # (B, T, N, n_point_features)
        metadata: torch.Tensor,  # (B, T, n_metadata_features)
        distances: torch.Tensor,  # (B, T, N)
        timestamps: Optional[torch.Tensor] = None,  # (B, T) - epoch indices
        return_spatial_attention: bool = False,
        return_temporal_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through spatio-temporal encoder.

        Args:
            point_features: Point-level features (B, T, N, n_point_features)
                Contains 12 features per point: distance_m, elevation_m, slope_deg,
                curvature, roughness, intensity, red, green, blue, classification,
                return_number, num_returns
            metadata: Transect-level metadata (B, T, n_metadata_features)
                Contains 12 fields: cliff_height_m, mean_slope_deg, max_slope_deg,
                toe_elevation_m, top_elevation_m, orientation_deg, transect_length_m,
                latitude, longitude, transect_id, mean_intensity, dominant_class
            distances: Distance from cliff toe in meters (B, T, N)
            timestamps: Temporal indices for positional encoding (B, T)
                If None, uses sequential indices [0, 1, 2, ..., T-1]
            return_spatial_attention: Whether to return spatial attention weights
            return_temporal_attention: Whether to return temporal attention weights

        Returns:
            Dictionary containing:
                - embeddings: Spatio-temporal embeddings (B, T, N, d_model)
                - temporal_embeddings: Pooled temporal embeddings (B, T, d_model)
                - pooled: Global pooled representation via CLS token (B, d_model)
                - spatial_attention: (optional) Attention weights (placeholder)
                - temporal_attention: (optional) Attention weights (placeholder)
        """
        B, T, N, _ = point_features.shape

        # Embed point features: (B, T, N, n_point_features) -> (B, T, N, d_model)
        x = self.point_embed(point_features)

        # Embed metadata and broadcast: (B, T, n_metadata_features) -> (B, T, N, d_model)
        meta_embed = self.metadata_embed(metadata).unsqueeze(2)  # (B, T, 1, d_model)
        meta_embed = repeat(meta_embed, 'b t 1 d -> b t n d', n=N)

        # Add metadata to point embeddings
        x = x + meta_embed

        # Add distance-based spatial positional encoding
        spatial_pe = self.distance_positional_encoding(distances, self.d_model)
        x = x + spatial_pe

        # Spatial attention (within each timestep)
        # Reshape to process all timesteps independently: (B*T, N, d_model)
        x_spatial = rearrange(x, 'b t n d -> (b t) n d')
        x_spatial = self.spatial_encoder(x_spatial)
        x_spatial = rearrange(x_spatial, '(b t) n d -> b t n d', b=B, t=T)

        # Pool over spatial dimension for temporal attention
        # Use mean pooling (attention pooling can be added later)
        x_temporal = x_spatial.mean(dim=2)  # (B, T, d_model)

        # Add temporal positional encoding
        if timestamps is None:
            timestamps = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)

        temporal_pe = self.temporal_pos_embed(timestamps)  # (B, T, d_model)
        x_temporal = x_temporal + temporal_pe

        # Prepend CLS token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x_temporal = torch.cat([cls_tokens, x_temporal], dim=1)  # (B, T+1, d_model)

        # Temporal attention (across timesteps)
        x_temporal = self.temporal_encoder(x_temporal)  # (B, T+1, d_model)

        # Extract CLS token as pooled representation
        pooled = x_temporal[:, 0]  # (B, d_model)
        pooled = self.norm(pooled)

        # Extract temporal embeddings (without CLS)
        temporal_embeddings = x_temporal[:, 1:]  # (B, T, d_model)

        outputs = {
            'embeddings': x_spatial,  # (B, T, N, d_model)
            'temporal_embeddings': temporal_embeddings,  # (B, T, d_model)
            'pooled': pooled,  # (B, d_model)
        }

        # TODO: Extract attention weights if requested
        # This requires custom transformer layers with attention weight return
        # or using hooks to capture intermediate attention maps
        if return_spatial_attention:
            outputs['spatial_attention'] = None  # Placeholder
        if return_temporal_attention:
            outputs['temporal_attention'] = None  # Placeholder

        return outputs
