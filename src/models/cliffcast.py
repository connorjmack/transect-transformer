"""
CliffCast: Transformer-based model for coastal cliff erosion prediction.

This module assembles all model components:
- SpatioTemporalTransectEncoder: Processes multi-temporal cliff geometry
- EnvironmentalEncoders: Process wave and atmospheric forcing data
- CrossAttentionFusion: Fuses cliff and environmental representations
- PredictionHeads: Multi-task predictions (risk, retreat, collapse, failure mode)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from src.models.environmental_encoder import EnvironmentalEncoder
from src.models.fusion import CrossAttentionFusion
from src.models.prediction_heads import PredictionHeads
from src.models.transect_encoder import SpatioTemporalTransectEncoder


class CliffCast(nn.Module):
    """
    CliffCast: Full model for cliff erosion prediction.

    Architecture:
        1. Encode cliff geometry (spatio-temporal attention)
        2. Encode wave forcing (temporal attention)
        3. Encode atmospheric forcing (temporal attention)
        4. Fuse cliff and environmental embeddings (cross-attention)
        5. Predict multiple targets via task-specific heads

    Args:
        # Shared
        d_model: Hidden dimension (default 256)
        n_heads: Number of attention heads (default 8)
        dropout: Dropout rate (default 0.1)

        # Transect encoder
        n_layers_spatial: Spatial transformer layers (default 2)
        n_layers_temporal: Temporal transformer layers (default 2)
        max_timesteps: Max LiDAR epochs (default 20)
        n_point_features: Point-level features (default 12)
        n_metadata_features: Transect metadata features (default 12)

        # Environmental encoders
        n_layers_env: Environmental transformer layers (default 3)
        n_wave_features: Wave features (default 4)
        n_atmos_features: Atmospheric features (default 24)

        # Fusion
        n_layers_fusion: Cross-attention layers (default 2)

        # Prediction heads
        enable_risk: Enable risk index head (default True)
        enable_retreat: Enable retreat head (default True)
        enable_collapse: Enable collapse probability head (default True)
        enable_failure_mode: Enable failure mode head (default True)
        n_collapse_horizons: Collapse time horizons (default 4)
        n_failure_modes: Failure mode classes (default 5)
    """

    def __init__(
        self,
        # Shared parameters
        d_model: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
        # Transect encoder
        n_layers_spatial: int = 2,
        n_layers_temporal: int = 2,
        max_timesteps: int = 20,
        n_point_features: int = 12,
        n_metadata_features: int = 12,
        # Environmental encoders
        n_layers_env: int = 3,
        n_wave_features: int = 4,
        n_atmos_features: int = 24,
        # Fusion
        n_layers_fusion: int = 2,
        # Prediction heads
        enable_risk: bool = True,
        enable_retreat: bool = True,
        enable_collapse: bool = True,
        enable_failure_mode: bool = True,
        n_collapse_horizons: int = 4,
        n_failure_modes: int = 5,
    ):
        super().__init__()
        self.d_model = d_model

        # 1. Transect encoder (spatio-temporal)
        self.transect_encoder = SpatioTemporalTransectEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers_spatial=n_layers_spatial,
            n_layers_temporal=n_layers_temporal,
            dropout=dropout,
            max_timesteps=max_timesteps,
            n_point_features=n_point_features,
            n_metadata_features=n_metadata_features,
        )

        # 2. Wave encoder
        self.wave_encoder = EnvironmentalEncoder(
            n_features=n_wave_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers_env,
            dropout=dropout,
            max_timesteps=512,  # Support up to 512 timesteps
            use_seasonality=True,
        )

        # 3. Atmospheric encoder
        self.atmos_encoder = EnvironmentalEncoder(
            n_features=n_atmos_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers_env,
            dropout=dropout,
            max_timesteps=512,  # Support up to 512 timesteps
            use_seasonality=True,
        )

        # 4. Cross-attention fusion
        self.fusion = CrossAttentionFusion(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers_fusion,
            dropout=dropout,
        )

        # 5. Prediction heads
        self.heads = PredictionHeads(
            d_model=d_model,
            enable_risk=enable_risk,
            enable_retreat=enable_retreat,
            enable_collapse=enable_collapse,
            enable_failure_mode=enable_failure_mode,
            n_collapse_horizons=n_collapse_horizons,
            n_failure_modes=n_failure_modes,
            dropout=dropout,
        )

    def forward(
        self,
        # Transect inputs (required)
        point_features: torch.Tensor,  # (B, T, N, n_point_features)
        metadata: torch.Tensor,  # (B, T, n_metadata_features)
        distances: torch.Tensor,  # (B, T, N)
        # Wave inputs (required)
        wave_features: torch.Tensor,  # (B, T_w, n_wave_features)
        # Atmospheric inputs (required)
        atmos_features: torch.Tensor,  # (B, T_a, n_atmos_features)
        # Transect inputs (optional)
        timestamps: Optional[torch.Tensor] = None,  # (B, T)
        # Wave inputs (optional)
        wave_doy: Optional[torch.Tensor] = None,  # (B, T_w)
        wave_timestamps: Optional[torch.Tensor] = None,  # (B, T_w)
        wave_padding_mask: Optional[torch.Tensor] = None,  # (B, T_w)
        # Atmospheric inputs (optional)
        atmos_doy: Optional[torch.Tensor] = None,  # (B, T_a)
        atmos_timestamps: Optional[torch.Tensor] = None,  # (B, T_a)
        atmos_padding_mask: Optional[torch.Tensor] = None,  # (B, T_a)
        # Attention visualization
        return_spatial_attention: bool = False,
        return_temporal_attention: bool = False,
        return_env_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through full CliffCast model.

        Args:
            # Transect inputs (cube format)
            point_features: Point-level features (B, T, N, 12)
            metadata: Transect-level metadata (B, T, 12)
            distances: Distance from cliff toe (B, T, N)
            timestamps: Temporal indices for LiDAR epochs (B, T)

            # Wave inputs
            wave_features: Wave time series (B, T_w, 4)
                Features: [hs, tp, dp, power]
            wave_doy: Day-of-year for seasonality (B, T_w)
            wave_timestamps: Temporal indices (B, T_w)
            wave_padding_mask: Padding mask (B, T_w)

            # Atmospheric inputs
            atmos_features: Atmospheric time series (B, T_a, 24)
            atmos_doy: Day-of-year for seasonality (B, T_a)
            atmos_timestamps: Temporal indices (B, T_a)
            atmos_padding_mask: Padding mask (B, T_a)

            # Attention visualization flags
            return_spatial_attention: Return spatial attention weights
            return_temporal_attention: Return temporal attention weights
            return_env_attention: Return cross-attention weights

        Returns:
            Dictionary containing:
                # Predictions (from enabled heads)
                - risk_index: (B,) if enabled
                - retreat_m: (B,) if enabled
                - p_collapse: (B, n_horizons) if enabled
                - failure_mode_logits: (B, n_modes) if enabled

                # Attention weights (if requested)
                - spatial_attention: (B, n_heads, T, N, N) if requested
                - temporal_attention: (B, n_heads, T, T) if requested
                - env_attention: (B, n_heads, T, T_w+T_a) if requested
        """
        # 1. Encode transect (spatio-temporal attention)
        transect_outputs = self.transect_encoder(
            point_features=point_features,
            metadata=metadata,
            distances=distances,
            timestamps=timestamps,
            return_spatial_attention=return_spatial_attention,
            return_temporal_attention=return_temporal_attention,
        )
        cliff_embeddings = transect_outputs['temporal_embeddings']  # (B, T, d_model)

        # 2. Encode wave forcing
        wave_outputs = self.wave_encoder(
            features=wave_features,
            day_of_year=wave_doy,
            timestamps=wave_timestamps,
            padding_mask=wave_padding_mask,
        )
        wave_embeddings = wave_outputs['embeddings']  # (B, T_w, d_model)

        # 3. Encode atmospheric forcing
        atmos_outputs = self.atmos_encoder(
            features=atmos_features,
            day_of_year=atmos_doy,
            timestamps=atmos_timestamps,
            padding_mask=atmos_padding_mask,
        )
        atmos_embeddings = atmos_outputs['embeddings']  # (B, T_a, d_model)

        # 4. Concatenate environmental embeddings
        env_embeddings = torch.cat(
            [wave_embeddings, atmos_embeddings], dim=1
        )  # (B, T_w+T_a, d_model)

        # Concatenate padding masks if either is provided
        if wave_padding_mask is not None or atmos_padding_mask is not None:
            B, T_w = wave_features.shape[:2]
            T_a = atmos_features.shape[1]

            # Create masks if not provided
            if wave_padding_mask is None:
                wave_padding_mask = torch.zeros(
                    B, T_w, dtype=torch.bool, device=wave_features.device
                )
            if atmos_padding_mask is None:
                atmos_padding_mask = torch.zeros(
                    B, T_a, dtype=torch.bool, device=atmos_features.device
                )

            env_padding_mask = torch.cat(
                [wave_padding_mask, atmos_padding_mask], dim=1
            )  # (B, T_w+T_a)
        else:
            env_padding_mask = None

        # 5. Cross-attention fusion
        fusion_outputs = self.fusion(
            cliff_embeddings=cliff_embeddings,
            env_embeddings=env_embeddings,
            env_padding_mask=env_padding_mask,
            return_attention=return_env_attention,
        )
        pooled = fusion_outputs['pooled']  # (B, d_model)

        # 6. Prediction heads
        predictions = self.heads(pooled)

        # Combine outputs
        outputs = predictions  # Start with predictions

        # Add attention weights if requested
        if return_spatial_attention:
            outputs['spatial_attention'] = transect_outputs.get('spatial_attention')
        if return_temporal_attention:
            outputs['temporal_attention'] = transect_outputs.get('temporal_attention')
        if return_env_attention:
            outputs['env_attention'] = fusion_outputs.get('attention')

        return outputs

    def get_attention_weights(
        self,
        # Required inputs
        point_features: torch.Tensor,
        metadata: torch.Tensor,
        distances: torch.Tensor,
        wave_features: torch.Tensor,
        atmos_features: torch.Tensor,
        # Optional inputs
        timestamps: Optional[torch.Tensor] = None,
        wave_doy: Optional[torch.Tensor] = None,
        wave_timestamps: Optional[torch.Tensor] = None,
        wave_padding_mask: Optional[torch.Tensor] = None,
        atmos_doy: Optional[torch.Tensor] = None,
        atmos_timestamps: Optional[torch.Tensor] = None,
        atmos_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Convenience method to extract all attention weights for visualization.

        Returns dictionary with spatial_attention, temporal_attention, and
        env_attention for interpretability analysis.
        """
        return self.forward(
            point_features=point_features,
            metadata=metadata,
            distances=distances,
            timestamps=timestamps,
            wave_features=wave_features,
            wave_doy=wave_doy,
            wave_timestamps=wave_timestamps,
            wave_padding_mask=wave_padding_mask,
            atmos_features=atmos_features,
            atmos_doy=atmos_doy,
            atmos_timestamps=atmos_timestamps,
            atmos_padding_mask=atmos_padding_mask,
            return_spatial_attention=True,
            return_temporal_attention=True,
            return_env_attention=True,
        )
