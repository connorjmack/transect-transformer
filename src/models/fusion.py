"""
Cross-attention fusion module for combining cliff geometry with environmental context.

This module implements cross-attention where cliff embeddings query environmental
embeddings to learn which environmental conditions (waves, precipitation) explain
the observed cliff state and predict future erosion.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion of cliff geometry and environmental forcing.

    The cliff temporal embeddings act as queries (Q), while the concatenated
    environmental embeddings (wave + atmospheric) act as keys (K) and values (V).
    This allows the model to learn "which environmental conditions explain
    each cliff location's state and drive erosion."

    Attention weights are extractable for interpretability - high attention to
    specific wave timesteps indicates those storms contributed to erosion.

    Args:
        d_model: Hidden dimension (default 256)
        n_heads: Number of attention heads (default 8)
        n_layers: Number of cross-attention layers (default 2)
        dropout: Dropout rate (default 0.1)
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Cross-attention layers
        # Each layer has MultiheadAttention + FFN with residual connections
        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Output normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        cliff_embeddings: torch.Tensor,  # (B, T_cliff, d_model) - queries
        env_embeddings: torch.Tensor,  # (B, T_env, d_model) - keys/values
        env_padding_mask: Optional[torch.Tensor] = None,  # (B, T_env)
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through cross-attention fusion.

        Args:
            cliff_embeddings: Temporal cliff embeddings (B, T_cliff, d_model)
                Typically from SpatioTemporalTransectEncoder's temporal_embeddings
            env_embeddings: Concatenated environmental embeddings (B, T_env, d_model)
                Concatenation of wave and atmospheric embeddings along time dimension
                T_env = T_wave + T_atmos (e.g., 360 + 90 = 450)
            env_padding_mask: Mask for padded environmental positions (B, T_env)
                True for positions that should be masked (padding)
            return_attention: Whether to return attention weights for visualization

        Returns:
            Dictionary containing:
                - fused: Fused embeddings (B, T_cliff, d_model)
                - pooled: Mean-pooled representation (B, d_model)
                - attention: (optional) Attention weights from last layer (B, n_heads, T_cliff, T_env)
        """
        x = cliff_embeddings  # (B, T_cliff, d_model)

        # Store attention weights from last layer if requested
        attn_weights = None

        # Apply cross-attention layers
        for i, layer in enumerate(self.layers):
            # Last layer: capture attention weights if requested
            if i == len(self.layers) - 1 and return_attention:
                x, attn_weights = layer(
                    x, env_embeddings, env_padding_mask, return_attention=True
                )
            else:
                x, _ = layer(x, env_embeddings, env_padding_mask, return_attention=False)

        # Output normalization
        x = self.norm(x)  # (B, T_cliff, d_model)

        # Compute pooled representation (mean over temporal dimension)
        pooled = x.mean(dim=1)  # (B, d_model)

        outputs = {
            'fused': x,  # (B, T_cliff, d_model)
            'pooled': pooled,  # (B, d_model)
        }

        if return_attention and attn_weights is not None:
            outputs['attention'] = attn_weights  # (B, n_heads, T_cliff, T_env)

        return outputs


class CrossAttentionLayer(nn.Module):
    """
    Single cross-attention layer with FFN and residual connections.

    Architecture:
        1. MultiheadAttention (cliff queries, env keys/values)
        2. Add & Norm (residual)
        3. Feed-forward network
        4. Add & Norm (residual)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        # Layer normalization (pre-norm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,  # (B, T_cliff, d_model)
        key_value: torch.Tensor,  # (B, T_env, d_model)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, T_env)
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through cross-attention layer.

        Args:
            query: Cliff embeddings (B, T_cliff, d_model)
            key_value: Environmental embeddings (B, T_env, d_model)
            key_padding_mask: Mask for padded positions (B, T_env)
            return_attention: Whether to return attention weights

        Returns:
            Tuple of (output, attention_weights)
                - output: (B, T_cliff, d_model)
                - attention_weights: (B, n_heads, T_cliff, T_env) or None
        """
        # Pre-norm
        query_norm = self.norm1(query)

        # Cross-attention (query=cliff, key=value=env)
        attn_output, attn_weights = self.cross_attention(
            query_norm,
            key_value,
            key_value,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False if return_attention else True,
        )

        # Residual connection
        x = query + self.dropout(attn_output)

        # Feed-forward network with residual
        x = x + self.ffn(self.norm2(x))

        return x, attn_weights if return_attention else None
