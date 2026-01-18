# CliffCast: Transect-Based Transformer for Coastal Cliff Erosion Prediction

## Executive Summary

CliffCast is a transformer-based deep learning model that predicts coastal cliff erosion risk by learning relationships between cliff geometry (from LiDAR transects), wave forcing, and precipitation history. The model uses cross-attention to identify which environmental conditions drive erosion, enabling both prediction and physical attribution.

**Core Innovation**: Cross-attention fusion allows the model to learn "which storm events matter for which cliff locations" - providing interpretable predictions grounded in physical processes.

**Scalability Target**: Process state-wide LiDAR datasets by operating on 1D transects rather than full 3D point clouds.

---

## Prediction Heads (4 Total)

The model outputs four complementary predictions from a shared encoder backbone:

| Head | Output | Type | Shape | Description |
|------|--------|------|-------|-------------|
| **1. Risk Index** | `risk_index` | Regression | (B, 1) | Normalized 0-1 composite risk score |
| **2. Collapse Probability** | `p_collapse` | Multi-label | (B, 4) | P(failure) at 1wk, 1mo, 3mo, 1yr horizons |
| **3. Expected Retreat** | `retreat_m` | Regression | (B, 1) | Predicted retreat distance (meters/year) |
| **4. Failure Mode** | `failure_mode` | Multi-class | (B, 5) | Probabilities: topple, planar, rotational, rockfall, stable |

### Implementation Order

Start with Head 1 (Risk Index) to validate the architecture, then incrementally add heads:

```
Phase 1: Risk Index only (validate encoder + fusion work)
Phase 2: Add Expected Retreat (second regression target)
Phase 3: Add Collapse Probability (time-horizon predictions)
Phase 4: Add Failure Mode (requires labeled failure types)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CLIFFCAST ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    INPUTS                                                                    │
│    ──────                                                                    │
│    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                  │
│    │  Transect   │     │    Wave     │     │   Precip    │                  │
│    │  (N, F_t)   │     │  (T_w, F_w) │     │  (T_p, F_p) │                  │
│    └──────┬──────┘     └──────┬──────┘     └──────┬──────┘                  │
│           │                   │                   │                          │
│    ENCODERS                   │                   │                          │
│    ────────                   │                   │                          │
│           ▼                   ▼                   ▼                          │
│    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                  │
│    │  Transect   │     │    Wave     │     │   Precip    │                  │
│    │  Encoder    │     │   Encoder   │     │   Encoder   │                  │
│    │ (N, d_model)│     │(T_w, d_model│     │(T_p, d_model│                  │
│    └──────┬──────┘     └──────┬──────┘     └──────┬──────┘                  │
│           │                   │                   │                          │
│           │                   └─────────┬─────────┘                          │
│           │                             │                                    │
│           │                    ┌────────▼────────┐                           │
│           │                    │   Concatenate   │                           │
│           │                    │ Environmental   │                           │
│           │                    │   Embeddings    │                           │
│           │                    └────────┬────────┘                           │
│           │                             │                                    │
│    FUSION │                             │                                    │
│    ──────                               │                                    │
│           └──────────────┬──────────────┘                                    │
│                          │                                                   │
│                 ┌────────▼────────┐                                          │
│                 │  Cross-Attention │                                         │
│                 │     Fusion       │                                         │
│                 │                  │                                         │
│                 │ Q: cliff tokens  │                                         │
│                 │ K,V: env tokens  │                                         │
│                 └────────┬────────┘                                          │
│                          │                                                   │
│                 ┌────────▼────────┐                                          │
│                 │  Global Pooling │                                          │
│                 │  (N, d) → (d,)  │                                          │
│                 └────────┬────────┘                                          │
│                          │                                                   │
│    PREDICTION HEADS      │                                                   │
│    ────────────────      │                                                   │
│           ┌──────────────┼──────────────┬──────────────┐                    │
│           ▼              ▼              ▼              ▼                    │
│    ┌─────────────┐┌─────────────┐┌─────────────┐┌─────────────┐            │
│    │    Risk     ││  Collapse   ││   Retreat   ││   Failure   │            │
│    │   Index     ││ Probability ││   Distance  ││    Mode     │            │
│    │             ││             ││             ││             │            │
│    │  (B, 1)     ││   (B, 4)    ││   (B, 1)    ││   (B, 5)    │            │
│    │  σ output   ││  σ per head ││  softplus   ││  softmax    │            │
│    └─────────────┘└─────────────┘└─────────────┘└─────────────┘            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

Legend:
  B = batch size
  N = number of points per transect (e.g., 128)
  T_w = wave history timesteps (e.g., 360 for 90 days @ 6hr)
  T_p = precip history timesteps (e.g., 90 for 90 days @ daily)
  F_t, F_w, F_p = input feature dimensions
  d_model = model hidden dimension (e.g., 256)
```

---

## Data Specifications

### Transect Input Features

Each transect is resampled to N=128 points along a shore-normal profile.


### A note for later, EVENTUALLY the point cloud data will have classes for beach, rip rap, cliff face, vegetation, etc ### !!!!

```python
# Per-point features (shape: [N, n_features])
transect_features = {
    'distance_m': float,        # Distance from cliff toe (m), used for positional encoding
    'elevation_m': float,       # Elevation (m NAVD88)
    'slope_deg': float,         # Local slope (degrees)
    'curvature': float,         # Profile curvature (1/m)
    'roughness': float,         # Local surface roughness (std of residuals)
}

# Transect-level metadata (shape: [n_meta])
transect_metadata = {
    'cliff_height_m': float,    # Total cliff height
    'mean_slope_deg': float,    # Average cliff face slope  
    'max_slope_deg': float,     # Maximum slope (overhang indicator)
    'toe_elevation_m': float,   # Beach/toe elevation
    'orientation_deg': float,   # Cliff face orientation (azimuth)
    'latitude': float,
    'longitude': float,
}
```

### Wave Input Features

Time series of wave conditions for T_w timesteps (e.g., 90 days at 6-hour intervals = 360 timesteps).

```python
# Per-timestep features (shape: [T_w, n_wave_features])
# CHECK these with BOR
wave_features = {
    'hs_m': float,              # Significant wave height (m)
    'tp_s': float,              # Peak period (s)
    'dp_deg': float,            # Peak direction (degrees)
    'wave_power_kw': float,     # Wave power flux (kW/m)
}

# Derived features (computed in preprocessing)
# ADD wave impacts with Raph
wave_derived = {
    'cumulative_energy': float, # Cumulative wave energy since last scan
    'max_hs_m': float,          # Maximum Hs in window
    'storm_hours': int,         # Hours with Hs > storm threshold
}
```

### Precipitation Input Features

Time series for T_p timesteps (e.g., 90 days at daily intervals = 90 timesteps).

```python
# Per-timestep features (shape: [T_p, n_precip_features])
precip_features = {
    'precip_mm': float,         # Daily precipitation (mm)
    'cumulative_mm': float,     # Cumulative precipitation
}

# Derived features
precip_derived = {
    'antecedent_precip_index': float,  # Exponentially weighted sum
    'wet_days': int,                    # Days with precip > 1mm
    'max_daily_mm': float,              # Maximum daily rainfall
}
```

### Target Labels

```python
# Computed from M3C2 or DoD change detection between scan epochs
labels = {
    # Primary targets
    'risk_index': float,        # Normalized 0-1 (see formula below)
    'retreat_m_yr': float,      # Annualized retreat rate (m/yr)
    'collapse_1wk': bool,       # Did failure occur within 1 week?
    'collapse_1mo': bool,       # Did failure occur within 1 month?
    'collapse_3mo': bool,       # Did failure occur within 3 months?
    'collapse_1yr': bool,       # Did failure occur within 1 year?
    'failure_mode': int,        # 0=stable, 1=topple, 2=planar, 3=rotational, 4=rockfall
    
    # Auxiliary (for analysis)
    'max_retreat_m': float,     # Maximum local retreat
    'mean_retreat_m': float,    # Mean retreat across transect
    'volume_loss_m3': float,    # Volume loss per meter of coastline
}
```

### Risk Index Formula

```python
def compute_risk_index(retreat_m_yr: float, cliff_height_m: float) -> float:
    """
    Compute normalized risk index from retreat rate and cliff height.
    
    Rationale:
    - Higher retreat rates = higher risk
    - Taller cliffs = higher consequence (more material, longer runout)
    - Sigmoid normalizes to 0-1 with 0.5 at "moderate" risk threshold
    
    Calibration:
    - 0.0-0.2: Low risk (retreat < 0.3 m/yr)
    - 0.2-0.4: Moderate-low risk
    - 0.4-0.6: Moderate risk (retreat ~ 1 m/yr)
    - 0.6-0.8: High risk
    - 0.8-1.0: Very high risk (retreat > 3 m/yr)
    """
    # needs to take slope into account and point cloud classes also when that's added
    # Height-weighted retreat (taller cliffs amplify risk)
    height_factor = 1 + 0.1 * (cliff_height_m - 20) / 20  # Normalized around 20m avg
    weighted_retreat = retreat_m_yr * max(height_factor, 0.5)
    
    # Sigmoid normalization centered at 1 m/yr
    risk = 1 / (1 + np.exp(-2 * (weighted_retreat - 1)))
    
    return float(np.clip(risk, 0, 1))
```

---

## Model Components (Detailed)

### 1. Transect Encoder

```python
class TransectEncoder(nn.Module):
    """
    Encode transect point sequence using self-attention.
    
    Key design choices:
    - Distance-based positional encoding (not index-based)
    - Pre-norm transformer layers for stability
    - Returns both per-point and pooled representations
    """
    
    def __init__(
        self,
        n_point_features: int = 5,
        n_meta_features: int = 7,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Project point features to model dimension
        self.point_embed = nn.Linear(n_point_features, d_model)
        
        # Project metadata and broadcast to all points
        self.meta_embed = nn.Linear(n_meta_features, d_model)
        
        # Distance-based sinusoidal positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Learnable [CLS] token for pooled representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(
        self, 
        point_features: torch.Tensor,  # (B, N, n_point_features)
        distances: torch.Tensor,        # (B, N) - distance from toe
        metadata: torch.Tensor,         # (B, n_meta_features)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            point_embeddings: (B, N, d_model) - per-point representations
            pooled: (B, d_model) - transect-level representation
        """
        B, N, _ = point_features.shape
        
        # Embed point features
        x = self.point_embed(point_features)  # (B, N, d_model)
        
        # Add metadata (broadcast to all points)
        meta = self.meta_embed(metadata)  # (B, d_model)
        x = x + meta.unsqueeze(1)  # (B, N, d_model)
        
        # Add positional encoding based on actual distances
        x = x + self.pos_encoding(distances)  # (B, N, d_model)
        
        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)  # (B, N+1, d_model)
        
        # Self-attention
        x = self.transformer(x)  # (B, N+1, d_model)
        
        # Split CLS and point embeddings
        pooled = x[:, 0, :]           # (B, d_model)
        point_embeddings = x[:, 1:, :] # (B, N, d_model)
        
        return point_embeddings, pooled
```

### 2. Environmental Encoder

```python
class EnvironmentalEncoder(nn.Module):
    """
    Encode wave or precipitation time series.
    
    Shared architecture for both modalities with separate instances.
    Uses learned temporal encoding to capture seasonality.
    """
    
    def __init__(
        self,
        n_features: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
        max_timesteps: int = 400,
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(n_features, d_model)
        
        # Learned temporal encoding (captures seasonality better than sinusoidal)
        self.temporal_embed = nn.Embedding(max_timesteps, d_model)
        
        # Optional: time-of-year encoding for seasonality
        self.season_embed = nn.Embedding(366, d_model // 4)
        self.season_proj = nn.Linear(d_model // 4, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
    def forward(
        self,
        features: torch.Tensor,     # (B, T, n_features)
        day_of_year: torch.Tensor,  # (B, T) - for seasonality
    ) -> torch.Tensor:
        """
        Returns:
            embeddings: (B, T, d_model) - per-timestep environmental embeddings
        """
        B, T, _ = features.shape
        
        # Project features
        x = self.input_proj(features)  # (B, T, d_model)
        
        # Add temporal position
        positions = torch.arange(T, device=features.device)
        x = x + self.temporal_embed(positions)  # (B, T, d_model)
        
        # Add seasonality
        season = self.season_embed(day_of_year)  # (B, T, d_model//4)
        x = x + self.season_proj(season)  # (B, T, d_model)
        
        # Self-attention over time
        x = self.transformer(x)  # (B, T, d_model)
        
        return x
```

### 3. Cross-Attention Fusion

```python
class CrossAttentionFusion(nn.Module):
    """
    Fuse cliff geometry with environmental conditions via cross-attention.
    
    Cliff tokens (queries) attend to environmental tokens (keys/values).
    This learns "which environmental conditions explain each cliff location's state".
    
    Attention weights are extractable for interpretation.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        cliff_embeddings: torch.Tensor,  # (B, N, d_model)
        env_embeddings: torch.Tensor,    # (B, T_env, d_model)
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Returns:
            fused: (B, N, d_model) - cliff embeddings enriched with environmental context
            attention_weights: (B, n_heads, N, T_env) or None - for interpretation
        """
        x = cliff_embeddings
        all_attention = []
        
        for layer in self.layers:
            x, attn = layer(x, env_embeddings)
            if return_attention:
                all_attention.append(attn)
        
        x = self.final_norm(x)
        
        # Average attention across layers if requested
        attention_weights = None
        if return_attention:
            attention_weights = torch.stack(all_attention).mean(dim=0)
        
        return x, attention_weights


class CrossAttentionLayer(nn.Module):
    """Single cross-attention layer with residual and FFN."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        
    def forward(
        self, 
        x: torch.Tensor,    # Queries (cliff)
        env: torch.Tensor,  # Keys/Values (environmental)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Cross-attention
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.cross_attn(
            query=x_norm, key=env, value=env,
            need_weights=True, average_attn_weights=False
        )
        x = x + attn_out
        
        # FFN
        x = x + self.ffn(self.norm2(x))
        
        return x, attn_weights
```

### 4. Prediction Heads

```python
class PredictionHeads(nn.Module):
    """
    Multi-task prediction heads from fused embeddings.
    
    All heads share the same pooled representation but have independent parameters.
    Heads can be enabled/disabled for phased training.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        enable_risk: bool = True,
        enable_retreat: bool = True,
        enable_collapse: bool = True,
        enable_failure_mode: bool = True,
        n_collapse_horizons: int = 4,
        n_failure_modes: int = 5,
    ):
        super().__init__()
        
        self.enable_risk = enable_risk
        self.enable_retreat = enable_retreat
        self.enable_collapse = enable_collapse
        self.enable_failure_mode = enable_failure_mode
        
        # Shared pooling
        self.pool_attention = nn.Linear(d_model, 1)  # Attention pooling
        
        # Head 1: Risk Index
        if enable_risk:
            self.risk_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid(),
            )
        
        # Head 2: Collapse Probability (multi-horizon)
        if enable_collapse:
            self.collapse_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 2, n_collapse_horizons),
                nn.Sigmoid(),
            )
        
        # Head 3: Expected Retreat
        if enable_retreat:
            self.retreat_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 2, 1),
                nn.Softplus(),  # Ensures positive output
            )
        
        # Head 4: Failure Mode Classification
        if enable_failure_mode:
            self.mode_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 2, n_failure_modes),
                # No activation - use CrossEntropyLoss which applies softmax
            )
    
    def forward(
        self, 
        fused_embeddings: torch.Tensor,  # (B, N, d_model)
    ) -> dict[str, torch.Tensor]:
        """
        Returns dict of predictions, only including enabled heads.
        """
        # Attention-weighted pooling
        attn_weights = F.softmax(self.pool_attention(fused_embeddings), dim=1)
        pooled = (fused_embeddings * attn_weights).sum(dim=1)  # (B, d_model)
        
        outputs = {}
        
        if self.enable_risk:
            outputs['risk_index'] = self.risk_head(pooled).squeeze(-1)  # (B,)
        
        if self.enable_collapse:
            outputs['p_collapse'] = self.collapse_head(pooled)  # (B, 4)
        
        if self.enable_retreat:
            outputs['retreat_m'] = self.retreat_head(pooled).squeeze(-1)  # (B,)
        
        if self.enable_failure_mode:
            outputs['failure_mode'] = self.mode_head(pooled)  # (B, 5)
        
        return outputs
```

### 5. Full Model

```python
class CliffCast(nn.Module):
    """
    Complete CliffCast model combining all components.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        d_model = config['d_model']
        
        # Encoders
        self.transect_encoder = TransectEncoder(
            n_point_features=config['n_point_features'],
            n_meta_features=config['n_meta_features'],
            d_model=d_model,
            n_heads=config['n_heads'],
            n_layers=config['n_layers_transect'],
            dropout=config['dropout'],
        )
        
        self.wave_encoder = EnvironmentalEncoder(
            n_features=config['n_wave_features'],
            d_model=d_model,
            n_heads=config['n_heads'],
            n_layers=config['n_layers_env'],
            dropout=config['dropout'],
        )
        
        self.precip_encoder = EnvironmentalEncoder(
            n_features=config['n_precip_features'],
            d_model=d_model,
            n_heads=config['n_heads'],
            n_layers=config['n_layers_env'],
            dropout=config['dropout'],
        )
        
        # Fusion
        self.fusion = CrossAttentionFusion(
            d_model=d_model,
            n_heads=config['n_heads'],
            n_layers=config['n_layers_fusion'],
            dropout=config['dropout'],
        )
        
        # Prediction heads
        self.heads = PredictionHeads(
            d_model=d_model,
            enable_risk=config.get('enable_risk', True),
            enable_retreat=config.get('enable_retreat', True),
            enable_collapse=config.get('enable_collapse', True),
            enable_failure_mode=config.get('enable_failure_mode', True),
        )
        
    def forward(
        self,
        transect_points: torch.Tensor,
        transect_distances: torch.Tensor,
        transect_metadata: torch.Tensor,
        wave_features: torch.Tensor,
        wave_doy: torch.Tensor,
        precip_features: torch.Tensor,
        precip_doy: torch.Tensor,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass.
        
        Returns dict with prediction keys + optional 'attention' key.
        """
        # Encode transect
        point_emb, transect_pooled = self.transect_encoder(
            transect_points, transect_distances, transect_metadata
        )
        
        # Encode environmental
        wave_emb = self.wave_encoder(wave_features, wave_doy)
        precip_emb = self.precip_encoder(precip_features, precip_doy)
        
        # Concatenate environmental embeddings
        env_emb = torch.cat([wave_emb, precip_emb], dim=1)
        
        # Cross-attention fusion
        fused, attention = self.fusion(point_emb, env_emb, return_attention)
        
        # Predictions
        outputs = self.heads(fused)
        
        if return_attention:
            outputs['attention'] = attention
        
        return outputs
```

---

## Loss Functions

```python
class CliffCastLoss(nn.Module):
    """
    Multi-task loss with configurable weights.
    """
    
    def __init__(
        self,
        weight_risk: float = 1.0,
        weight_retreat: float = 1.0,
        weight_collapse: float = 2.0,  # Higher: safety-critical
        weight_mode: float = 0.5,       # Lower: fewer labels
    ):
        super().__init__()
        self.weights = {
            'risk': weight_risk,
            'retreat': weight_retreat,
            'collapse': weight_collapse,
            'mode': weight_mode,
        }
        
    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Returns total loss and dict of individual losses for logging.
        """
        losses = {}
        
        # Risk Index: Smooth L1 (less sensitive to outliers)
        if 'risk_index' in predictions:
            losses['risk'] = F.smooth_l1_loss(
                predictions['risk_index'], 
                targets['risk_index']
            )
        
        # Expected Retreat: Smooth L1
        if 'retreat_m' in predictions:
            losses['retreat'] = F.smooth_l1_loss(
                predictions['retreat_m'],
                targets['retreat_m_yr']
            )
        
        # Collapse Probability: Binary cross-entropy per horizon
        if 'p_collapse' in predictions:
            collapse_targets = torch.stack([
                targets['collapse_1wk'],
                targets['collapse_1mo'],
                targets['collapse_3mo'],
                targets['collapse_1yr'],
            ], dim=1).float()
            
            losses['collapse'] = F.binary_cross_entropy(
                predictions['p_collapse'],
                collapse_targets
            )
        
        # Failure Mode: Cross-entropy (only on samples with failures)
        if 'failure_mode' in predictions:
            has_failure = targets['failure_mode'] > 0  # 0 = stable
            if has_failure.any():
                losses['mode'] = F.cross_entropy(
                    predictions['failure_mode'][has_failure],
                    targets['failure_mode'][has_failure]
                )
            else:
                losses['mode'] = torch.tensor(0.0, device=predictions['failure_mode'].device)
        
        # Weighted sum
        total = sum(
            self.weights.get(k, 1.0) * v 
            for k, v in losses.items()
        )
        
        return total, {k: v.item() for k, v in losses.items()}
```

---

## Project Structure

```
cliffcast/
├── configs/
│   ├── default.yaml           # Full model config
│   ├── phase1_risk_only.yaml  # Phase 1: Risk index only
│   ├── phase2_add_retreat.yaml
│   ├── phase3_add_collapse.yaml
│   └── phase4_full.yaml       # All heads enabled
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── transect_extractor.py   # Extract transects from point clouds
│   │   ├── wave_loader.py          # Load/process CDIP/WW3 wave data
│   │   ├── precip_loader.py        # Load/process PRISM precip data
│   │   ├── label_generator.py      # Compute labels from change detection
│   │   ├── dataset.py              # PyTorch Dataset class
│   │   └── transforms.py           # Data augmentation
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── positional_encoding.py  # Sinusoidal & learned encodings
│   │   ├── transect_encoder.py     # Transect self-attention encoder
│   │   ├── environmental_encoder.py # Wave/precip encoder
│   │   ├── fusion.py               # Cross-attention fusion
│   │   ├── prediction_heads.py     # Multi-task heads
│   │   ├── cliffcast.py            # Full model assembly
│   │   └── context_3d.py           # [Future] Option C: 3D context
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py               # Multi-task loss
│   │   ├── trainer.py              # Training loop
│   │   ├── scheduler.py            # LR scheduling
│   │   └── callbacks.py            # Checkpointing, early stopping
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py              # RMSE, MAE, R², AUC, etc.
│   │   ├── calibration.py          # Reliability diagrams
│   │   └── baseline.py             # Baseline models for comparison
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── attention_maps.py       # Visualize cross-attention
│   │   ├── transect_plots.py       # Plot predictions on profiles
│   │   └── spatial_maps.py         # Map predictions geographically
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predictor.py            # Single transect prediction
│   │   └── batch_processor.py      # State-wide batch inference
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py               # Config loading/validation
│       ├── logging.py              # Logging setup
│       └── io.py                   # File I/O utilities
│
├── tests/
│   ├── __init__.py
│   ├── test_data/
│   │   ├── test_transect_extractor.py
│   │   ├── test_wave_loader.py
│   │   ├── test_precip_loader.py
│   │   └── test_dataset.py
│   ├── test_models/
│   │   ├── test_encoders.py
│   │   ├── test_fusion.py
│   │   ├── test_heads.py
│   │   └── test_full_model.py
│   └── test_training/
│       ├── test_losses.py
│       └── test_trainer.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_debugging.ipynb
│   ├── 03_attention_analysis.ipynb
│   └── 04_evaluation_report.ipynb
│
├── scripts/
│   ├── download_wave_data.py       # Fetch CDIP data
│   ├── download_precip_data.py     # Fetch PRISM data
│   ├── prepare_dataset.py          # Full preprocessing pipeline
│   └── export_predictions.py       # Export to GeoJSON/shapefile
│
├── train.py                        # Main training script
├── evaluate.py                     # Evaluation script
├── predict.py                      # Inference script
├── requirements.txt
├── setup.py
├── pyproject.toml
└── README.md
```

---

## Configuration

```yaml
# configs/default.yaml

# Model architecture
model:
  d_model: 256
  n_heads: 8
  n_layers_transect: 4
  n_layers_env: 3
  n_layers_fusion: 2
  dropout: 0.1
  
  # Input dimensions
  n_point_features: 5    # elevation, slope, curvature, roughness, aspect
  n_meta_features: 7     # cliff_height, mean_slope, etc.
  n_wave_features: 4     # hs, tp, dp, power
  n_precip_features: 2   # daily_precip, cumulative
  
  # Prediction heads (toggle for phased training)
  enable_risk: true
  enable_retreat: true
  enable_collapse: true
  enable_failure_mode: true

# Data
data:
  n_transect_points: 128
  wave_history_days: 90
  wave_timestep_hours: 6
  precip_history_days: 90
  precip_timestep_hours: 24
  
  # Normalization
  normalize_elevation: true
  normalize_to_toe: true  # Set toe as origin

# Training
training:
  batch_size: 32
  learning_rate: 1.0e-4
  weight_decay: 1.0e-5
  max_epochs: 100
  warmup_epochs: 5
  
  # Loss weights
  loss_weight_risk: 1.0
  loss_weight_retreat: 1.0
  loss_weight_collapse: 2.0
  loss_weight_mode: 0.5
  
  # Early stopping
  early_stopping_patience: 15
  early_stopping_metric: val_loss

# Logging
logging:
  project_name: cliffcast
  log_every_n_steps: 10
  save_top_k: 3
```

---

## Implementation Checklist

### Phase 1: Project Setup & Data Pipeline
**Goal**: Robust data loading and preprocessing
**Estimated time**: 1-2 weeks

#### 1.1 Project Initialization
- [ ] Create directory structure as specified above
- [ ] Set up `pyproject.toml` with dependencies
- [ ] Create `requirements.txt`
- [ ] Initialize git repository with `.gitignore`
- [ ] Set up basic logging utility

**Test checkpoint**: 
```bash
python -c "import src; print('Package imports work')"
# Should print: Package imports work
```

#### 1.2 Configuration System
- [ ] Implement `src/utils/config.py` with YAML loading
- [ ] Create `configs/default.yaml`
- [ ] Add config validation (check required fields)
- [ ] Support config inheritance/overrides

**Test checkpoint**: 
```bash
python -c "from src.utils.config import load_config; cfg = load_config('configs/default.yaml'); print(cfg['model']['d_model'])"
# Should print: 256
```

#### 1.3 Transect Extraction
- [ ] Implement `src/data/transect_extractor.py`
  - [ ] Load LAZ/LAS files via laspy
  - [ ] Define shore-normal direction from coastline
  - [ ] Extract profiles at specified spacing (e.g., 10m)
  - [ ] Resample each profile to fixed N points (128)
  - [ ] Compute derived features: slope, curvature, roughness
  - [ ] Handle edge cases: data gaps, vegetation points, water returns
  - [ ] Save transects to standardized format (NPZ or Parquet)
- [ ] Write unit tests in `tests/test_data/test_transect_extractor.py`

**Test checkpoint**: 
```python
from src.data.transect_extractor import TransectExtractor
extractor = TransectExtractor(n_points=128, spacing_m=10)
transects = extractor.extract_from_file("test_data/sample.laz")
assert transects['points'].shape[1] == 128, "Wrong number of points"
assert transects['points'].shape[2] >= 5, "Missing features"
print(f"Extracted {transects['points'].shape[0]} transects")
```

#### 1.4 Wave Data Loader
- [ ] Implement `src/data/wave_loader.py`
  - [ ] Fetch from CDIP API (historic/realtime)
  - [ ] Alternative: load WaveWatch III NetCDF
  - [ ] Resample to consistent timesteps (6-hourly)
  - [ ] Compute derived: wave power, cumulative energy, storm flags
  - [ ] Handle missing data (interpolation or flagging)
- [ ] Write unit tests

**Test checkpoint**:
```python
from src.data.wave_loader import WaveLoader
loader = WaveLoader(source='cdip', buoy_id='100')
waves = loader.load(start_date='2023-10-01', end_date='2024-01-01')
assert waves.shape[0] == 368, "Expected ~92 days * 4 = 368 timesteps"
assert waves.shape[1] == 4, "Expected 4 features: hs, tp, dp, power"
assert not np.isnan(waves).any(), "Found NaN values"
```

#### 1.5 Precipitation Data Loader
- [ ] Implement `src/data/precip_loader.py`
  - [ ] Load PRISM daily grids
  - [ ] Extract values at transect coordinates
  - [ ] Compute: cumulative, antecedent precip index (API)
  - [ ] Handle missing data
- [ ] Write unit tests

**Test checkpoint**:
```python
from src.data.precip_loader import PrecipLoader
loader = PrecipLoader(source='prism')
precip = loader.load(
    lat=32.87, lon=-117.25,
    start_date='2023-10-01', end_date='2024-01-01'
)
assert precip.shape[0] == 92, "Expected 92 days"
assert precip.shape[1] == 2, "Expected 2 features: daily, cumulative"
```

#### 1.6 Label Generation
- [ ] Implement `src/data/label_generator.py`
  - [ ] M3C2 or DoD change detection between epoch pairs
  - [ ] Compute retreat at each transect point
  - [ ] Aggregate to transect-level stats: mean, max, volume
  - [ ] Compute risk_index using formula
  - [ ] Assign collapse labels based on thresholds
  - [ ] (Manual step) Failure mode annotation workflow
- [ ] Write unit tests

**Test checkpoint**:
```python
from src.data.label_generator import LabelGenerator, compute_risk_index

# Test risk index formula
risk = compute_risk_index(retreat_m_yr=1.5, cliff_height_m=25)
assert 0.5 < risk < 0.8, f"Expected moderate-high risk, got {risk}"

risk_low = compute_risk_index(retreat_m_yr=0.2, cliff_height_m=15)
assert risk_low < 0.3, f"Expected low risk, got {risk_low}"

risk_high = compute_risk_index(retreat_m_yr=4.0, cliff_height_m=30)
assert risk_high > 0.85, f"Expected high risk, got {risk_high}"
```

#### 1.7 PyTorch Dataset
- [ ] Implement `src/data/dataset.py`
  - [ ] `CliffCastDataset` class with `__getitem__`, `__len__`
  - [ ] Lazy loading for memory efficiency
  - [ ] Temporal alignment: match transect scan date to environmental window
  - [ ] Train/val/test split logic (spatial + temporal)
  - [ ] Collate function for DataLoader
- [ ] Implement `src/data/transforms.py`
  - [ ] `RandomHorizontalFlip` - flip transect direction
  - [ ] `ElevationNoise` - add Gaussian noise to elevations
  - [ ] `TemporalShift` - shift environmental window slightly
- [ ] Write unit tests

**Test checkpoint**:
```python
from src.data.dataset import CliffCastDataset
from torch.utils.data import DataLoader

dataset = CliffCastDataset(data_dir="data/processed", split="train")
print(f"Dataset size: {len(dataset)}")

loader = DataLoader(dataset, batch_size=4, collate_fn=dataset.collate_fn)
batch = next(iter(loader))

# Check all expected keys present
assert batch['transect_points'].shape == (4, 128, 5)
assert batch['transect_distances'].shape == (4, 128)
assert batch['transect_metadata'].shape == (4, 7)
assert batch['wave_features'].shape == (4, 360, 4)
assert batch['precip_features'].shape == (4, 90, 2)
assert 'risk_index' in batch
print("Dataset test passed!")
```

---

### Phase 2: Model Implementation (Risk Index Only)
**Goal**: Working model with single prediction head to validate architecture
**Estimated time**: 1 week

#### 2.1 Positional Encodings
- [ ] Implement `src/models/positional_encoding.py`
  - [ ] `SinusoidalPositionalEncoding`: distance-based for transects
  - [ ] `LearnedTemporalEncoding`: for environmental sequences
- [ ] Write unit tests

**Test checkpoint**:
```python
from src.models.positional_encoding import SinusoidalPositionalEncoding
import torch

pe = SinusoidalPositionalEncoding(d_model=256)
distances = torch.linspace(0, 100, 128).unsqueeze(0)  # (1, 128)
encoding = pe(distances)

assert encoding.shape == (1, 128, 256), f"Wrong shape: {encoding.shape}"
assert not torch.isnan(encoding).any(), "NaN in positional encoding"
assert encoding[0, 0, :].abs().mean() > 0.1, "Encoding seems too small"
print("Positional encoding test passed!")
```

#### 2.2 Transect Encoder
- [ ] Implement `src/models/transect_encoder.py`
  - [ ] Input projection layer
  - [ ] Metadata embedding + broadcast
  - [ ] CLS token (learnable)
  - [ ] Transformer encoder layers (pre-norm)
  - [ ] Return both per-point and pooled outputs
- [ ] Write unit tests

**Test checkpoint**:
```python
from src.models.transect_encoder import TransectEncoder
import torch

encoder = TransectEncoder(
    n_point_features=5, n_meta_features=7, 
    d_model=256, n_heads=8, n_layers=4
)
print(f"TransectEncoder params: {sum(p.numel() for p in encoder.parameters()):,}")

# Test forward pass
points = torch.randn(2, 128, 5)
distances = torch.linspace(0, 100, 128).unsqueeze(0).expand(2, -1)
meta = torch.randn(2, 7)

point_emb, pooled = encoder(points, distances, meta)

assert point_emb.shape == (2, 128, 256), f"Wrong point_emb shape: {point_emb.shape}"
assert pooled.shape == (2, 256), f"Wrong pooled shape: {pooled.shape}"
assert not torch.isnan(point_emb).any(), "NaN in point embeddings"
print("TransectEncoder test passed!")
```

#### 2.3 Environmental Encoder
- [ ] Implement `src/models/environmental_encoder.py`
  - [ ] Input projection
  - [ ] Learned temporal position embedding
  - [ ] Seasonality embedding (day-of-year)
  - [ ] Transformer encoder layers
- [ ] Write unit tests

**Test checkpoint**:
```python
from src.models.environmental_encoder import EnvironmentalEncoder
import torch

encoder = EnvironmentalEncoder(n_features=4, d_model=256, n_heads=8, n_layers=3)
print(f"EnvironmentalEncoder params: {sum(p.numel() for p in encoder.parameters()):,}")

features = torch.randn(2, 360, 4)
doy = torch.randint(1, 366, (2, 360))

emb = encoder(features, doy)

assert emb.shape == (2, 360, 256), f"Wrong shape: {emb.shape}"
assert not torch.isnan(emb).any(), "NaN in environmental embeddings"
print("EnvironmentalEncoder test passed!")
```

#### 2.4 Cross-Attention Fusion
- [ ] Implement `src/models/fusion.py`
  - [ ] `CrossAttentionLayer`: single layer with residual + FFN
  - [ ] `CrossAttentionFusion`: stack of layers
  - [ ] Attention weight extraction for interpretability
- [ ] Write unit tests

**Test checkpoint**:
```python
from src.models.fusion import CrossAttentionFusion
import torch

fusion = CrossAttentionFusion(d_model=256, n_heads=8, n_layers=2)
print(f"Fusion params: {sum(p.numel() for p in fusion.parameters()):,}")

cliff = torch.randn(2, 128, 256)
env = torch.randn(2, 450, 256)  # 360 wave + 90 precip

fused, attn = fusion(cliff, env, return_attention=True)

assert fused.shape == (2, 128, 256), f"Wrong fused shape: {fused.shape}"
assert attn.shape == (2, 8, 128, 450), f"Wrong attn shape: {attn.shape}"
assert attn.sum(dim=-1).allclose(torch.ones(2, 8, 128)), "Attention doesn't sum to 1"
print("CrossAttentionFusion test passed!")
```

#### 2.5 Prediction Heads (Risk Only First)
- [ ] Implement `src/models/prediction_heads.py`
  - [ ] Attention-weighted pooling
  - [ ] Risk index head (Linear → GELU → Linear → Sigmoid)
  - [ ] Enable/disable flags for other heads (disabled for now)
- [ ] Write unit tests

**Test checkpoint**:
```python
from src.models.prediction_heads import PredictionHeads
import torch

heads = PredictionHeads(
    d_model=256,
    enable_risk=True,
    enable_retreat=False,
    enable_collapse=False,
    enable_failure_mode=False,
)
print(f"Heads params: {sum(p.numel() for p in heads.parameters()):,}")

fused = torch.randn(2, 128, 256)
outputs = heads(fused)

assert 'risk_index' in outputs, "Missing risk_index"
assert outputs['risk_index'].shape == (2,), f"Wrong shape: {outputs['risk_index'].shape}"
assert (outputs['risk_index'] >= 0).all(), "Risk below 0"
assert (outputs['risk_index'] <= 1).all(), "Risk above 1"
assert 'retreat_m' not in outputs, "retreat_m should be disabled"
print("PredictionHeads test passed!")
```

#### 2.6 Full Model Assembly
- [ ] Implement `src/models/cliffcast.py`
  - [ ] Instantiate all components from config
  - [ ] Full forward pass
  - [ ] Optional attention return
- [ ] Write integration test

**Test checkpoint**:
```python
from src.models.cliffcast import CliffCast
import torch

config = {
    'd_model': 256,
    'n_heads': 8,
    'n_layers_transect': 4,
    'n_layers_env': 3,
    'n_layers_fusion': 2,
    'dropout': 0.1,
    'n_point_features': 5,
    'n_meta_features': 7,
    'n_wave_features': 4,
    'n_precip_features': 2,
    'enable_risk': True,
    'enable_retreat': False,
    'enable_collapse': False,
    'enable_failure_mode': False,
}

model = CliffCast(config)
print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

# Test forward
batch = {
    'transect_points': torch.randn(2, 128, 5),
    'transect_distances': torch.linspace(0, 100, 128).unsqueeze(0).expand(2, -1),
    'transect_metadata': torch.randn(2, 7),
    'wave_features': torch.randn(2, 360, 4),
    'wave_doy': torch.randint(1, 366, (2, 360)),
    'precip_features': torch.randn(2, 90, 2),
    'precip_doy': torch.randint(1, 366, (2, 90)),
}

outputs = model(**batch, return_attention=True)

assert 'risk_index' in outputs
assert 'attention' in outputs
print(f"Risk predictions: {outputs['risk_index']}")
print("Full model test passed!")
```

---

### Phase 3: Training Infrastructure
**Goal**: Complete training pipeline, train on real data
**Estimated time**: 1 week

#### 3.1 Loss Functions
- [ ] Implement `src/training/losses.py`
  - [ ] `CliffCastLoss` class
  - [ ] Risk loss: Smooth L1
  - [ ] Weighted combination
  - [ ] Return individual losses for logging
- [ ] Write unit tests

**Test checkpoint**:
```python
from src.training.losses import CliffCastLoss
import torch

loss_fn = CliffCastLoss(weight_risk=1.0)

preds = {'risk_index': torch.tensor([0.3, 0.7, 0.5])}
targets = {'risk_index': torch.tensor([0.4, 0.6, 0.8])}

total_loss, loss_dict = loss_fn(preds, targets)

assert total_loss.requires_grad, "Loss doesn't require grad"
assert 'risk' in loss_dict, "Missing risk loss"
assert total_loss.item() > 0, "Loss should be positive"
print(f"Total loss: {total_loss.item():.4f}")
print("Loss function test passed!")
```

#### 3.2 Learning Rate Scheduler
- [ ] Implement `src/training/scheduler.py`
  - [ ] Linear warmup
  - [ ] Cosine annealing decay
- [ ] Write unit tests

**Test checkpoint**:
```python
from src.training.scheduler import get_scheduler
import torch

optimizer = torch.optim.Adam([torch.randn(10, requires_grad=True)], lr=1e-4)
scheduler = get_scheduler(optimizer, warmup_epochs=5, max_epochs=100)

lrs = []
for epoch in range(100):
    lrs.append(scheduler.get_last_lr()[0])
    scheduler.step()

assert lrs[0] < lrs[4], "LR should increase during warmup"
assert lrs[5] > lrs[99], "LR should decrease after warmup"
print("Scheduler test passed!")
```

#### 3.3 Training Loop
- [ ] Implement `src/training/trainer.py`
  - [ ] `Trainer` class with train/validate methods
  - [ ] Gradient clipping
  - [ ] Mixed precision (AMP) support
  - [ ] Progress bar (tqdm)
  - [ ] Metric logging
- [ ] Implement `src/training/callbacks.py`
  - [ ] `CheckpointCallback`: save best model
  - [ ] `EarlyStoppingCallback`: stop if no improvement
  - [ ] `LoggingCallback`: W&B integration
- [ ] Write integration test

**Test checkpoint**:
```bash
# Create small synthetic dataset first
python scripts/create_synthetic_data.py --n_samples 100 --output data/synthetic/

# Run training for 2 epochs
python train.py \
    --config configs/phase1_risk_only.yaml \
    --data_dir data/synthetic/ \
    --max_epochs 2 \
    --batch_size 8

# Should complete without errors and print loss values
```

#### 3.4 Main Training Script
- [ ] Implement `train.py`
  - [ ] Argument parsing (config path, overrides)
  - [ ] Config loading
  - [ ] Data loader setup
  - [ ] Model instantiation
  - [ ] Trainer instantiation
  - [ ] Run training
- [ ] Test end-to-end

**Test checkpoint**:
```bash
python train.py \
    --config configs/phase1_risk_only.yaml \
    --data_dir data/processed/san_diego \
    --max_epochs 50 \
    --wandb_project cliffcast_dev

# After training:
# - checkpoints/best.pt should exist
# - W&B should show decreasing loss curve
# - Final val loss < 0.5
```

#### 3.5 Validate Learning on Synthetic Data
- [ ] Create synthetic dataset with known relationships
  - [ ] risk = f(slope, wave_energy) with noise
  - [ ] Known ground truth for debugging
- [ ] Train model on synthetic data
- [ ] Verify model learns the relationship

**Test checkpoint**:
```python
# After training on synthetic data where risk = 0.5 * normalized_slope + 0.5 * normalized_wave_power

# Model should achieve R² > 0.8 on synthetic test set
from src.evaluation.metrics import compute_r2

r2 = compute_r2(predictions, targets)
assert r2 > 0.8, f"Model failed to learn synthetic relationship: R²={r2}"
print(f"Synthetic data R²: {r2:.3f} - Model is learning!")
```

---

### Phase 4: Add Remaining Prediction Heads
**Goal**: Enable all 4 prediction heads incrementally
**Estimated time**: 1-2 weeks

#### 4.1 Add Expected Retreat Head
- [ ] Update config: `enable_retreat: true`
- [ ] Add retreat loss to `CliffCastLoss`
- [ ] Prepare retreat labels (annualized from change detection)
- [ ] Train with both heads
- [ ] Validate retreat predictions

**Test checkpoint**:
```python
# After training with retreat enabled
outputs = model(**batch)
assert 'risk_index' in outputs
assert 'retreat_m' in outputs
assert (outputs['retreat_m'] >= 0).all(), "Retreat should be positive (softplus)"

# Retreat predictions should be in reasonable range
assert outputs['retreat_m'].mean() < 10, "Mean retreat seems too high"
print(f"Retreat predictions: {outputs['retreat_m']}")
```

#### 4.2 Add Collapse Probability Head
- [ ] Update config: `enable_collapse: true`
- [ ] Prepare collapse labels
  - [ ] Define thresholds (e.g., max_retreat > 1m = collapse)
  - [ ] Assign to time horizons based on when scan occurred
- [ ] Add collapse BCE loss
- [ ] Train with 3 heads
- [ ] Check temporal monotonicity (P(1yr) >= P(3mo) >= ...)

**Test checkpoint**:
```python
outputs = model(**batch)
p_collapse = outputs['p_collapse']  # (B, 4)

assert p_collapse.shape[1] == 4, "Expected 4 horizons"
assert (p_collapse >= 0).all() and (p_collapse <= 1).all(), "Probabilities out of range"

# Check approximate monotonicity (not strict due to noise)
# P(failure by 1yr) should generally >= P(failure by 1wk)
# This is a soft check - may not hold for all samples
monotonic_violations = (p_collapse[:, 0] > p_collapse[:, 3]).sum()
print(f"Monotonicity violations: {monotonic_violations}/{len(p_collapse)}")
```

#### 4.3 Add Failure Mode Head
- [ ] Update config: `enable_failure_mode: true`
- [ ] Prepare failure mode labels
  - [ ] Manual annotation workflow (or heuristic)
  - [ ] Classes: stable(0), topple(1), planar(2), rotational(3), rockfall(4)
- [ ] Add cross-entropy loss (only on failure samples)
- [ ] Handle class imbalance (most are "stable")
- [ ] Train full model

**Test checkpoint**:
```python
outputs = model(**batch)
failure_logits = outputs['failure_mode']  # (B, 5)

assert failure_logits.shape[1] == 5, "Expected 5 classes"

# Convert to probabilities
probs = F.softmax(failure_logits, dim=-1)
assert probs.sum(dim=-1).allclose(torch.ones(B)), "Probs don't sum to 1"

# Check predicted classes
predicted_modes = failure_logits.argmax(dim=-1)
print(f"Predicted modes: {predicted_modes}")
```

#### 4.4 Full Multi-Task Training
- [ ] Create `configs/phase4_full.yaml` with all heads
- [ ] Tune loss weights
- [ ] Train on full dataset
- [ ] Monitor all loss terms (should all decrease)
- [ ] Save best checkpoint

**Test checkpoint**:
```bash
python train.py \
    --config configs/phase4_full.yaml \
    --data_dir data/processed/san_diego \
    --max_epochs 100

# All loss terms should decrease:
# - loss/risk: decreasing
# - loss/retreat: decreasing  
# - loss/collapse: decreasing
# - loss/mode: decreasing (when failures present)
```

---

### Phase 5: Evaluation & Interpretation
**Goal**: Comprehensive metrics, baselines, interpretability
**Estimated time**: 1 week

#### 5.1 Evaluation Metrics
- [ ] Implement `src/evaluation/metrics.py`
  - [ ] Regression: RMSE, MAE, R², Pearson r
  - [ ] Classification: AUC-ROC, precision, recall, F1
  - [ ] Multi-class: confusion matrix, per-class accuracy
  - [ ] `compute_all_metrics()` function
- [ ] Write unit tests

**Test checkpoint**:
```python
from src.evaluation.metrics import compute_all_metrics
import numpy as np

# Fake predictions/targets
preds = {
    'risk_index': np.random.rand(100),
    'retreat_m': np.random.rand(100) * 2,
    'p_collapse': np.random.rand(100, 4),
    'failure_mode': np.random.randint(0, 5, 100),
}
targets = {
    'risk_index': np.random.rand(100),
    'retreat_m_yr': np.random.rand(100) * 2,
    'collapse_1wk': np.random.randint(0, 2, 100),
    'collapse_1mo': np.random.randint(0, 2, 100),
    'collapse_3mo': np.random.randint(0, 2, 100),
    'collapse_1yr': np.random.randint(0, 2, 100),
    'failure_mode': np.random.randint(0, 5, 100),
}

metrics = compute_all_metrics(preds, targets)

assert 'risk_rmse' in metrics
assert 'risk_r2' in metrics
assert 'retreat_mae' in metrics
assert 'collapse_auc_1yr' in metrics
print("Metrics computation test passed!")
```

#### 5.2 Calibration Analysis
- [ ] Implement `src/evaluation/calibration.py`
  - [ ] Reliability diagram plotting
  - [ ] Expected Calibration Error (ECE)
  - [ ] Maximum Calibration Error (MCE)
- [ ] Generate calibration plots for collapse probabilities

**Test checkpoint**:
```python
from src.evaluation.calibration import reliability_diagram, compute_ece

# Should generate a plot file
reliability_diagram(
    predicted_probs=preds['p_collapse'][:, 3],  # 1-year horizon
    true_labels=targets['collapse_1yr'],
    save_path='results/calibration_1yr.png'
)

ece = compute_ece(preds['p_collapse'][:, 3], targets['collapse_1yr'])
print(f"ECE (1yr): {ece:.3f}")
assert ece < 0.3, "Calibration seems poor"
```

#### 5.3 Baseline Comparisons
- [ ] Implement `src/evaluation/baseline.py`
  - [ ] `HistoricalAverageBaseline`: predict site's historical mean
  - [ ] `LinearRegressionBaseline`: simple features → risk
  - [ ] `RandomForestBaseline`: sklearn RF
- [ ] Train baselines on same data
- [ ] Compare all metrics

**Test checkpoint**:
```python
from src.evaluation.baseline import HistoricalAverageBaseline, RandomForestBaseline

# Train baselines
hist_baseline = HistoricalAverageBaseline()
hist_baseline.fit(train_data)
hist_preds = hist_baseline.predict(test_data)
hist_metrics = compute_all_metrics(hist_preds, test_targets)

rf_baseline = RandomForestBaseline()
rf_baseline.fit(train_data, train_targets)
rf_preds = rf_baseline.predict(test_data)
rf_metrics = compute_all_metrics(rf_preds, test_targets)

# CliffCast should beat baselines
assert model_metrics['risk_r2'] > hist_metrics['risk_r2'], "CliffCast worse than historical avg"
assert model_metrics['risk_r2'] > rf_metrics['risk_r2'], "CliffCast worse than RF"
print(f"CliffCast R²: {model_metrics['risk_r2']:.3f}")
print(f"RF R²: {rf_metrics['risk_r2']:.3f}")
print(f"Historical R²: {hist_metrics['risk_r2']:.3f}")
```

#### 5.4 Attention Visualization
- [ ] Implement `src/visualization/attention_maps.py`
  - [ ] Plot attention weights on transect profile
  - [ ] Plot temporal attention on environmental time series
  - [ ] Identify peak attention timesteps (storms)
- [ ] Generate visualizations for test samples

**Test checkpoint**:
```python
from src.visualization.attention_maps import plot_transect_attention, plot_temporal_attention

# Get attention from model
outputs = model(**batch, return_attention=True)
attn = outputs['attention']  # (B, n_heads, N_cliff, T_env)

# Plot for first sample
plot_transect_attention(
    transect_profile=batch['transect_points'][0],
    attention=attn[0].mean(dim=0),  # Average over heads
    save_path='results/attention_transect_0.png'
)

plot_temporal_attention(
    wave_data=batch['wave_features'][0],
    precip_data=batch['precip_features'][0],
    attention=attn[0].mean(dim=(0, 1)),  # Average over heads and cliff points
    save_path='results/attention_temporal_0.png'
)

# Visual check: attention should highlight storm events
print("Generated attention visualizations - check manually")
```

#### 5.5 Evaluation Script
- [ ] Implement `evaluate.py`
  - [ ] Load checkpoint
  - [ ] Run inference on test set
  - [ ] Compute all metrics
  - [ ] Generate visualizations
  - [ ] Save results to JSON/CSV

**Test checkpoint**:
```bash
python evaluate.py \
    --checkpoint checkpoints/best.pt \
    --data_dir data/processed/san_diego \
    --split test \
    --output results/

# Should generate:
# - results/metrics.json
# - results/calibration/
# - results/attention_samples/
# - results/predictions.csv
```

---

### Phase 6: Inference Pipeline
**Goal**: Production-ready inference for state-wide data
**Estimated time**: 1 week

#### 6.1 Single-Sample Predictor
- [ ] Implement `src/inference/predictor.py`
  - [ ] `CliffCastPredictor` class
  - [ ] Load from checkpoint
  - [ ] Preprocess single transect
  - [ ] Run inference
  - [ ] Return predictions + attention
- [ ] Write unit tests

**Test checkpoint**:
```python
from src.inference.predictor import CliffCastPredictor

predictor = CliffCastPredictor.from_checkpoint('checkpoints/best.pt')

result = predictor.predict(
    transect=my_transect,
    wave_data=wave_history,
    precip_data=precip_history,
)

assert 'risk_index' in result
assert 'retreat_m' in result
assert 'p_collapse' in result
assert 'failure_mode' in result
assert 'attention' in result
print(f"Risk: {result['risk_index']:.3f}")
```

#### 6.2 Batch Processor
- [ ] Implement `src/inference/batch_processor.py`
  - [ ] Memory-efficient batch iteration
  - [ ] Parallel data loading
  - [ ] Progress tracking
  - [ ] Checkpoint/resume for long runs
  - [ ] GPU memory management
- [ ] Test on medium dataset

**Test checkpoint**:
```python
from src.inference.batch_processor import BatchProcessor

processor = BatchProcessor(
    checkpoint='checkpoints/best.pt',
    batch_size=64,
    num_workers=4,
)

results = processor.process(
    transect_dir='data/statewide/transects/',
    wave_dir='data/statewide/waves/',
    precip_dir='data/statewide/precip/',
    output_path='results/statewide_predictions.parquet',
    resume=True,  # Resume if interrupted
)

print(f"Processed {len(results)} transects")
# Should process 10,000+ transects in reasonable time
```

#### 6.3 Output Formatters
- [ ] Implement output exporters:
  - [ ] GeoJSON with predictions per transect
  - [ ] CSV for tabular analysis
  - [ ] GeoTIFF raster (gridded predictions)
- [ ] Handle coordinate transformations

**Test checkpoint**:
```python
from src.output.formatters import to_geojson, to_csv

# Export to GeoJSON
to_geojson(
    predictions=results,
    output_path='results/predictions.geojson',
    crs='EPSG:4326'
)

# Verify valid GeoJSON
import geopandas as gpd
gdf = gpd.read_file('results/predictions.geojson')
assert len(gdf) == len(results)
assert 'risk_index' in gdf.columns
print("GeoJSON export successful!")
```

#### 6.4 Main Prediction Script
- [ ] Implement `predict.py`
  - [ ] CLI with argparse
  - [ ] Config loading
  - [ ] Input validation
  - [ ] Batch processing orchestration
  - [ ] Output format selection
  - [ ] Logging and error handling

**Test checkpoint**:
```bash
python predict.py \
    --input data/san_diego_county/ \
    --checkpoint checkpoints/best.pt \
    --output results/san_diego/ \
    --format geojson \
    --batch_size 64

# Verify outputs
ls results/san_diego/
# predictions.geojson
# predictions.csv
# metadata.json
```

---

### Phase 7: Documentation & Polish
**Goal**: Publication-ready code and documentation
**Estimated time**: 3-5 days

#### 7.1 README
- [ ] Project overview and motivation
- [ ] Installation instructions (conda/pip)
- [ ] Quick start guide
- [ ] Data format specifications
- [ ] Training instructions
- [ ] Inference instructions
- [ ] Citation information

#### 7.2 API Documentation
- [ ] Docstrings for all public functions/classes
- [ ] Type hints throughout codebase
- [ ] Generate docs with MkDocs or Sphinx
- [ ] Host on GitHub Pages

#### 7.3 Example Notebooks
- [ ] `01_data_exploration.ipynb` - Visualize transects, environmental data
- [ ] `02_model_debugging.ipynb` - Inspect model internals, gradients
- [ ] `03_attention_analysis.ipynb` - Interpret attention patterns
- [ ] `04_evaluation_report.ipynb` - Full evaluation with plots

#### 7.4 Testing & CI
- [ ] Achieve >80% test coverage
- [ ] Set up pytest with coverage
- [ ] GitHub Actions workflow
  - [ ] Run tests on push
  - [ ] Type checking with mypy
  - [ ] Linting with ruff

**Test checkpoint**:
```bash
pytest tests/ --cov=src --cov-report=html
# Coverage should be >80%

# CI should pass on GitHub
```

---

## Future: Option C (3D Context Enhancement)

When the transect-only approach is insufficient for complex geometries.

### Architecture Extension

```python
class Context3DExtractor(nn.Module):
    """
    Extract 3D neighborhood features for each transect point.
    
    For each point on the transect, find k-nearest neighbors in the
    full point cloud and aggregate their features.
    """
    
    def __init__(self, k_neighbors: int = 32, d_context: int = 64):
        super().__init__()
        self.k = k_neighbors
        
        # Mini-PointNet to aggregate neighbor features
        self.mini_pointnet = nn.Sequential(
            nn.Linear(6, 64),   # xyz + normals (or rgb)
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, d_context),
        )
        
    def forward(
        self,
        transect_xyz: torch.Tensor,     # (B, N, 3)
        point_cloud: torch.Tensor,       # (B, M, 6) - full point cloud
    ) -> torch.Tensor:
        """
        Returns context features (B, N, d_context) for each transect point.
        """
        B, N, _ = transect_xyz.shape
        
        # Find k-nearest neighbors for each transect point
        # (requires torch_cluster or custom CUDA kernel)
        dists = torch.cdist(transect_xyz, point_cloud[:, :, :3])  # (B, N, M)
        _, indices = dists.topk(self.k, dim=-1, largest=False)    # (B, N, k)
        
        # Gather neighbor features
        neighbors = torch.gather(
            point_cloud.unsqueeze(1).expand(-1, N, -1, -1),
            dim=2,
            index=indices.unsqueeze(-1).expand(-1, -1, -1, 6)
        )  # (B, N, k, 6)
        
        # Convert to relative coordinates
        relative = neighbors.clone()
        relative[:, :, :, :3] -= transect_xyz.unsqueeze(2)
        
        # Aggregate with mini-PointNet (max pool)
        features = self.mini_pointnet(relative)  # (B, N, k, d_context)
        context = features.max(dim=2).values      # (B, N, d_context)
        
        return context
```

### Integration

```python
# In CliffCast.__init__:
if config.get('use_3d_context', False):
    self.context_3d = Context3DExtractor(
        k_neighbors=config.get('k_neighbors', 32),
        d_context=config.get('d_context', 64)
    )
    # Expand input projection to handle additional features
    self.transect_encoder.point_embed = nn.Linear(
        config['n_point_features'] + config['d_context'],
        config['d_model']
    )

# In CliffCast.forward:
if hasattr(self, 'context_3d'):
    context = self.context_3d(transect_xyz, point_cloud)
    transect_points = torch.cat([transect_points, context], dim=-1)
```

### When to Enable

- Coastal cliffs with caves, sea arches, or complex notching
- High alongshore variability in failure patterns
- Wide transect spacing relative to failure zone width
- When 2D transect representation loses critical information

---

## Success Criteria

### Minimum Viable Product (MVP)
- [ ] Model trains without errors on San Diego LiDAR data
- [ ] Risk index R² > 0.30 on held-out test set
- [ ] Collapse probability (1yr horizon) AUC-ROC > 0.70
- [ ] Expected retreat MAE < 1.0 m/yr
- [ ] Inference throughput > 10,000 transects/hour on single GPU
- [ ] Attention maps show physically plausible patterns

### Target Performance
- [ ] Risk index R² > 0.50
- [ ] Collapse probability AUC-ROC > 0.85 (all horizons)
- [ ] Expected retreat MAE < 0.5 m/yr
- [ ] Failure mode accuracy > 70% (on failure samples)
- [ ] Outperforms all baselines on all primary metrics
- [ ] Well-calibrated probabilities (ECE < 0.1)

### Stretch Goals
- [ ] Real-time inference capability (<100ms per transect)
- [ ] Option C (3D context) integration
- [ ] Transfer learning to other coastlines (Oregon, Malibu)
- [ ] Published peer-reviewed paper
- [ ] Web dashboard for predictions

---

## Dependencies

```
# requirements.txt

# Core ML
torch>=2.0
numpy>=1.24
scipy>=1.10
einops>=0.7

# Data handling
pandas>=2.0
xarray>=2023.1
netCDF4>=1.6
h5py>=3.8

# Point cloud
laspy>=2.5
# pdal>=3.0  # Optional, for complex processing

# Geospatial
rasterio>=1.3
geopandas>=0.14
shapely>=2.0
pyproj>=3.5

# Config & logging
pyyaml>=6.0
omegaconf>=2.3
wandb>=0.16

# Visualization
matplotlib>=3.7
seaborn>=0.13
plotly>=5.18

# ML utilities
scikit-learn>=1.3
tqdm>=4.65

# Testing
pytest>=7.4
pytest-cov>=4.1

# Code quality
black>=23.0
ruff>=0.1
mypy>=1.7
```

---

## Notes for Claude Code CLI

1. **Complete Phase 1 fully before Phase 2**. Data pipeline bugs surface late and are painful to debug.

2. **Test shapes obsessively**. Add `assert tensor.shape == expected` everywhere. Transformer bugs are almost always shape mismatches.

3. **Use `einops`** for tensor ops: `rearrange(x, 'b n d -> b d n')` >> `x.permute(0, 2, 1)`

4. **Start small**: Debug with `d_model=32, n_layers=1`. Scale up only after correctness verified.

5. **Synthetic data first**: Create toy data with known `risk = f(slope, waves)`. Verify model learns before using real data.

6. **Visualize attention early**: Don't wait for Phase 5. Attention plots during training catch bugs.

7. **Log everything to W&B**: Loss curves, learning rates, gradient norms, sample predictions.

8. **Memory profile early**: State-wide inference is memory-bound. Use `torch.cuda.memory_stats()`.

9. **Checkpoint frequently**: Name checkpoints with config hash + epoch + val_loss.

10. **One head at a time**: Get risk working perfectly before adding retreat, then collapse, then mode.
