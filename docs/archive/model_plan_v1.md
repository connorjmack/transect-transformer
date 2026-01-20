# CliffCast Model Plan: Event-Supervised Coastal Erosion Prediction

## Executive Summary

CliffCast is a transformer-based deep learning model that predicts coastal cliff erosion by learning from multi-temporal cliff geometry (LiDAR transects across epochs), environmental forcing (waves, precipitation), and **supervised event labels** (observed volume losses from M3C2 change detection).

### Core Innovations

1. **Spatio-temporal attention**: Learns cliff evolution patterns across multiple LiDAR epochs — progressive weakening, crack development, and precursor deformation are highly predictive of future failures.

2. **Event-supervised training**: Uses real observed erosion events with measured volumes as ground truth, not just derived proxies.

3. **M3C2 change integration**: Incorporates point-level change distances as input features, giving the model direct access to recent erosion patterns.

4. **Hybrid labeling**: Prioritizes observed events where available, falls back to LiDAR-derived labels elsewhere, with confidence weighting.

### Key Design Principle: Predict the Future from the Past

```
Timeline:  ──[Epoch 0]────[Epoch 1]────[Epoch 2]────[Epoch 3]────[Epoch 4]──
                │                                    │            │
                └──────── MODEL INPUT ───────────────┘            │
                         (context epochs)                         │
                                                                  ▼
                                                          MODEL TARGET
                                                     (events between E3→E4)
```

The model **never sees the target epoch** in its input. This enforces true prediction rather than reconstruction.

---

## Table of Contents

1. [Study Site & Coordinate Systems](#study-site--coordinate-systems)
2. [Data Specifications](#data-specifications)
3. [Architecture](#architecture)
4. [Event Integration Pipeline](#event-integration-pipeline)
5. [Training Pipeline](#training-pipeline)
6. [Loss Functions](#loss-functions)
7. [Evaluation Strategy](#evaluation-strategy)
8. [Implementation Phases](#implementation-phases)
9. [Success Metrics](#success-metrics)

---

## Study Site & Coordinate Systems

### Beach Definitions

| Beach | MOP Range | Local Coord Origin | Alongshore Span |
|-------|-----------|-------------------|-----------------|
| **Blacks** | 520-567 | MOP 520 (south) | ~4700m |
| **Torrey** | 567-581 | MOP 567 (south) | ~1400m |
| **Del Mar** | 595-620 | MOP 595 (south) | ~2500m |
| **Solana** | 637-666 | MOP 637 (south) | ~2900m |
| **San Elijo** | 683-708 | MOP 683 (south) | ~2500m |
| **Encinitas** | 708-764 | MOP 708 (south) | ~5600m |

### Coordinate System Mapping

Event CSVs use local alongshore coordinates (meters from southern end of each beach). Transects are at 1m spacing within the cube.

```python
def alongshore_to_transect_idx(alongshore_m: float, beach: str, cube: dict) -> int:
    """
    Convert local alongshore coordinate to transect index in unified cube.
    
    Args:
        alongshore_m: Distance from southern end of beach (meters)
        beach: Beach name (lowercase)
        cube: Loaded cube with 'beach_slices' and 'mop_ids'
    
    Returns:
        Transect index in the unified cube
    """
    # Beach origins (MOP at local coord 0)
    beach_origins = {
        'blacks': 520,
        'torrey': 567,
        'delmar': 595,
        'solana': 637,
        'sanelijo': 683,
        'encinitas': 708,
    }
    
    # MOP spacing is 100m, transects within are at 1m or 10m spacing
    # For 1m transects: transect_offset = alongshore_m
    # For 10m transects: transect_offset = alongshore_m / 10
    
    beach_start_idx, beach_end_idx = cube['beach_slices'][beach]
    n_transects_in_beach = beach_end_idx - beach_start_idx
    
    # Assuming 1m spacing within beach
    transect_offset = int(round(alongshore_m))
    transect_offset = np.clip(transect_offset, 0, n_transects_in_beach - 1)
    
    return beach_start_idx + transect_offset


def transect_idx_to_mop(transect_idx: int, cube: dict) -> int:
    """Convert transect index to approximate MOP ID."""
    return cube['mop_ids'][transect_idx]
```

---

## Data Specifications

### Transect Cube Format (Updated with M3C2)

```python
# Unified cube structure: (n_transects, n_epochs, N, n_features)
# Stored in: data/processed/unified_cube.npz

cube = {
    # === POINT DATA ===
    'points': (n_transects, n_epochs, 128, 13),  # 13 features now (was 12)
    'distances': (n_transects, n_epochs, 128),    # Distance along transect
    
    # === METADATA ===
    'metadata': (n_transects, n_epochs, 12),      # Per-epoch transect metadata
    'timestamps': (n_epochs,),                    # Ordinal dates (shared)
    'coverage_mask': (n_transects, n_epochs),     # Boolean: data exists
    
    # === IDENTIFIERS ===
    'transect_ids': (n_transects,),               # Full ID strings
    'mop_ids': (n_transects,),                    # Integer MOP IDs
    'beach_slices': dict,                         # {'delmar': (start, end), ...}
    
    # === EPOCH INFO ===
    'epoch_files': (n_epochs,),                   # Original LAS filenames
    'epoch_dates': (n_epochs,),                   # ISO date strings
    
    # === FEATURE NAMES ===
    'feature_names': [...],                       # 13 feature names
    'metadata_names': [...],                      # 12 metadata field names
}
```

### Per-Point Features (13 total)

| Index | Feature | Description | Source |
|-------|---------|-------------|--------|
| 0 | `distance_m` | Distance from transect start | Geometry |
| 1 | `elevation_m` | Elevation (NAVD88) | LAS Z |
| 2 | `slope_deg` | Local slope | Derived |
| 3 | `curvature` | Profile curvature (1/m) | Derived |
| 4 | `roughness` | Surface roughness (std) | Derived |
| 5 | `intensity` | LAS intensity [0,1] | LAS |
| 6 | `red` | Red channel [0,1] | LAS RGB |
| 7 | `green` | Green channel [0,1] | LAS RGB |
| 8 | `blue` | Blue channel [0,1] | LAS RGB |
| 9 | `classification` | LAS class code | LAS |
| 10 | `return_number` | Return number | LAS |
| 11 | `num_returns` | Total returns | LAS |
| **12** | **`m3c2_distance`** | **Change distance to previous epoch (m)** | **M3C2** |

#### M3C2 Distance Feature

The `m3c2_distance` field contains the signed distance to the previous epoch's surface:
- **Negative values**: Material loss (erosion/retreat)
- **Positive values**: Material gain (deposition/error)
- **Zero/NaN**: No change or no previous epoch

```python
# M3C2 distance handling
def normalize_m3c2(m3c2_raw: np.ndarray) -> np.ndarray:
    """
    Normalize M3C2 distances for model input.
    
    Erosion events typically range from -0.1m to -5m.
    We clip and scale to [-1, 1] range.
    """
    # Clip extreme values
    clipped = np.clip(m3c2_raw, -5.0, 2.0)
    
    # Scale to roughly [-1, 1]
    # Erosion (-5m) → -1, Deposition (+2m) → +0.4
    normalized = clipped / 5.0
    
    # Handle NaN (first epoch has no M3C2)
    normalized = np.nan_to_num(normalized, nan=0.0)
    
    return normalized
```

**First Epoch Handling**: The first epoch in any sequence has no previous surface to compare against. Set `m3c2_distance = 0` for all points in the first epoch.

### Transect-Level Metadata (12 fields)

| Index | Field | Description |
|-------|-------|-------------|
| 0 | `cliff_height_m` | Total cliff height |
| 1 | `mean_slope_deg` | Average cliff face slope |
| 2 | `max_slope_deg` | Maximum slope (overhang indicator) |
| 3 | `toe_elevation_m` | Transect start elevation |
| 4 | `top_elevation_m` | Transect end elevation |
| 5 | `orientation_deg` | Transect azimuth from N |
| 6 | `transect_length_m` | Total transect length |
| 7 | `latitude` | Transect midpoint Y |
| 8 | `longitude` | Transect midpoint X |
| 9 | `transect_id` | Numeric ID |
| 10 | `mean_intensity` | Mean LAS intensity |
| 11 | `dominant_class` | Most common LAS class |

### Environmental Features

#### Wave Features (CDIP MOP System)

```python
# Shape: (n_samples, T_w, 4) where T_w = 360 (90 days @ 6hr)
wave_features = {
    0: 'hs',      # Significant wave height (m)
    1: 'tp',      # Peak period (s)
    2: 'dp',      # Peak direction (deg from N)
    3: 'power',   # Wave power flux (kW/m)
}

# Day-of-year for seasonality encoding
wave_doy: (n_samples, T_w)  # 1-366
```

#### Atmospheric Features (PRISM + Derived)

```python
# Shape: (n_samples, T_a, 24) where T_a = 90 (90 days @ daily)
atmos_features = {
    # Raw PRISM
    0: 'precip_mm',           # Daily precipitation
    1: 'tmax_c',              # Max temperature
    2: 'tmin_c',              # Min temperature
    3: 'tmean_c',             # Mean temperature
    4: 'vpdmax_hpa',          # Max vapor pressure deficit
    5: 'vpdmin_hpa',          # Min vapor pressure deficit
    
    # Derived features
    6: 'precip_cumsum_7d',    # 7-day cumulative precip
    7: 'precip_cumsum_30d',   # 30-day cumulative precip
    8: 'api',                 # Antecedent Precipitation Index
    9: 'wet_days_7d',         # Wet days in past 7 days
    10: 'wet_days_30d',       # Wet days in past 30 days
    11: 'dry_spell_days',     # Days since last rain
    12: 'vpd_mean',           # Mean VPD
    13: 'freeze_thaw',        # Freeze-thaw cycles (binary)
    # ... additional derived features up to 24
}

# Day-of-year for seasonality encoding
atmos_doy: (n_samples, T_a)  # 1-366
```

---

## Architecture

### High-Level Overview

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    CLIFFCAST ARCHITECTURE (v2 with M3C2)                          │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  INPUTS                                                                           │
│  ──────                                                                           │
│  ┌────────────────────┐   ┌─────────────┐   ┌─────────────┐                      │
│  │  Transect Cube     │   │    Wave     │   │   Atmos     │                      │
│  │  (T_ctx, 128, 13)  │   │  (360, 4)   │   │   (90, 24)  │                      │
│  │                    │   │             │   │             │                      │
│  │  includes M3C2     │   │  90 days    │   │  90 days    │                      │
│  │  distances!        │   │  @ 6hr      │   │  @ daily    │                      │
│  └──────────┬─────────┘   └──────┬──────┘   └──────┬──────┘                      │
│             │                    │                  │                             │
│  ENCODERS   │                    │                  │                             │
│  ────────   │                    │                  │                             │
│             ▼                    │                  │                             │
│  ┌────────────────────┐          │                  │                             │
│  │ Spatio-Temporal    │          │                  │                             │
│  │ Transect Encoder   │          │                  │                             │
│  │                    │          │                  │                             │
│  │ 1. Spatial Attn    │          │                  │                             │
│  │    (per epoch)     │          │                  │                             │
│  │ 2. Temporal Attn   │          │                  │                             │
│  │    (across epochs) │          │                  │                             │
│  │                    │          │                  │                             │
│  │ Output: (T, d)     │          ▼                  ▼                             │
│  └──────────┬─────────┘   ┌─────────────┐   ┌─────────────┐                      │
│             │             │    Wave     │   │   Atmos     │                      │
│             │             │   Encoder   │   │   Encoder   │                      │
│             │             │  (360, d)   │   │   (90, d)   │                      │
│             │             └──────┬──────┘   └──────┬──────┘                      │
│             │                    │                  │                             │
│             │                    └────────┬─────────┘                             │
│             │                             │                                       │
│             │                    ┌────────▼────────┐                              │
│             │                    │   Concatenate   │                              │
│             │                    │  Environmental  │                              │
│             │                    │    (450, d)     │                              │
│             │                    └────────┬────────┘                              │
│             │                             │                                       │
│  FUSION     │                             │                                       │
│  ──────     └─────────────┬───────────────┘                                       │
│                           │                                                       │
│                  ┌────────▼────────┐                                              │
│                  │  Cross-Attention │                                             │
│                  │      Fusion      │                                             │
│                  │                  │                                             │
│                  │  Q: cliff (T,d)  │                                             │
│                  │  K,V: env (450,d)│                                             │
│                  └────────┬────────┘                                              │
│                           │                                                       │
│                  ┌────────▼────────┐                                              │
│                  │  Global Pooling │                                              │
│                  │   CLS token     │                                              │
│                  │     (d,)        │                                              │
│                  └────────┬────────┘                                              │
│                           │                                                       │
│  PREDICTION HEADS         │                                                       │
│  ────────────────         │                                                       │
│         ┌─────────────────┼─────────────────┬─────────────────┐                  │
│         ▼                 ▼                 ▼                 ▼                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│  │   Volume    │   │    Event    │   │    Risk     │   │  Collapse   │          │
│  │ Regression  │   │   Class     │   │    Index    │   │Probability  │          │
│  │             │   │             │   │             │   │             │          │
│  │ log(vol+1)  │   │  4 classes  │   │   [0, 1]    │   │  4 horizons │          │
│  │  softplus   │   │  softmax    │   │   sigmoid   │   │   sigmoid   │          │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘          │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Model Dimensions

| Parameter | Value | Notes |
|-----------|-------|-------|
| `d_model` | 256 | Hidden dimension |
| `n_heads` | 8 | Attention heads |
| `n_spatial_layers` | 4 | Spatial attention depth |
| `n_temporal_layers` | 2 | Temporal attention depth |
| `n_env_layers` | 3 | Environmental encoder depth |
| `n_fusion_layers` | 2 | Cross-attention fusion depth |
| `max_context_epochs` | 10 | Maximum context length |
| `n_points` | 128 | Points per transect |
| `dropout` | 0.1 | Regularization |

### Spatio-Temporal Transect Encoder

```python
class SpatioTemporalTransectEncoder(nn.Module):
    """
    Hierarchical attention over multi-epoch cliff geometry.
    
    Key features:
    - Distance-based sinusoidal positional encoding (spatial)
    - Learned positional encoding (temporal)
    - M3C2 distances as explicit input features
    - Returns per-epoch embeddings + pooled representation
    """
    
    def __init__(
        self,
        n_point_features: int = 13,   # Now 13 with M3C2
        n_meta_features: int = 12,
        d_model: int = 256,
        n_heads: int = 8,
        n_spatial_layers: int = 4,
        n_temporal_layers: int = 2,
        max_epochs: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        # ... (implementation as in CLAUDE.md)
    
    def forward(
        self,
        point_features: torch.Tensor,   # (B, T_ctx, 128, 13)
        distances: torch.Tensor,         # (B, T_ctx, 128)
        metadata: torch.Tensor,          # (B, T_ctx, 12)
        context_mask: torch.Tensor,      # (B, T_ctx) - valid epochs
        return_attention: bool = False,
    ) -> dict:
        """
        Returns:
            temporal_embeddings: (B, T_ctx, d_model)
            pooled: (B, d_model)
            attention: dict (optional)
        """
        # Spatial attention processes each epoch independently
        # M3C2 distances in point_features[:, :, :, 12] inform the model
        # about recent change at each point
        pass
```

### Prediction Heads (Updated)

| Head | Output Shape | Activation | Target Source |
|------|--------------|------------|---------------|
| **Volume** | (B,) | Softplus on log(vol+1) | Event CSVs |
| **Event Class** | (B, 4) | Softmax | Event CSVs (thresholded) |
| **Risk Index** | (B,) | Sigmoid | Computed from volume + height |
| **Collapse Prob** | (B, 4) | Sigmoid per horizon | Event frequency analysis |

#### Event Classification

```python
# Event classes based on total volume in prediction window
# maybe update to be large upper cliff, cliff toe, small rockfall, beach change, construction
EVENT_CLASSES = {
    0: 'stable',      # volume < 10 m³
    1: 'minor',       # 10 ≤ volume < 50 m³
    2: 'major',       # 50 ≤ volume < 200 m³
    3: 'failure',     # volume ≥ 200 m³
}

def volume_to_class(total_volume: float) -> int:
    """Convert total volume to event class."""
    if total_volume < 10:
        return 0
    elif total_volume < 50:
        return 1
    elif total_volume < 200:
        return 2
    else:
        return 3
```

---

## Event Integration Pipeline

### Event Data Structure

```python
# Event CSV columns (per beach)
event_columns = {
    'mid_date': datetime,           # Event midpoint date
    'start_date': datetime,         # Detection window start
    'end_date': datetime,           # Detection window end
    'volume': float,                # Eroded volume (m³)
    'vol_unc': float,               # Volume uncertainty (m³)
    'elevation': float,             # Centroid elevation (m)
    'alongshore_centroid_m': float, # Local alongshore coordinate
    'alongshore_start_m': float,    # Event extent start
    'alongshore_end_m': float,      # Event extent end
    'width': float,                 # Alongshore width (m)
    'height': float,                # Vertical extent (m)
    'month': int,                   # Month of year
}
```

### Event-to-Transect Alignment

```python
def align_events_to_cube(
    events_df: pd.DataFrame,
    cube: dict,
    beach: str,
) -> pd.DataFrame:
    """
    Map events from local coordinates to transect indices and epoch pairs.
    
    Args:
        events_df: Event CSV loaded as DataFrame
        cube: Unified transect cube
        beach: Beach name (lowercase)
    
    Returns:
        DataFrame with added columns: transect_idx, epoch_before, epoch_after
    """
    events = events_df.copy()
    
    # 1. Parse dates
    events['start_date'] = pd.to_datetime(events['start_date'])
    events['end_date'] = pd.to_datetime(events['end_date'])
    events['mid_date'] = pd.to_datetime(events['mid_date'])
    
    # 2. Map alongshore coordinate to transect index
    beach_start, beach_end = cube['beach_slices'][beach]
    n_transects = beach_end - beach_start
    
    def alongshore_to_idx(alongshore_m):
        # 1m transect spacing
        offset = int(round(alongshore_m))
        offset = np.clip(offset, 0, n_transects - 1)
        return beach_start + offset
    
    events['transect_idx'] = events['alongshore_centroid_m'].apply(alongshore_to_idx)
    
    # 3. Find bracketing epochs
    timestamps = cube['timestamps']  # Ordinal dates
    
    def find_epochs(row):
        t_idx = row['transect_idx']
        event_date = row['mid_date'].toordinal()
        
        # Get valid epochs for this transect
        valid = np.where(cube['coverage_mask'][t_idx])[0]
        if len(valid) < 2:
            return pd.Series([np.nan, np.nan])
        
        valid_times = timestamps[valid]
        
        # Find bracketing epochs
        before_mask = valid_times < event_date
        after_mask = valid_times > event_date
        
        if not before_mask.any() or not after_mask.any():
            return pd.Series([np.nan, np.nan])
        
        epoch_before = valid[np.where(before_mask)[0][-1]]
        epoch_after = valid[np.where(after_mask)[0][0]]
        
        return pd.Series([epoch_before, epoch_after])
    
    events[['epoch_before', 'epoch_after']] = events.apply(find_epochs, axis=1)
    
    # 4. Drop events that couldn't be aligned
    events = events.dropna(subset=['epoch_before', 'epoch_after'])
    events['epoch_before'] = events['epoch_before'].astype(int)
    events['epoch_after'] = events['epoch_after'].astype(int)
    
    return events


def aggregate_events_by_sample(aligned_events: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate multiple events per (transect, epoch_pair) into single labels.
    
    Some transect-epoch pairs may have multiple events (different elevations
    or alongshore positions within the same transect). Aggregate these.
    """
    grouped = aligned_events.groupby(
        ['transect_idx', 'epoch_before', 'epoch_after']
    ).agg({
        'volume': 'sum',              # Total volume
        'vol_unc': lambda x: np.sqrt((x**2).sum()),  # Propagate uncertainty
        'height': 'max',              # Maximum vertical extent
        'width': 'max',               # Maximum alongshore extent
        'mid_date': 'first',          # Representative date
    }).reset_index()
    
    grouped.columns = [
        'transect_idx', 'epoch_before', 'epoch_after',
        'total_volume', 'volume_unc', 'max_height', 'max_width', 'event_date'
    ]
    
    # Add event class
    grouped['event_class'] = grouped['total_volume'].apply(volume_to_class)
    
    # Add event count
    event_counts = aligned_events.groupby(
        ['transect_idx', 'epoch_before', 'epoch_after']
    ).size().reset_index(name='n_events')
    
    grouped = grouped.merge(event_counts, on=['transect_idx', 'epoch_before', 'epoch_after'])
    
    return grouped
```

### Hybrid Label Generation

```python
def create_training_labels(
    cube: dict,
    aggregated_events: pd.DataFrame,
    min_context_epochs: int = 3,
) -> list[dict]:
    """
    Create training samples with hybrid labeling.
    
    For each valid (transect, context_epochs, target_epoch):
    1. If observed events exist → use event volumes (high confidence)
    2. Else → derive from M3C2/LiDAR (lower confidence)
    
    Args:
        cube: Unified transect cube
        aggregated_events: Output of aggregate_events_by_sample()
        min_context_epochs: Minimum context length
    
    Returns:
        List of sample dictionaries
    """
    samples = []
    timestamps = cube['timestamps']
    
    # Index events for fast lookup
    event_index = {}
    for _, row in aggregated_events.iterrows():
        key = (row['transect_idx'], row['epoch_before'], row['epoch_after'])
        event_index[key] = row
    
    # Iterate over all transects
    n_transects = cube['coverage_mask'].shape[0]
    
    for t_idx in range(n_transects):
        valid_epochs = np.where(cube['coverage_mask'][t_idx])[0]
        
        if len(valid_epochs) < min_context_epochs + 1:
            continue
        
        # Sliding window: each position generates one sample
        for end_pos in range(min_context_epochs, len(valid_epochs)):
            context_epochs = valid_epochs[end_pos - min_context_epochs : end_pos]
            target_epoch = valid_epochs[end_pos]
            last_context_epoch = context_epochs[-1]
            
            # Look up events
            event_key = (t_idx, int(last_context_epoch), int(target_epoch))
            
            if event_key in event_index:
                # === OBSERVED EVENT ===
                event = event_index[event_key]
                
                sample = {
                    'transect_idx': t_idx,
                    'context_epochs': context_epochs.tolist(),
                    'target_epoch': int(target_epoch),
                    'prediction_window': (
                        int(timestamps[last_context_epoch]),
                        int(timestamps[target_epoch]),
                    ),
                    # Labels
                    'total_volume': float(event['total_volume']),
                    'volume_unc': float(event['volume_unc']),
                    'n_events': int(event['n_events']),
                    'event_class': int(event['event_class']),
                    'max_height': float(event['max_height']),
                    # Metadata
                    'label_source': 'observed',
                    'confidence': 1.0 - min(event['volume_unc'] / (event['total_volume'] + 1), 0.5),
                }
            else:
                # === DERIVED LABEL ===
                # Use M3C2 distances from target epoch to estimate change
                m3c2_distances = cube['points'][t_idx, target_epoch, :, 12]
                
                # Sum of negative (erosion) distances × transect width
                erosion_mask = m3c2_distances < -0.1  # Threshold noise
                if erosion_mask.any():
                    mean_erosion = np.abs(m3c2_distances[erosion_mask]).mean()
                    erosion_width = erosion_mask.sum() * (cube['metadata'][t_idx, target_epoch, 6] / 128)
                    cliff_height = cube['metadata'][t_idx, target_epoch, 0]
                    
                    # Rough volume estimate: erosion_depth × width × height
                    estimated_volume = mean_erosion * erosion_width * cliff_height * 0.5
                else:
                    estimated_volume = 0.0
                
                sample = {
                    'transect_idx': t_idx,
                    'context_epochs': context_epochs.tolist(),
                    'target_epoch': int(target_epoch),
                    'prediction_window': (
                        int(timestamps[last_context_epoch]),
                        int(timestamps[target_epoch]),
                    ),
                    # Labels (derived)
                    'total_volume': float(estimated_volume),
                    'volume_unc': float(estimated_volume * 0.5),  # 50% uncertainty
                    'n_events': 1 if estimated_volume > 10 else 0,
                    'event_class': volume_to_class(estimated_volume),
                    'max_height': float(cube['metadata'][t_idx, target_epoch, 0]),
                    # Metadata
                    'label_source': 'derived',
                    'confidence': 0.5,  # Lower confidence for derived
                }
            
            # Compute risk index
            cliff_height = cube['metadata'][t_idx, target_epoch, 0]
            sample['risk_index'] = compute_risk_index(
                sample['total_volume'], cliff_height
            )
            
            samples.append(sample)
    
    return samples


def compute_risk_index(total_volume: float, cliff_height: float) -> float:
    """
    Compute risk index from event volume and cliff height.
    
    Larger volumes and taller cliffs → higher risk.
    Sigmoid-normalized to [0, 1].
    """
    # Log-transform volume (handles wide range)
    log_vol = np.log1p(total_volume)  # log(1 + vol)
    
    # Height factor: taller cliffs have higher consequence
    height_factor = 1 + 0.05 * (cliff_height - 15)  # Centered at 15m
    height_factor = np.clip(height_factor, 0.5, 2.0)
    
    # Combined score
    score = log_vol * height_factor
    
    # Sigmoid normalization
    # Calibrated so: vol=10m³ → ~0.3, vol=50m³ → ~0.5, vol=200m³ → ~0.75
    risk = 1 / (1 + np.exp(-0.5 * (score - 4)))
    
    return float(np.clip(risk, 0, 1))
```

---

## Training Pipeline

### Sample Generation Workflow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING DATA PREPARATION                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  INPUTS                                                                          │
│  ──────                                                                          │
│  ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐          │
│  │   Unified Cube    │   │    Event CSVs     │   │   Wave/Atmos      │          │
│  │   (all beaches)   │   │   (per beach)     │   │     Loaders       │          │
│  └─────────┬─────────┘   └─────────┬─────────┘   └─────────┬─────────┘          │
│            │                       │                       │                     │
│            ▼                       ▼                       │                     │
│  ┌──────────────────────────────────────────┐              │                     │
│  │     STEP 1: Align Events to Cube         │              │                     │
│  │                                          │              │                     │
│  │  For each beach:                         │              │                     │
│  │  1. Load event CSV                       │              │                     │
│  │  2. Map alongshore → transect_idx        │              │                     │
│  │  3. Find bracketing epochs               │              │                     │
│  │  4. Aggregate per (transect, epoch_pair) │              │                     │
│  └────────────────────┬─────────────────────┘              │                     │
│                       │                                    │                     │
│                       ▼                                    │                     │
│  ┌──────────────────────────────────────────┐              │                     │
│  │     STEP 2: Generate Training Samples    │              │                     │
│  │                                          │              │                     │
│  │  For each transect with ≥4 valid epochs: │              │                     │
│  │  - Sliding window (context=3, target=1)  │              │                     │
│  │  - Hybrid labeling (event priority)      │              │                     │
│  │  - Track label source & confidence       │              │                     │
│  └────────────────────┬─────────────────────┘              │                     │
│                       │                                    │                     │
│                       ▼                                    ▼                     │
│  ┌─────────────────────────────────────────────────────────────────────┐        │
│  │     STEP 3: Load Environmental Features                              │        │
│  │                                                                      │        │
│  │  For each sample:                                                    │        │
│  │  - Wave: 90 days before target_epoch from CDIP                       │        │
│  │  - Atmos: 90 days before target_epoch from PRISM                     │        │
│  │  - Handle missing data (interpolate or flag)                         │        │
│  └────────────────────┬────────────────────────────────────────────────┘        │
│                       │                                                          │
│                       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐        │
│  │     STEP 4: Create Training Tensors                                  │        │
│  │                                                                      │        │
│  │  Output: training_data.npz                                           │        │
│  │  ├── point_features: (N, T_ctx, 128, 13)                            │        │
│  │  ├── metadata: (N, T_ctx, 12)                                       │        │
│  │  ├── distances: (N, T_ctx, 128)                                     │        │
│  │  ├── context_mask: (N, T_ctx)                                       │        │
│  │  ├── wave_features: (N, 360, 4)                                     │        │
│  │  ├── wave_doy: (N, 360)                                             │        │
│  │  ├── atmos_features: (N, 90, 24)                                    │        │
│  │  ├── atmos_doy: (N, 90)                                             │        │
│  │  ├── total_volume: (N,)                                             │        │
│  │  ├── event_class: (N,)                                              │        │
│  │  ├── risk_index: (N,)                                               │        │
│  │  ├── label_source: (N,)  # 0=derived, 1=observed                    │        │
│  │  ├── confidence: (N,)                                               │        │
│  │  └── sample_info: transect_idx, target_epoch, beach, mop_id         │        │
│  └─────────────────────────────────────────────────────────────────────┘        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Dataset Class

```python
class CliffCastDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for CliffCast training.
    
    Loads pre-processed training data and provides samples
    with proper padding and masking.
    """
    
    def __init__(
        self,
        data_path: str,
        max_context_epochs: int = 10,
        transform: callable = None,
    ):
        data = np.load(data_path, allow_pickle=True)
        
        self.point_features = data['point_features']
        self.metadata = data['metadata']
        self.distances = data['distances']
        self.context_mask = data['context_mask']
        self.wave_features = data['wave_features']
        self.wave_doy = data['wave_doy']
        self.atmos_features = data['atmos_features']
        self.atmos_doy = data['atmos_doy']
        
        # Labels
        self.total_volume = data['total_volume']
        self.event_class = data['event_class']
        self.risk_index = data['risk_index']
        self.label_source = data['label_source']
        self.confidence = data['confidence']
        
        # Info
        self.sample_info = data['sample_info'].item()
        
        self.max_context_epochs = max_context_epochs
        self.transform = transform
    
    def __len__(self):
        return len(self.total_volume)
    
    def __getitem__(self, idx):
        # Get context length for this sample
        ctx_len = self.context_mask[idx].sum()
        
        # Pad to max_context_epochs if needed
        point_feat = self._pad_temporal(self.point_features[idx], ctx_len)
        meta = self._pad_temporal(self.metadata[idx], ctx_len)
        dist = self._pad_temporal(self.distances[idx], ctx_len)
        mask = self._pad_mask(self.context_mask[idx], ctx_len)
        
        sample = {
            # Inputs
            'point_features': torch.tensor(point_feat, dtype=torch.float32),
            'metadata': torch.tensor(meta, dtype=torch.float32),
            'distances': torch.tensor(dist, dtype=torch.float32),
            'context_mask': torch.tensor(mask, dtype=torch.bool),
            'wave_features': torch.tensor(self.wave_features[idx], dtype=torch.float32),
            'wave_doy': torch.tensor(self.wave_doy[idx], dtype=torch.long),
            'atmos_features': torch.tensor(self.atmos_features[idx], dtype=torch.float32),
            'atmos_doy': torch.tensor(self.atmos_doy[idx], dtype=torch.long),
            
            # Labels
            'total_volume': torch.tensor(self.total_volume[idx], dtype=torch.float32),
            'event_class': torch.tensor(self.event_class[idx], dtype=torch.long),
            'risk_index': torch.tensor(self.risk_index[idx], dtype=torch.float32),
            'confidence': torch.tensor(self.confidence[idx], dtype=torch.float32),
            'label_source': torch.tensor(self.label_source[idx], dtype=torch.long),
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _pad_temporal(self, arr, valid_len):
        """Pad temporal dimension to max_context_epochs."""
        if len(arr) >= self.max_context_epochs:
            return arr[:self.max_context_epochs]
        
        pad_shape = list(arr.shape)
        pad_shape[0] = self.max_context_epochs - len(arr)
        padding = np.zeros(pad_shape, dtype=arr.dtype)
        
        return np.concatenate([arr, padding], axis=0)
    
    def _pad_mask(self, mask, valid_len):
        """Pad context mask."""
        if len(mask) >= self.max_context_epochs:
            return mask[:self.max_context_epochs]
        
        padding = np.zeros(self.max_context_epochs - len(mask), dtype=bool)
        return np.concatenate([mask, padding])
```

### Training Loop

```python
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> dict:
    """
    Train for one epoch.
    
    Returns dict of loss components for logging.
    """
    model.train()
    
    losses = defaultdict(float)
    n_batches = 0
    
    for batch in tqdm(dataloader, desc=f'Epoch {epoch}'):
        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(
            point_features=batch['point_features'],
            metadata=batch['metadata'],
            distances=batch['distances'],
            context_mask=batch['context_mask'],
            wave_features=batch['wave_features'],
            wave_doy=batch['wave_doy'],
            atmos_features=batch['atmos_features'],
            atmos_doy=batch['atmos_doy'],
        )
        
        # Compute losses
        loss_dict = criterion(
            outputs=outputs,
            targets={
                'total_volume': batch['total_volume'],
                'event_class': batch['event_class'],
                'risk_index': batch['risk_index'],
            },
            confidence=batch['confidence'],
            label_source=batch['label_source'],
        )
        
        total_loss = loss_dict['total']
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate losses
        for k, v in loss_dict.items():
            losses[k] += v.item()
        n_batches += 1
    
    # Average losses
    return {k: v / n_batches for k, v in losses.items()}
```

---

## Loss Functions

### Multi-Task Loss with Confidence Weighting

```python
class CliffCastLoss(nn.Module):
    """
    Multi-task loss for CliffCast.
    
    Components:
    - Volume regression (log-transformed)
    - Event classification
    - Risk index regression
    
    Features:
    - Per-sample confidence weighting
    - Label source awareness (observed vs derived)
    - Phased head enabling
    """
    
    def __init__(
        self,
        volume_weight: float = 1.0,
        class_weight: float = 1.0,
        risk_weight: float = 1.0,
        use_confidence_weighting: bool = True,
        observed_boost: float = 1.5,  # Boost observed labels
    ):
        super().__init__()
        
        self.volume_weight = volume_weight
        self.class_weight = class_weight
        self.risk_weight = risk_weight
        self.use_confidence_weighting = use_confidence_weighting
        self.observed_boost = observed_boost
        
        # Loss functions
        self.volume_loss = nn.SmoothL1Loss(reduction='none')
        self.class_loss = nn.CrossEntropyLoss(reduction='none')
        self.risk_loss = nn.SmoothL1Loss(reduction='none')
    
    def forward(
        self,
        outputs: dict,
        targets: dict,
        confidence: torch.Tensor,
        label_source: torch.Tensor,
    ) -> dict:
        """
        Compute weighted multi-task loss.
        
        Args:
            outputs: Model outputs dict with keys:
                - volume_pred: (B,) log(volume + 1) predictions
                - class_logits: (B, 4) event class logits
                - risk_pred: (B,) risk index predictions
            targets: Target dict with keys:
                - total_volume: (B,)
                - event_class: (B,)
                - risk_index: (B,)
            confidence: (B,) per-sample confidence weights
            label_source: (B,) 0=derived, 1=observed
        
        Returns:
            Dict with loss components and total
        """
        batch_size = confidence.shape[0]
        
        # Compute sample weights
        if self.use_confidence_weighting:
            weights = confidence.clone()
            # Boost observed samples
            observed_mask = label_source == 1
            weights[observed_mask] *= self.observed_boost
        else:
            weights = torch.ones(batch_size, device=confidence.device)
        
        # Normalize weights
        weights = weights / weights.sum() * batch_size
        
        # === Volume Loss ===
        # Transform to log space
        vol_target = torch.log1p(targets['total_volume'])
        vol_loss_raw = self.volume_loss(outputs['volume_pred'], vol_target)
        vol_loss = (vol_loss_raw * weights).mean()
        
        # === Classification Loss ===
        class_loss_raw = self.class_loss(outputs['class_logits'], targets['event_class'])
        class_loss = (class_loss_raw * weights).mean()
        
        # === Risk Index Loss ===
        risk_loss_raw = self.risk_loss(outputs['risk_pred'], targets['risk_index'])
        risk_loss = (risk_loss_raw * weights).mean()
        
        # === Total ===
        total = (
            self.volume_weight * vol_loss +
            self.class_weight * class_loss +
            self.risk_weight * risk_loss
        )
        
        return {
            'total': total,
            'volume': vol_loss,
            'class': class_loss,
            'risk': risk_loss,
        }
```

### Alternative: Focal Loss for Class Imbalance

```python
class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance in event classification.
    
    Down-weights easy examples (stable transects), focuses on hard examples.
    """
    
    def __init__(self, alpha: list = None, gamma: float = 2.0):
        super().__init__()
        # Class weights: [stable, minor, major, failure]
        # Higher weight for rarer classes
        self.alpha = torch.tensor(alpha or [0.25, 0.5, 1.0, 2.0])
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 4) class logits
            targets: (B,) class indices
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Get alpha for each sample's target class
        alpha = self.alpha.to(logits.device)[targets]
        
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()
```

---

## Evaluation Strategy

### Data Splits

```python
def create_splits(
    samples: list[dict],
    cube: dict,
    strategy: str = 'temporal',
) -> tuple[list, list, list]:
    """
    Create train/val/test splits.
    
    Strategies:
    - 'temporal': Last year = test, second-to-last = val
    - 'spatial': Leave-one-beach-out for test
    - 'random': Stratified random split (baseline)
    """
    if strategy == 'temporal':
        # Sort by target epoch date
        samples_sorted = sorted(samples, key=lambda x: x['prediction_window'][1])
        
        # Last 15% = test, 15% before that = val
        n = len(samples_sorted)
        test_start = int(n * 0.85)
        val_start = int(n * 0.70)
        
        train = samples_sorted[:val_start]
        val = samples_sorted[val_start:test_start]
        test = samples_sorted[test_start:]
    
    elif strategy == 'spatial':
        # Leave one beach out for testing
        test_beach = 'delmar'  # Or parameterize
        beach_slices = cube['beach_slices']
        
        test_range = beach_slices[test_beach]
        
        test = [s for s in samples if test_range[0] <= s['transect_idx'] < test_range[1]]
        non_test = [s for s in samples if s not in test]
        
        # Split non-test into train/val
        np.random.shuffle(non_test)
        val_size = int(len(non_test) * 0.15)
        val = non_test[:val_size]
        train = non_test[val_size:]
    
    else:  # random
        np.random.shuffle(samples)
        n = len(samples)
        train = samples[:int(n * 0.70)]
        val = samples[int(n * 0.70):int(n * 0.85)]
        test = samples[int(n * 0.85):]
    
    return train, val, test
```

### Metrics

```python
class CliffCastMetrics:
    """
    Evaluation metrics for CliffCast.
    """
    
    @staticmethod
    def volume_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
        """Regression metrics for volume prediction."""
        # Transform back from log space
        pred_vol = np.expm1(pred)
        target_vol = target
        
        mae = np.mean(np.abs(pred_vol - target_vol))
        rmse = np.sqrt(np.mean((pred_vol - target_vol) ** 2))
        
        # Log-space metrics (better for wide range)
        log_mae = np.mean(np.abs(pred - np.log1p(target_vol)))
        
        # Correlation
        corr = np.corrcoef(pred_vol, target_vol)[0, 1]
        
        return {
            'volume_mae': mae,
            'volume_rmse': rmse,
            'volume_log_mae': log_mae,
            'volume_corr': corr,
        }
    
    @staticmethod
    def classification_metrics(
        pred_logits: np.ndarray,
        target: np.ndarray,
    ) -> dict:
        """Classification metrics for event class."""
        pred_class = np.argmax(pred_logits, axis=1)
        
        accuracy = (pred_class == target).mean()
        
        # Per-class metrics
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
        
        precision, recall, f1, support = precision_recall_fscore_support(
            target, pred_class, labels=[0, 1, 2, 3], zero_division=0
        )
        
        # Binary: any event (class > 0) vs stable
        pred_binary = pred_class > 0
        target_binary = target > 0
        
        from sklearn.metrics import roc_auc_score
        
        # AUC for binary detection
        pred_prob = 1 - softmax(pred_logits, axis=1)[:, 0]  # P(any event)
        try:
            auc = roc_auc_score(target_binary, pred_prob)
        except ValueError:
            auc = 0.5
        
        return {
            'class_accuracy': accuracy,
            'class_f1_macro': f1.mean(),
            'class_f1_stable': f1[0],
            'class_f1_minor': f1[1],
            'class_f1_major': f1[2],
            'class_f1_failure': f1[3],
            'event_detection_auc': auc,
        }
    
    @staticmethod
    def risk_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
        """Regression metrics for risk index."""
        mae = np.mean(np.abs(pred - target))
        rmse = np.sqrt(np.mean((pred - target) ** 2))
        corr = np.corrcoef(pred, target)[0, 1]
        
        return {
            'risk_mae': mae,
            'risk_rmse': rmse,
            'risk_corr': corr,
        }
    
    @staticmethod
    def stratified_metrics(
        predictions: dict,
        targets: dict,
        label_source: np.ndarray,
    ) -> dict:
        """
        Compute metrics separately for observed vs derived labels.
        
        This helps assess whether the model performs better on
        high-quality observed labels.
        """
        observed_mask = label_source == 1
        derived_mask = label_source == 0
        
        results = {}
        
        for source, mask in [('observed', observed_mask), ('derived', derived_mask)]:
            if mask.sum() == 0:
                continue
            
            vol_metrics = CliffCastMetrics.volume_metrics(
                predictions['volume'][mask],
                targets['total_volume'][mask],
            )
            
            for k, v in vol_metrics.items():
                results[f'{source}_{k}'] = v
        
        return results
```

---

## Implementation Phases

### Phase 1: Data Pipeline (Current Priority)

**Goal**: Create training-ready dataset with aligned events and features.

```bash
# Step 1: Verify cube structure
python scripts/processing/verify_cube.py --input data/processed/unified_cube.npz

# Step 2: Align events to cube
python scripts/processing/align_events.py \
    --cube data/processed/unified_cube.npz \
    --events-dir data/raw/events/ \
    --output data/processed/aligned_events.parquet

# Step 3: Generate training samples
python scripts/processing/prepare_training_data.py \
    --cube data/processed/unified_cube.npz \
    --events data/processed/aligned_events.parquet \
    --cdip-dir data/raw/cdip/ \
    --atmos-dir data/processed/atmospheric/ \
    --output data/processed/training_data.npz \
    --min-context 3 \
    --max-context 10
```

**Deliverables**:
- [ ] `scripts/processing/align_events.py` — Event-to-cube alignment
- [ ] `scripts/processing/prepare_training_data.py` — Full pipeline
- [ ] `data/processed/training_data.npz` — Training-ready tensors
- [ ] Coverage report: % observed vs derived labels per beach

### Phase 2: Model Implementation

**Goal**: Implement and test all model components with M3C2 input.

**Deliverables**:
- [ ] Update `SpatioTemporalTransectEncoder` for 13 input features
- [ ] Implement volume prediction head
- [ ] Implement event classification head
- [ ] Update `CliffCast` forward pass
- [ ] Unit tests for all components

### Phase 3: Training Infrastructure

**Goal**: End-to-end training with logging and checkpointing.

**Deliverables**:
- [ ] `CliffCastLoss` with confidence weighting
- [ ] Training script with W&B logging
- [ ] Learning rate scheduling
- [ ] Checkpointing and early stopping

### Phase 4: Evaluation & Analysis

**Goal**: Comprehensive evaluation and interpretability.

**Deliverables**:
- [ ] Evaluation script with all metrics
- [ ] Stratified analysis (observed vs derived)
- [ ] Attention visualization
- [ ] Error analysis by beach/volume/class

### Phase 5: Refinement

**Goal**: Improve performance based on Phase 4 analysis.

**Deliverables**:
- [ ] Hyperparameter tuning
- [ ] Class balancing strategies
- [ ] Ensemble methods (optional)
- [ ] Final model selection

---

## Success Metrics

### Minimum Viable Performance

| Metric | Target | Notes |
|--------|--------|-------|
| Event Detection AUC | > 0.70 | Binary: any event vs stable |
| Volume Log-MAE | < 1.0 | In log(m³ + 1) space |
| Risk Index Correlation | > 0.50 | Pearson r |
| Class F1 (macro) | > 0.40 | Across all 4 classes |

### Target Performance

| Metric | Target | Notes |
|--------|--------|-------|
| Event Detection AUC | > 0.85 | |
| Volume Log-MAE | < 0.5 | |
| Risk Index Correlation | > 0.70 | |
| Class F1 (macro) | > 0.55 | |
| Failure Class F1 | > 0.50 | Most important class |

### Interpretability Goals

- Temporal attention identifies which past epochs are most predictive
- Spatial attention highlights critical cliff locations (overhangs, notches)
- Cross-attention reveals which environmental events drive erosion
- M3C2 feature importance confirms change detection value

---

## Appendix: Event Statistics by Beach

*To be populated after running alignment pipeline*

```
Beach       | Events | Coverage | Observed % | Derived %
------------|--------|----------|------------|----------
Blacks      |        |          |            |
Torrey      |        |          |            |
Del Mar     | 411    |          |            |
Solana      |        |          |            |
San Elijo   |        |          |            |
Encinitas   |        |          |            |
------------|--------|----------|------------|----------
TOTAL       |        |          |            |
```

---

## Appendix: Configuration Template

```yaml
# configs/cliffcast_v2.yaml

model:
  d_model: 256
  n_heads: 8
  n_spatial_layers: 4
  n_temporal_layers: 2
  n_env_layers: 3
  n_fusion_layers: 2
  dropout: 0.1
  max_context_epochs: 10
  n_point_features: 13  # Updated for M3C2
  n_meta_features: 12

data:
  cube_path: data/processed/unified_cube.npz
  training_path: data/processed/training_data.npz
  min_context_epochs: 3
  max_context_epochs: 10
  wave_lookback_days: 90
  atmos_lookback_days: 90

training:
  batch_size: 32
  learning_rate: 1e-4
  weight_decay: 1e-5
  max_epochs: 100
  early_stopping_patience: 10
  gradient_clip: 1.0

loss:
  volume_weight: 1.0
  class_weight: 1.0
  risk_weight: 0.5
  use_confidence_weighting: true
  observed_boost: 1.5

evaluation:
  split_strategy: temporal  # or 'spatial', 'random'
  test_beach: null  # for spatial split
```

---

## References

- CLAUDE.md: Project overview and architecture details
- plan.md: Original implementation plan
- DATA_REQUIREMENTS.md: Data collection specifications
- Event CSV format: M3C2-derived erosion events with volumes
