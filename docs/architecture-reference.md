# CliffCast Architecture Reference

This document contains detailed API documentation, usage examples, and implementation details for the CliffCast model. Read this when working on model code, data loaders, or training infrastructure.

> **Quick links**: See `CLAUDE.md` for commands and quick reference. See `docs/model_plan.md` for target architecture.

---

## Model Components

All components fully implemented with 100% test coverage (216 tests passing). Import with:
```python
from src.models import CliffCast, SpatioTemporalTransectEncoder, WaveEncoder, AtmosphericEncoder
```

### 1. SpatioTemporalTransectEncoder

**File**: `src/models/transect_encoder.py`

Hierarchical attention encoder for multi-temporal cliff geometry.

**Input**: Cube format (B, T, N, 12) where T = number of LiDAR epochs, N = 128 points per transect

**Architecture**:
- **Spatial attention** (within each timestep): Self-attention over N=128 points to learn cliff geometry
- **Temporal attention** (across timesteps): Self-attention over T epochs to learn cliff evolution
- Core geometric features: distance_m, elevation_m, slope_deg, curvature, roughness
- LAS attributes: intensity, RGB, classification, return_number, num_returns
- Uses **distance-based sinusoidal positional encoding** for spatial dimension (not sequential indices!)
- Uses **learned temporal positional encoding** for time dimension
- Embeds transect-level metadata (12 fields) and broadcasts to all points/timesteps
- Returns fused spatio-temporal embeddings and a pooled representation via learnable CLS token

**Key insight**: Temporal evolution of cliff geometry (progressive weakening, crack development) is highly predictive of future failures.

**Usage**:
```python
encoder = SpatioTemporalTransectEncoder(d_model=256, n_heads=8)
outputs = encoder(point_features, metadata, distances, timestamps)
# Returns: embeddings (B,T,N,d), temporal_embeddings (B,T,d), pooled (B,d)
```

### 2. EnvironmentalEncoder

**File**: `src/models/environmental_encoder.py`

Time-series encoder for forcing data. Shared architecture for both wave and atmospheric inputs.

**Features**:
- Uses learned temporal position embeddings
- Includes day-of-year seasonality embedding (handles leap years with 367 days)
- Padding mask support for variable-length sequences

**WaveEncoder**: processes T_w timesteps (~360 for 90 days @ 6hr intervals)
- Basic (n_features=4, default): [hs, tp, dp, power]
- With derived (n_features=6): [hs, tp, dp, power, shore_normal, runup]

**AtmosphericEncoder**: processes T_a timesteps (~90 for 90 days @ daily intervals), 24 features

**Usage**:
```python
# Basic wave encoder (4 features)
wave_encoder = WaveEncoder(n_features=4, d_model=256, n_heads=8)

# With derived features (6 features)
wave_encoder = WaveEncoder(n_features=6, d_model=256, n_heads=8)

atmos_encoder = AtmosphericEncoder(d_model=256, n_heads=8)

wave_outputs = wave_encoder(wave_features, day_of_year, padding_mask=None)
atmos_outputs = atmos_encoder(atmos_features, day_of_year, padding_mask=None)
# Returns: embeddings (B,T,d), pooled (B,d)
```

### 3. CrossAttentionFusion

**File**: `src/models/fusion.py`

Fuses cliff geometry with environmental context.

**Architecture**:
- Cliff temporal embeddings are queries (Q)
- Concatenated environmental embeddings (wave + atmos) are keys/values (K,V)
- Learns "which environmental conditions explain each cliff location's state"
- Multi-layer cross-attention with residual connections and FFN
- Attention weights are extractable for interpretability (return_attention=True)

**Usage**:
```python
fusion = CrossAttentionFusion(d_model=256, n_heads=8, n_layers=2)

# Concatenate environmental embeddings
env_embeddings = torch.cat([wave_embeddings, atmos_embeddings], dim=1)

outputs = fusion(cliff_embeddings, env_embeddings, return_attention=True)
# Returns: fused (B,T,d), pooled (B,d), attention (B,n_heads,T_cliff,T_env)
```

### 4. SusceptibilityHead

**File**: `src/models/susceptibility_head.py`

5-class susceptibility classification head. Operates on pooled representation from fusion module.

**Output**:
- **logits**: (B, 5) raw class logits
- **probs**: (B, 5) softmax probabilities
- **predicted_class**: (B,) argmax class index
- **risk_score**: (B,) derived from probability-weighted consequences

**Erosion Mode Classes**:

| Class | Name | Physical Process | Risk Weight |
|-------|------|------------------|-------------|
| 0 | Stable | No significant change | 0.0 |
| 1 | Beach erosion | Sediment transport, tidal processes | 0.1 |
| 2 | Cliff toe erosion | Wave undercutting at cliff base | 0.4 |
| 3 | Small rockfall | Weathering-driven small failures | 0.6 |
| 4 | Large upper cliff failure | Major structural collapse | 1.0 |

**Risk Score Derivation**:
```python
risk_weights = torch.tensor([0.0, 0.1, 0.4, 0.6, 1.0])
risk_score = (probs * risk_weights).sum(dim=-1)
```

**Usage**:
```python
head = SusceptibilityHead(d_model=256, n_classes=5)

outputs = head(pooled_embeddings)
# Returns dict with: logits, probs, predicted_class, risk_score
```

### 5. CliffCast

**File**: `src/models/cliffcast.py`

Full model assembly for susceptibility classification.

**Features**:
- Instantiates all encoders, fusion module, and susceptibility head
- Flexible configuration for all hyperparameters
- Attention extraction via `get_attention_weights()` method
- End-to-end forward pass from raw inputs to predictions
- Risk score derived from class probabilities (not learned separately)
- **Default model size**: ~2-10M parameters (depends on configuration)

**Usage**:
```python
# Full model
model = CliffCast(
    d_model=256,
    n_heads=8,
    n_layers_spatial=3,
    n_layers_temporal=2,
    n_layers_env=2,
    n_layers_fusion=2,
    n_classes=5,
)

# Forward pass
outputs = model(
    point_features=point_features,  # (B, T, 128, 12)
    metadata=metadata,              # (B, T, 12)
    distances=distances,            # (B, T, 128)
    wave_features=wave_features,    # (B, 360, 4)
    atmos_features=atmos_features,  # (B, 90, 24)
)

# Outputs dict contains:
# - logits: (B, 5) raw class logits
# - probs: (B, 5) softmax probabilities for 5 erosion mode classes
# - predicted_class: (B,) argmax class index (0-4)
# - risk_score: (B,) derived from probs, range [0, 1]

# Extract attention weights for interpretability
attn_outputs = model.get_attention_weights(...)
# Contains: spatial_attention, temporal_attention, env_attention
```

---

## Key Design Patterns

- **Spatio-temporal hierarchy**: Spatial attention within timesteps, then temporal attention across timesteps
- **Pre-norm transformers**: All transformer layers use pre-normalization for training stability
- **Phased training**: Enable prediction heads incrementally (risk → retreat → collapse → failure mode)
- **Distance-based spatial encoding**: Transects use actual distance from cliff toe, not sequential indices
- **Learned temporal encoding**: LiDAR epochs use learned positional embeddings to capture temporal relationships
- **Multi-scale attention for interpretability**:
  - Temporal attention → which past scans matter for prediction
  - Spatial attention → which cliff locations are critical
  - Cross-attention → which environmental events drive erosion
- **Multi-task learning**: Shared encoder backbone with task-specific heads and weighted loss combination
- **Cube data format**: Transects stored as (n_transects, T, N, 12) to enable efficient temporal batching

---

## Data Components

### 1. ShapefileTransectExtractor

**File**: `src/data/shapefile_transect_extractor.py`

Transect extraction from LiDAR.

- Uses predefined transect lines from a shapefile (e.g., MOPS transects)
- Extracts points within buffer distance, projects onto transect line
- Resamples to N=128 points with 12 features per point

**Point features (12)**: distance_m, elevation_m, slope_deg, curvature, roughness, intensity, red, green, blue, classification, return_number, num_returns

**Metadata (12)**: cliff_height_m, mean_slope_deg, max_slope_deg, toe_elevation_m, top_elevation_m, orientation_deg, transect_length_m, latitude, longitude, transect_id, mean_intensity, dominant_class

### 2. CDIPWaveLoader

**File**: `src/data/cdip_wave_loader.py`

CDIP wave data access.

- Fetches nearshore wave predictions from CDIP MOP system via THREDDS/OPeNDAP
- Supports local NetCDF files (recommended) or remote THREDDS access
- Data at 100m alongshore spacing, 10m water depth, hourly from 2000-present
- **Features**: hs (wave height), tp (peak period), dp (direction), ta (average period), power (computed)
- `to_tensor()` method converts to CliffCast format: (T_w, 4) with 6-hour resampling
- Handles circular mean for direction, fill values, and temporal alignment

### 3. WaveLoader

**File**: `src/data/wave_loader.py`

Wave data integration for training.

- Manages loading CDIP data aligned to LiDAR scan dates
- LRU caching of WaveData objects (default: 50 MOPs)
- Graceful degradation: returns zeros if data unavailable (with warning)
- Mirrors `AtmosphericLoader` API for consistency
- Methods: `get_wave_for_scan()`, `get_batch_wave()`, `validate_coverage()`
- **Optional derived features**: Set `compute_derived_features=True` to compute shore-normal and runup (6 features total)
- **WaveDataset** helper class for PyTorch integration

**Usage**:
```python
# Basic 4 features (default)
loader = WaveLoader('data/raw/cdip/', lookback_days=90)
features, doy = loader.get_wave_for_scan(mop_id=582, scan_date='2023-12-15')
# features.shape: (360, 4) - [hs, tp, dp, power]

# With derived features (requires cliff orientations)
orientations = {582: 270, 583: 265}  # MOP ID -> degrees from N
loader = WaveLoader(
    'data/raw/cdip/',
    compute_derived_features=True,
    cliff_orientations=orientations
)
features, doy = loader.get_wave_for_scan(mop_id=582, scan_date='2023-12-15')
# features.shape: (360, 6) - [hs, tp, dp, power, shore_normal, runup]
```

### 4. WaveMetricsCalculator

**File**: `src/data/wave_features.py`

Derived wave feature engineering. Computes 19 physically-motivated features from raw CDIP wave data.

**Features computed**:
- **Raw (4)**: hs, tp, dp, power
- **Physical (2)**: shore_normal_hs (directional impact), runup_2pct (Stockdon 2006)
- **Integrated (3)**: cumulative_energy_mj, cumulative_power_kwh, mean_power_kw
- **Extreme (3)**: max_hs, hs_p90, hs_p99
- **Storm (5)**: storm_hours, storm_count, max_storm_duration_hr, time_since_storm_hr, mean_storm_duration_hr
- **Temporal (2)**: rolling_max_7d, hs_trend_slope

Can output summary metrics (for analysis) or time series features (for model input). Integrated with WaveLoader via `compute_derived_features` flag.

Has CLI for batch processing: `python src/data/wave_features.py --input <nc_file> --scan-date <date>`

### 5. AtmosphericLoader

**File**: `src/data/atmos_loader.py`

Atmospheric data integration.

- Loads pre-computed atmospheric features from per-beach parquet files
- 90-day lookback window at daily intervals (T_a=90)
- 24 features: precip, temperature, API, wet/dry cycles, freeze-thaw, VPD
- **AtmosphericDataset** helper class for PyTorch integration

### 6. Parsers

**Directory**: `src/data/parsers/`

I/O logic for geospatial formats.
- `kml_parser.py`: Parse KML/KMZ files for transect lines or regions
- `shapefile_parser.py`: Parse ESRI shapefiles

### 7. TransectVoxelizer

**File**: `src/data/transect_voxelizer.py`

Alternative voxel-based extraction (unused).
- Bins points along transect into 1D segments
- More robust to variable point density but currently not used

### 8. CliffDelineation

**Directory**: `src/data/cliff_delineation/`

Cliff toe/top detection via CNN-BiLSTM. Uses pre-trained CliffDelineaTool v2.0 model for automatic cliff edge detection.

**Prerequisite**: Install CliffDelineaTool as editable package: `pip install -e /path/to/CliffDelineaTool_2.0/v2`

**Components**:
- **CliffFeatureAdapter**: Transforms 12 transect-transformer features → 13 CliffDelineaTool features
  - Computes: normalized elevation/distance, gradient, curvature, seaward/landward slopes, trendline deviation, convexity, relative elevation, low elevation zone, shore proximity, max local slope
- **CliffDelineationModel**: Loads checkpoint, runs CNN-BiLSTM inference, extracts toe/top from segmentation probabilities
- **detect_cliff_edges()**: Main entry point, processes NPZ files and saves results to sidecar file

**Output format**: `*.cliff.npz` sidecar file containing:
- `toe_distances`, `top_distances`: (n_transects, T) distances along transect (-1 if none)
- `toe_indices`, `top_indices`: (n_transects, T) point indices (0-127)
- `toe_confidences`, `top_confidences`: (n_transects, T) model confidence [0,1]
- `has_cliff`: (n_transects, T) boolean detection flag

**Usage**:
```python
from src.data.cliff_delineation import detect_cliff_edges, load_cliff_results
from src.data.cliff_delineation.detector import get_cliff_metrics

# Run detection
results = detect_cliff_edges(
    npz_path='data/processed/delmar.npz',
    checkpoint_path='/path/to/best_model.pth',
    confidence_threshold=0.5,
    n_vert=20,  # Must match training config
)

# Load existing results
results = load_cliff_results('data/processed/delmar.cliff.npz')

# Get summary metrics
metrics = get_cliff_metrics(results)
print(f"Detection rate: {metrics['detection_rate']:.1%}")
print(f"Mean cliff width: {metrics['mean_cliff_width_m']:.1f} m")
```

**TODO (Future)**: Option to merge cliff results directly into main NPZ file

---

## Data Flow

```
Inputs (Cube Format):
  - Transect: (B, T, N, 12) point features + (B, T, 12) metadata + (B, T, N) distances
    where T = number of context LiDAR epochs
  - M3C2: (B, N) change distances from most recent epoch pair
  - Timestamps: (B, T) scan dates for temporal encoding
  - Wave: (B, T_w, 4) features + (B, T_w) day-of-year (aligned to most recent scan)
    Features: [hs, tp, dp, power] - 90 days @ 6hr = 360 timesteps
  - Atmospheric: (B, T_a, 24) features + (B, T_a) day-of-year (aligned to most recent scan)
    Features: precip, temp, API, wet/dry cycles, VPD, freeze-thaw - 90 days @ daily = 90 timesteps

Pipeline (CliffCast forward pass):
  1. Transect encoder:
     a. Spatial attention (per timestep) → (B, T, N, d_model)
     b. Temporal attention (across timesteps) → (B, T, d_model)
     c. CLS token pooling → (B, d_model)
  2. Wave encoder → (B, T_w, d_model)
  3. Atmospheric encoder → (B, T_a, d_model)
  4. Concatenate environmental → (B, T_w+T_a, d_model)
  5. Cross-attention fusion (cliff queries environment) → (B, d_model)
  6. Susceptibility head → dict of outputs

Outputs:
  - logits: (B, 5) - raw class logits for 5 erosion modes
  - probs: (B, 5) - softmax probabilities [stable, beach, toe, rockfall, large_failure]
  - predicted_class: (B,) - argmax class index (0-4)
  - risk_score: (B,) - derived: sum(probs * [0.0, 0.1, 0.4, 0.6, 1.0])

  Optional (for interpretability):
  - spatial_attention: (B, n_heads, T, N, N) - attention within timesteps
  - temporal_attention: (B, n_heads, T, T) - attention across timesteps
  - env_attention: (B, n_heads, T, T_w+T_a) - cross-attention to environment
```

---

## Wave Data (CDIP MOP System)

**Overview**: Wave forcing data comes from CDIP's MOP (Monitoring and Prediction) system, providing nearshore wave predictions at 100m alongshore spacing at 10m water depth. Data is hourly from 2000-present and stored as NetCDF files.

### Data Source

CDIP THREDDS server:
- THREDDS catalog: https://thredds.cdip.ucsd.edu/thredds/catalog/cdip/model/MOP_alongshore/catalog.html
- OPeNDAP access: Direct remote access for on-demand loading
- Local caching: Pre-download files to `data/raw/cdip/` for training (recommended)

### File Naming Convention

- San Diego MOPs: `D0520_hindcast.nc`, `D0521_hindcast.nc`, ..., `D0764_hindcast.nc`
- File size: ~10-100 MB per MOP (depends on date range)
- Each file contains full time series for one MOP location

### Features (4 per timestep)

1. **hs** - Significant wave height (m)
2. **tp** - Peak period (s)
3. **dp** - Peak direction (deg from N, 0=north, 90=east)
4. **power** - Wave power flux (kW/m), computed as `(ρ*g²/64π) * Hs² * Tp`

### Temporal Configuration

- Lookback window: 90 days before scan date
- Resampling: 6-hour intervals (captures storm dynamics)
- Total timesteps: T_w = 360 for 90 days @ 6hr
- Circular mean for direction during resampling (sin/cos components)

### Download Workflow

1. **Batch download**: Use `download_cdip_data.py` to populate `data/raw/cdip/`
2. **Verify coverage**: Check that >80% of MOPs have data
3. **Train with local files**: WaveLoader reads from disk (fast, no network dependency)

### Coverage Requirements

- Minimum 80% coverage for training (safety-critical predictions need reliable forcing)
- Missing data handling: Graceful degradation (log warning, return zeros)
- Pre-training validation: Run `validate_coverage()` to check all beaches

### Critical Implementation Details

- **Fill value handling**: CDIP uses -999.99 for invalid data; these are filtered to NaN
- **Circular mean resampling**: Direction uses sin/cos components, not simple averaging
- **Temporal alignment**: Wave lookback is always 90 days BEFORE scan date (no future leakage)
- **Cache strategy**: WaveData objects cached in memory (LRU eviction, default 50 MOPs)
- **MOP-to-site-label mapping**: Files may use D0582, D582, or D582 formats; loader tries all

### Expected Tensor Shapes

- Input to model: `(B, T_w, 4)` where B = batch size, T_w = 360
- Day-of-year: `(B, T_w)` for seasonality embedding
- Feature order: [hs, tp, dp, power] (always in this order)

### Data Organization

```
data/
├── raw/
│   ├── cdip/                  # Wave data (NetCDF files) - 183 MOPs downloaded
│   │   ├── D0520_hindcast.nc  # Black's Beach (~141MB each)
│   │   ├── D0582_hindcast.nc  # Torrey Pines
│   │   ├── D0595_hindcast.nc  # Del Mar
│   │   └── ...                # 183 of 245 San Diego MOPs (74.7% coverage)
│   ├── prism/                 # Raw PRISM climate data
│   └── master_list.csv        # LiDAR survey metadata
├── processed/
│   ├── atmospheric/           # Computed atmospheric features (parquet)
│   ├── delmar.npz            # Transect cubes
│   └── ...
```

---

## Attention Interpretation

Multiple attention mechanisms reveal different physical attributions:

### Temporal Attention (across LiDAR epochs)

- High attention to specific past scans → those epochs show critical cliff evolution
- Identifies progressive weakening patterns, crack development, or precursor deformation
- Use `return_temporal_attention=True` to visualize which historical scans matter most

### Spatial Attention (within each timestep)

- Identifies which cliff locations are most predictive (e.g., overhangs, notches)
- Use `return_spatial_attention=True` to visualize per-point importance

### Cross-Attention (cliff → environment)

- High attention to specific wave timesteps → those storms contributed to erosion
- Spatial patterns in cliff points attending to events → local vs. regional forcing
- Use `return_env_attention=True` to extract weights for visualization

---

## Risk Score Derivation

Risk scores are **derived from class probabilities**, not learned as a separate target. This ensures risk has physical meaning: high risk = high probability of high-consequence erosion modes.

```python
def compute_risk_score(class_probs: torch.Tensor) -> torch.Tensor:
    """
    Derive risk score from 5-class susceptibility probabilities.

    Args:
        class_probs: (B, 5) softmax probabilities for erosion modes

    Returns:
        risk_score: (B,) values in [0, 1]
    """
    # Risk weights reflect consequence severity
    risk_weights = torch.tensor([0.0, 0.1, 0.4, 0.6, 1.0])
    #                           stable, beach, toe, rockfall, large_failure

    risk_score = (class_probs * risk_weights).sum(dim=-1)
    return risk_score
```

**Risk Categories for Management**:

| Risk Score | Category | Management Action |
|------------|----------|-------------------|
| 0.0 - 0.2 | Low | Routine monitoring |
| 0.2 - 0.4 | Moderate | Enhanced monitoring |
| 0.4 - 0.6 | High | Access restrictions, signage |
| 0.6 - 1.0 | Critical | Closures, setback enforcement |

---

## Loss Function

`SusceptibilityLoss` in `src/training/susceptibility_loss.py` uses weighted cross-entropy for 5-class classification:

```python
class SusceptibilityLoss(nn.Module):
    def __init__(self):
        # Class weights: higher for rarer, more important classes
        # Lower weight for "stable" (weak negative evidence)
        class_weights = torch.tensor([0.3, 1.0, 2.0, 2.0, 5.0])
        #                             stable, beach, toe, rockfall, large_failure

        self.loss_fn = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=0.1,  # Helps with label uncertainty
        )

    def forward(self, logits, targets):
        return self.loss_fn(logits, targets)
```

**Weight Rationale**:
- **Stable (0.3)**: Weak negative evidence; don't overfit to "nothing happened"
- **Beach erosion (1.0)**: Baseline weight
- **Toe erosion (2.0)**: Important precursor signal
- **Small rockfall (2.0)**: Safety concern
- **Large failure (5.0)**: Critical; must not miss these events

---

## Test Coverage Details

**Test Suite**: 216 tests passing with 100% code coverage for all model components

**Test Organization**:
- `tests/test_models/` - 151 tests
- `tests/test_data/` - 30+ tests
- `tests/test_apps/` - 27+ tests

**Coverage Areas**:
- Shape validation: All tensor shapes verified at each stage
- Value ranges: Risk [0,1], retreat positive, probabilities [0,1]
- No NaN/Inf: All outputs checked for numerical stability
- Gradient flow: Backpropagation verified through entire model
- Attention extraction: Attention weights can be extracted and visualized
- Padding masks: Variable-length sequences handled correctly
- Phased training: Selective head enabling works as expected
- Edge cases: Single timestep, large batches, zero inputs, eval mode

---

## Future Extensions

- **3D Context Enhancement**: For complex geometries (caves, arches), add `Context3DExtractor` module using k-nearest neighbors
- **Transfer Learning**: Fine-tune on other coastlines (Oregon, Malibu, Great Lakes) by freezing encoders and retraining heads
