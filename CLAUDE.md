# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CliffCast** is a transformer-based deep learning model for predicting coastal cliff erosion risk. The model processes 1D transect data from LiDAR scans along with environmental forcing data (wave conditions and precipitation) to predict multiple targets: risk index, collapse probability at multiple time horizons, expected retreat distance, and failure mode classification.

**Core Architecture**: Cross-attention fusion between cliff geometry embeddings (from transect encoder) and environmental embeddings (wave + precipitation encoders), followed by multi-task prediction heads.

## Directory Structure

```
transect-transformer/
├── docs/                          # Project documentation and planning
│   ├── plan.md                    # Implementation phases and roadmap
│   ├── todo.md                    # Current tasks and progress
│   └── DATA_REQUIREMENTS.md       # Data collection requirements
├── apps/                          # Interactive applications
│   └── transect_viewer/           # Streamlit transect visualization app
│       ├── app.py                 # Main entry point
│       ├── config.py              # App configuration
│       ├── components/            # UI components (sidebar, views)
│       ├── utils/                 # Data loading, validation
│       └── plots/                 # Plotting functions
├── src/
│   ├── data/
│   │   ├── parsers/               # I/O logic for various formats
│   │   │   ├── kml_parser.py      # Parse KML/KMZ files
│   │   │   ├── shapefile_parser.py # Parse shapefiles
│   │   │   └── __init__.py
│   │   ├── shapefile_transect_extractor.py  # Core transect extraction
│   │   ├── transect_voxelizer.py  # Alternative voxel-based extraction (unused)
│   │   ├── spatial_filter.py      # Spatial filtering utilities
│   │   ├── README.md              # Data module documentation
│   │   └── __init__.py
│   ├── models/                    # Model architecture components
│   ├── training/                  # Training infrastructure
│   └── utils/                     # Shared utilities
├── scripts/
│   ├── processing/                # Data pipeline scripts
│   │   └── extract_transects.py   # Transect extraction CLI
│   ├── visualization/             # Visualization and plotting
│   │   └── study_site_fig.py      # Generate study site figures
│   ├── setup/                     # Environment and admin scripts
│   │   └── verify_setup.py        # Verify installation
│   └── debug_orientation.py       # Debug script for orientation issues
├── configs/                       # Model and training configurations
├── tests/                         # Test suite
│   └── test_data/                 # Data module tests
├── README.md                      # Project README
└── CLAUDE.md                      # This file - AI assistant instructions
```

## Commands

### Environment Setup
```bash
# Create environment and install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ --cov=src --cov-report=html

# Type checking
mypy src/

# Linting
ruff check src/
```

### Training
```bash
# Phase 1: Risk index only
python train.py --config configs/phase1_risk_only.yaml --data_dir data/processed/

# Phase 2: Risk + Retreat
python train.py --config configs/phase2_add_retreat.yaml --data_dir data/processed/

# Phase 3: Risk + Retreat + Collapse
python train.py --config configs/phase3_add_collapse.yaml --data_dir data/processed/

# Phase 4: All heads (full model)
python train.py --config configs/phase4_full.yaml --data_dir data/processed/ --wandb_project cliffcast
```

### Evaluation
```bash
# Evaluate on test set
python evaluate.py --checkpoint checkpoints/best.pt --data_dir data/processed/ --split test --output results/

# Run single test file
pytest tests/test_models/test_encoders.py -v

# Run tests for specific module
pytest tests/test_data/ -v
```

### Inference
```bash
# Predict on new data
python predict.py --input data/new_site/ --checkpoint checkpoints/best.pt --output results/predictions/ --format geojson --batch_size 64
```

### Data Preprocessing
```bash
# Extract transects from LiDAR using predefined transect lines (shapefile)
python scripts/processing/extract_transects.py \
    --transects data/mops/transects_10m/transect_lines.shp \
    --las-dir data/raw/lidar/ \
    --output data/processed/transects.npz \
    --buffer 1.0 \
    --n-points 128 \
    --visualize

# Download wave data
python scripts/download_wave_data.py --buoy_id 100 --start_date 2023-01-01 --end_date 2024-01-01

# Download precipitation data
python scripts/download_precip_data.py --region san_diego --start_date 2023-01-01 --end_date 2024-01-01

# Full preprocessing pipeline
python scripts/prepare_dataset.py --lidar_dir data/raw/lidar/ --output data/processed/ --spacing_m 10
```

### Interactive Apps
```bash
# Launch transect viewer for visual inspection of NPZ files
streamlit run apps/transect_viewer/app.py

# With specific port
streamlit run apps/transect_viewer/app.py --server.port 8501
```

The transect viewer supports:
- **Data Dashboard**: Overview statistics, feature distributions, quality checks
- **Single Transect Inspector**: Detailed view of individual transects with all 12 features
- **Transect Evolution**: Compare transects across time epochs (multi-file)
- **Cross-Transect View**: Spatial analysis and multi-transect comparison

## Architecture Overview

### Model Components

1. **TransectEncoder** (`src/models/transect_encoder.py`): Self-attention encoder for cliff geometry
   - Processes N=128 points per transect with 12 features from `ShapefileTransectExtractor`
   - Core geometric features: distance_m, elevation_m, slope_deg, curvature, roughness
   - LAS attributes: intensity, RGB, classification, return_number, num_returns
   - Uses distance-based sinusoidal positional encoding (not index-based)
   - Embeds transect-level metadata (12 fields) and broadcasts to all points
   - Returns per-point embeddings and a pooled representation via learnable CLS token

2. **EnvironmentalEncoder** (`src/models/environmental_encoder.py`): Time-series encoder for forcing data
   - Shared architecture for both wave and precipitation inputs
   - Uses learned temporal position embeddings
   - Includes day-of-year seasonality embedding
   - Wave encoder: processes T_w timesteps (~360 for 90 days @ 6hr intervals)
   - Precip encoder: processes T_p timesteps (~90 for 90 days @ daily intervals)

3. **CrossAttentionFusion** (`src/models/fusion.py`): Fuses cliff geometry with environmental context
   - Cliff embeddings are queries (Q)
   - Concatenated environmental embeddings (wave + precip) are keys/values (K,V)
   - Learns "which environmental conditions explain each cliff location's state"
   - Attention weights are extractable for interpretability

4. **PredictionHeads** (`src/models/prediction_heads.py`): Multi-task prediction outputs
   - Uses attention-weighted pooling (not simple mean/max)
   - Head 1: Risk Index - sigmoid output, range [0,1]
   - Head 2: Collapse Probability - 4 time horizons (1wk, 1mo, 3mo, 1yr), multi-label classification
   - Head 3: Expected Retreat - softplus activation ensures positive values (m/yr)
   - Head 4: Failure Mode - 5 classes (stable, topple, planar, rotational, rockfall), multi-class classification

5. **CliffCast** (`src/models/cliffcast.py`): Full model assembly
   - Instantiates all encoders, fusion module, and prediction heads
   - Heads can be selectively enabled/disabled via config for phased training

### Key Design Patterns

- **Pre-norm transformers**: All transformer layers use pre-normalization for training stability
- **Phased training**: Enable prediction heads incrementally (risk → retreat → collapse → failure mode)
- **Distance-based positional encoding**: Transects use actual distance from cliff toe, not sequential indices
- **Attention for interpretability**: Cross-attention weights identify which storms/events matter for predictions
- **Multi-task learning**: Shared encoder backbone with task-specific heads and weighted loss combination

### Data Components

1. **ShapefileTransectExtractor** (`src/data/shapefile_transect_extractor.py`): Transect extraction from LiDAR
   - Uses predefined transect lines from a shapefile (e.g., MOPS transects)
   - Extracts points within buffer distance, projects onto transect line
   - Resamples to N=128 points with 12 features per point
   - **Point features (12)**: distance_m, elevation_m, slope_deg, curvature, roughness, intensity, red, green, blue, classification, return_number, num_returns
   - **Metadata (12)**: cliff_height_m, mean_slope_deg, max_slope_deg, toe_elevation_m, top_elevation_m, orientation_deg, transect_length_m, latitude, longitude, transect_id, mean_intensity, dominant_class

2. **Parsers** (`src/data/parsers/`): I/O logic for geospatial formats
   - `kml_parser.py`: Parse KML/KMZ files for transect lines or regions
   - `shapefile_parser.py`: Parse ESRI shapefiles

3. **TransectVoxelizer** (`src/data/transect_voxelizer.py`): Alternative voxel-based extraction (unused)
   - Bins points along transect into 1D segments
   - More robust to variable point density but currently not used

### Data Flow

```
Inputs:
  - Transect: (B, N, 12) point features + (B, 12) metadata + (B, N) distances
  - Wave: (B, T_w, 4) features + (B, T_w) day-of-year
  - Precip: (B, T_p, 2) features + (B, T_p) day-of-year

Pipeline:
  1. Encode transect → (B, N, d_model)
  2. Encode wave → (B, T_w, d_model)
  3. Encode precip → (B, T_p, d_model)
  4. Concatenate environmental → (B, T_w+T_p, d_model)
  5. Cross-attention fusion → (B, N, d_model)
  6. Attention pooling → (B, d_model)
  7. Prediction heads → dict of outputs

Outputs:
  - risk_index: (B,)
  - p_collapse: (B, 4)
  - retreat_m: (B,)
  - failure_mode: (B, 5) logits
  - attention: (B, n_heads, N, T_w+T_p) [optional]
```

## Critical Implementation Details

### Shape Expectations
Always validate tensor shapes. Common shapes:
- `B` = batch size (typically 32)
- `N` = transect points (128)
- `T_w` = wave timesteps (360 for 90 days @ 6hr)
- `T_p` = precip timesteps (90 for 90 days @ daily)
- `d_model` = hidden dimension (256)

### Loss Function
`CliffCastLoss` in `src/training/losses.py` combines multiple objectives:
- Risk Index: Smooth L1 loss (weight=1.0)
- Expected Retreat: Smooth L1 loss (weight=1.0)
- Collapse Probability: Binary cross-entropy per horizon (weight=2.0, higher for safety-critical)
- Failure Mode: Cross-entropy, only on samples where failure occurred (weight=0.5, fewer labels)

### Risk Index Formula
```python
def compute_risk_index(retreat_m_yr: float, cliff_height_m: float) -> float:
    height_factor = 1 + 0.1 * (cliff_height_m - 20) / 20
    weighted_retreat = retreat_m_yr * max(height_factor, 0.5)
    risk = 1 / (1 + np.exp(-2 * (weighted_retreat - 1)))
    return float(np.clip(risk, 0, 1))
```
Sigmoid-normalized, centered at 1 m/yr retreat. Taller cliffs amplify risk.

### Data Processing Conventions
- **Transect extraction**: Use `ShapefileTransectExtractor` with predefined transect lines from shapefile
- **Transect resampling**: Always N=128 points, uniformly spaced along transect profile
- **Buffer distance**: Default 1.0m around transect line for point collection
- **Feature normalization**: Intensity and RGB normalized to [0,1], classification as discrete codes
- **Wave timesteps**: 6-hourly for capturing storm dynamics
- **Precip timesteps**: Daily for antecedent moisture
- **Missing data**: Interpolate or flag - never silently fill with zeros

### Attention Interpretation
Cross-attention weights reveal physical attribution:
- High attention to specific wave timesteps → those storms contributed to erosion
- Spatial patterns in cliff points attending to events → local vs. regional forcing
- Use `return_attention=True` in model forward pass to extract weights for visualization

## Testing Strategy

### Unit Tests
Each module has corresponding tests in `tests/`:
- `test_data/`: Test data loaders, transect extraction, label generation
- `test_models/`: Test encoders, fusion, heads, full model (shape checks, NaN detection)
- `test_training/`: Test loss functions, trainer, scheduler

### Integration Tests
- Full forward/backward pass through model
- Training loop for 2 epochs on synthetic data
- Inference on batch data

### Checkpoints
- Test shapes obsessively: `assert tensor.shape == expected_shape`
- Verify no NaN: `assert not torch.isnan(tensor).any()`
- Check value ranges (e.g., sigmoid outputs in [0,1])
- Validate attention weights sum to 1 along key dimension

## Development Workflow

### Phased Implementation
Follow the phase structure from docs/plan.md:
1. **Phase 1**: Data pipeline (transect extraction, wave/precip loaders, dataset class)
2. **Phase 2**: Model implementation with risk index head only
3. **Phase 3**: Training infrastructure and validate on synthetic data
4. **Phase 4**: Add remaining prediction heads incrementally
5. **Phase 5**: Evaluation metrics, baselines, attention visualization
6. **Phase 6**: Batch inference pipeline for state-wide predictions
7. **Phase 7**: Documentation and polish

### Start Small, Scale Up
- Debug with `d_model=32, n_layers=1` before using full config
- Use synthetic data with known relationships before real data
- Train on small subset (100 samples) to verify learning before full dataset

### Memory Management
State-wide inference is memory-bound:
- Use `batch_size=64` or lower on single GPU
- Implement checkpointing for long batch jobs
- Monitor with `torch.cuda.memory_stats()`

### Logging
Log to Weights & Biases:
- All loss components separately
- Learning rate
- Gradient norms
- Sample predictions (numerical + attention maps)
- Hardware utilization

## Common Pitfalls

1. **Index vs. Distance**: Transect positional encoding uses distance from toe, NOT point index
2. **Concatenation order**: Environmental embeddings are [wave, precip] not [precip, wave]
3. **CLS token**: Prepended to sequence, so output has shape (B, N+1, d_model)
4. **Failure mode loss**: Only computed on samples with `failure_mode > 0` (not stable)
5. **Attention pooling**: Use learned attention weights, not simple mean pooling
6. **Temporal alignment**: Match transect scan date to environmental window end date
7. **Batch_first=True**: All transformers use `batch_first=True` convention

## Future Extensions

### Option C: 3D Context Enhancement
When transects are insufficient for complex geometries (caves, arches):
- Add `Context3DExtractor` module using k-nearest neighbors in full point cloud
- Mini-PointNet aggregates 3D neighborhood features
- Concatenate with transect point features before encoder
- Enable via config: `use_3d_context: true`

### Transfer Learning
Model trained on San Diego can be fine-tuned on other coastlines:
- Oregon, Malibu, Great Lakes
- Freeze encoders, retrain prediction heads
- Or full fine-tuning with lower learning rate

## Success Metrics

### Minimum Viable Product
- Risk index R² > 0.30
- Collapse probability (1yr) AUC-ROC > 0.70
- Expected retreat MAE < 1.0 m/yr
- Inference throughput > 10,000 transects/hour

### Target Performance
- Risk index R² > 0.50
- Collapse probability AUC-ROC > 0.85 (all horizons)
- Expected retreat MAE < 0.5 m/yr
- Failure mode accuracy > 70%
- Well-calibrated probabilities (ECE < 0.1)

## Notes

- **Use einops**: `rearrange(x, 'b n d -> b d n')` is clearer than `x.permute(0, 2, 1)`
- **Visualize attention early**: Don't wait for final evaluation - catch bugs during training
- **Checkpoint naming**: Include config hash + epoch + val_loss in filename
- **One head at a time**: Perfect risk index before adding retreat, then collapse, then mode
- **Point cloud classes**: Future enhancement will include beach, rip rap, cliff face, vegetation classifications
