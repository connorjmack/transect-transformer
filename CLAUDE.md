# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Critical Safety

This project may connect to cloud servers with production data. NEVER:
- Delete files in bulk (no `rm -rf`, `rm -r`, or recursive deletes)
- Modify anything under `/data/production/`
- Run commands that affect multiple files without explicit user approval
- Force push to git repositories (`git push --force`)
- Execute destructive operations on remote servers

## Output Formatting

When providing Python or shell commands for the user to copy and paste, **always format them as a single line**. Do not split commands across multiple lines with backslashes (`\`) - put everything on one line so it can be copied and pasted directly without issues.

## Project Overview

**CliffCast** is a transformer-based deep learning model for predicting coastal cliff erosion risk. The model processes multi-temporal 1D transect data from LiDAR scans along with environmental forcing data (wave conditions and precipitation) to predict multiple targets: risk index, collapse probability at multiple time horizons, expected retreat distance, and failure mode classification.

**Core Architecture**: Spatio-temporal attention over cliff geometry sequences (multiple LiDAR epochs per transect), followed by cross-attention fusion with environmental embeddings (wave + precipitation encoders), and multi-task prediction heads. The temporal dimension captures cliff evolution over time, which is critical for predicting future failures.

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
│   │   ├── cdip_wave_loader.py    # CDIP THREDDS/OPeNDAP wave data access ✅
│   │   ├── wave_loader.py         # Wave data integration for training ✅
│   │   ├── atmos_loader.py        # Atmospheric data integration ✅
│   │   ├── spatial_filter.py      # Spatial filtering utilities
│   │   ├── README.md              # Data module documentation
│   │   └── __init__.py
│   ├── models/                    # Model architecture components
│   ├── training/                  # Training infrastructure
│   └── utils/                     # Shared utilities
├── scripts/
│   ├── processing/                # Data pipeline scripts
│   │   ├── extract_transects.py   # Transect extraction CLI
│   │   └── download_cdip_data.py  # Batch download CDIP wave data ✅
│   ├── visualization/             # Visualization and plotting ✅
│   │   ├── README.md              # Visualization documentation ✅
│   │   ├── quick_wave_summary.py  # Quick 4-panel wave overview ✅
│   │   ├── wave_climate_figures.py # 8 comprehensive wave appendix figures ✅
│   │   ├── plot_prism_coverage.py  # 3 comprehensive atmospheric figures ✅
│   │   └── study_site_fig.py      # Generate study site figures
│   ├── setup/                     # Environment and admin scripts
│   │   └── verify_setup.py        # Verify installation
│   └── debug_orientation.py       # Debug script for orientation issues
├── configs/                       # Model and training configurations
├── tests/                         # Test suite (83+ tests)
│   ├── test_data/                 # Data module tests
│   │   ├── test_transect_extractor.py  # Extractor + cube format tests (30 tests)
│   │   ├── test_wave_loader.py         # Wave loader tests (18 tests) ✅
│   │   ├── test_atmos_loader.py        # Atmospheric loader tests (27 tests)
│   │   └── test_prism_download.py      # PRISM download tests
│   ├── test_apps/                 # Application tests
│   │   └── test_transect_viewer.py     # Viewer data_loader + validators (27 tests)
│   └── test_utils.py              # Config utility tests (8 tests)
├── README.md                      # Project README
└── CLAUDE.md                      # This file - AI assistant instructions
```

## Commands

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run all tests (57 tests)
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test modules
pytest tests/test_data/ -v
pytest tests/test_apps/ -v

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

### Evaluation & Testing
```bash
# Evaluate on test set (TODO: not yet implemented)
python evaluate.py --checkpoint checkpoints/best.pt --data_dir data/processed/ --split test --output results/

# Run all tests (216 tests)
pytest tests/ -v

# Run model tests only (151 tests)
pytest tests/test_models/ -v

# Run specific test file
pytest tests/test_models/test_cliffcast.py -v
pytest tests/test_models/test_transect_encoder.py -v
pytest tests/test_models/test_environmental_encoder.py -v
pytest tests/test_models/test_fusion.py -v
pytest tests/test_models/test_prediction_heads.py -v

# Run tests for data pipeline
pytest tests/test_data/ -v

# Run tests with coverage report
pytest tests/test_models/ --cov=src/models --cov-report=html

# Type checking
mypy src/models/
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

# Extract transects from survey CSV (recommended for large datasets)
python scripts/processing/extract_transects.py \
    --transects data/mops/transects_10m/transect_lines.shp \
    --survey-csv data/raw/master_list.csv \
    --beach delmar \
    --output data/processed/delmar.npz

# Available --beach options: blacks, torrey, delmar, solana, sanelijo, encinitas
# Or use --mop-min and --mop-max for custom ranges

# Subset existing cube by MOP range (recommended workflow)
# 1. Extract all transects at once
# 2. Subset into beach-specific files
python scripts/processing/subset_transects.py \
    --input data/processed/all_transects.npz \
    --output data/processed/delmar.npz \
    --beach delmar

# List transects in a cube file
python scripts/processing/subset_transects.py \
    --input data/processed/all_transects.npz \
    --list

# Cross-platform: paths auto-convert between Mac (/Volumes/group/...) and Linux (/project/group/...)
# Force Linux paths when running on Linux with Mac-formatted CSV:
python scripts/processing/extract_transects.py \
    --transects data/mops/transects_10m/transect_lines.shp \
    --survey-csv data/raw/master_list.csv \
    --target-os linux \
    --output data/processed/all_transects.npz

# Use LAZ files instead of LAS (faster loading, smaller files)
# Automatically substitutes .las paths with .laz if they exist:
python scripts/processing/extract_transects.py \
    --transects data/mops/transects_10m/transect_lines.shp \
    --survey-csv data/raw/master_list.csv \
    --prefer-laz \
    --output data/processed/all_transects.npz

# Download CDIP wave data (all San Diego MOPs 520-764)
python scripts/processing/download_cdip_data.py --output data/raw/cdip/ --start-date 2017-01-01 --end-date 2025-12-31

# Download wave data for specific beach
python scripts/processing/download_cdip_data.py --output data/raw/cdip/ --beach delmar

# Download custom MOP range
python scripts/processing/download_cdip_data.py --output data/raw/cdip/ --mop-min 595 --mop-max 620

# Verify existing CDIP downloads
python scripts/processing/download_cdip_data.py --output data/raw/cdip/ --verify-only

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

The transect viewer supports (cube format NPZ files):
- **Data Dashboard**: Overview statistics, feature distributions, quality checks, temporal coverage
- **Single Transect Inspector**: Detailed view of individual transects with all 12 features at any epoch
- **Temporal Slider**: Scrub through time for a single transect with fixed y-axis for comparison
- **Transect Evolution**: Overlaid profiles, change detection, temporal heatmap across all epochs
- **Cross-Transect View**: Spatial analysis and multi-transect comparison at selected epoch

### Visualization Scripts (Publication Figures)
```bash
# Generate wave climate appendix figures (8 comprehensive figures)
python scripts/visualization/wave_climate_figures.py --cdip-dir data/raw/cdip/ --output figures/appendix/

# Quick wave summary (4-panel overview)
python scripts/visualization/quick_wave_summary.py --cdip-dir data/raw/cdip/ --output figures/appendix/

# Generate PRISM atmospheric data figures (3 comprehensive figures)
python scripts/visualization/plot_prism_coverage.py --atmos-dir data/processed/atmospheric/ --output-dir figures/appendix/

# Generate specific figure types
python scripts/visualization/wave_climate_figures.py --figures A1 A3 A5  # Specific wave figures
python scripts/visualization/plot_prism_coverage.py --figure-type overview  # Only overview
```

**Wave Climate Figures** (2017-2025, 193 MOPs):
- **Figure A1**: Wave height distributions with Weibull fits (6 beaches)
- **Figure A2**: Wave period characteristics (Hs vs Tp hexbin with marginals)
- **Figure A3**: Wave direction roses (6 polar plots, weighted by height)
- **Figure A4**: Wave power statistics (box plots, CDFs, distributions, table)
- **Figure A5**: Seasonal patterns (monthly means, seasonal boxes, annual heatmap)
- **Figure A6**: Storm climatology (time series, duration, frequency, intensity)
- **Figure A7**: Spatial wave climate (latitudinal profiles, summary table)
- **Figure A8**: Extreme value analysis (GEV fit, return periods, design levels)

**PRISM Atmospheric Figures** (2017-2025, 6 beaches):
- **prism_overview.png**: 3x3 grid with beach map, long-term trends (monthly), seasonal climatology, annual totals, spatio-temporal heatmap, coverage table
- **prism_feature_distributions.png**: 5x3 grid with histograms for 15 derived features (cumulative precip, API, wet-dry cycles, VPD, freeze-thaw)
- **prism_extreme_events.png**: 2x2 grid with extreme precipitation events (>25mm, >50mm), API time series, VPD time series

**Output Location**: All figures save to `figures/appendix/` by default
**Requirements**: See `scripts/visualization/README.md` for detailed documentation

## Architecture Overview

### Model Components (IMPLEMENTED & TESTED)

All model components are fully implemented with 100% test coverage. Import with:
```python
from src.models import CliffCast, SpatioTemporalTransectEncoder, WaveEncoder, AtmosphericEncoder
```

1. **SpatioTemporalTransectEncoder** (`src/models/transect_encoder.py`): Hierarchical attention encoder for multi-temporal cliff geometry
   - **Status**: ✅ Implemented, 28 tests passing, 100% coverage
   - **Input**: Cube format (B, T, N, 12) where T = number of LiDAR epochs, N = 128 points per transect
   - **Spatial attention** (within each timestep): Self-attention over N=128 points to learn cliff geometry
   - **Temporal attention** (across timesteps): Self-attention over T epochs to learn cliff evolution
   - Core geometric features: distance_m, elevation_m, slope_deg, curvature, roughness
   - LAS attributes: intensity, RGB, classification, return_number, num_returns
   - Uses **distance-based sinusoidal positional encoding** for spatial dimension (not sequential indices!)
   - Uses **learned temporal positional encoding** for time dimension
   - Embeds transect-level metadata (12 fields) and broadcasts to all points/timesteps
   - Returns fused spatio-temporal embeddings and a pooled representation via learnable CLS token
   - **Key insight**: Temporal evolution of cliff geometry (progressive weakening, crack development) is highly predictive of future failures
   - **Usage**:
     ```python
     encoder = SpatioTemporalTransectEncoder(d_model=256, n_heads=8)
     outputs = encoder(point_features, metadata, distances, timestamps)
     # Returns: embeddings (B,T,N,d), temporal_embeddings (B,T,d), pooled (B,d)
     ```

2. **EnvironmentalEncoder** (`src/models/environmental_encoder.py`): Time-series encoder for forcing data
   - **Status**: ✅ Implemented, 37 tests passing, 100% coverage
   - Shared architecture for both wave and atmospheric inputs
   - Uses learned temporal position embeddings
   - Includes day-of-year seasonality embedding (handles leap years with 367 days)
   - Padding mask support for variable-length sequences
   - **WaveEncoder**: processes T_w timesteps (~360 for 90 days @ 6hr intervals), 4 or 6 features
     - Basic (n_features=4, default): [hs, tp, dp, power]
     - With derived (n_features=6): [hs, tp, dp, power, shore_normal, runup]
   - **AtmosphericEncoder**: processes T_a timesteps (~90 for 90 days @ daily intervals), 24 features
   - **Usage**:
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

3. **CrossAttentionFusion** (`src/models/fusion.py`): Fuses cliff geometry with environmental context
   - **Status**: ✅ Implemented, 25 tests passing, 100% coverage
   - Cliff temporal embeddings are queries (Q)
   - Concatenated environmental embeddings (wave + atmos) are keys/values (K,V)
   - Learns "which environmental conditions explain each cliff location's state"
   - Multi-layer cross-attention with residual connections and FFN
   - Attention weights are extractable for interpretability (return_attention=True)
   - **Usage**:
     ```python
     fusion = CrossAttentionFusion(d_model=256, n_heads=8, n_layers=2)

     # Concatenate environmental embeddings
     env_embeddings = torch.cat([wave_embeddings, atmos_embeddings], dim=1)

     outputs = fusion(cliff_embeddings, env_embeddings, return_attention=True)
     # Returns: fused (B,T,d), pooled (B,d), attention (B,n_heads,T_cliff,T_env)
     ```

4. **PredictionHeads** (`src/models/prediction_heads.py`): Multi-task prediction outputs
   - **Status**: ✅ Implemented, 35 tests passing, 100% coverage
   - All heads operate on pooled representation from fusion module
   - **RiskIndexHead**: Sigmoid output, range [0,1]
   - **CollapseProbabilityHead**: 4 time horizons (1wk, 1mo, 3mo, 1yr), multi-label binary (sigmoid per horizon)
   - **ExpectedRetreatHead**: Softplus activation ensures positive values (m/yr)
   - **FailureModeHead**: 5 classes (stable, topple, planar, rotational, rockfall), returns logits for cross-entropy
   - Heads can be selectively enabled/disabled for phased training
   - **Usage**:
     ```python
     # Phase 1: Risk only
     heads_phase1 = PredictionHeads(
         enable_risk=True, enable_retreat=False,
         enable_collapse=False, enable_failure_mode=False
     )

     # Phase 4: All heads
     heads_full = PredictionHeads()  # All enabled by default

     outputs = heads_full(pooled_embeddings)
     # Returns dict with: risk_index, retreat_m, p_collapse, failure_mode_logits
     ```

5. **CliffCast** (`src/models/cliffcast.py`): Full model assembly
   - **Status**: ✅ Implemented, 26 tests passing, 100% coverage
   - Instantiates all encoders, fusion module, and prediction heads
   - Flexible configuration for all hyperparameters
   - Heads can be selectively enabled/disabled for phased training
   - Attention extraction via `get_attention_weights()` method
   - End-to-end forward pass from raw inputs to predictions
   - **Default model size**: ~2-10M parameters (depends on configuration)
   - **Usage**:
     ```python
     # Full model
     model = CliffCast(
         d_model=256,
         n_heads=8,
         n_layers_spatial=2,
         n_layers_temporal=2,
         n_layers_env=3,
         n_layers_fusion=2,
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
     # - risk_index: (B,) in [0,1]
     # - retreat_m: (B,) positive values
     # - p_collapse: (B,4) probabilities per horizon
     # - failure_mode_logits: (B,5) logits

     # Extract attention weights for interpretability
     attn_outputs = model.get_attention_weights(...)
     # Contains: spatial_attention, temporal_attention, env_attention
     ```

### Key Design Patterns

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

### Data Components

1. **ShapefileTransectExtractor** (`src/data/shapefile_transect_extractor.py`): Transect extraction from LiDAR
   - **Status**: ✅ Implemented, 30 tests passing
   - Uses predefined transect lines from a shapefile (e.g., MOPS transects)
   - Extracts points within buffer distance, projects onto transect line
   - Resamples to N=128 points with 12 features per point
   - **Point features (12)**: distance_m, elevation_m, slope_deg, curvature, roughness, intensity, red, green, blue, classification, return_number, num_returns
   - **Metadata (12)**: cliff_height_m, mean_slope_deg, max_slope_deg, toe_elevation_m, top_elevation_m, orientation_deg, transect_length_m, latitude, longitude, transect_id, mean_intensity, dominant_class

2. **CDIPWaveLoader** (`src/data/cdip_wave_loader.py`): CDIP wave data access
   - **Status**: ✅ Implemented and tested
   - Fetches nearshore wave predictions from CDIP MOP system via THREDDS/OPeNDAP
   - Supports local NetCDF files (recommended) or remote THREDDS access
   - Data at 100m alongshore spacing, 10m water depth, hourly from 2000-present
   - **Features**: hs (wave height), tp (peak period), dp (direction), ta (average period), power (computed)
   - `to_tensor()` method converts to CliffCast format: (T_w, 4) with 6-hour resampling
   - Handles circular mean for direction, fill values, and temporal alignment
   - **Usage**: See "Wave Data (CDIP MOP System)" section below

3. **WaveLoader** (`src/data/wave_loader.py`): Wave data integration for training
   - **Status**: ✅ Implemented, 18 tests passing
   - Manages loading CDIP data aligned to LiDAR scan dates
   - LRU caching of WaveData objects (default: 50 MOPs)
   - Graceful degradation: returns zeros if data unavailable (with warning)
   - Mirrors `AtmosphericLoader` API for consistency
   - Methods: `get_wave_for_scan()`, `get_batch_wave()`, `validate_coverage()`
   - **Optional derived features**: Set `compute_derived_features=True` to compute shore-normal and runup (6 features total)
   - **WaveDataset** helper class for PyTorch integration
   - **Usage**:
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

4. **WaveMetricsCalculator** (`src/data/wave_features.py`): Derived wave feature engineering
   - **Status**: ✅ Implemented, 25+ tests passing
   - Computes 19 physically-motivated features from raw CDIP wave data
   - **Features computed**:
     - **Raw (4)**: hs, tp, dp, power
     - **Physical (2)**: shore_normal_hs (directional impact), runup_2pct (Stockdon 2006)
     - **Integrated (3)**: cumulative_energy_mj, cumulative_power_kwh, mean_power_kw
     - **Extreme (3)**: max_hs, hs_p90, hs_p99
     - **Storm (5)**: storm_hours, storm_count, max_storm_duration_hr, time_since_storm_hr, mean_storm_duration_hr
     - **Temporal (2)**: rolling_max_7d, hs_trend_slope
   - Can output summary metrics (for analysis) or time series features (for model input)
   - Integrated with WaveLoader via `compute_derived_features` flag
   - **Usage**:
     ```python
     from src.data.wave_features import WaveMetricsCalculator, WaveMetricsConfig

     # Compute time series features (for model input)
     config = WaveMetricsConfig(lookback_days=90, resample_hours=6)
     calculator = WaveMetricsCalculator(config)
     features, doy = calculator.compute_timeseries_features(
         df,  # pandas DataFrame with hs, tp, dp columns
         cliff_orientation_deg=270,
         beach_slope=0.1
     )
     # Returns: (T, 6) array [hs, tp, dp, power, shore_normal, runup]

     # Compute summary metrics (for analysis/visualization)
     metrics = calculator.compute_all_metrics(df, cliff_orientation_deg=270)
     # Returns: dict with all 19 metrics

     # CLI for batch processing
     python src/data/wave_features.py \
         --input data/raw/cdip/D0582_hindcast.nc \
         --scan-date 2023-12-15 \
         --lookback-days 90 \
         --cliff-orientation 270
     ```

5. **AtmosphericLoader** (`src/data/atmos_loader.py`): Atmospheric data integration
   - **Status**: ✅ Implemented, 27 tests passing
   - Loads pre-computed atmospheric features from per-beach parquet files
   - 90-day lookback window at daily intervals (T_a=90)
   - 24 features: precip, temperature, API, wet/dry cycles, freeze-thaw, VPD
   - **AtmosphericDataset** helper class for PyTorch integration

6. **Parsers** (`src/data/parsers/`): I/O logic for geospatial formats
   - `kml_parser.py`: Parse KML/KMZ files for transect lines or regions
   - `shapefile_parser.py`: Parse ESRI shapefiles

7. **TransectVoxelizer** (`src/data/transect_voxelizer.py`): Alternative voxel-based extraction (unused)
   - Bins points along transect into 1D segments
   - More robust to variable point density but currently not used

### Data Flow

```
Inputs (Cube Format):
  - Transect: (B, T, N, 12) point features + (B, T, 12) metadata + (B, T, N) distances
    where T = number of LiDAR epochs (e.g., 10 for 2017-2025 annual scans)
  - Timestamps: (B, T) scan dates for temporal encoding
  - Wave: (B, T_w, n_wave) features + (B, T_w) day-of-year (aligned to most recent scan)
    Basic (n_wave=4): [hs, tp, dp, power]
    With derived (n_wave=6): [hs, tp, dp, power, shore_normal, runup]
  - Atmospheric: (B, T_a, 24) features + (B, T_a) day-of-year (aligned to most recent scan)
    Features: precip_mm, temp, cumulative precip, API, intensity, wet/dry cycles, VPD, freeze-thaw

Pipeline (CliffCast forward pass):
  1. Transect encoder:
     a. Spatial attention (per timestep) → (B, T, N, d_model)
     b. Temporal attention (across timesteps) → (B, T, d_model)
     c. CLS token pooling → (B, d_model)
  2. Wave encoder → (B, T_w, d_model)
  3. Atmospheric encoder → (B, T_a, d_model)
  4. Concatenate environmental → (B, T_w+T_a, d_model)
  5. Cross-attention fusion (cliff queries environment) → (B, d_model)
  6. Prediction heads → dict of outputs

Outputs:
  - risk_index: (B,) - values in [0, 1]
  - retreat_m: (B,) - positive values (m/yr)
  - p_collapse: (B, 4) - probabilities per horizon [1wk, 1mo, 3mo, 1yr]
  - failure_mode_logits: (B, 5) - logits for [stable, topple, planar, rotational, rockfall]

  Optional (for interpretability):
  - spatial_attention: (B, n_heads, T, N, N) - attention within timesteps
  - temporal_attention: (B, n_heads, T, T) - attention across timesteps
  - env_attention: (B, n_heads, T, T_w+T_a) - cross-attention to environment
```

## Critical Implementation Details

### Shape Expectations
Always validate tensor shapes. Common shapes:
- `B` = batch size (typically 32)
- `T` = LiDAR epochs per transect (e.g., 10 for annual scans 2017-2025)
- `N` = transect points (128)
- `T_w` = wave timesteps (360 for 90 days @ 6hr)
- `T_a` = atmospheric timesteps (90 for 90 days @ daily)
- `d_model` = hidden dimension (256)

**Cube data format**: Transects stored as (n_transects, T, N, 12) where each transect has full temporal coverage across all LiDAR epochs.

### Testing Strategy

**Test Suite**: 216 tests passing with 100% code coverage for all model components

Run tests with:
```bash
# All tests
pytest tests/ -v

# Model tests only (151 tests)
pytest tests/test_models/ -v

# With coverage report
pytest tests/test_models/ --cov=src/models --cov-report=html
```

**Test Organization**:
- `tests/test_models/test_transect_encoder.py` - 28 tests for spatio-temporal encoder
- `tests/test_models/test_environmental_encoder.py` - 37 tests for wave/atmospheric encoders
- `tests/test_models/test_fusion.py` - 25 tests for cross-attention fusion
- `tests/test_models/test_prediction_heads.py` - 35 tests for all prediction heads
- `tests/test_models/test_cliffcast.py` - 26 tests for full model integration
- `tests/test_data/test_transect_extractor.py` - 30 tests for data pipeline
- `tests/test_apps/test_transect_viewer.py` - 27 tests for viewer app

**Test Coverage**:
- Shape validation: All tensor shapes verified at each stage
- Value ranges: Risk [0,1], retreat positive, probabilities [0,1]
- No NaN/Inf: All outputs checked for numerical stability
- Gradient flow: Backpropagation verified through entire model
- Attention extraction: Attention weights can be extracted and visualized
- Padding masks: Variable-length sequences handled correctly
- Phased training: Selective head enabling works as expected
- Edge cases: Single timestep, large batches, zero inputs, eval mode

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

### Study Site: San Diego County Beaches

The project covers coastal cliffs in San Diego County, organized by beach with MOP (Monitoring Point) ID ranges:

| Beach | MOP Range | Notes |
|-------|-----------|-------|
| Blacks | 520-567 | Black's Beach, steep cliffs |
| Torrey | 567-581 | Torrey Pines State Beach |
| Del Mar | 595-620 | Del Mar city beaches |
| Solana | 637-666 | Solana Beach |
| San Elijo | 683-708 | San Elijo State Beach |
| Encinitas | 708-764 | Encinitas/Moonlight Beach |

**IMPORTANT**: These MOP ranges are canonical and match the transect shapefile. Always use these ranges when:
- Filtering survey data by beach
- Subsetting transects for training/evaluation
- Reporting results by location
- Running extraction with `--beach` flag

Survey CSV files use MOP1/MOP2 columns to indicate coverage range per survey.

### Data Processing Conventions
- **Transect extraction**: Use `ShapefileTransectExtractor` with predefined transect lines from shapefile
- **Transect resampling**: Always N=128 points, uniformly spaced along transect profile
- **Buffer distance**: Default 1.0m around transect line for point collection
- **Cube format**: Transects organized as (n_transects, T, N, 12) data cube for spatio-temporal attention
  - T = number of LiDAR epochs (all time steps for each transect)
  - Each transect has full temporal coverage (no missing timesteps expected)
  - `las_sources` array maps to timestamps for temporal encoding
- **Feature normalization**: Intensity and RGB normalized to [0,1], classification as discrete codes
- **Wave timesteps**: 6-hourly for capturing storm dynamics (T_w = 360 for 90 days)
- **Precip timesteps**: Daily for antecedent moisture (T_p = 90 for 90 days)
- **Temporal alignment**: Environmental data aligned to most recent scan date; model learns from full temporal sequence
- **Missing data**: Interpolate or flag - never silently fill with zeros

### Wave Data (CDIP MOP System)

**Overview**: Wave forcing data comes from CDIP's MOP (Monitoring and Prediction) system, providing nearshore wave predictions at 100m alongshore spacing at 10m water depth. Data is hourly from 2000-present and stored as NetCDF files.

**Data Source**: CDIP THREDDS server
- THREDDS catalog: https://thredds.cdip.ucsd.edu/thredds/catalog/cdip/model/MOP_alongshore/catalog.html
- OPeNDAP access: Direct remote access for on-demand loading
- Local caching: Pre-download files to `data/raw/cdip/` for training (recommended)

**File Naming Convention**:
- San Diego MOPs: `D0520_hindcast.nc`, `D0521_hindcast.nc`, ..., `D0764_hindcast.nc`
- File size: ~10-100 MB per MOP (depends on date range)
- Each file contains full time series for one MOP location

**Features** (4 per timestep):
1. **hs** - Significant wave height (m)
2. **tp** - Peak period (s)
3. **dp** - Peak direction (deg from N, 0=north, 90=east)
4. **power** - Wave power flux (kW/m), computed as `(ρ*g²/64π) * Hs² * Tp`

**Temporal Configuration**:
- Lookback window: 90 days before scan date
- Resampling: 6-hour intervals (captures storm dynamics)
- Total timesteps: T_w = 360 for 90 days @ 6hr
- Circular mean for direction during resampling (sin/cos components)

**Integration Pattern**:
```python
from src.data.wave_loader import WaveLoader

# Initialize loader with local CDIP directory
loader = WaveLoader('data/raw/cdip/')

# Get wave data for single scan (MOP 582, Dec 15, 2023)
features, doy = loader.get_wave_for_scan(mop_id=582, scan_date='2023-12-15')
print(features.shape)  # (360, 4) - 90 days @ 6hr
print(doy.shape)       # (360,) - day-of-year for seasonality

# Batch loading for multiple transects
mop_ids = [582, 583, 584]
scan_dates = ['2023-12-15', '2023-12-15', '2023-12-15']
features, doy = loader.get_batch_wave(mop_ids, scan_dates)
print(features.shape)  # (3, 360, 4)

# Validate coverage before training
coverage = loader.validate_coverage(mop_ids, scan_dates)
print(f"Coverage: {coverage['coverage_pct']:.1f}%")
```

**Download Workflow**:
1. **Batch download**: Use `download_cdip_data.py` to populate `data/raw/cdip/`
2. **Verify coverage**: Check that >80% of MOPs have data
3. **Train with local files**: WaveLoader reads from disk (fast, no network dependency)

**Coverage Requirements**:
- Minimum 80% coverage for training (safety-critical predictions need reliable forcing)
- Missing data handling: Graceful degradation (log warning, return zeros)
- Pre-training validation: Run `validate_coverage()` to check all beaches

**Critical Implementation Details**:
- **Fill value handling**: CDIP uses -999.99 for invalid data; these are filtered to NaN
- **Circular mean resampling**: Direction uses sin/cos components, not simple averaging
- **Temporal alignment**: Wave lookback is always 90 days BEFORE scan date (no future leakage)
- **Cache strategy**: WaveData objects cached in memory (LRU eviction, default 50 MOPs)
- **MOP-to-site-label mapping**: Files may use D0582, D582, or D582 formats; loader tries all

**Expected Tensor Shapes**:
- Input to model: `(B, T_w, 4)` where B = batch size, T_w = 360
- Day-of-year: `(B, T_w)` for seasonality embedding
- Feature order: [hs, tp, dp, power] (always in this order)

**Data Organization**:
```
data/
├── raw/
│   ├── cdip/                  # Wave data (NetCDF files) ✅ 183 MOPs downloaded
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

**Current Integration Status** (as of 2026-01-19):
- ✅ **CDIPWaveLoader**: Fully implemented and tested
- ✅ **WaveLoader**: Fully implemented with 18 unit tests passing
- ✅ **Download Script**: Successfully downloaded 183/245 San Diego MOPs (~26GB)
- ✅ **Configuration**: Wave data settings added to configs/default.yaml
- ✅ **Documentation**: Complete usage examples and API documentation
- 🔄 **Next Step**: Integrate WaveLoader into PyTorch Dataset class for training

### Attention Interpretation
Multiple attention mechanisms reveal different physical attributions:

**Temporal attention** (across LiDAR epochs):
- High attention to specific past scans → those epochs show critical cliff evolution
- Identifies progressive weakening patterns, crack development, or precursor deformation
- Use `return_temporal_attention=True` to visualize which historical scans matter most

**Spatial attention** (within each timestep):
- Identifies which cliff locations are most predictive (e.g., overhangs, notches)
- Use `return_spatial_attention=True` to visualize per-point importance

**Cross-attention** (cliff → environment):
- High attention to specific wave timesteps → those storms contributed to erosion
- Spatial patterns in cliff points attending to events → local vs. regional forcing
- Use `return_env_attention=True` to extract weights for visualization

## Testing Strategy

### Current Test Suite (65 tests)
Run all tests: `pytest tests/ -v`

**test_data/test_transect_extractor.py** (30 tests):
- `TestShapefileTransectExtractorInit`: Initialization, feature/metadata names
- `TestTransectDirection`: Direction vector computation from LineString
- `TestTransectExtraction`: Point extraction and resampling
- `TestFeatureComputation`: Feature value ranges, metadata computation
- `TestSaveLoad`: NPZ save/load roundtrip
- `TestEdgeCases`: Insufficient points, edge case handling
- `TestFullPipeline`: End-to-end extraction with mocked LAS
- `TestCubeFormat`: Flat-to-cube conversion, temporal ordering, save/load
- `TestSubsetTransects`: MOP number parsing, cube subsetting by range
- `TestPathConversion`: Cross-platform path conversion (Mac/Linux)
- `TestDateParsing`: LAS filename date extraction

**test_apps/test_transect_viewer.py** (27 tests):
- `TestDataLoader`: Cube format detection, dimension extraction, transect retrieval, temporal slicing
- `TestValidators`: NaN checking, value range validation, dataset validation, statistics
- `TestSaveLoadRoundtrip`: NPZ roundtrip with string arrays

**test_utils.py** (8 tests):
- Config loading, validation, overrides, save/load

### Test Conventions
- Test shapes obsessively: `assert tensor.shape == expected_shape`
- Verify no NaN: `assert not torch.isnan(tensor).any()`
- Check value ranges (e.g., sigmoid outputs in [0,1])
- Use synthetic data fixtures for deterministic testing

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

1. **Index vs. Distance**: Transect spatial positional encoding uses distance from toe, NOT point index
2. **Temporal ordering**: LiDAR epochs must be sorted chronologically; oldest first, newest last
3. **Cube vs. Flat format**: Model expects (B, T, N, 12) cube format, not flat (B*T, N, 12)
4. **Concatenation order**: Environmental embeddings are [wave, precip] not [precip, wave]
5. **CLS token**: Prepended to sequence, so output has shape (B, T+1, d_model) after temporal attention
6. **Failure mode loss**: Only computed on samples with `failure_mode > 0` (not stable)
7. **Attention pooling**: Use learned attention weights, not simple mean pooling
8. **Temporal alignment**: Environmental data aligned to most recent scan; model sees full history
9. **Batch_first=True**: All transformers use `batch_first=True` convention
10. **Full temporal coverage**: Each transect in batch should have same T epochs (pad if needed)

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
