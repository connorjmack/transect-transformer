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

**CliffCast** is a transformer-based deep learning model for predicting coastal cliff erosion risk. The model processes multi-temporal 1D transect data from LiDAR scans along with environmental forcing data (wave conditions and precipitation) to predict: risk index, collapse probability, event volume, and event classification.

**Core Architecture**: Spatio-temporal attention over cliff geometry sequences, cross-attention fusion with environmental embeddings, and multi-task prediction heads.

## Project History & Reference Docs

When starting a new phase or needing detailed context, read these on-demand:
- `docs/model_plan.md` - Target architecture spec
- `docs/plan.md` - Actionable implementation checklist
- `docs/project-evolution.md` - Completed phases, key decisions, lessons learned
- `docs/architecture-reference.md` - Detailed API docs, usage examples, data loaders
- `docs/data_requirements.md` - Data schemas and validation rules

## Directory Structure

```
transect-transformer/
├── docs/                    # Documentation (read on-demand)
├── apps/transect_viewer/    # Streamlit visualization app
├── src/
│   ├── data/                # Data loaders and extractors
│   ├── models/              # Model architecture components
│   ├── training/            # Training infrastructure
│   └── utils/               # Shared utilities
├── scripts/
│   ├── processing/          # Data pipeline scripts
│   └── visualization/       # Publication figures
├── configs/                 # Model configurations
└── tests/                   # Test suite (216 tests)
```

## Commands

### Environment & Testing
```bash
# Install
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Run all tests (216 tests)
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Type checking / Linting
mypy src/
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
# Evaluate on test set (TODO: not yet implemented)
python evaluate.py --checkpoint checkpoints/best.pt --data_dir data/processed/ --split test --output results/
```

### Data Preprocessing
```bash
# Extract transects for a single beach
python scripts/processing/extract_transects.py --transects data/mops/transects_10m/transect_lines.shp --survey-csv data/raw/master_list.csv --beach delmar --output data/processed/delmar.npz
# Beaches: blacks, torrey, delmar, solana, sanelijo, encinitas
# Add --prefer-laz for faster loading, --target-os linux for path conversion

# UNIFIED CUBE MODE (recommended)
python scripts/processing/extract_transects.py --transects data/mops/transects_10m/transect_lines.shp --survey-csv data/raw/master_list.csv --output data/processed/unified_cube.npz --unified --prefer-laz --workers 8

# Subset existing cube by beach
python scripts/processing/subset_transects.py --input data/processed/all_transects.npz --output data/processed/delmar.npz --beach delmar

# Download CDIP wave data
python scripts/processing/download_cdip_data.py --output data/raw/cdip/ --beach delmar

# Detect cliff toe/top edges
python scripts/processing/detect_cliff_edges.py --input data/processed/delmar.npz --checkpoint /path/to/best_model.pth
```

### Interactive Apps
```bash
# Launch transect viewer
streamlit run apps/transect_viewer/app.py
```

### Visualization Scripts
```bash
# Wave climate figures (8 figures)
python scripts/visualization/wave_climate_figures.py --cdip-dir data/raw/cdip/ --output figures/appendix/

# PRISM atmospheric figures (3 figures)
python scripts/visualization/plot_prism_coverage.py --atmos-dir data/processed/atmospheric/ --output-dir figures/appendix/
```

## Architecture Summary

> **Full details**: See `docs/architecture-reference.md` for API docs and usage examples.

**Model Components** (all implemented with 100% test coverage):
- `SpatioTemporalTransectEncoder` - Hierarchical attention for cliff geometry
- `WaveEncoder` / `AtmosphericEncoder` - Environmental time series
- `CrossAttentionFusion` - Fuses cliff with environment
- `PredictionHeads` - Multi-task outputs (volume, event class, risk, collapse)
- `CliffCast` - Full model assembly

**Data Components**:
- `ShapefileTransectExtractor` - LiDAR transect extraction
- `WaveLoader` / `AtmosphericLoader` - Environmental data for training
- `CliffDelineation` - Cliff toe/top detection

## Shape Expectations

| Symbol | Meaning | Typical Value |
|--------|---------|---------------|
| B | Batch size | 32 |
| T | LiDAR epochs | 10 |
| N | Transect points | 128 |
| T_w | Wave timesteps | 360 (90 days @ 6hr) |
| T_a | Atmospheric timesteps | 90 (90 days @ daily) |
| d_model | Hidden dimension | 256 |

**Cube format**: `(n_transects, T, N, 12)` point features + `(n_transects, T, 12)` metadata

## Study Site: San Diego County Beaches

| Beach | MOP Range | Notes |
|-------|-----------|-------|
| Blacks | 520-567 | Black's Beach, steep cliffs |
| Torrey | 567-581 | Torrey Pines State Beach |
| Del Mar | 595-620 | Del Mar city beaches |
| Solana | 637-666 | Solana Beach |
| San Elijo | 683-708 | San Elijo State Beach |
| Encinitas | 708-764 | Encinitas/Moonlight Beach |

**IMPORTANT**: These MOP ranges are canonical. Use them for filtering, subsetting, and the `--beach` flag.

## Data Processing Conventions

- **Transect spacing**: 10m aligned with MOP lines (~1958 transects total)
- **Transect resampling**: Always N=128 points
- **Unified cube format**: Includes `coverage_mask`, `beach_slices`, `mop_ids`
- **Wave data**: 90-day lookback, 6-hour intervals, 4 features [hs, tp, dp, power]
- **Atmospheric data**: 90-day lookback, daily intervals, 24 features
- **Temporal alignment**: Environmental data aligned to scan date (no future leakage)

## Common Pitfalls

1. **Index vs. Distance**: Spatial positional encoding uses distance from toe, NOT point index
2. **Temporal ordering**: LiDAR epochs must be chronological; oldest first, newest last
3. **Cube vs. Flat format**: Model expects `(B, T, N, 12)`, not `(B*T, N, 12)`
4. **Concatenation order**: Environmental embeddings are `[wave, atmos]` not `[atmos, wave]`
5. **CLS token**: Prepended to sequence, so output shape is `(B, T+1, d_model)`
6. **Event class loss**: Use focal loss for class imbalance (most samples are stable)
7. **Temporal alignment**: Environmental data aligned to most recent scan
8. **Batch_first=True**: All transformers use this convention
9. **Full temporal coverage**: Each transect in batch needs same T epochs (pad if needed)

## Development Tips

- **Debug small**: Use `d_model=32, n_layers=1` before full config
- **Synthetic first**: Validate on data with known relationships
- **Visualize attention early**: Catch encoder bugs during training
- **Use einops**: `rearrange(x, 'b n d -> b d n')` is clearer than `x.permute(0, 2, 1)`
- **One head at a time**: Perfect risk index before adding other heads
