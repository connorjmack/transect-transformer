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

**CliffCast** is a transformer-based deep learning model that classifies coastal cliff erosion susceptibility. The model processes multi-temporal 1D transect data from LiDAR scans along with environmental forcing data (wave conditions and precipitation) to classify transects into 5 erosion mode classes: stable, beach erosion, cliff toe erosion, small rockfall, and large upper cliff failure.

**Core Philosophy**: Nowcast susceptibility, not event forecasting. The model answers "What erosion mode is this transect susceptible to?" rather than predicting when specific events will occur.

**Core Architecture**: Spatio-temporal attention over cliff geometry sequences, cross-attention fusion with environmental embeddings, and 5-class susceptibility classification head with derived risk scores.

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
# Prepare training data with water year splits
python scripts/processing/prepare_susceptibility_data.py --cube data/processed/unified_cube.npz --labels data/labels/erosion_mode_labels.csv --cdip-dir data/raw/cdip/ --atmos-dir data/processed/atmospheric/ --output-dir data/processed/

# Validate data before training
python scripts/processing/validate_susceptibility_data.py --data-dir data/processed/

# Train susceptibility classifier
python scripts/train_susceptibility.py --config configs/susceptibility_v1.yaml --wandb-project cliffcast
```

### Evaluation
```bash
# Evaluate susceptibility model on test set
python scripts/evaluate_susceptibility.py --checkpoint checkpoints/best.pt --data data/processed/susceptibility_test.npz --output results/

# Generate risk map from latest survey
python scripts/generate_risk_map.py --checkpoint checkpoints/best.pt --cube data/processed/latest_survey.npz --output results/risk_map.csv
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

# Launch labeling interface (for creating erosion mode labels)
streamlit run apps/labeling/app.py

# Launch risk map viewer (for deployment)
streamlit run apps/risk_map_viewer/app.py
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
- `SusceptibilityHead` - 5-class classification (stable, beach, toe, rockfall, large failure)
- `CliffCast` - Full model assembly with derived risk scores

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

## Erosion Mode Classes

| Class | Name | Physical Process | Risk Weight |
|-------|------|------------------|-------------|
| 0 | Stable | No significant change | 0.0 |
| 1 | Beach erosion | Sediment transport, tidal processes | 0.1 |
| 2 | Cliff toe erosion | Wave undercutting at cliff base | 0.4 |
| 3 | Small rockfall | Weathering-driven small failures | 0.6 |
| 4 | Large upper cliff failure | Major structural collapse | 1.0 |

**Dominance hierarchy** (for labeling): Large failure > Small rockfall > Toe erosion > Beach erosion > Stable

## Data Splits (Water Year Based)

| Set | Water Years | Date Range | Purpose |
|-----|-------------|------------|---------|
| Train | WY2017-WY2023 | Oct 2016 - Sep 2023 | Learn susceptibility patterns |
| Validation | WY2024 | Oct 2023 - Sep 2024 | Hyperparameter tuning, early stopping |
| Test | WY2025 | Oct 2024 - Sep 2025 | Final held-out evaluation |

**Note**: Water year N runs from October 1 of year N-1 to September 30 of year N.

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
6. **Susceptibility loss**: Use weighted cross-entropy with class weights [0.3, 1.0, 2.0, 2.0, 5.0] and label smoothing (0.1)
7. **Temporal alignment**: Environmental data aligned to most recent scan
8. **Batch_first=True**: All transformers use this convention
9. **Full temporal coverage**: Each transect in batch needs same T epochs (pad if needed)

## Development Tips

- **Debug small**: Use `d_model=32, n_layers=1` before full config
- **Synthetic first**: Validate on data with known relationships
- **Visualize attention early**: Catch encoder bugs during training
- **Use einops**: `rearrange(x, 'b n d -> b d n')` is clearer than `x.permute(0, 2, 1)`
- **Risk from probs**: Risk score is derived from class probabilities: `risk = sum(probs * [0.0, 0.1, 0.4, 0.6, 1.0])`
