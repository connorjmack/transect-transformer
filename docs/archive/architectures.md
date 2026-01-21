# CliffCast Architecture Diagrams

This document contains visual diagrams of data pipelines, model architecture, and processing workflows.

---

## Table of Contents

1. [Transect Processing Pipeline](#transect-processing-pipeline)
2. [File Structure](#file-structure)

---

## Transect Processing Pipeline

The transect extraction and processing workflow converts raw LAS files into cliff-centered, model-ready tensors.

```
Raw LAS files (weekly/biweekly surveys)
     │
     ▼
┌─────────────────────────────────────┐
│  ShapefileTransectExtractor         │
│                                     │
│  • Input: LAS + transect shapefile  │
│  • Output: 256 points × 12 features │
│  • Full transect (beach → inland)   │
│  • 10m spacing along coast          │
└─────────────────┬───────────────────┘
                  │
                  ▼
           data/raw/transects/
           {beach}_{survey_date}.npz
           (256 points × 12 features)
                  │
                  ▼
┌─────────────────────────────────────┐
│  CliffDelineation                   │
│  (CNN-BiLSTM model)                 │
│                                     │
│  • Input: raw transect profiles     │
│  • Output: toe/top indices + conf   │
│  • Per transect-epoch detection     │
└─────────────────┬───────────────────┘
                  │
                  ▼
           data/raw/transects/
           {beach}_{survey_date}.cliff.npz (sidecar)
           (toe_indices, top_indices, confidences)
                  │
                  ▼
┌─────────────────────────────────────┐
│  TransectProcessor                  │
│                                     │
│  • Crop to cliff window:            │
│    [toe - 10m] to [top + 5m]        │
│  • Resample to 128 points           │
│  • Drop features: RGB, return info  │
│  • Keep: dist, elev, slope, curv,   │
│          roughness, intensity, class│
│  • Recompute slope/curvature        │
│  • Add window metadata              │
└─────────────────┬───────────────────┘
                  │
                  ▼
           data/processed/transects/
           {beach}_{survey_date}.npz
           (128 points × 7 features, cliff-centered)
                  │
                  ▼
┌─────────────────────────────────────┐
│  M3C2 Change Detection              │
│  (CloudCompare or similar)          │
│                                     │
│  • Compare consecutive epochs       │
│  • Compute signed distance field    │
│  • Negative = erosion               │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  CubeBuilder                        │
│                                     │
│  • Combine all surveys into cube    │
│  • Align transects across epochs    │
│  • Add M3C2 as 8th feature          │
│  • Add coverage mask                │
│  • Output unified training cube     │
└─────────────────┬───────────────────┘
                  │
                  ▼
           data/processed/
           unified_cube.npz
           (n_transects × n_epochs × 128 × 8)
```

---

## Feature Evolution Through Pipeline

### Raw Extraction (12 features)

| Index | Feature | Status |
|-------|---------|--------|
| 0 | distance_m | Keep |
| 1 | elevation_m | Keep |
| 2 | slope_deg | Keep (recompute) |
| 3 | curvature | Keep (recompute) |
| 4 | roughness | Keep |
| 5 | intensity | Keep |
| 6 | red | **Drop** |
| 7 | green | **Drop** |
| 8 | blue | **Drop** |
| 9 | classification | Keep |
| 10 | return_number | **Drop** |
| 11 | num_returns | **Drop** |

### Processed Output (8 features)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | distance_m | Distance along transect (relative to window start) |
| 1 | elevation_m | Elevation (NAVD88) |
| 2 | slope_deg | Local slope (recomputed after resampling) |
| 3 | curvature | Profile curvature (recomputed after resampling) |
| 4 | roughness | Surface roughness |
| 5 | intensity | LAS intensity [0,1] |
| 6 | classification | LAS class code |
| 7 | m3c2_distance | Change from previous epoch (0 for first epoch) |

---

## File Structure

```
data/
├── raw/
│   ├── las/                          # Original LAS/LAZ files (not in repo)
│   │   ├── 2017/
│   │   │   ├── 20170115_delmar.laz
│   │   │   └── ...
│   │   └── 2025/
│   │       └── ...
│   │
│   ├── transects/                    # Raw extracted transects (256 pts × 12 feat)
│   │   ├── delmar_20170115.npz
│   │   ├── delmar_20170115.cliff.npz # Cliff delineation sidecar
│   │   ├── delmar_20170122.npz
│   │   ├── delmar_20170122.cliff.npz
│   │   └── ...
│   │
│   ├── cdip/                         # Wave data (NetCDF)
│   │   ├── D0520_hindcast.nc
│   │   └── ...
│   │
│   ├── prism/                        # Raw climate data
│   │   └── ...
│   │
│   ├── events/                       # M3C2-derived event CSVs (for reference)
│   │   ├── delmar_events.csv
│   │   └── ...
│   │
│   └── master_list.csv               # Survey metadata
│
├── processed/
│   ├── transects/                    # Cliff-centered transects (128 pts × 7 feat)
│   │   ├── delmar_20170115.npz
│   │   ├── delmar_20170122.npz
│   │   └── ...
│   │
│   ├── atmospheric/                  # Computed atmospheric features (parquet)
│   │   ├── delmar_atmos.parquet
│   │   └── ...
│   │
│   ├── unified_cube.npz              # All transects × all epochs × 128 × 8
│   │
│   └── splits/                       # Train/val/test indices
│       ├── train_indices.npy
│       ├── val_indices.npy
│       └── test_indices.npy
│
└── labels/
    ├── erosion_mode_labels.csv       # Manual class labels
    └── labeling_progress.json        # Tracking file
```

---

## Metadata Structure

### Per-Survey NPZ (raw/transects/)

```python
{
    'points': (n_transects, 256, 12),      # Full transect, all features
    'distances': (n_transects, 256),        # Distance along transect
    'metadata': (n_transects, 12),          # Transect-level metadata
    'transect_ids': (n_transects,),         # String IDs
    'survey_date': str,                     # ISO format
    'las_sources': list,                    # Source files
    'feature_names': list,                  # 12 feature names
    'metadata_names': list,                 # 12 metadata names
}
```

### Cliff Delineation Sidecar (raw/transects/*.cliff.npz)

```python
{
    'toe_indices': (n_transects,),          # Point index of cliff toe (0-255)
    'top_indices': (n_transects,),          # Point index of cliff top (0-255)
    'toe_distances': (n_transects,),        # Distance along transect to toe
    'top_distances': (n_transects,),        # Distance along transect to top
    'toe_confidences': (n_transects,),      # Detection confidence [0,1]
    'top_confidences': (n_transects,),      # Detection confidence [0,1]
    'has_cliff': (n_transects,),            # Boolean: cliff detected
}
```

### Per-Survey NPZ (processed/transects/)

```python
{
    'points': (n_transects, 128, 7),        # Cliff-centered, reduced features
    'distances': (n_transects, 128),        # Distance (relative to window start)
    'metadata': (n_transects, 14),          # Extended metadata
    'transect_ids': (n_transects,),         # String IDs
    'survey_date': str,                     # ISO format
    'feature_names': list,                  # 7 feature names
    'metadata_names': list,                 # 14 metadata names

    # Window information (new)
    'window_start_m': (n_transects,),       # Original distance to window start
    'window_end_m': (n_transects,),         # Original distance to window end
    'toe_distance_m': (n_transects,),       # Cliff toe in original coordinates
    'top_distance_m': (n_transects,),       # Cliff top in original coordinates
    'delineation_confidence': (n_transects,), # Mean of toe/top confidence
}
```

### Unified Cube (processed/unified_cube.npz)

```python
{
    # Core data
    'points': (n_transects, n_epochs, 128, 8),  # Includes M3C2 as 8th feature
    'distances': (n_transects, n_epochs, 128),
    'metadata': (n_transects, n_epochs, 14),

    # Temporal info
    'timestamps': (n_epochs,),               # Ordinal dates
    'epoch_dates': (n_epochs,),              # ISO date strings
    'coverage_mask': (n_transects, n_epochs), # Bool: data exists

    # Spatial info
    'transect_ids': (n_transects,),
    'mop_ids': (n_transects,),
    'beach_slices': dict,                    # {'delmar': (start, end), ...}

    # Feature info
    'feature_names': list,                   # 8 feature names
    'metadata_names': list,
}
```

---

## Processing Commands

```bash
# Step 1: Extract raw transects from LAS (256 points, full coverage)
python scripts/processing/extract_transects.py \
    --transects data/mops/transect_lines.shp \
    --las-dir data/raw/las/2023/ \
    --output data/raw/transects/ \
    --n-points 256 \
    --workers 8

# Step 2: Run cliff delineation
python scripts/processing/detect_cliffs.py \
    --input data/raw/transects/ \
    --checkpoint models/cliff_delineation/best_model.pth \
    --confidence-threshold 0.5

# Step 3: Process transects (crop, resample, clean features)
python scripts/processing/process_transects.py \
    --input data/raw/transects/ \
    --output data/processed/transects/ \
    --n-points 128 \
    --toe-buffer 10 \
    --top-buffer 5

# Step 4: Build unified cube
python scripts/processing/build_cube.py \
    --input data/processed/transects/ \
    --output data/processed/unified_cube.npz \
    --add-m3c2

# Step 5: Validate
python scripts/processing/validate_cube.py \
    --cube data/processed/unified_cube.npz
```
