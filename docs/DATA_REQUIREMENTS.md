# CliffCast Data Requirements

> **Purpose**: This document defines all data contracts, schemas, validation rules, and processing specifications for the CliffCast coastal erosion prediction system. It serves as the authoritative reference for agentic code workflows.

---

## Table of Contents

1. [Data Sources Overview](#1-data-sources-overview)
2. [Transect Cube Specification](#2-transect-cube-specification)
3. [Event Data Specification](#3-event-data-specification)
4. [Wave Data Specification](#4-wave-data-specification)
5. [Atmospheric Data Specification](#5-atmospheric-data-specification)
6. [Coordinate System Mappings](#6-coordinate-system-mappings)
7. [Training Data Specification](#7-training-data-specification)
8. [Validation Rules](#8-validation-rules)
9. [File Locations & Naming Conventions](#9-file-locations--naming-conventions)
10. [Processing Pipeline Specifications](#10-processing-pipeline-specifications)

---

## 1. Data Sources Overview

### Summary Table

| Data Source | Format | Location | Update Frequency | Size |
|-------------|--------|----------|------------------|------|
| Transect Cube | NPZ | `data/processed/unified_cube.npz` | Per LiDAR campaign | ~2-5 GB |
| Event CSVs | CSV | `data/raw/events/{beach}_events_sig.csv` | Per M3C2 analysis | ~50 KB each |
| Wave Data | NetCDF | `data/raw/cdip/D{MOP}_hindcast.nc` | Static (2000-present) | ~50 MB each |
| Atmospheric Data | Parquet | `data/processed/atmospheric/{beach}_atmos.parquet` | Static (2017-2025) | ~5 MB each |
| Training Data | NPZ | `data/processed/training_data.npz` | Generated | ~1-2 GB |

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW OVERVIEW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   RAW DATA                                                                   │
│   ────────                                                                   │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│   │   LiDAR      │  │   Event      │  │    CDIP      │  │    PRISM     │   │
│   │   (LAS/LAZ)  │  │   (CSV)      │  │   (NetCDF)   │  │   (BIL)      │   │
│   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
│          │                 │                 │                 │            │
│          ▼                 │                 │                 ▼            │
│   PROCESSED DATA           │                 │          ┌──────────────┐   │
│   ──────────────           │                 │          │   Atmos      │   │
│   ┌──────────────┐         │                 │          │  (Parquet)   │   │
│   │  Transect    │         │                 │          └──────┬───────┘   │
│   │    Cube      │         │                 │                 │            │
│   │   (NPZ)      │         │                 │                 │            │
│   └──────┬───────┘         │                 │                 │            │
│          │                 │                 │                 │            │
│          └────────┬────────┴────────┬────────┴─────────────────┘            │
│                   │                 │                                        │
│                   ▼                 ▼                                        │
│            ┌─────────────────────────────────┐                              │
│            │      TRAINING DATA              │                              │
│            │         (NPZ)                   │                              │
│            │                                 │                              │
│            │  - Aligned transects            │                              │
│            │  - Matched events               │                              │
│            │  - Environmental features       │                              │
│            │  - Labels & confidence          │                              │
│            └─────────────────────────────────┘                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Transect Cube Specification

### 2.1 File Format

- **Format**: NumPy NPZ (compressed)
- **Primary file**: `data/processed/unified_cube.npz`
- **Per-beach files**: `data/processed/{beach}.npz` (optional, for development)

### 2.2 Schema

```python
# Load with: data = np.load('unified_cube.npz', allow_pickle=True)

cube_schema = {
    # === CORE ARRAYS ===
    'points': {
        'dtype': 'float32',
        'shape': (n_transects, n_epochs, 128, 13),
        'description': 'Per-point features for all transects across all epochs',
    },
    'distances': {
        'dtype': 'float32',
        'shape': (n_transects, n_epochs, 128),
        'description': 'Distance along transect from start (meters)',
    },
    'metadata': {
        'dtype': 'float32',
        'shape': (n_transects, n_epochs, 12),
        'description': 'Transect-level metadata per epoch',
    },
    'coverage_mask': {
        'dtype': 'bool',
        'shape': (n_transects, n_epochs),
        'description': 'True where valid data exists',
    },
    
    # === TEMPORAL INFO ===
    'timestamps': {
        'dtype': 'int32',
        'shape': (n_epochs,),
        'description': 'Ordinal dates (days since 0001-01-01) for each epoch',
    },
    'epoch_dates': {
        'dtype': 'str',  # stored as object array
        'shape': (n_epochs,),
        'description': 'ISO format date strings (YYYY-MM-DD)',
    },
    'epoch_files': {
        'dtype': 'str',
        'shape': (n_epochs,),
        'description': 'Source LAS filenames',
    },
    
    # === SPATIAL INFO ===
    'transect_ids': {
        'dtype': 'str',
        'shape': (n_transects,),
        'description': 'Full transect ID strings (e.g., "MOP 595", "MOP 595_001")',
    },
    'mop_ids': {
        'dtype': 'int32',
        'shape': (n_transects,),
        'description': 'Integer MOP IDs extracted from transect strings',
    },
    'beach_slices': {
        'dtype': 'dict',  # stored as object
        'description': 'Index ranges per beach: {"delmar": (start, end), ...}',
    },
    
    # === FEATURE NAMES ===
    'feature_names': {
        'dtype': 'str',
        'shape': (13,),
        'description': 'Names of the 13 per-point features',
    },
    'metadata_names': {
        'dtype': 'str',
        'shape': (12,),
        'description': 'Names of the 12 metadata fields',
    },
}
```

### 2.3 Point Features (13 total)

| Index | Name | Dtype | Units | Range | Description |
|-------|------|-------|-------|-------|-------------|
| 0 | `distance_m` | float32 | meters | [0, ~100] | Distance from transect start |
| 1 | `elevation_m` | float32 | meters (NAVD88) | [-5, 50] | Point elevation |
| 2 | `slope_deg` | float32 | degrees | [0, 90] | Local slope angle |
| 3 | `curvature` | float32 | 1/m | [-1, 1] | Profile curvature |
| 4 | `roughness` | float32 | meters | [0, 2] | Local surface roughness (std) |
| 5 | `intensity` | float32 | normalized | [0, 1] | LAS intensity |
| 6 | `red` | float32 | normalized | [0, 1] | Red channel |
| 7 | `green` | float32 | normalized | [0, 1] | Green channel |
| 8 | `blue` | float32 | normalized | [0, 1] | Blue channel |
| 9 | `classification` | float32 | code | {0-31} | LAS classification |
| 10 | `return_number` | float32 | count | [1, 5] | Return number |
| 11 | `num_returns` | float32 | count | [1, 5] | Total returns |
| **12** | **`m3c2_distance`** | **float32** | **meters** | **[-5, 2]** | **Change to previous epoch** |

#### M3C2 Distance Conventions

```python
# M3C2 distance interpretation:
#   Negative: Material loss (erosion) - cliff retreated
#   Positive: Material gain (deposition or error)
#   Zero: No change
#   NaN: No previous epoch or no correspondence

# First epoch handling:
#   Set m3c2_distance = 0.0 for all points (no previous surface)

# Normalization for model input:
def normalize_m3c2(raw: np.ndarray) -> np.ndarray:
    """Normalize M3C2 to [-1, 1] range."""
    clipped = np.clip(raw, -5.0, 2.0)
    normalized = clipped / 5.0
    return np.nan_to_num(normalized, nan=0.0)
```

### 2.4 Metadata Fields (12 total)

| Index | Name | Dtype | Units | Range | Description |
|-------|------|-------|-------|-------|-------------|
| 0 | `cliff_height_m` | float32 | meters | [5, 40] | Total cliff height |
| 1 | `mean_slope_deg` | float32 | degrees | [30, 80] | Average cliff face slope |
| 2 | `max_slope_deg` | float32 | degrees | [45, 90] | Maximum slope (overhang indicator) |
| 3 | `toe_elevation_m` | float32 | meters | [-2, 10] | Transect start elevation |
| 4 | `top_elevation_m` | float32 | meters | [10, 45] | Transect end elevation |
| 5 | `orientation_deg` | float32 | degrees | [180, 360] | Azimuth from north |
| 6 | `transect_length_m` | float32 | meters | [20, 150] | Total transect length |
| 7 | `latitude` | float32 | degrees | [32.8, 33.1] | Midpoint latitude |
| 8 | `longitude` | float32 | degrees | [-117.4, -117.2] | Midpoint longitude |
| 9 | `transect_id` | float32 | integer | [520, 764] | Numeric MOP ID |
| 10 | `mean_intensity` | float32 | normalized | [0, 1] | Mean LAS intensity |
| 11 | `dominant_class` | float32 | code | {0-31} | Most common classification |

### 2.5 Beach Slices

**Important: Two Transect Spacing Contexts**

| Context | Spacing | Total Transects | Used For |
|---------|---------|-----------------|----------|
| **Volume Estimation** | 1m | ~19,600 | M3C2 change detection, event volume calculations |
| **Model Training** | 10m | ~1,958 | CliffCast transformer input (target spacing) |

The supervised learning labels (event volumes) were computed using 1m transect spacing for accuracy.
The model itself uses 10m spacing to reduce computation and match MOP monitoring points.

```python
# Canonical beach definitions - MOP ID ranges (fixed, 100m spacing between MOPs)
BEACH_MOP_RANGES = {
    'blacks': (520, 567),
    'torrey': (567, 581),
    'delmar': (595, 620),
    'solana': (637, 666),
    'sanelijo': (683, 708),
    'encinitas': (708, 764),
}

# Beach slices for 10m transect spacing (MODEL TRAINING - target format)
# Total: ~1,958 transects
BEACH_SLICES_10M = {
    'blacks': (0, 479),         # MOP 520-567, 479 transects
    'torrey': (479, 609),       # MOP 568-581, 130 transects
    'delmar': (609, 857),       # MOP 595-620, 248 transects
    'solana': (857, 1145),      # MOP 637-666, 288 transects
    'sanelijo': (1145, 1408),   # MOP 683-708, 263 transects
    'encinitas': (1408, 1958),  # MOP 709-764, 550 transects
}

# Beach slices for 1m transect spacing (VOLUME ESTIMATION only)
# Total: ~19,600 transects
BEACH_SLICES_1M = {
    'blacks': (0, 4700),        # MOP 520-567, ~4700 transects
    'torrey': (4700, 6100),     # MOP 567-581, ~1400 transects
    'delmar': (6100, 8600),     # MOP 595-620, ~2500 transects
    'solana': (8600, 11500),    # MOP 637-666, ~2900 transects
    'sanelijo': (11500, 14000), # MOP 683-708, ~2500 transects
    'encinitas': (14000, 19600),# MOP 708-764, ~5600 transects
}

# IMPORTANT: Always verify with cube['beach_slices'] from actual file.
# The cube stores the actual indices used during extraction.
```

---

## 3. Event Data Specification

### 3.1 File Format

- **Format**: CSV
- **Location**: `data/raw/events/{beach}_events_sig.csv`
- **Naming**: `{beach}_events_sig.csv` where beach is lowercase
- **Encoding**: UTF-8
- **Delimiter**: Comma

### 3.2 Available Files

| Beach | Filename Options | Status |
|-------|------------------|--------|
| Blacks | `blacks_events_sig.csv` or `Blacks_events_sig.csv` | **NOT YET AVAILABLE** - will be added when M3C2 analysis is complete |
| Torrey | `torrey_events_sig.csv` or `Torrey_events_sig.csv` | Available |
| Del Mar | `delmar_events_sig.csv` or `DelMar_events_sig.csv` | Available |
| Solana | `solana_events_sig.csv` or `Solana_events_sig.csv` | Available |
| San Elijo | `sanelijo_events_sig.csv` or `SanElijo_events_sig.csv` | Available |
| Encinitas | `encinitas_events_sig.csv` or `Encinitas_events_sig.csv` | Available |

**Note**: The loader supports both lowercase (`delmar_events_sig.csv`) and mixed-case (`DelMar_events_sig.csv`) filenames for flexibility.

### 3.3 Schema

```python
event_schema = {
    'mid_date': {
        'dtype': 'datetime64',
        'format': 'YYYY-MM-DD HH:MM:SS',
        'nullable': False,
        'description': 'Event midpoint date (center of detection window)',
    },
    'start_date': {
        'dtype': 'datetime64',
        'format': 'YYYY-MM-DD',
        'nullable': False,
        'description': 'Detection window start (earlier LiDAR scan)',
    },
    'end_date': {
        'dtype': 'datetime64',
        'format': 'YYYY-MM-DD',
        'nullable': False,
        'description': 'Detection window end (later LiDAR scan)',
    },
    'volume': {
        'dtype': 'float64',
        'units': 'cubic meters (m³)',
        'range': [10, 1000],  # Significant events only
        'nullable': False,
        'description': 'Eroded volume from M3C2 gridded analysis',
    },
    'vol_unc': {
        'dtype': 'float64',
        'units': 'cubic meters (m³)',
        'range': [0, 100],
        'nullable': False,
        'description': 'Volume uncertainty (propagated from M3C2)',
    },
    'elevation': {
        'dtype': 'float64',
        'units': 'meters (NAVD88)',
        'range': [5, 25],
        'nullable': False,
        'description': 'Centroid elevation of event',
    },
    'alongshore_centroid_m': {
        'dtype': 'float64',
        'units': 'meters',
        'range': [0, 6000],  # Varies by beach
        'nullable': False,
        'description': 'Local alongshore coordinate (from south end)',
    },
    'alongshore_start_m': {
        'dtype': 'float64',
        'units': 'meters',
        'nullable': False,
        'description': 'Southern extent of event',
    },
    'alongshore_end_m': {
        'dtype': 'float64',
        'units': 'meters',
        'nullable': False,
        'description': 'Northern extent of event',
    },
    'width': {
        'dtype': 'float64',
        'units': 'meters',
        'range': [1, 60],
        'nullable': False,
        'description': 'Alongshore width of event',
    },
    'height': {
        'dtype': 'float64',
        'units': 'meters',
        'range': [0.5, 20],
        'nullable': False,
        'description': 'Vertical extent of event',
    },
    'month': {
        'dtype': 'int64',
        'range': [1, 12],
        'nullable': False,
        'description': 'Month of year (for seasonality)',
    },
}
```

### 3.4 Erosion Mode Classification

The model uses **manually labeled erosion mode classes** based on physical process, not volume thresholds. Labels are assigned during the labeling phase using the dominance hierarchy.

```python
# 5-class erosion mode classification
# Based on physical process, not volume

EROSION_MODE_CLASSES = {
    0: 'stable',              # No significant change
    1: 'beach_erosion',       # Sediment transport, tidal processes
    2: 'toe_erosion',         # Wave undercutting at cliff base
    3: 'small_rockfall',      # Weathering-driven small failures
    4: 'large_failure',       # Major structural collapse
}

EROSION_MODE_NAMES = ['stable', 'beach_erosion', 'toe_erosion', 'small_rockfall', 'large_failure']

# Dominance hierarchy for labeling (when multiple processes occur)
# Large failure > Small rockfall > Toe erosion > Beach erosion > Stable
DOMINANCE_ORDER = [4, 3, 2, 1, 0]

# Risk weights for derived risk score
RISK_WEIGHTS = [0.0, 0.1, 0.4, 0.6, 1.0]
```

**Note**: Event CSVs from M3C2 analysis are used for pre-filtering candidates and providing M3C2 change visualization during labeling, but the final erosion mode labels are assigned manually.

### 3.5 Loading Events

```python
import pandas as pd
from pathlib import Path

# Filename patterns to try (supports both lowercase and mixed-case)
EVENT_FILENAME_PATTERNS = {
    'blacks': ['blacks_events_sig.csv', 'Blacks_events_sig.csv'],
    'torrey': ['torrey_events_sig.csv', 'Torrey_events_sig.csv'],
    'delmar': ['delmar_events_sig.csv', 'DelMar_events_sig.csv'],
    'solana': ['solana_events_sig.csv', 'Solana_events_sig.csv'],
    'sanelijo': ['sanelijo_events_sig.csv', 'SanElijo_events_sig.csv'],
    'encinitas': ['encinitas_events_sig.csv', 'Encinitas_events_sig.csv'],
}


def load_events(beach: str, events_dir: str = 'data/raw/events') -> pd.DataFrame:
    """
    Load event CSV for a specific beach.

    Supports both lowercase and mixed-case filenames (e.g., delmar_events_sig.csv
    or DelMar_events_sig.csv).

    Args:
        beach: Beach name (lowercase): blacks, torrey, delmar, solana, sanelijo, encinitas
        events_dir: Directory containing event CSVs

    Returns:
        DataFrame with parsed dates and validated columns

    Raises:
        FileNotFoundError: If no matching event file found
        ValueError: If required columns are missing
    """
    events_path = Path(events_dir)

    # Try each filename pattern for this beach
    patterns = EVENT_FILENAME_PATTERNS.get(beach.lower(), [f'{beach}_events_sig.csv'])

    path = None
    for pattern in patterns:
        candidate = events_path / pattern
        if candidate.exists():
            path = candidate
            break

    if path is None:
        tried = ', '.join(patterns)
        raise FileNotFoundError(f'Event file not found for {beach}. Tried: {tried}')

    df = pd.read_csv(path)

    # Parse dates
    df['mid_date'] = pd.to_datetime(df['mid_date'])
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])

    # Validate required columns
    required = ['mid_date', 'start_date', 'end_date', 'volume', 'vol_unc',
                'elevation', 'alongshore_centroid_m', 'width', 'height']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    # Add beach identifier (normalized to lowercase)
    df['beach'] = beach.lower()

    # Note: erosion_mode labels are assigned manually during labeling,
    # not derived from volume. Event CSVs are used for pre-filtering
    # and M3C2 visualization during the labeling process.

    return df


def load_all_events(events_dir: str = 'data/raw/events') -> pd.DataFrame:
    """
    Load and concatenate events from all beaches.

    Note: Blacks beach events are not yet available and will be skipped.
    """
    beaches = ['blacks', 'torrey', 'delmar', 'solana', 'sanelijo', 'encinitas']

    dfs = []
    for beach in beaches:
        try:
            df = load_events(beach, events_dir)
            dfs.append(df)
            print(f'  {beach}: {len(df)} events')
        except FileNotFoundError:
            # Blacks events not yet available - this is expected
            if beach == 'blacks':
                print(f'  {beach}: NOT YET AVAILABLE (pending M3C2 analysis)')
            else:
                print(f'  {beach}: NOT FOUND')

    if not dfs:
        raise ValueError('No event files found')

    combined = pd.concat(dfs, ignore_index=True)
    print(f'Total: {len(combined)} events')

    return combined
```

---

## 4. Wave Data Specification

### 4.1 File Format

- **Format**: NetCDF4
- **Source**: CDIP THREDDS server
- **Location**: `data/raw/cdip/D{MOP:04d}_hindcast.nc`
- **Coverage**: MOP 520-764 (San Diego region)

### 4.2 File Naming

```python
# MOP ID to filename
def mop_to_cdip_filename(mop_id: int) -> str:
    """Convert MOP ID to CDIP hindcast filename."""
    return f'D{mop_id:04d}_hindcast.nc'

# Examples:
#   MOP 520 → 'D0520_hindcast.nc'
#   MOP 595 → 'D0595_hindcast.nc'

# Alternative formats (loader should try all):
CDIP_FILENAME_PATTERNS = [
    'D{mop:04d}_hindcast.nc',  # D0520_hindcast.nc
    'D{mop:03d}_hindcast.nc',  # D520_hindcast.nc
    'D{mop}_hindcast.nc',      # D520_hindcast.nc
]
```

### 4.3 NetCDF Variables

| Variable | Dimensions | Units | Description |
|----------|------------|-------|-------------|
| `waveTime` | (time,) | seconds since 1970-01-01 | Unix timestamp |
| `waveHs` | (time,) | meters | Significant wave height |
| `waveTp` | (time,) | seconds | Peak period |
| `waveDp` | (time,) | degrees | Peak direction (from, compass) |
| `waveTa` | (time,) | seconds | Average period |

### 4.4 Derived Features

```python
# Wave power computation
def compute_wave_power(hs: np.ndarray, tp: np.ndarray) -> np.ndarray:
    """
    Compute wave power flux (kW/m).
    
    Formula: P = (ρ * g² / 64π) * Hs² * Tp
    With ρ=1025 kg/m³, g=9.81 m/s²:
    P ≈ 0.49 * Hs² * Tp (kW/m)
    """
    return 0.49 * hs**2 * tp
```

### 4.5 Model Input Format

```python
# Shape: (n_samples, T_w, 4) where T_w = 360
# 90 days @ 6-hour intervals = 360 timesteps

WAVE_FEATURES = {
    0: 'hs',     # Significant wave height (m)
    1: 'tp',     # Peak period (s)
    2: 'dp',     # Peak direction (degrees, 0=N, 90=E)
    3: 'power',  # Wave power flux (kW/m)
}

WAVE_LOOKBACK_DAYS = 90
WAVE_INTERVAL_HOURS = 6
WAVE_TIMESTEPS = (WAVE_LOOKBACK_DAYS * 24) // WAVE_INTERVAL_HOURS  # 360

# Day-of-year array for seasonality encoding
# Shape: (n_samples, T_w)
# Values: 1-366 (handles leap years)
```

### 4.6 Fill Value Handling

```python
# CDIP fill values
CDIP_FILL_VALUE = -999.99

def clean_cdip_data(arr: np.ndarray) -> np.ndarray:
    """Replace CDIP fill values with NaN."""
    arr = arr.copy()
    arr[arr <= -999] = np.nan
    return arr
```

---

## 5. Atmospheric Data Specification

### 5.1 File Format

- **Format**: Parquet
- **Source**: PRISM daily climate data (processed)
- **Location**: `data/processed/atmospheric/{beach}_atmos.parquet`

### 5.2 Schema

```python
# Parquet columns
atmos_schema = {
    'date': 'datetime64[ns]',
    'mop_id': 'int32',
    
    # Raw PRISM variables
    'precip_mm': 'float32',      # Daily precipitation
    'tmax_c': 'float32',         # Max temperature
    'tmin_c': 'float32',         # Min temperature  
    'tmean_c': 'float32',        # Mean temperature
    'vpdmax_hpa': 'float32',     # Max vapor pressure deficit
    'vpdmin_hpa': 'float32',     # Min vapor pressure deficit
    
    # Derived features (24 total for model input)
    'precip_cumsum_7d': 'float32',
    'precip_cumsum_14d': 'float32',
    'precip_cumsum_30d': 'float32',
    'precip_cumsum_90d': 'float32',
    'api': 'float32',            # Antecedent Precipitation Index
    'wet_days_7d': 'int32',
    'wet_days_30d': 'int32',
    'dry_spell_days': 'int32',
    'precip_intensity': 'float32',
    'vpd_mean': 'float32',
    'vpd_range': 'float32',
    'temp_range': 'float32',
    'freeze_thaw': 'int32',      # Binary: min < 0 and max > 0
    # ... additional derived features
}
```

### 5.3 Model Input Format

```python
# Shape: (n_samples, T_a, 24) where T_a = 90
# 90 days @ daily intervals = 90 timesteps

ATMOS_LOOKBACK_DAYS = 90
ATMOS_TIMESTEPS = 90
ATMOS_FEATURES = 24

# Feature indices (order in model input)
ATMOS_FEATURE_NAMES = [
    'precip_mm',
    'tmax_c',
    'tmin_c', 
    'tmean_c',
    'vpdmax_hpa',
    'vpdmin_hpa',
    'precip_cumsum_7d',
    'precip_cumsum_14d',
    'precip_cumsum_30d',
    'precip_cumsum_90d',
    'api',
    'wet_days_7d',
    'wet_days_30d',
    'dry_spell_days',
    'precip_intensity',
    'vpd_mean',
    'vpd_range',
    'temp_range',
    'freeze_thaw',
    # Padding features if needed to reach 24
]
```

---

## 6. Coordinate System Mappings

### 6.1 Overview

Three coordinate systems must be aligned:

1. **Local alongshore (meters)**: Used in event CSVs, origin at south end of each beach
2. **MOP ID (integer)**: CDIP monitoring points at ~100m spacing
3. **Transect index (integer)**: Position in unified cube array

### 6.2 Beach Reference Points

```python
# Southern endpoint MOP for each beach (local coord origin)
BEACH_ORIGIN_MOP = {
    'blacks': 520,
    'torrey': 567,
    'delmar': 595,
    'solana': 637,
    'sanelijo': 683,
    'encinitas': 708,
}

# Northern endpoint MOP for each beach
BEACH_END_MOP = {
    'blacks': 567,
    'torrey': 581,
    'delmar': 620,
    'solana': 666,
    'sanelijo': 708,
    'encinitas': 764,
}

# MOP spacing (meters between MOP IDs)
MOP_SPACING_M = 100
```

### 6.3 Coordinate Conversion Functions

```python
def alongshore_to_mop(
    alongshore_m: float, 
    beach: str,
) -> int:
    """
    Convert local alongshore coordinate to nearest MOP ID.
    
    Args:
        alongshore_m: Distance from southern end of beach (meters)
        beach: Beach name (lowercase)
    
    Returns:
        Integer MOP ID
    
    Example:
        alongshore_to_mop(1500, 'delmar') → 610
        # 595 + (1500 / 100) = 610
    """
    origin_mop = BEACH_ORIGIN_MOP[beach]
    end_mop = BEACH_END_MOP[beach]
    
    mop_float = origin_mop + (alongshore_m / MOP_SPACING_M)
    mop_int = int(round(mop_float))
    
    # Clamp to valid range
    return max(origin_mop, min(end_mop, mop_int))


def mop_to_alongshore(
    mop_id: int,
    beach: str,
) -> float:
    """
    Convert MOP ID to local alongshore coordinate.
    
    Args:
        mop_id: Integer MOP ID
        beach: Beach name (lowercase)
    
    Returns:
        Alongshore distance in meters from beach origin
    """
    origin_mop = BEACH_ORIGIN_MOP[beach]
    return (mop_id - origin_mop) * MOP_SPACING_M


def alongshore_to_transect_idx(
    alongshore_m: float,
    beach: str,
    cube: dict,
    transect_spacing_m: float = 10.0,  # 10m for model training
) -> int:
    """
    Convert local alongshore coordinate to transect index in cube.
    
    Args:
        alongshore_m: Distance from southern end of beach (meters)
        beach: Beach name (lowercase)
        cube: Loaded cube dict with 'beach_slices'
        transect_spacing_m: Spacing between transects (default 1m)
    
    Returns:
        Integer index into cube arrays
    """
    beach_start, beach_end = cube['beach_slices'][beach]
    n_transects = beach_end - beach_start
    
    # Convert to transect offset within beach
    transect_offset = int(round(alongshore_m / transect_spacing_m))
    transect_offset = max(0, min(n_transects - 1, transect_offset))
    
    return beach_start + transect_offset


def transect_idx_to_mop(
    transect_idx: int,
    cube: dict,
) -> int:
    """
    Get MOP ID for a transect index.
    
    Args:
        transect_idx: Index into cube arrays
        cube: Loaded cube dict with 'mop_ids'
    
    Returns:
        Integer MOP ID
    """
    return int(cube['mop_ids'][transect_idx])


def transect_idx_to_beach(
    transect_idx: int,
    cube: dict,
) -> str:
    """
    Get beach name for a transect index.
    
    Args:
        transect_idx: Index into cube arrays
        cube: Loaded cube dict with 'beach_slices'
    
    Returns:
        Beach name (lowercase)
    """
    for beach, (start, end) in cube['beach_slices'].items():
        if start <= transect_idx < end:
            return beach
    raise ValueError(f'Transect {transect_idx} not in any beach')
```

### 6.4 Event-to-Cube Alignment

```python
def align_event_to_cube(
    event: pd.Series,
    cube: dict,
    beach: str,
) -> dict:
    """
    Align a single event to cube coordinates.
    
    Args:
        event: Row from event DataFrame
        cube: Loaded cube dict
        beach: Beach name
    
    Returns:
        Dict with transect_idx, epoch_before, epoch_after, or None if alignment fails
    """
    # 1. Get transect index
    transect_idx = alongshore_to_transect_idx(
        event['alongshore_centroid_m'], 
        beach, 
        cube
    )
    
    # 2. Get event date as ordinal
    event_date = event['mid_date'].toordinal()
    
    # 3. Find bracketing epochs
    timestamps = cube['timestamps']
    valid_epochs = np.where(cube['coverage_mask'][transect_idx])[0]
    
    if len(valid_epochs) < 2:
        return None  # Insufficient data
    
    valid_times = timestamps[valid_epochs]
    
    # Find epochs before and after event
    before_mask = valid_times < event_date
    after_mask = valid_times > event_date
    
    if not before_mask.any() or not after_mask.any():
        return None  # Event outside observation window
    
    epoch_before = valid_epochs[np.where(before_mask)[0][-1]]
    epoch_after = valid_epochs[np.where(after_mask)[0][0]]
    
    return {
        'transect_idx': int(transect_idx),
        'epoch_before': int(epoch_before),
        'epoch_after': int(epoch_after),
        'mop_id': transect_idx_to_mop(transect_idx, cube),
    }
```

---

## 7. Training Data Specification

### 7.1 File Format

- **Format**: NumPy NPZ (compressed)
- **Location**: `data/processed/susceptibility_{train,val,test}.npz`

### 7.2 Schema

```python
training_data_schema = {
    # === INPUT FEATURES ===
    'point_features': {
        'dtype': 'float32',
        'shape': (n_samples, max_context_epochs, 128, 12),
        'description': 'Padded transect point features (context epochs only)',
    },
    'metadata': {
        'dtype': 'float32',
        'shape': (n_samples, max_context_epochs, 12),
        'description': 'Padded transect metadata',
    },
    'distances': {
        'dtype': 'float32',
        'shape': (n_samples, max_context_epochs, 128),
        'description': 'Padded distance arrays',
    },
    'm3c2_recent': {
        'dtype': 'float32',
        'shape': (n_samples, 128),
        'description': 'M3C2 distance from most recent epoch pair',
    },
    'context_mask': {
        'dtype': 'bool',
        'shape': (n_samples, max_context_epochs),
        'description': 'True for valid context epochs (before padding)',
    },
    'wave_features': {
        'dtype': 'float32',
        'shape': (n_samples, 360, 4),
        'description': '90 days of wave data @ 6hr before current epoch',
    },
    'wave_doy': {
        'dtype': 'int32',
        'shape': (n_samples, 360),
        'description': 'Day-of-year for wave timesteps (1-366)',
    },
    'atmos_features': {
        'dtype': 'float32',
        'shape': (n_samples, 90, 24),
        'description': '90 days of atmospheric data @ daily before current epoch',
    },
    'atmos_doy': {
        'dtype': 'int32',
        'shape': (n_samples, 90),
        'description': 'Day-of-year for atmospheric timesteps (1-366)',
    },

    # === LABELS ===
    'erosion_class': {
        'dtype': 'int32',
        'shape': (n_samples,),
        'values': [0, 1, 2, 3, 4],
        'description': 'Erosion mode: 0=stable, 1=beach, 2=toe, 3=rockfall, 4=large_failure',
    },

    # === SAMPLE INFO ===
    'transect_idx': {
        'dtype': 'int32',
        'shape': (n_samples,),
        'description': 'Index into unified cube',
    },
    'epoch_before': {
        'dtype': 'int32',
        'shape': (n_samples,),
        'description': 'Epoch index before the labeled transition',
    },
    'epoch_after': {
        'dtype': 'int32',
        'shape': (n_samples,),
        'description': 'Epoch index after the labeled transition',
    },
    'mop_id': {
        'dtype': 'int32',
        'shape': (n_samples,),
        'description': 'MOP ID for sample',
    },
    'beach': {
        'dtype': 'object',
        'shape': (n_samples,),
        'description': 'Beach name for sample',
    },
    'water_year': {
        'dtype': 'int32',
        'shape': (n_samples,),
        'description': 'Water year (Oct 1 of year N-1 to Sep 30 of year N)',
    },
}
```

### 7.3 Sample Generation Rules

```python
# Minimum requirements for a valid sample
MIN_CONTEXT_EPOCHS = 3      # Need at least 3 epochs of history
MAX_CONTEXT_EPOCHS = 10     # Pad shorter sequences to this length
MIN_DAYS_BETWEEN_EPOCHS = 7 # Ignore epoch pairs closer than 1 week

# Sample generation algorithm:
# 1. For each transect with coverage_mask.sum() >= MIN_CONTEXT_EPOCHS + 1:
# 2.   Find all valid epochs (sorted chronologically)
# 3.   Sliding window: context = [i-MIN_CONTEXT:i], target = i
# 4.   For each window:
#        a. Check if events exist for (transect, last_context, target) → observed label
#        b. Otherwise → derived label from M3C2/LiDAR
#        c. Load wave/atmos aligned to target epoch date
#        d. Compute risk index from volume + cliff height
# 5. Pad context to MAX_CONTEXT_EPOCHS, create context_mask
```

---

## 8. Validation Rules

### 8.1 Cube Validation

```python
def validate_cube(cube: dict) -> list[str]:
    """
    Validate transect cube structure and values.
    
    Returns list of validation errors (empty if valid).
    """
    errors = []
    
    # Required keys
    required = ['points', 'distances', 'metadata', 'coverage_mask', 
                'timestamps', 'mop_ids', 'beach_slices']
    for key in required:
        if key not in cube:
            errors.append(f'Missing required key: {key}')
    
    if errors:
        return errors  # Can't continue without required keys
    
    # Shape consistency
    n_transects, n_epochs, n_points, n_features = cube['points'].shape
    
    if cube['distances'].shape != (n_transects, n_epochs, n_points):
        errors.append(f'distances shape mismatch: {cube["distances"].shape}')
    
    if cube['metadata'].shape[:2] != (n_transects, n_epochs):
        errors.append(f'metadata shape mismatch: {cube["metadata"].shape}')
    
    if cube['coverage_mask'].shape != (n_transects, n_epochs):
        errors.append(f'coverage_mask shape mismatch: {cube["coverage_mask"].shape}')
    
    if len(cube['timestamps']) != n_epochs:
        errors.append(f'timestamps length mismatch: {len(cube["timestamps"])}')
    
    if len(cube['mop_ids']) != n_transects:
        errors.append(f'mop_ids length mismatch: {len(cube["mop_ids"])}')
    
    # Feature count
    if n_features != 13:
        errors.append(f'Expected 13 features, got {n_features}')
    
    # Value ranges (where coverage_mask is True)
    valid_mask = cube['coverage_mask']
    
    # Elevation reasonable
    elevations = cube['points'][..., 1][valid_mask[:, :, np.newaxis].repeat(n_points, axis=2)]
    if elevations.min() < -10 or elevations.max() > 100:
        errors.append(f'Elevation out of range: [{elevations.min()}, {elevations.max()}]')
    
    # No all-NaN transects that claim to have coverage
    for t in range(n_transects):
        for e in range(n_epochs):
            if valid_mask[t, e]:
                if np.isnan(cube['points'][t, e]).all():
                    errors.append(f'Transect {t} epoch {e} marked valid but all NaN')
    
    return errors
```

### 8.2 Event Validation

```python
def validate_events(df: pd.DataFrame, beach: str) -> list[str]:
    """
    Validate event DataFrame structure and values.
    
    Returns list of validation errors.
    """
    errors = []
    
    # Required columns
    required = ['mid_date', 'start_date', 'end_date', 'volume', 
                'alongshore_centroid_m', 'width', 'height']
    missing = set(required) - set(df.columns)
    if missing:
        errors.append(f'Missing columns: {missing}')
        return errors
    
    # Date ordering
    invalid_dates = df[df['start_date'] > df['end_date']]
    if len(invalid_dates) > 0:
        errors.append(f'{len(invalid_dates)} events with start_date > end_date')
    
    # Volume positive
    negative_vol = df[df['volume'] <= 0]
    if len(negative_vol) > 0:
        errors.append(f'{len(negative_vol)} events with non-positive volume')
    
    # Alongshore in valid range
    beach_length = (BEACH_END_MOP[beach] - BEACH_ORIGIN_MOP[beach]) * MOP_SPACING_M
    out_of_range = df[(df['alongshore_centroid_m'] < 0) | 
                      (df['alongshore_centroid_m'] > beach_length * 1.1)]
    if len(out_of_range) > 0:
        errors.append(f'{len(out_of_range)} events outside beach extent')
    
    # Width/height positive
    invalid_dims = df[(df['width'] <= 0) | (df['height'] <= 0)]
    if len(invalid_dims) > 0:
        errors.append(f'{len(invalid_dims)} events with non-positive dimensions')
    
    return errors
```

### 8.3 Training Data Validation

```python
def validate_training_data(data: dict) -> list[str]:
    """
    Validate training data structure and values.
    
    Returns list of validation errors.
    """
    errors = []
    
    n_samples = len(data['erosion_class'])

    # Shape checks
    expected_shapes = {
        'point_features': (n_samples, -1, 128, 12),  # -1 = max_context
        'metadata': (n_samples, -1, 12),
        'm3c2_recent': (n_samples, 128),
        'wave_features': (n_samples, 360, 4),
        'atmos_features': (n_samples, 90, 24),
        'erosion_class': (n_samples,),
    }
    
    for key, expected in expected_shapes.items():
        if key not in data:
            errors.append(f'Missing key: {key}')
            continue
        
        actual = data[key].shape
        for i, (e, a) in enumerate(zip(expected, actual)):
            if e != -1 and e != a:
                errors.append(f'{key} shape mismatch at dim {i}: expected {e}, got {a}')
    
    # Value ranges
    if (data['erosion_class'] < 0).any() or (data['erosion_class'] > 4).any():
        errors.append('erosion_class values outside [0, 4]')

    # No NaN in labels
    if np.isnan(data['erosion_class']).any():
        errors.append('NaN values in erosion_class')

    # Check label distribution
    class_counts = np.bincount(data['erosion_class'], minlength=5)
    if class_counts[0] == n_samples:
        errors.append('WARNING: All samples are class 0 (stable)')
    if class_counts[4] == 0:
        errors.append('WARNING: No class 4 (large_failure) samples')
    
    return errors
```

---

## 9. File Locations & Naming Conventions

### 9.1 Directory Structure

```
data/
├── raw/
│   ├── lidar/                          # Source LAS/LAZ files
│   │   └── {survey_date}/              # Organized by survey date
│   ├── cdip/                           # CDIP wave data (NetCDF)
│   │   ├── D0520_hindcast.nc
│   │   ├── D0521_hindcast.nc
│   │   └── ...
│   ├── prism/                          # Raw PRISM climate data
│   │   └── {variable}/                 # ppt, tmax, tmin, etc.
│   ├── events/                         # Event CSVs from M3C2
│   │   ├── blacks_events_sig.csv
│   │   ├── torrey_events_sig.csv
│   │   ├── delmar_events_sig.csv
│   │   ├── solana_events_sig.csv
│   │   ├── sanelijo_events_sig.csv
│   │   └── encinitas_events_sig.csv
│   └── master_list.csv                 # Survey metadata
│
├── processed/
│   ├── unified_cube.npz                # All transects, all epochs
│   ├── atmospheric/                    # Processed atmospheric data
│   │   ├── blacks_atmos.parquet
│   │   ├── torrey_atmos.parquet
│   │   └── ...
│   ├── aligned_events.parquet          # Events aligned to cube
│   ├── training_data.npz               # Final training tensors
│   └── splits/                         # Train/val/test splits
│       ├── train_indices.npy
│       ├── val_indices.npy
│       └── test_indices.npy
│
└── external/                           # External reference data
    └── mops/
        └── transects_10m/
            └── transect_lines.shp      # Transect geometries
```

### 9.2 Naming Conventions

```python
# Beach names: always lowercase, no spaces
BEACH_NAMES = ['blacks', 'torrey', 'delmar', 'solana', 'sanelijo', 'encinitas']

# File naming patterns
NAMING_PATTERNS = {
    'event_csv': '{beach}_events_sig.csv',
    'atmos_parquet': '{beach}_atmos.parquet',
    'cdip_nc': 'D{mop:04d}_hindcast.nc',
    'transect_cube': '{beach}.npz',  # Per-beach
    'unified_cube': 'unified_cube.npz',
    'training_data': 'training_data.npz',
}
```

---

## 10. Processing Pipeline Specifications

### 10.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PROCESSING PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STEP 1: Extract Transects                                                   │
│  ─────────────────────────                                                   │
│  Input:  LAS files, transect shapefile                                       │
│  Output: data/processed/unified_cube.npz                                     │
│  Script: scripts/processing/extract_transects.py                             │
│                                                                              │
│  STEP 2: Process Atmospheric Data                                            │
│  ────────────────────────────────                                            │
│  Input:  PRISM BIL files                                                     │
│  Output: data/processed/atmospheric/{beach}_atmos.parquet                    │
│  Script: scripts/processing/process_prism.py                                 │
│                                                                              │
│  STEP 3: Download Wave Data                                                  │
│  ──────────────────────────                                                  │
│  Input:  CDIP THREDDS server                                                 │
│  Output: data/raw/cdip/D{MOP}_hindcast.nc                                    │
│  Script: scripts/processing/download_cdip_data.py                            │
│                                                                              │
│  STEP 4: Align Events to Cube                                                │
│  ────────────────────────────                                                │
│  Input:  unified_cube.npz, events/*.csv                                      │
│  Output: data/processed/aligned_events.parquet                               │
│  Script: scripts/processing/align_events.py                                  │
│                                                                              │
│  STEP 5: Generate Training Data                                              │
│  ──────────────────────────────                                              │
│  Input:  unified_cube.npz, aligned_events.parquet, cdip/, atmospheric/       │
│  Output: data/processed/training_data.npz                                    │
│  Script: scripts/processing/prepare_training_data.py                         │
│                                                                              │
│  STEP 6: Create Data Splits                                                  │
│  ──────────────────────────                                                  │
│  Input:  training_data.npz                                                   │
│  Output: data/processed/splits/{train,val,test}_indices.npy                  │
│  Script: scripts/processing/create_splits.py                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Step 4: Align Events to Cube

```bash
python scripts/processing/align_events.py \
    --cube data/processed/unified_cube.npz \
    --events-dir data/raw/events/ \
    --output data/processed/aligned_events.parquet \
    --transect-spacing 1.0
```

**Input Requirements**:
- Unified cube with `beach_slices`, `coverage_mask`, `timestamps`, `mop_ids`
- Event CSVs for all beaches in `events-dir`

**Output Schema**:
```python
aligned_events_schema = {
    # Original event fields
    'volume': 'float32',
    'vol_unc': 'float32',
    'elevation': 'float32',
    'alongshore_centroid_m': 'float32',
    'width': 'float32',
    'height': 'float32',
    'mid_date': 'datetime64[ns]',
    'start_date': 'datetime64[ns]',
    'end_date': 'datetime64[ns]',
    
    # Alignment fields (added)
    'beach': 'str',
    'transect_idx': 'int32',
    'epoch_before': 'int32',
    'epoch_after': 'int32',
    'mop_id': 'int32',
    'event_class': 'int32',
}
```

**Aggregation**:
Events are aggregated per `(transect_idx, epoch_before, epoch_after)`:
- `total_volume`: sum of volumes
- `volume_unc`: sqrt(sum of squared uncertainties)
- `n_events`: count of events
- `max_height`: maximum height
- `max_width`: maximum width

### 10.3 Step 5: Generate Training Data

```bash
python scripts/processing/prepare_training_data.py \
    --cube data/processed/unified_cube.npz \
    --events data/processed/aligned_events.parquet \
    --cdip-dir data/raw/cdip/ \
    --atmos-dir data/processed/atmospheric/ \
    --output data/processed/training_data.npz \
    --min-context 3 \
    --max-context 10 \
    --workers 8
```

**Processing Steps**:

1. **Find valid sequences**: Transects with ≥4 consecutive valid epochs
2. **Generate samples**: Sliding window (context + target)
3. **Match events**: Look up observed events for each sample
4. **Load environmental data**: Wave (90d @ 6hr), Atmos (90d @ daily)
5. **Compute labels**: Volume, class, risk index
6. **Pad and stack**: Create uniform tensors

**Parallelization**:
- Wave/atmos loading parallelized across samples
- Use `--workers` to control thread pool size

### 10.4 Step 6: Create Data Splits

```bash
python scripts/processing/create_splits.py \
    --input data/processed/training_data.npz \
    --output-dir data/processed/splits/ \
    --strategy temporal \
    --val-ratio 0.15 \
    --test-ratio 0.15
```

**Split Strategies**:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `temporal` | Last 15% = test, prior 15% = val | Production (recommended) |
| `spatial` | Leave one beach out for test | Generalization testing |
| `random` | Stratified random split | Baseline comparison |

**Output**:
```
splits/
├── train_indices.npy    # (n_train,) int32
├── val_indices.npy      # (n_val,) int32
├── test_indices.npy     # (n_test,) int32
└── split_info.json      # Metadata: strategy, sizes, class balance
```

---

## Appendix A: Quick Reference

### Data Shapes Summary

| Array | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `cube['points']` | (n_transects, n_epochs, 128, 12) | float32 | Point features |
| `cube['coverage_mask']` | (n_transects, n_epochs) | bool | Valid data mask |
| `training['point_features']` | (n_samples, max_ctx, 128, 12) | float32 | Padded context |
| `training['m3c2_recent']` | (n_samples, 128) | float32 | Recent M3C2 change |
| `training['wave_features']` | (n_samples, 360, 4) | float32 | 90d wave @ 6hr |
| `training['atmos_features']` | (n_samples, 90, 24) | float32 | 90d atmos @ daily |
| `training['erosion_class']` | (n_samples,) | int32 | Target class [0-4] |

### Key Constants

```python
# Transect features
N_POINT_FEATURES = 12
N_META_FEATURES = 12
N_POINTS_PER_TRANSECT = 128

# Environmental features
WAVE_LOOKBACK_DAYS = 90
WAVE_INTERVAL_HOURS = 6
WAVE_TIMESTEPS = 360
WAVE_FEATURES = 4

ATMOS_LOOKBACK_DAYS = 90
ATMOS_TIMESTEPS = 90
ATMOS_FEATURES = 24

# Training
MIN_CONTEXT_EPOCHS = 2
MAX_CONTEXT_EPOCHS = 5

# Erosion mode classification
N_CLASSES = 5
CLASS_NAMES = ['stable', 'beach_erosion', 'toe_erosion', 'small_rockfall', 'large_failure']
RISK_WEIGHTS = [0.0, 0.1, 0.4, 0.6, 1.0]
CLASS_WEIGHTS = [0.3, 1.0, 2.0, 2.0, 5.0]  # For loss function

# Coordinates
MOP_SPACING_M = 100
```

### Beach MOP Ranges

| Beach | MOP Start | MOP End | Span (m) |
|-------|-----------|---------|----------|
| Blacks | 520 | 567 | 4700 |
| Torrey | 567 | 581 | 1400 |
| Del Mar | 595 | 620 | 2500 |
| Solana | 637 | 666 | 2900 |
| San Elijo | 683 | 708 | 2500 |
| Encinitas | 708 | 764 | 5600 |

---

## Appendix B: Validation Checklist

Before training, verify:

- [ ] Unified cube loads without errors
- [ ] All 6 event CSVs present and valid
- [ ] CDIP files cover MOP range 520-764
- [ ] Atmospheric parquet files for all beaches
- [ ] Event alignment produces >0 observed samples per beach
- [ ] Training data has balanced class distribution
- [ ] No NaN in labels
- [ ] Wave/atmos features have no gaps in required date ranges

```bash
# Run all validations
python scripts/processing/validate_all.py --data-dir data/
```
