# Data Loading and Preprocessing

This module provides tools for loading and preprocessing data for CliffCast.

## Transect Extraction

The `ShapefileTransectExtractor` class extracts shore-normal 2D profiles from 3D LiDAR point clouds using predefined transect lines from a shapefile.

### Features

- **Shapefile-based extraction** using predefined transect LineStrings
- **Buffer-based point collection** around transect lines
- **Resampling** to fixed number of points (default: 128)
- **12 per-point features**:
  - Distance from transect start (m)
  - Elevation (m)
  - Slope (degrees)
  - Profile curvature
  - Surface roughness
  - LAS intensity (normalized 0-1)
  - RGB colors (normalized 0-1)
  - Point classification
  - Return number
  - Number of returns
- **12 transect-level metadata fields**:
  - Cliff height, mean/max slope
  - Toe/top elevation, orientation
  - Geographic position (lat/lon or xy)
  - Transect ID, mean intensity, dominant classification

### Quick Start

```python
from src.data.shapefile_transect_extractor import ShapefileTransectExtractor
from pathlib import Path

# Initialize extractor
extractor = ShapefileTransectExtractor(
    n_points=128,    # Points per transect
    buffer_m=1.0,    # Buffer around transect line
    min_points=20,   # Minimum points for valid transect
)

# Load transect lines from shapefile
transect_gdf = extractor.load_transect_lines("transects.shp")

# Extract transects from LAS files
las_files = [Path("scan1.las"), Path("scan2.las")]
transects = extractor.extract_from_shapefile_and_las(
    transect_gdf,
    las_files,
    transect_id_col='tr_id'
)

# Access extracted data
points = transects['points']        # (N, 128, 12) features
distances = transects['distances']  # (N, 128) distances along transect
metadata = transects['metadata']    # (N, 12) transect-level metadata
transect_ids = transects['transect_ids']  # (N,) original IDs
las_sources = transects['las_sources']    # List of source files

print(f"Extracted {len(points)} transects")

# Save for later use
extractor.save_transects(transects, "transects.npz")

# Load saved transects
loaded = extractor.load_transects("transects.npz")
```

### Using the Extraction Script

A command-line script is provided for batch processing:

```bash
# Extract from shapefile and LAS directory
python scripts/extract_transects.py \
    --transects data/mops/transects_10m/transect_lines.shp \
    --las-dir data/raw/lidar/ \
    --output data/processed/transects.npz \
    --buffer 1.0 \
    --n-points 128

# Extract from specific LAS files
python scripts/extract_transects.py \
    --transects data/transects.shp \
    --las-files data/scan1.las data/scan2.las \
    --output data/transects.npz \
    --transect-id-col tr_id

# With visualization
python scripts/extract_transects.py \
    --transects data/transects.shp \
    --las-dir data/raw/lidar/ \
    --output data/transects.npz \
    --visualize
```

### Data Format

#### Input: Shapefile with Transect Lines

Shapefile containing LineString geometries:
- **Geometry**: LineString (2D or 3D)
- **Attributes**: Must include transect ID column (default: 'tr_id')
- **CRS**: Should match LAS file coordinates

#### Input: LAS/LAZ Point Cloud

Standard LAS format with XYZ coordinates. Supports:
- LAS 1.2, 1.3, 1.4
- LAZ compressed files
- Optional: intensity, RGB, classification, return info

#### Output: Transect NPZ File

NumPy compressed archive containing:

**`points`**: (N_transects, n_points, 12) array
- Feature 0: Distance from start (m)
- Feature 1: Elevation (m)
- Feature 2: Slope (degrees)
- Feature 3: Curvature (1/m)
- Feature 4: Roughness (m)
- Feature 5: Intensity (0-1)
- Feature 6: Red (0-1)
- Feature 7: Green (0-1)
- Feature 8: Blue (0-1)
- Feature 9: Classification code
- Feature 10: Return number
- Feature 11: Number of returns

**`distances`**: (N_transects, n_points) array
- Distance along transect from start (m)

**`metadata`**: (N_transects, 12) array
- [0] Cliff height (m)
- [1] Mean slope (degrees)
- [2] Max slope (degrees)
- [3] Toe elevation (m)
- [4] Top elevation (m)
- [5] Orientation (azimuth, degrees from N)
- [6] Transect length (m)
- [7] Latitude (or y-coordinate)
- [8] Longitude (or x-coordinate)
- [9] Transect ID
- [10] Mean intensity
- [11] Dominant classification

**`transect_ids`**: (N_transects,) array
- Original transect IDs from shapefile

**`las_sources`**: List of strings
- Source LAS file name for each transect

**`feature_names`**: List of 12 feature names
**`metadata_names`**: List of 12 metadata names

### Parameters

#### ShapefileTransectExtractor Parameters

- **`n_points`** (int, default=128): Number of points to resample each transect to
- **`buffer_m`** (float, default=1.0): Buffer distance around transect line in meters
- **`min_points`** (int, default=20): Minimum points required for valid transect

### Algorithm Details

#### 1. Transect Line Processing

For each LineString in shapefile:
1. Extract start and end points
2. Compute direction vector and length
3. Normalize to unit vector

#### 2. Point Collection

For each transect:
1. Sample query points along transect line
2. Query LAS points within buffer distance
3. Project points onto transect line
4. Filter by perpendicular distance and along-transect bounds
5. Sort by distance along transect

#### 3. Resampling

Interpolate to fixed number of points:
1. Linear interpolation of xyz coordinates
2. Linear interpolation of intensity and RGB
3. Nearest neighbor for classification and returns
4. Uniform spacing from start to end

#### 4. Feature Computation

**Slope**: Finite difference of elevation
```python
slope = arctan(dz / dd) in degrees
```

**Curvature**: Second derivative of elevation
```python
curvature = d²z / dd²
```

**Roughness**: Local elevation standard deviation
```python
roughness = std(detrended_elevation in 5-point window)
```

### Edge Cases Handled

- **Sparse data**: Transects with fewer than `min_points` are skipped
- **Data gaps**: Linear interpolation across gaps
- **Irregular sampling**: Resampling handles variable point density
- **Noisy data**: Roughness feature captures local variability
- **Endpoint effects**: Features at edges use one-sided differences
- **Missing attributes**: Defaults to zeros for missing intensity/RGB/classification

### Quality Checks

Before using extracted transects, verify:

1. **Sufficient transects**: `len(transects['points']) > 0`
2. **No NaN values**: `not np.any(np.isnan(transects['points']))`
3. **Monotonic distances**: `np.all(np.diff(distances) >= 0)` for each transect
4. **Reasonable cliff heights**: Typically 5-50m for coastal cliffs
5. **Reasonable slopes**: Typically 20-90 degrees for cliff faces

### Example: Data Inspection

```python
import numpy as np

# Load transects
transects = extractor.load_transects("transects.npz")

# Basic statistics
print(f"Number of transects: {len(transects['points'])}")
print(f"Points per transect: {transects['points'].shape[1]}")
print(f"Features per point: {transects['points'].shape[2]}")

# Cliff height statistics
heights = transects['metadata'][:, 0]
print(f"Cliff height: {heights.mean():.1f} ± {heights.std():.1f} m")
print(f"Range: {heights.min():.1f} - {heights.max():.1f} m")

# Slope statistics
slopes = transects['metadata'][:, 1]
print(f"Mean slope: {slopes.mean():.1f} ± {slopes.std():.1f} degrees")

# Check for quality issues
has_nan = np.any(np.isnan(transects['points']))
print(f"Contains NaN: {has_nan}")

# Check distance monotonicity
for i in range(len(transects['distances'])):
    is_monotonic = np.all(np.diff(transects['distances'][i]) >= 0)
    if not is_monotonic:
        print(f"Warning: Transect {i} has non-monotonic distances")
```

### Visualization

```python
import matplotlib.pyplot as plt

# Plot first 5 transects
fig, ax = plt.subplots(figsize=(12, 6))

for i in range(min(5, len(transects['points']))):
    distances = transects['distances'][i]
    elevations = transects['points'][i, :, 1]  # Feature 1 is elevation
    ax.plot(distances, elevations, label=f"Transect {i}")

ax.set_xlabel("Distance along Transect (m)")
ax.set_ylabel("Elevation (m)")
ax.set_title("Extracted Transect Profiles")
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

## Alternative Extractors

### TransectVoxelizer (Unused)

The `TransectVoxelizer` class provides an alternative voxel-based extraction approach. It bins points along transects into 1D segments and is more robust to variable point density. Currently not used in the pipeline but available in `src/data/transect_voxelizer.py`.

## Future Data Modules

The following modules will be implemented in subsequent phases:

- **wave_loader.py**: Load wave data from CDIP/WaveWatch III
- **precip_loader.py**: Load precipitation data from PRISM
- **label_generator.py**: Compute labels from change detection
- **dataset.py**: PyTorch Dataset class for training
- **transforms.py**: Data augmentation

## Testing

Run tests for transect extraction:

```bash
# All data tests
pytest tests/test_data/ -v

# Transect extractor only
pytest tests/test_data/test_transect_extractor.py -v

# Specific test class
pytest tests/test_data/test_transect_extractor.py::TestShapefileTransectExtractorInit -v
```
