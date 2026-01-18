# Data Loading and Preprocessing

This module provides tools for loading and preprocessing data for CliffCast.

## Transect Extraction

The `TransectExtractor` class extracts shore-normal 2D profiles from 3D LiDAR point clouds.

### Features

- **Automatic coastline detection** from point cloud elevation
- **Shore-normal profile extraction** with configurable spacing
- **Resampling** to fixed number of points (default: 128)
- **Derived features** computation:
  - Distance from cliff toe
  - Elevation
  - Slope (degrees)
  - Profile curvature
  - Surface roughness (local elevation std)
- **Transect metadata** extraction:
  - Cliff height
  - Mean/max slope
  - Toe elevation
  - Orientation
  - Geographic position

### Quick Start

```python
from src.data.transect_extractor import TransectExtractor
import numpy as np

# Initialize extractor
extractor = TransectExtractor(
    n_points=128,          # Points per transect
    spacing_m=10.0,        # Alongshore spacing
    profile_length_m=150.0, # Max distance from toe
)

# Extract transects from LAS file
transects = extractor.extract_from_file("cliff_scan.laz")

# Access extracted data
points = transects['points']        # (N, 128, 5) features
distances = transects['distances']  # (N, 128) distances from toe
metadata = transects['metadata']    # (N, 7) transect-level stats
positions = transects['positions']  # (N, 2) transect origins
normals = transects['normals']      # (N, 2) shore-normal directions

print(f"Extracted {len(points)} transects")

# Save for later use
extractor.save_transects(transects, "transects.npz")

# Load saved transects
loaded = extractor.load_transects("transects.npz")
```

### With Custom Coastline

If you have coastline data (e.g., from manual digitization or GIS):

```python
# Define coastline points (cliff toe positions)
coastline_points = np.array([
    [100, 200],  # x, y coordinates
    [100, 210],
    [100, 220],
    # ...
])

# Define shore-normal vectors (pointing inland)
coastline_normals = np.array([
    [1, 0],  # Unit vectors
    [1, 0],
    [1, 0],
    # ...
])

# Extract transects
transects = extractor.extract_from_file(
    "cliff_scan.laz",
    coastline_points=coastline_points,
    coastline_normals=coastline_normals,
)
```

### Using the Extraction Script

A command-line script is provided for batch processing:

```bash
# Basic usage (automatic coastline detection)
python scripts/extract_transects.py \
    --input data/raw/lidar/cliff_scan.laz \
    --output data/processed/transects.npz

# With custom coastline file (CSV format: x,y,normal_x,normal_y)
python scripts/extract_transects.py \
    --input data/raw/lidar/cliff_scan.laz \
    --output data/processed/transects.npz \
    --coastline data/raw/coastline.csv \
    --n-points 128 \
    --spacing 10.0

# With visualization
python scripts/extract_transects.py \
    --input data/raw/lidar/cliff_scan.laz \
    --output data/processed/transects.npz \
    --visualize
```

### Data Format

#### Input: LAS/LAZ Point Cloud

Standard LAS format with XYZ coordinates. Supports:
- LAS 1.2, 1.3, 1.4
- LAZ compressed files
- Point classifications (if available, used for filtering)

#### Output: Transect NPZ File

NumPy compressed archive containing:

**`points`**: (N_transects, n_points, 5) array
- Feature 0: Distance from toe (m)
- Feature 1: Elevation (m)
- Feature 2: Slope (degrees)
- Feature 3: Curvature (1/m)
- Feature 4: Roughness (m)

**`distances`**: (N_transects, n_points) array
- Distance along transect from toe (m)

**`metadata`**: (N_transects, 7) array
- [0] Cliff height (m)
- [1] Mean slope (degrees)
- [2] Max slope (degrees)
- [3] Toe elevation (m)
- [4] Orientation (azimuth, degrees)
- [5] Latitude (or y-coordinate)
- [6] Longitude (or x-coordinate)

**`positions`**: (N_transects, 2) array
- Transect origin xy coordinates

**`normals`**: (N_transects, 2) array
- Shore-normal unit vectors

### Parameters

#### TransectExtractor Parameters

- **`n_points`** (int, default=128): Number of points to resample each transect to
- **`spacing_m`** (float, default=10.0): Alongshore spacing between transects in meters
- **`profile_length_m`** (float, default=150.0): Maximum profile length from toe
- **`min_points`** (int, default=20): Minimum points required for valid transect
- **`search_radius_m`** (float, default=2.0): Radius for local neighborhood queries

### Algorithm Details

#### 1. Coastline Detection (if not provided)

Uses elevation quantile to identify cliff toe:
1. Find low points (bottom 5% elevation)
2. Sort alongshore
3. Sample at `spacing_m` intervals

#### 2. Shore-Normal Computation

Computes perpendicular vectors to coastline tangent:
1. Fit local tangent using neighbors
2. Rotate 90° to get normal
3. Orient landward (positive direction)

#### 3. Profile Extraction

For each coastline point:
1. Define ray from toe along shore-normal
2. Query points within `search_radius_m` of ray
3. Project points onto ray (get alongshore distance)
4. Filter to valid range [0, profile_length_m]
5. Sort by distance

#### 4. Resampling

Interpolate to fixed number of points:
1. Linear interpolation of xyz coordinates
2. Uniform spacing from 0 to max distance
3. Handles irregular point density

#### 5. Feature Computation

**Slope**: Finite difference of elevation
```python
slope = arctan(dz / dx) in degrees
```

**Curvature**: Second derivative of elevation
```python
curvature = d²z / dx²
```

**Roughness**: Local elevation standard deviation
```python
roughness = std(detrended_elevation in 5-point window)
```

### Edge Cases Handled

- **Sparse data**: Transects with fewer than `min_points` are discarded
- **Data gaps**: Linear interpolation across small gaps
- **Irregular sampling**: Resampling handles variable point density
- **Noisy data**: Roughness feature captures local variability
- **Endpoint effects**: Features at edges use one-sided differences

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
    elevations = transects['points'][i, :, 1]
    ax.plot(distances, elevations, label=f"Transect {i}")

ax.set_xlabel("Distance from Toe (m)")
ax.set_ylabel("Elevation (m)")
ax.set_title("Extracted Transect Profiles")
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

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

# Specific test
pytest tests/test_data/test_transect_extractor.py::test_full_pipeline_synthetic -v
```
