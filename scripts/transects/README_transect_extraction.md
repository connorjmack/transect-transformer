# Transect Extraction Methods

This directory contains tools for extracting shore-normal transects from LiDAR point clouds using two different methods:

## Methods

### 1. Interpolation-based (Original)
- **File**: `src/data/transect_extractor.py`
- **Approach**: Extracts points along transect, resamples to fixed number of points via interpolation
- **Output**: 128 uniformly-spaced points with features [distance, elevation, slope, curvature, roughness]
- **Best for**: Continuous profiles, smooth terrain

### 2. Voxelized (New)
- **File**: `src/data/transect_voxelizer.py`
- **Approach**: Bins points into 1D segments along transect, aggregates features per bin
- **Output**: 128 bins with features [mean_elevation, roughness, height_range, slope, curvature, point_density]
- **Best for**: Variable point density, complex geometry, overhang/caves

## Quick Start

### Prerequisites
```bash
# Ensure dependencies are installed
pip install -r requirements.txt
```

### Using the KML Test Script

The main test script (`test_transect_extraction.py`) compares both methods:

```bash
# Run both methods on your LiDAR data
python scripts/test_transect_extraction.py \
    --lidar /path/to/your/lidar_scan.laz \
    --kml data/mops/MOPs-SD.kml \
    --output results/transects/ \
    --method both

# Run only voxelized method
python scripts/test_transect_extraction.py \
    --lidar /path/to/your/lidar_scan.laz \
    --kml data/mops/MOPs-SD.kml \
    --method voxelized

# Limit to first 10 transects (for testing)
python scripts/test_transect_extraction.py \
    --lidar /path/to/your/lidar_scan.laz \
    --kml data/mops/MOPs-SD.kml \
    --n-transects 10 \
    --method both
```

### Using Directly in Python

#### Voxelized Method with KML Transects
```python
from src.data.kml_parser import KMLParser
from src.data.transect_voxelizer import TransectVoxelizer

# Parse KML to get transect lines
parser = KMLParser(utm_zone=11, hemisphere='N')
kml_transects = parser.parse("data/mops/MOPs-SD.kml")

# Extract voxelized transects
voxelizer = TransectVoxelizer(
    bin_size_m=1.0,
    corridor_width_m=2.0,
    max_bins=128,
)

transects = voxelizer.extract_from_file(
    "path/to/lidar.laz",
    transect_origins=kml_transects['origins'],
    transect_normals=kml_transects['normals'],
    transect_names=kml_transects['names'],
)

# transects contains:
# - bin_features: (N, 128, 6) voxelized features
# - bin_centers: (N, 128) distances from origin
# - bin_mask: (N, 128) boolean mask for valid bins
# - metadata: (N, 7) transect-level metadata
```

#### Interpolation Method (Auto-detection)
```python
from src.data.transect_extractor import TransectExtractor

extractor = TransectExtractor(
    n_points=128,
    spacing_m=10.0,
    profile_length_m=150.0,
)

transects = extractor.extract_from_file("path/to/lidar.laz")

# transects contains:
# - points: (N, 128, 5) interpolated features
# - distances: (N, 128) distances from toe
# - metadata: (N, 7) transect-level metadata
# - positions: (N, 2) transect origins
# - normals: (N, 2) shore-normal vectors
```

## File Formats

### Input

**LiDAR files**: `.las` or `.laz` format
- Must contain X, Y, Z coordinates
- Coordinate system should match KML (if using KML transects)

**KML files**: Standard KML with LineString Placemarks
- Each Placemark represents one shore-normal transect
- LineString must have at least 2 coordinate pairs (start, end)
- Coordinates in lon, lat, elevation format

### Output

Both methods save to `.npz` (compressed numpy) format:

```python
# Load saved transects
data = np.load("results/transects/transects_voxelized.npz")

# Access arrays
bin_features = data['bin_features']
bin_mask = data['bin_mask']
```

## Comparison

| Aspect | Interpolation | Voxelized |
|--------|--------------|-----------|
| **Speed** | Moderate | Faster |
| **Memory** | Lower | Moderate |
| **Sparse data** | Interpolates (risky) | Masks invalid bins |
| **Dense data** | May oversample | Aggregates naturally |
| **Overhangs/caves** | Poor | Better (via point density) |
| **Interpretability** | High (real points) | Moderate (aggregated) |
| **Output size** | (N, 128, 5) | (N, 128, 6) |

## Next Steps

After extracting transects:

1. **Inspect outputs**: Check shapes and valid data ratio
2. **Visualize**: Plot sample transects to verify quality
3. **Create dataset**: Use extracted transects in PyTorch dataset (Phase 1.7)
4. **Train model**: Feed to CliffCast transformer

## Troubleshooting

**"No valid transects extracted"**
- Check that KML coordinates overlap with LiDAR extent
- Verify coordinate reference systems match (use correct UTM zone)
- Try increasing `corridor_width_m` or decreasing `min_points_per_bin`

**"Empty point cloud"**
- Ensure LiDAR file is valid (check with `laspy` or CloudCompare)
- Verify file path is correct

**"pyproj not installed" warning**
- Install with `pip install pyproj` for accurate coordinate transformation
- Without it, uses approximate conversion (less accurate for large areas)

## Files

```
src/data/
├── kml_parser.py           # Parse KML transect lines
├── transect_extractor.py   # Interpolation-based extraction
└── transect_voxelizer.py   # Voxelized extraction

scripts/
├── test_transect_extraction.py  # Compare both methods
└── README_transect_extraction.md  # This file
```
