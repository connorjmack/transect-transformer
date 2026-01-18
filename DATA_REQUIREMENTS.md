# Data Requirements for CliffCast

This document describes the data needed to train and test CliffCast.

## Current Status

✅ **Transect Extraction Module**: Complete and ready to process LiDAR data
⏳ **Wave Data Loader**: Not yet implemented
⏳ **Precipitation Data Loader**: Not yet implemented
⏳ **Label Generator**: Not yet implemented

## Phase 1: Transect Extraction (READY FOR DATA)

### What We Need

**LiDAR Point Clouds** in LAS/LAZ format containing cliff profiles:

**File Format:**
- LAS 1.2, 1.3, or 1.4
- LAZ compressed format supported
- Must contain XYZ coordinates (required)
- Point classifications optional but helpful

**Spatial Coverage:**
- Coastal cliff sections
- Should include beach/toe, cliff face, and cliff top
- Recommended extent: 100-200m cross-shore × variable alongshore
- Multiple scans from different dates (for change detection) preferred

**Data Quality:**
- Point density: >10 points/m² recommended
- Vertical accuracy: <0.5m
- Coordinate system: Any projected CRS (UTM, State Plane, etc.)

### Optional but Helpful

**Coastline/Cliff Toe Digitization:**
- CSV file with format: `x,y,normal_x,normal_y`
- Points defining cliff toe location
- Shore-normal unit vectors pointing inland
- If not provided, automatic detection will be attempted

**Example coastline CSV:**
```csv
x,y,normal_x,normal_y
500100.0,4100200.0,1.0,0.0
500100.0,4100210.0,1.0,0.0
500100.0,4100220.0,0.9,0.1
```

### Testing the Transect Extractor

Once you provide LiDAR data, we can test with:

```bash
python scripts/extract_transects.py \
    --input /path/to/your/cliff_scan.laz \
    --output data/processed/test_transects.npz \
    --n-points 128 \
    --spacing 10.0 \
    --visualize
```

**Expected Output:**
- `test_transects.npz` - Extracted transect data
- `transects_viz.png` - Visualization of profiles
- Console output with statistics (cliff heights, slopes, etc.)

### What to Provide

Please share:
1. **Sample LiDAR file** (even a small subset for testing)
2. **Coordinate system info** (EPSG code or WKT)
3. **Approximate location** (helps validate results)
4. **Optional: Coastline file** (if available)

## Phase 2: Environmental Data (NOT YET READY)

We will need wave and precipitation data, but the loaders are not yet implemented. These will be addressed in upcoming steps.

### Wave Data (Future)

**Sources:**
- CDIP (Coastal Data Information Program) buoy data
- WaveWatch III (WW3) model output
- NOAA buoy data

**Required Fields:**
- Significant wave height (Hs, meters)
- Peak period (Tp, seconds)
- Peak direction (Dp, degrees)
- Timestamp

**Time Coverage:**
- Historical: Match LiDAR scan dates ± 90 days
- Temporal resolution: 6-hourly preferred

### Precipitation Data (Future)

**Sources:**
- PRISM (Parameter-elevation Regressions on Independent Slopes Model)
- Local rain gauge networks
- NOAA precipitation grids

**Required Fields:**
- Daily precipitation (mm)
- Timestamp
- Location (lat/lon or xy)

**Time Coverage:**
- Historical: Match LiDAR scan dates ± 90 days
- Temporal resolution: Daily

## Phase 3: Change Detection Labels (NOT YET READY)

For supervised learning, we need:

**Multi-Temporal LiDAR:**
- At least 2 scans of the same area
- Time separation: weeks to years
- Must be co-registered or in same coordinate system

**Change Detection:**
- M3C2 algorithm (preferred)
- DoD (DEM of Difference)
- Manual annotation of failures (if available)

**Label Requirements:**
- Retreat distances at each transect
- Timing of failures (for collapse probability)
- Failure mode classification (if available): topple, planar, rotational, rockfall

## Data Organization

Recommended directory structure:

```
data/
├── raw/
│   ├── lidar/
│   │   ├── site1_2023-01-15.laz
│   │   ├── site1_2023-06-20.laz
│   │   ├── site2_2023-01-15.laz
│   │   └── ...
│   ├── coastline/
│   │   ├── site1_coastline.csv
│   │   └── site2_coastline.csv
│   ├── waves/
│   │   └── (future - wave data files)
│   └── precipitation/
│       └── (future - precip data files)
├── processed/
│   ├── transects/
│   │   ├── site1_2023-01-15_transects.npz
│   │   └── ...
│   ├── environmental/
│   │   └── (future)
│   └── labels/
│       └── (future)
└── metadata/
    └── coordinate_systems.txt
```

## When to Provide Data

### NOW (Phase 1.3 - Transect Extraction)
✅ **LiDAR point clouds** (LAS/LAZ files)
✅ **Coastline data** (optional)
✅ **Coordinate system info**

We can immediately test the `TransectExtractor` with your data.

### LATER (Phase 1.4-1.6)
⏳ Wave data
⏳ Precipitation data
⏳ Multi-temporal scans for change detection

I'll let you know when we're ready for these.

## Data Privacy & Sharing

If your data is:
- **Public domain**: Can be included in test datasets
- **Proprietary/Sensitive**: Will only use for development, not distributed
- **Embargoed**: Can work with anonymized subsets or synthetic data

Please let me know any data restrictions.

## Questions?

When providing data, please include:
- Brief description of the site
- Date of acquisition
- Known issues or data quality concerns
- Any specific areas of interest to test

## Contact for Data Submission

Ready to share data? Please provide:
1. Download link or file transfer method
2. Metadata (location, date, coordinate system)
3. Any known issues or special handling requirements

I'll process it through the TransectExtractor and share results/visualizations for validation.
