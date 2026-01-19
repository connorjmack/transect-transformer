# Streamlit Transect Viewer - Implementation Plan

## Overview

A Streamlit-based GUI for visual inspection of transect NPZ files before training. Supports both **data validation** (single epoch QA) and **temporal analysis** (multi-epoch comparison).

**Location**: `apps/transect_viewer/`

---

## Directory Structure

```
apps/
  transect_viewer/
    __init__.py
    app.py                      # Main Streamlit entry point
    config.py                   # App configuration (colors, defaults)

    components/
      __init__.py
      sidebar.py                # File loading & navigation
      transect_inspector.py     # Single transect detail view
      evolution_view.py         # Multi-epoch temporal comparison
      cross_transect_view.py    # Spatial multi-transect view
      data_dashboard.py         # Summary statistics dashboard

    utils/
      __init__.py
      data_loader.py            # NPZ loading with caching
      date_parser.py            # Extract dates from las_sources
      validators.py             # Quality checks (NaN, range validation)
      map_utils.py              # UTM to lat/lon transforms

    plots/
      __init__.py
      profile_plots.py          # Feature vs distance line plots
      rgb_visualization.py      # RGB color bar visualization
      map_plots.py              # Plotly scatter mapbox
      histograms.py             # Distribution plots
      difference_plots.py       # Temporal change visualization
```

---

## View Modes

### 1. Data Summary Dashboard
- Dataset overview (transect count, points, epoch date)
- Feature distributions (histogram grid)
- Metadata distributions
- Data quality report (NaN count, outliers, range validation)

### 2. Single Transect Inspector
- Transect ID selector with prev/next navigation
- All 12 features as interactive line plots vs distance
- Metadata summary card
- RGB color visualization along transect
- Quality check results for selected transect

### 3. Transect Evolution View (Temporal)
- Load multiple NPZ files from different scan dates
- Select transect ID (must exist in all epochs)
- Overlay profiles from different dates (color-coded)
- Difference/change visualization
- Change statistics (max retreat, mean elevation change)

### 4. Cross-Transect View (Spatial)
- Interactive map showing transect locations
- Color by any metadata field (cliff height, slope, etc.)
- Click to select transects for comparison
- Overlay selected transect profiles
- Statistical summaries across selection

---

## Data Structure Reference

**NPZ Contents** (from ShapefileTransectExtractor):
- `points`: (N, 128, 12) - 12 features per point
- `distances`: (N, 128) - distance along transect
- `metadata`: (N, 12) - transect-level stats
- `transect_ids`: (N,) - unique IDs for cross-epoch tracking
- `las_sources`: (N,) - filenames with YYYYMMDD date prefix
- `feature_names`: list of 12 feature names
- `metadata_names`: list of 12 metadata names

**12 Features**: distance_m, elevation_m, slope_deg, curvature, roughness, intensity, red, green, blue, classification, return_number, num_returns

**12 Metadata**: cliff_height_m, mean_slope_deg, max_slope_deg, toe_elevation_m, top_elevation_m, orientation_deg, transect_length_m, latitude, longitude, transect_id, mean_intensity, dominant_class

---

## Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Plotting library | Plotly | Native Streamlit support, interactivity |
| View organization | Sidebar selectbox | Simple navigation, clear separation |
| Multi-epoch handling | Session state dict | Flexible, supports N epochs |
| Caching | `@st.cache_data` | NPZ data is serializable |
| Map library | Plotly Mapbox | Consistent with other plots |
| File loading | Both upload + path | Flexibility for dev vs production |

---

## Dependencies to Add

```
streamlit>=1.30
```

Already available: plotly, numpy, pandas, pyproj

---

## Phased Implementation

### Phase 1: MVP - Core Infrastructure + Dashboard
**Goal**: Basic app running with data loading and overview

- [ ] Create `apps/transect_viewer/` directory structure
- [ ] Implement `utils/data_loader.py` with `@st.cache_data`
- [ ] Implement `utils/date_parser.py` for las_sources parsing
- [ ] Create `app.py` with basic navigation
- [ ] Create `components/sidebar.py` for file loading
- [ ] Create `components/data_dashboard.py`:
  - Dataset overview metrics
  - Feature histograms (Plotly)
  - Basic data quality checks
- [ ] Update requirements.txt with streamlit

**Test**: `streamlit run apps/transect_viewer/app.py`

### Phase 2: Single Transect Inspector
**Goal**: Detailed inspection of individual transects

- [ ] Implement `plots/profile_plots.py`
- [ ] Implement `plots/rgb_visualization.py`
- [ ] Implement `utils/validators.py`
- [ ] Create `components/transect_inspector.py`:
  - Transect selector with navigation
  - Feature tabs with line plots
  - Metadata summary card
  - RGB color bar
  - Quality check display

### Phase 3: Cross-Transect View
**Goal**: Spatial exploration across transects

- [ ] Implement `utils/map_utils.py` (UTM Zone 11N → lat/lon)
- [ ] Implement `plots/map_plots.py`
- [ ] Create `components/cross_transect_view.py`:
  - Interactive location map
  - Click-to-select functionality
  - Multi-transect comparison
  - Feature distributions

### Phase 4: Temporal Evolution View
**Goal**: Multi-epoch comparison and change detection

- [ ] Implement `plots/difference_plots.py`
- [ ] Create `components/evolution_view.py`:
  - Multi-file loading UI
  - Common transect detection
  - Temporal overlay plots
  - Difference visualization
  - Change statistics

### Phase 5: Polish + Future Hooks
**Goal**: Production-ready with extensibility

- [ ] Add export functionality (CSV, PNG)
- [ ] Add help/documentation within app
- [ ] Create hooks for future wave/precip data overlays
- [ ] Performance optimization

---

## Future Extensibility

The app will be designed with hooks for future environmental data:

```python
# config.py - placeholder for future data types
SUPPORTED_DATA_TYPES = {
    'transects': True,
    'wave': False,      # Future: wave forcing time series
    'precipitation': False,  # Future: precip time series
}
```

When wave/precip data is ready, new views can be added:
- Wave Forcing Overlay: Show wave time series alongside transect changes
- Precipitation Overlay: Show precip history with erosion events
- Combined Environmental View: Correlate forcing with transect evolution

---

## UI Mockup

```
+------------------+----------------------------------------+
|  SIDEBAR         |  MAIN PANEL                            |
|                  |                                        |
|  [CliffCast]     |  +----------------------------------+  |
|                  |  | View Title                       |  |
|  DATA LOADING    |  +----------------------------------+  |
|  [File Upload]   |  |                                  |  |
|  [Path Input]    |  |  Content varies by view:         |  |
|  [Load Button]   |  |  - Dashboard: stat cards + hists |  |
|                  |  |  - Inspector: plots + metadata   |  |
|  Loaded: 248     |  |  - Evolution: overlay plots      |  |
|  Date: 2025-11   |  |  - Cross: map + comparisons      |  |
|                  |  |                                  |  |
|  VIEW MODE       |  |                                  |  |
|  [Dashboard   v] |  |                                  |  |
|                  |  |                                  |  |
|  VIEW OPTIONS    |  |                                  |  |
|  (contextual)    |  +----------------------------------+  |
+------------------+----------------------------------------+
```

---

## Run Command

```bash
# After implementation
streamlit run apps/transect_viewer/app.py

# Or with specific port
streamlit run apps/transect_viewer/app.py --server.port 8501
```
