# CliffCast Visualization Scripts

Publication-quality figure generation for wave climate analysis and appendix figures.

## Quick Wave Summary

**Script:** `quick_wave_summary.py`

Generates a simple 4-panel overview figure showing wave climate characteristics across all San Diego beaches.

**Usage:**
```bash
python scripts/visualization/quick_wave_summary.py --cdip-dir data/raw/cdip/ --output figures/appendix/
```

**Output:**
- `wave_summary_overview_2017-2025.png` - 4-panel figure with:
  - Wave height distributions by beach
  - Wave height vs period scatter
  - Wave height box plots
  - Wave power distributions

**Date Range:** 2017-2025 (aligned with LiDAR data)

**Requirements:** WaveLoader with downloaded CDIP data

---

## Comprehensive Wave Climate Figures

**Script:** `wave_climate_figures.py`

Generates 8 publication-quality appendix figures for comprehensive wave climate characterization.

**Date Range:** 2017-2025 (aligned with LiDAR data)

**Usage:**
```bash
# Generate all figures
python scripts/visualization/wave_climate_figures.py --cdip-dir data/raw/cdip/ --output figures/appendix/

# Generate specific figures
python scripts/visualization/wave_climate_figures.py --cdip-dir data/raw/cdip/ --output figures/appendix/ --figures A1 A3 A5
```

**Generated Figures:**

### Figure A1: Wave Height Distributions
- **File:** `wave_A1_wave_height_distributions.png`
- **Content:** Histograms with Weibull fits for each beach (6-panel)
- **Shows:** Distribution characteristics, mean, std, 95th percentile per beach

### Figure A2: Wave Period Characteristics
- **File:** `wave_A2_wave_period_characteristics.png`
- **Content:** Hexbin scatter (Hs vs Tp) with marginal histograms
- **Shows:** Wave steepness lines, period-height relationships

### Figure A3: Wave Direction Roses
- **File:** `wave_A3_wave_direction_roses.png`
- **Content:** Polar rose diagrams for each beach (6-panel)
- **Shows:** Directional distribution weighted by wave height, mean direction

### Figure A4: Wave Power Statistics
- **File:** `wave_A4_wave_power_statistics.png`
- **Content:** Box plots, CDFs, distributions, and statistics table (4-panel)
- **Shows:** Wave power characteristics across beaches

### Figure A5: Seasonal Patterns
- **File:** `wave_A5_wave_seasonal_patterns.png`
- **Content:** Monthly means, seasonal boxes, annual heatmap, power (4-panel)
- **Shows:** Intra-annual and inter-annual wave climate variability

### Figure A6: Storm Climatology
- **File:** `wave_A6_wave_storm_climatology.png`
- **Content:** Time series, duration distribution, frequency, intensity vs duration
- **Shows:** Storm identification (95th percentile threshold), characteristics

### Figure A7: Spatial Wave Climate
- **File:** `wave_A7_wave_spatial_climate.png`
- **Content:** Latitudinal profiles and summary table (4-panel)
- **Shows:** Alongshore wave climate variation, beach-averaged statistics

### Figure A8: Extreme Value Analysis
- **File:** `wave_A8_wave_extreme_value_analysis.png`
- **Content:** Annual maxima, GEV fit, return periods, design levels (4-panel)
- **Shows:** Extreme value statistics, return period curves, design wave heights

**Requirements:**
- scipy (for extreme value distributions)
- seaborn (for enhanced styling)

---

## Wave Data Summary

**Current Status (2026-01-19):**
- ✅ 193 MOPs downloaded with complete data
- ✅ Coverage: All 6 San Diego beaches
- ✅ Time range: 2000-2025 (25 years, ~221,328 hourly records per MOP)
- ✅ **Analysis period: 2017-2025 (aligned with LiDAR data)**
- ✅ Data quality: 100% coverage per MOP
- ✅ Features: Hs, Tp, Dp, Ta, wave power

**Beach Coverage:**
| Beach | MOPs | MOP Range | Sample Statistics |
|-------|------|-----------|-------------------|
| Blacks | 48 | 520-567 | Mean Hs: 0.58 m, Max: 3.52 m |
| Torrey | 15 | 567-581 | Mean Hs: 0.82 m, Max: 4.66 m |
| Del Mar | 26 | 595-620 | Mean Hs: 0.89 m, Max: 4.74 m |
| Solana | 30 | 637-666 | Mean Hs: 0.86 m, Max: 5.57 m |
| San Elijo | 26 | 683-708 | Mean Hs: 0.86 m, Max: 4.69 m |
| Encinitas | 5 | 708-712 | Mean Hs: 0.82 m, Max: 4.89 m |

---

## PRISM Atmospheric Data Visualization

**Script:** `plot_prism_coverage.py`

Generates comprehensive multi-panel figures showing atmospheric forcing data across all San Diego beaches.

**Date Range:** 2017-2025 (9 years, 3,287 days per beach)

**Usage:**
```bash
# Generate all three figures (default: figures/appendix/)
python scripts/visualization/plot_prism_coverage.py

# Generate specific figure types
python scripts/visualization/plot_prism_coverage.py --figure-type overview
python scripts/visualization/plot_prism_coverage.py --figure-type features
python scripts/visualization/plot_prism_coverage.py --figure-type extremes

# Custom output directory
python scripts/visualization/plot_prism_coverage.py --output-dir figures/custom/

# Interactive display (does not save)
python scripts/visualization/plot_prism_coverage.py --show
```

**Generated Figures:**

### Figure 1: Overview (prism_overview.png)
**Size:** 18" x 14" (3x3 grid)
**Content:**
1. Beach locations map with PRISM grid overlay
2. Precipitation long-term trend (monthly mean, 2017-2025)
3. Temperature long-term trend (monthly mean, 2017-2025)
4. Precipitation seasonal climatology (mean annual cycle)
5. Temperature seasonal climatology (mean annual cycle)
6. Annual precipitation totals by beach and year
7. Single beach (Del Mar) raw variables with dual y-axes
8. Precipitation spatio-temporal heatmap (beaches vs time)
9. Data coverage summary table with completeness statistics

**Purpose:** High-level overview of atmospheric forcing across all beaches, showing spatial and temporal patterns.

### Figure 2: Feature Distributions (prism_feature_distributions.png)
**Size:** 18" x 12" (5x3 grid)
**Content:** Histograms for 15 derived atmospheric features:
- Cumulative precipitation windows: 7d, 30d, 90d
- Antecedent Precipitation Index (API)
- Dry period metrics: days since rain, consecutive dry days
- Extreme metrics: max 7-day and 30-day precipitation
- Wet-dry cycle counts: 30-day and 90-day windows
- Vapor Pressure Deficit (VPD): daily and 7-day mean
- Freeze-thaw cycles: 30-day and seasonal
- Rain day flag

**Purpose:** Understand the distribution and variability of engineered features used for cliff erosion modeling.

### Figure 3: Extreme Events (prism_extreme_events.png)
**Size:** 16" x 10" (2x2 grid)
**Content:**
1. Extreme precipitation events (>25mm/day) scatter plot
2. Extreme precipitation events (>50mm/day) scatter plot
3. Antecedent Precipitation Index (API) time series
4. Vapor Pressure Deficit (VPD) 7-day mean time series

**Purpose:** Identify storm events, drought periods, and environmental stress indicators that may trigger cliff failures.

**Data Features:**
- **Raw variables (5):** Daily precipitation, min/mean/max temperature, dewpoint
- **Derived features (20):** Cumulative precip, API, wet/dry cycles, VPD, freeze-thaw cycles, intensity classes
- **Total features:** 25 per day per beach
- **Completeness:** ~100% (no missing data in processed files)
- **Spatial coverage:** 6 beaches spanning 30km of coastline

**Key Insights Revealed:**
- Strong seasonal pattern: wet winters (Nov-Mar), dry summers (Jun-Sep)
- Spatial homogeneity: All 6 beaches show similar precipitation patterns (PRISM 4km grid smooths local variation)
- Extreme events: ~10-20 days/year exceed 25mm, ~2-5 days/year exceed 50mm
- Inter-annual variability: 2017 (wet), 2018 (wet), 2020-2022 (dry), 2023 (wet), 2024-2025 (moderate)
- VPD seasonality: High in summer (plant stress, soil desiccation), low in winter

**Requirements:**
- Processed atmospheric parquet files in `data/processed/atmospheric/`
- Files: `{beach}_atmos.parquet` for each of 6 beaches

---

## Installation

Ensure all dependencies are installed:
```bash
pip install matplotlib numpy pandas seaborn scipy xarray netCDF4
```

Or use the project environment:
```bash
source venv/bin/activate
```

---

## Customization

### Modifying Figure Style

Both scripts use `seaborn-paper` style for publication quality. To change:

```python
plt.style.use('seaborn-v0_8-whitegrid')  # Or other style
```

### Beach Colors

Consistent colors defined in `BEACH_COLORS` dict:
```python
BEACH_COLORS = {
    'Blacks': '#1f77b4',
    'Torrey': '#ff7f0e',
    'Del Mar': '#2ca02c',
    'Solana': '#d62728',
    'San Elijo': '#9467bd',
    'Encinitas': '#8c564b',
}
```

### DPI and Format

Change DPI or format in save commands:
```python
plt.savefig(output_path, dpi=600, format='pdf', bbox_inches='tight')  # High-res PDF
```

---

## Troubleshooting

### Issue: Missing MOPs in figures
**Solution:** Some MOPs may not have complete data. The scripts skip MOPs that fail to load.

### Issue: Memory errors with large datasets
**Solution:** Scripts subsample data for scatter plots (max 10,000 points). Reduce further if needed.

### Issue: Corrupt NetCDF files
**Solution:** Download may have been interrupted. Re-run download script to complete missing files.

---

## Future Enhancements

Potential additions for wave climate analysis:
- [ ] Wave-by-wave analysis (individual wave statistics)
- [ ] Directional spectra visualization
- [ ] Multi-year comparisons (El Niño vs La Niña)
- [ ] Joint probability distributions (Hs-Tp-Dp)
- [ ] Wave climate change trends
- [ ] Downtime analysis for construction planning

---

## References

- CDIP MOP System: https://cdip.ucsd.edu/MOP_v1.1/
- Extreme value theory: Coles, S. (2001). An Introduction to Statistical Modeling of Extreme Values.
- Wave climate analysis: Hemer et al. (2013). Projected changes in wave climate from a multi-model ensemble.

---

**Author:** CliffCast Team
**Date:** 2026-01-19
**Version:** 1.0
