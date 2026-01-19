# CliffCast: Coastal Cliff Erosion Prediction

**CliffCast** is a transformer-based deep learning model designed to predict coastal cliff erosion risk. It processes 1D transect data from LiDAR scans along with environmental forcing data (waves and precipitation) to predict erosion risk, collapse probability, and retreat distance.

## Current Status: Phase 1 (Data Pipeline & Visualization)

The project is currently completing **Phase 1**. The data extraction pipeline is fully functional, capable of converting raw LiDAR (LAS/LAZ) data into structured 4D data cubes. A comprehensive interactive viewer has been built to validate and explore the processed data.

**Implemented Features:**
- **LiDAR Transect Extraction**: Extracts elevation profiles from point clouds along defined shapefile normals.
- **4D Data Cube Format**: Standardized `(N_transects, Time, Space, Features)` format.
- **Interactive Transect Viewer**: Streamlit app for visualizing profile evolution, heatmaps, and data coverage.
- **Cross-Platform Support**: Auto-handling of path differences between macOS and Linux environments.
- **Data Utilities**: Scripts for subsetting data and managing survey metadata.

## Project Structure

```
cliffcast/
├── apps/
│   └── transect_viewer/      # Streamlit application for data inspection
├── configs/                  # YAML configuration files
├── data/                     # Data directory (metadata, processed, raw)
├── docs/
│   └── ...
├── scripts/
│   ├── processing/
│   └── visualization/
├── src/
│   ├── data/
│   ├── utils/
│   └── ...
└── tests/
```

## Installation

```bash
# Clone the repository
git clone https://github.com/connorjmack/transect-transformer.git
cd transect-transformer

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Interactive Transect Viewer
Launch the Streamlit app to inspect processed `.npz` files:

```bash
streamlit run apps/transect_viewer/app.py
```

### 2. Extract Transects from LiDAR
Extract profiles from LAS/LAZ files using a shapefile of transect lines:

```bash
python scripts/processing/extract_transects.py \
    --transects data/mops/transects_10m/transect_lines.shp \
    --las-dir /path/to/las/files \
    --output data/processed/my_beach.npz \
    --survey-csv data/metadata/master_survey_list.csv \
    --beach "Torrey Pines"
```

### 3. Subset Processed Data
Filter an existing NPZ file by MOP range:

```bash
python scripts/processing/subset_transects.py \
    --input data/processed/all_beaches.npz \
    --output data/processed/torrey_pines.npz \
    --beach "Torrey Pines"
```

## Development

### Running Tests
The project maintains a high standard of testing with over 60 passing tests.

```bash
pytest tests/
```

### Documentation
- [CLAUDE.md](CLAUDE.md): Developer guidelines and commands.
- [docs/todo.md](docs/todo.md): Current task list and roadmap.
- [docs/plan.md](docs/plan.md): Detailed architectural plan.

## Next Steps (Phase 2)
- Implement Model Architecture (`src/models`)
    - SpatioTemporalTransectEncoder
    - EnvironmentalEncoder
    - CrossAttentionFusion
- Develop Training Infrastructure (`src/training`)

## License
MIT