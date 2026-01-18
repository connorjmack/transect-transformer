# CliffCast: Coastal Cliff Erosion Prediction

## Project Overview
**CliffCast** is a transformer-based deep learning model designed to predict coastal cliff erosion risk. It processes 1D transect data from LiDAR scans along with environmental forcing data (waves and precipitation) to predict erosion risk, collapse probability, and retreat distance.

The project uses a cross-attention mechanism to learn the relationship between cliff geometry and environmental conditions.

## Current Status: Phase 1 (Data Pipeline)
The project is currently in the **Data Pipeline** phase.
- **Implemented**: LiDAR transect extraction logic, shapefile parsing, and initial testing infrastructure.
- **In Progress**: Wave and precipitation data loaders.
- **Not Started**: Model architecture implementation (`src/models`), training loops (`src/training`), and inference scripts (`train.py`, `predict.py`).

**Note:** The `README.md` and `CLAUDE.md` contain forward-looking instructions for commands like `train.py` and `predict.py` which **do not yet exist**.

## Project Structure

```
cliffcast/
├── src/                  # Source code
│   ├── data/             # Data processing (Active Development)
│   │   ├── transect_extractor.py  # Core extraction logic
│   │   └── shapefile_parser.py    # Shapefile handling
│   ├── models/           # Model architectures (Empty/Planned)
│   └── training/         # Training loops (Empty/Planned)
├── scripts/              # Executable scripts
│   └── extract_transects.py  # Main script for data extraction
├── configs/              # YAML configuration files
├── tests/                # Unit and integration tests
├── requirements.txt      # Project dependencies
├── CLAUDE.md             # Developer guidelines
└── plan.md               # Detailed implementation roadmap
```

## Setup and Usage

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Transect Extraction (Current Workflow)
The primary functional script allows extracting transects from LiDAR data (LAS/LAZ) using lines defined in a shapefile.

```bash
python scripts/extract_transects.py \
    --transects data/mops/transects_10m/transect_lines.shp \
    --las-dir data/testing/ \
    --output data/processed/transects.npz \
    --buffer 1.0 \
    --n-points 128 \
    --visualize
```

**Arguments:**
- `--transects`: Path to shapefile containing transect lines.
- `--las-dir`: Directory containing LAS/LAZ point cloud files.
- `--output`: Path to save the extracted data (.npz).
- `--visualize`: (Optional) Generate plots of the extracted profiles.

### 3. Testing
Run the test suite using `pytest`:

```bash
pytest tests/
```

## Development Conventions
*   **Style**: Adhere to `ruff` and `mypy` standards.
*   **Testing**: All new data modules must have corresponding tests in `tests/`.
*   **Configuration**: Use YAML files in `configs/` for parameters.
*   **Documentation**: See `CLAUDE.md` for detailed coding guidelines.

## Roadmap (Immediate Next Steps)
1.  Verify transect extraction on real data samples.
2.  Implement `src/data/wave_loader.py` for CDIP/WW3 data.
3.  Implement `src/data/precip_loader.py` for PRISM data.
4.  Create the PyTorch `Dataset` class to combine these inputs.
