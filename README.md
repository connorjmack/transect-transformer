# CliffCast: Transformer-Based Coastal Cliff Erosion Prediction

CliffCast is a transformer-based deep learning model that predicts coastal cliff erosion risk by learning relationships between cliff geometry (from LiDAR transects), wave forcing, and precipitation history.

## Core Innovation

Cross-attention fusion allows the model to learn "which storm events matter for which cliff locations" - providing interpretable predictions grounded in physical processes.

## Features

- **Multi-task prediction**: Risk index, collapse probability (4 time horizons), expected retreat, and failure mode classification
- **Interpretable attention**: Cross-attention weights reveal which environmental conditions drive erosion
- **Scalable**: Operates on 1D transects rather than full 3D point clouds for state-wide processing

## Installation

### Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

### Install from source

```bash
# Clone the repository
git clone https://github.com/connorjmack/transect-transformer.git
cd transect-transformer

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e ".[dev]"
```

## Quick Start

### Training

```bash
# Phase 1: Risk index only (validate architecture)
python train.py --config configs/phase1_risk_only.yaml --data_dir data/processed/

# Phase 4: Full model (all prediction heads)
python train.py --config configs/phase4_full.yaml --data_dir data/processed/
```

### Inference

```bash
# Predict on new data
python predict.py \
    --input data/new_site/ \
    --checkpoint checkpoints/best.pt \
    --output results/ \
    --format geojson
```

### Evaluation

```bash
# Evaluate on test set
python evaluate.py \
    --checkpoint checkpoints/best.pt \
    --data_dir data/processed/ \
    --split test \
    --output results/
```

## Project Structure

```
cliffcast/
├── src/
│   ├── data/          # Data loading and preprocessing
│   ├── models/        # Model architectures (encoders, fusion, heads)
│   ├── training/      # Training loop, losses, callbacks
│   ├── evaluation/    # Metrics, calibration, baselines
│   ├── visualization/ # Attention maps, prediction plots
│   ├── inference/     # Single and batch prediction
│   └── utils/         # Logging, config, I/O utilities
├── tests/             # Unit and integration tests
├── configs/           # Model and training configurations
├── scripts/           # Data download and preprocessing scripts
└── notebooks/         # Jupyter notebooks for analysis

```

## Model Architecture

The model consists of:

1. **Transect Encoder**: Self-attention over cliff geometry points
2. **Environmental Encoders**: Separate encoders for wave and precipitation time series
3. **Cross-Attention Fusion**: Cliff embeddings attend to environmental conditions
4. **Multi-Task Heads**: Four prediction heads sharing the fused representation

See [plan.md](plan.md) for detailed architecture specifications.

## Documentation

- [CLAUDE.md](CLAUDE.md): Developer guide for working with this codebase
- [plan.md](plan.md): Comprehensive implementation plan and architecture details

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use CliffCast in your research, please cite:

```bibtex
@software{cliffcast2026,
  title = {CliffCast: Transformer-Based Coastal Cliff Erosion Prediction},
  author = {Mack, Connor J.},
  year = {2026},
  url = {https://github.com/connorjmack/transect-transformer}
}
```

## Development Status

This project is currently in **Phase 1: Project Setup & Data Pipeline**. See [plan.md](plan.md) for the full implementation roadmap.
