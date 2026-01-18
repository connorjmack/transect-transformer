# CliffCast Configuration Files

This directory contains YAML configuration files for different training phases and model configurations.

## Configuration Files

### `default.yaml`
Full model configuration with all prediction heads enabled. This is the complete configuration used for final training (Phase 4).

**Use when**: Training the full model with all prediction capabilities.

### `phase1_risk_only.yaml`
Phase 1 configuration with only the risk index prediction head enabled.

**Purpose**: Validate the architecture (encoders + fusion) with a single regression task before adding complexity.

**Use when**: Starting development or debugging the core architecture.

### `phase2_add_retreat.yaml`
Phase 2 configuration with risk index + expected retreat heads.

**Purpose**: Add a second regression head to test multi-task learning.

**Use when**: After Phase 1 succeeds and you want to add retreat prediction.

### `phase3_add_collapse.yaml`
Phase 3 configuration with risk + retreat + collapse probability heads.

**Purpose**: Add time-horizon collapse predictions (multi-label classification).

**Use when**: After Phase 2 succeeds and you want to add temporal risk forecasting.

### `phase4_full.yaml`
Phase 4 configuration with all prediction heads enabled.

**Purpose**: Full model with risk, retreat, collapse, and failure mode predictions.

**Use when**: Ready for complete multi-task training with all targets.

## Configuration Structure

Each configuration file has the following sections:

- **model**: Architecture parameters (d_model, n_heads, n_layers, etc.)
- **data**: Data loading parameters (history lengths, normalization, splits)
- **training**: Training hyperparameters (batch size, learning rate, loss weights)
- **logging**: Logging and visualization settings (W&B, checkpoints)
- **evaluation**: Evaluation metrics to compute
- **paths**: Directory paths for data, checkpoints, outputs

## Using Configurations

### Load a configuration in Python

```python
from src.utils.config import load_config

# Load default config
cfg = load_config('configs/default.yaml')
print(cfg.model.d_model)  # 256

# Load with overrides
cfg = load_config(
    'configs/phase1_risk_only.yaml',
    overrides={'training.batch_size': 64}
)
```

### Use in training

```bash
# Phase 1: Risk index only
python train.py --config configs/phase1_risk_only.yaml

# Phase 4: Full model
python train.py --config configs/phase4_full.yaml --data_dir data/processed/
```

### Override values from command line

```bash
# Override batch size and learning rate
python train.py \
    --config configs/default.yaml \
    --override training.batch_size=64 \
    --override training.learning_rate=5e-5
```

## Key Configuration Parameters

### Model Architecture

- `d_model`: Hidden dimension (default: 256)
- `n_heads`: Number of attention heads (default: 8)
- `n_layers_transect`: Transect encoder depth (default: 4)
- `n_layers_env`: Environmental encoder depth (default: 3)
- `n_layers_fusion`: Cross-attention fusion depth (default: 2)

### Prediction Heads

Enable/disable heads for phased training:
- `enable_risk`: Risk index (0-1 normalized)
- `enable_retreat`: Expected retreat distance (m/yr)
- `enable_collapse`: Collapse probability (4 time horizons)
- `enable_failure_mode`: Failure mode classification (5 classes)

### Loss Weights

Balance multi-task objectives:
- `loss_weight_risk`: 1.0 (baseline)
- `loss_weight_retreat`: 1.0 (same importance as risk)
- `loss_weight_collapse`: 2.0 (higher - safety critical)
- `loss_weight_mode`: 0.5 (lower - fewer training labels)

### Data Parameters

- `n_transect_points`: Number of points per transect (default: 128)
- `wave_history_days`: Wave lookback window (default: 90 days)
- `wave_timestep_hours`: Wave temporal resolution (default: 6 hours)
- `precip_history_days`: Precipitation lookback window (default: 90 days)
- `precip_timestep_hours`: Precip temporal resolution (default: 24 hours)

## Creating Custom Configurations

You can create custom configurations by:

1. **Copying an existing config**: Start with the phase config closest to your needs
2. **Modifying parameters**: Adjust model size, training settings, etc.
3. **Using inheritance**: Load base config and override specific values in code

Example custom config:

```yaml
# configs/custom_large.yaml
# Inherit most settings from default but use larger model

model:
  d_model: 512  # Larger hidden dimension
  n_heads: 16   # More attention heads
  # ... other model params

# Use training config from default
training:
  batch_size: 16  # Smaller batch for larger model
  # ... other training params
```

## Validation

All configs are validated on load to ensure required fields are present:
- Model fields: d_model, n_heads, n_layers_*, dropout, n_*_features
- Data fields: n_transect_points, wave_history_days, etc.
- Training fields: batch_size, learning_rate, max_epochs

Validation can be disabled by passing `validate=False` to `load_config()`.
