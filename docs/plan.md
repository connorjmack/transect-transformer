# CliffCast Implementation Plan

> **Target Architecture**: See `docs/model_plan.md` for the full technical specification.
> **Data Contracts**: See `docs/DATA_REQUIREMENTS.md` for schemas and validation rules.

This document is an actionable checklist to reach the event-supervised CliffCast model described in `model_plan.md`.

---

## Quick Reference

| Resource | Purpose |
|----------|---------|
| `docs/model_plan.md` | Target architecture, model design, training pipeline |
| `docs/DATA_REQUIREMENTS.md` | Data schemas, validation rules, file formats |
| `docs/todo.md` | Current sprint tasks and recent completions |
| `CLAUDE.md` | Codebase overview, commands, conventions |

---

## Phase 0: Foundation (COMPLETE)

Core infrastructure that's already built.

- [x] **Transect Extraction Pipeline**
  - `src/data/shapefile_transect_extractor.py` - Extract 128-point transects from LAS files
  - `scripts/processing/extract_transects.py` - CLI for batch extraction
  - Cube format output: `(n_transects, n_epochs, 128, 12)` with metadata
  - 10m transect spacing aligned with MOP monitoring points

- [x] **Model Architecture (v1)**
  - `src/models/transect_encoder.py` - Spatio-temporal attention encoder
  - `src/models/environmental_encoder.py` - Wave and atmospheric encoders
  - `src/models/fusion.py` - Cross-attention fusion
  - `src/models/prediction_heads.py` - Multi-task prediction heads (v1)
  - `src/models/cliffcast.py` - Full model assembly
  - 151 tests passing, 100% coverage

- [x] **Environmental Data Loaders**
  - `src/data/cdip_wave_loader.py` - CDIP wave data via THREDDS/OPeNDAP
  - `src/data/wave_loader.py` - Wave integration for training
  - `src/data/atmos_loader.py` - Atmospheric features from PRISM

- [x] **Transect Viewer App**
  - `apps/transect_viewer/` - Streamlit app for NPZ inspection
  - Dashboard, single transect, temporal slider, evolution, cross-transect views

---

## Phase 1: Cube Format v2

Update the transect cube to support event-supervised training.

### 1.1 Add M3C2 Distance Feature
- [ ] **Update `ShapefileTransectExtractor` to output 13 features**
  - Add `m3c2_distance` as feature index 12
  - First epoch: set to 0.0 (no previous surface)
  - Subsequent epochs: compute change from previous epoch's surface
  - File: `src/data/shapefile_transect_extractor.py`

- [ ] **Update feature normalization**
  - Clip M3C2 to [-5, 2] meters, normalize to [-1, 1]
  - Handle NaN values (set to 0.0)
  - Add to `normalize_m3c2()` function per `DATA_REQUIREMENTS.md` Section 2.3

### 1.2 Add Cube Metadata
- [ ] **Add `coverage_mask` array**
  - Shape: `(n_transects, n_epochs)`, dtype: bool
  - True where valid data exists, False for missing transect-epoch pairs
  - Enables partial-coverage surveys in unified cube

- [ ] **Add `beach_slices` dict**
  - Maps beach name to `(start_idx, end_idx)` tuple
  - Use canonical 10m spacing indices from `DATA_REQUIREMENTS.md` Section 2.5

- [ ] **Add `mop_ids` array**
  - Shape: `(n_transects,)`, dtype: int32
  - Extract integer MOP ID from transect string (e.g., "MOP 595" → 595)

### 1.3 Update Tests
- [ ] **Update `tests/test_data/test_transect_extractor.py`**
  - Test 13 features instead of 12
  - Test coverage_mask, beach_slices, mop_ids presence
  - Test M3C2 distance normalization

### 1.4 Regenerate Cube
- [ ] **Extract unified cube with new format**
  ```bash
  python scripts/processing/extract_transects.py --transects data/mops/transects_10m/transect_lines.shp --survey-csv data/raw/master_list.csv --output data/processed/unified_cube.npz --unified --prefer-laz --workers 8
  ```
- [ ] **Validate with transect viewer**

---

## Phase 2: Event Alignment Pipeline

Connect M3C2-derived event CSVs to the transect cube.

### 2.1 Event Loading
- [ ] **Create `src/data/event_loader.py`**
  - Implement `load_events(beach, events_dir)` per `DATA_REQUIREMENTS.md` Section 3.5
  - Support both lowercase and CamelCase filenames
  - Parse dates, validate required columns
  - Add `event_class` based on volume thresholds

- [ ] **Create `load_all_events(events_dir)` function**
  - Load and concatenate all beach event CSVs
  - Skip missing beaches gracefully (blacks not yet available)
  - Return combined DataFrame with beach column

### 2.2 Event-to-Cube Alignment
- [ ] **Create `scripts/processing/align_events.py`**
  - Map `alongshore_centroid_m` → `transect_idx` using coordinate conversion
  - Find bracketing epochs (epoch_before, epoch_after) for each event
  - Aggregate multiple events per (transect, epoch_pair)
  - Output: `data/processed/aligned_events.parquet`
  - See `model_plan.md` Section "Event Integration Pipeline" for algorithm

- [ ] **Implement aggregation logic**
  - Sum volumes, propagate uncertainties (sqrt of sum of squares)
  - Track event counts, max height/width
  - Compute event_class from total_volume

### 2.3 Tests
- [ ] **Create `tests/test_data/test_event_loader.py`**
  - Test case-insensitive filename loading
  - Test date parsing and validation
  - Test event classification thresholds
  - Test alignment to cube coordinates

---

## Phase 3: Training Data Generation

Create the final training tensors with aligned features and labels.

### 3.1 Sample Generation
- [ ] **Create `scripts/processing/prepare_training_data.py`**
  - Input: unified_cube.npz, aligned_events.parquet, cdip/, atmospheric/
  - Output: training_data.npz per `DATA_REQUIREMENTS.md` Section 7

- [ ] **Implement sliding window sample generation**
  - For each transect with ≥4 valid epochs
  - Context = 3-10 epochs, target = next epoch
  - Model never sees target epoch in input (true prediction)

- [ ] **Implement hybrid labeling**
  - Priority 1: Observed events from CSVs (confidence=1.0)
  - Priority 2: Derived from M3C2 distances (confidence=0.5)
  - Track `label_source` (0=derived, 1=observed)

- [ ] **Load and align environmental features**
  - Wave: 90 days before target epoch, 6hr intervals (360 timesteps)
  - Atmos: 90 days before target epoch, daily (90 timesteps)
  - Compute day-of-year arrays for seasonality encoding

### 3.2 Compute Labels
- [ ] **Compute risk index from volume + cliff height**
  - Use `compute_risk_index()` formula from `model_plan.md`
  - Log-transform volume, height factor, sigmoid normalization

- [ ] **Assign event classes**
  - Class 0 (stable): volume < 10 m³
  - Class 1 (minor): 10 ≤ volume < 50 m³
  - Class 2 (major): 50 ≤ volume < 200 m³
  - Class 3 (failure): volume ≥ 200 m³

### 3.3 Data Splits
- [ ] **Create `scripts/processing/create_splits.py`**
  - Temporal split (default): last 15% = test, prior 15% = val
  - Spatial split option: leave one beach out
  - Output: train_indices.npy, val_indices.npy, test_indices.npy

### 3.4 Validation
- [ ] **Create `scripts/processing/validate_all.py`**
  - Verify cube structure and value ranges
  - Check event alignment coverage
  - Validate training data shapes and label distributions
  - Report observed vs derived label percentages

---

## Phase 4: Model Updates

Update model architecture to match `model_plan.md`.

### 4.1 Update Transect Encoder
- [ ] **Update `src/models/transect_encoder.py`**
  - Change default `n_point_features` from 12 to 13
  - Update docstrings to reference M3C2 distance feature

### 4.2 Replace Prediction Heads
- [ ] **Create `VolumeHead` in `src/models/prediction_heads.py`**
  - Input: pooled embedding (B, d_model)
  - Output: log(volume+1) prediction (B,)
  - Activation: Softplus to ensure positive values

- [ ] **Create `EventClassHead` in `src/models/prediction_heads.py`**
  - Input: pooled embedding (B, d_model)
  - Output: logits for 4 classes (B, 4)
  - Classes: stable, minor, major, failure

- [ ] **Update `PredictionHeads` class**
  - Replace `ExpectedRetreatHead` with `VolumeHead`
  - Replace `FailureModeHead` with `EventClassHead`
  - Update enable flags: `enable_volume`, `enable_event_class`

- [ ] **Update `CliffCast` forward pass**
  - Change output keys: `volume_pred`, `event_class_logits`
  - Update docstrings and type hints

### 4.3 Update Tests
- [ ] **Update `tests/test_models/test_prediction_heads.py`**
  - Test VolumeHead output shape and range
  - Test EventClassHead output shape
  - Test selective head enabling

- [ ] **Update `tests/test_models/test_cliffcast.py`**
  - Test with 13 input features
  - Test new output keys

---

## Phase 5: Training Infrastructure

Build the training loop with confidence weighting.

### 5.1 Dataset Class
- [ ] **Create `src/data/cliffcast_dataset.py`**
  - Load training_data.npz
  - Implement `__getitem__` returning dict of tensors
  - Pad context to max_context_epochs
  - Handle context_mask for variable-length sequences

### 5.2 Loss Function
- [ ] **Create `src/training/losses.py`**
  - Implement `CliffCastLoss` per `model_plan.md` Section "Loss Functions"
  - Volume: Smooth L1 on log(vol+1)
  - Event class: Cross-entropy (with optional focal loss)
  - Risk index: Smooth L1
  - Collapse probability: Binary cross-entropy per horizon
  - Confidence weighting: weight samples by confidence score
  - Observed boost: multiply observed sample weights by 1.5x

### 5.3 Training Script
- [ ] **Create `train.py`**
  - Load config from YAML
  - Initialize model, optimizer (AdamW), scheduler (cosine)
  - Training loop with gradient clipping (max_norm=1.0)
  - Validation after each epoch
  - W&B logging: losses, metrics, sample predictions
  - Checkpointing: save top-k by validation loss
  - Early stopping: patience=10 epochs

### 5.4 Config Files
- [ ] **Create `configs/cliffcast_v2.yaml`**
  - Model hyperparameters per `model_plan.md` Appendix
  - Training settings: batch_size=32, lr=1e-4, epochs=100
  - Loss weights: volume=1.0, class=1.0, risk=0.5, collapse=2.0

---

## Phase 6: Evaluation & Analysis

Comprehensive evaluation with stratified metrics.

### 6.1 Metrics Implementation
- [ ] **Create `src/training/metrics.py`**
  - `volume_metrics()`: MAE, RMSE, Log-MAE, correlation
  - `classification_metrics()`: accuracy, per-class F1, macro F1, detection AUC
  - `risk_metrics()`: MAE, RMSE, correlation
  - `stratified_metrics()`: separate metrics for observed vs derived labels

### 6.2 Evaluation Script
- [ ] **Create `evaluate.py`**
  - Load checkpoint and test data
  - Compute all metrics
  - Generate confusion matrix
  - Save predictions to CSV for analysis

### 6.3 Attention Visualization
- [ ] **Create `scripts/visualization/attention_analysis.py`**
  - Extract spatial attention: which cliff locations are critical
  - Extract temporal attention: which past epochs matter
  - Extract cross-attention: which environmental events drive erosion
  - Generate attention heatmaps overlaid on cliff profiles

---

## Phase 7: Refinement

Iterate based on evaluation results.

### 7.1 Class Imbalance
- [ ] **Implement focal loss option for event classification**
  - Alpha weights: [0.25, 0.5, 1.0, 2.0] for [stable, minor, major, failure]
  - Gamma=2.0 for focusing on hard examples

- [ ] **Experiment with oversampling rare classes**
  - Duplicate failure samples in training set
  - Or use weighted random sampler

### 7.2 Hyperparameter Tuning
- [ ] **Grid search key hyperparameters**
  - d_model: [128, 256, 384]
  - n_heads: [4, 8]
  - n_layers: [2, 3, 4]
  - dropout: [0.1, 0.2]

### 7.3 Ensemble (Optional)
- [ ] **Train multiple models with different seeds**
- [ ] **Implement prediction averaging**

---

## Success Criteria

From `model_plan.md` Section "Success Metrics":

### Minimum Viable Performance
| Metric | Target |
|--------|--------|
| Event Detection AUC | > 0.70 |
| Volume Log-MAE | < 1.0 |
| Risk Index Correlation | > 0.50 |
| Event Class F1 (macro) | > 0.40 |

### Target Performance
| Metric | Target |
|--------|--------|
| Event Detection AUC | > 0.85 |
| Volume Log-MAE | < 0.5 |
| Risk Index Correlation | > 0.70 |
| Event Class F1 (macro) | > 0.55 |
| Failure Class F1 | > 0.50 |

---

## Commands Reference

```bash
# Extract unified cube (Phase 1)
python scripts/processing/extract_transects.py --transects data/mops/transects_10m/transect_lines.shp --survey-csv data/raw/master_list.csv --output data/processed/unified_cube.npz --unified --prefer-laz --workers 8

# Align events to cube (Phase 2)
python scripts/processing/align_events.py --cube data/processed/unified_cube.npz --events-dir data/raw/events/ --output data/processed/aligned_events.parquet

# Generate training data (Phase 3)
python scripts/processing/prepare_training_data.py --cube data/processed/unified_cube.npz --events data/processed/aligned_events.parquet --cdip-dir data/raw/cdip/ --atmos-dir data/processed/atmospheric/ --output data/processed/training_data.npz --min-context 3 --max-context 10

# Validate all data (Phase 3)
python scripts/processing/validate_all.py --data-dir data/

# Train model (Phase 5)
python train.py --config configs/cliffcast_v2.yaml --wandb-project cliffcast

# Evaluate model (Phase 6)
python evaluate.py --checkpoint checkpoints/best.pt --data data/processed/training_data.npz --output results/
```

---

## Notes

- **Start with Phase 1-3** (data pipeline) before touching model code
- **Test incrementally**: validate each phase before moving on
- **Use small subsets first**: test with `--limit 5` or single beach before full runs
- Blacks beach events not yet available - will be added when M3C2 analysis completes
