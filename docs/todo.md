# CliffCast Todo List

# Decisions to make
- should we limit the transect to be only between the cliff toe and cliff top? Can use cliffdelinea2.0 for this. 
- include beach or no? (beach erosion could be one of the failure classes)
- failure classes (large upper cliff failure, cliff toe erosion, minor rockfall, beach erosion, construction should be one also)
- need transect inputs to be refined (m3c2 distance, horizontal/inland change from previous survey? what else?)
- 

## Recently Completed

### Phase 1: Data Pipeline (COMPLETE)
- [x] ShapefileTransectExtractor for LiDAR point extraction
- [x] Cube format conversion (n_transects, T, N, 12)
- [x] Survey CSV input with --beach argument for MOP filtering
- [x] Date parsing from LAS filenames
- [x] Canonical beach/MOP range definitions

### Phase 2: Model Implementation (COMPLETE)
- [x] SpatioTemporalTransectEncoder (spatial then temporal attention)
  - Distance-based sinusoidal positional encoding
  - Learned temporal positional encoding
  - Metadata embedding and broadcasting
  - CLS token pooling
  - 28 tests passing
- [x] EnvironmentalEncoder for wave/atmospheric data
  - Shared architecture for time series
  - Day-of-year seasonality embedding
  - Padding mask support
  - WaveEncoder and AtmosphericEncoder wrappers
  - 37 tests passing
- [x] CrossAttentionFusion module
  - Multi-layer cross-attention (cliff queries environment)
  - Attention weight extraction for interpretability
  - Padding mask support
  - 25 tests passing
- [x] PredictionHeads (all four heads implemented)
  - RiskIndexHead: Sigmoid output [0,1]
  - ExpectedRetreatHead: Softplus for positive values
  - CollapseProbabilityHead: Multi-label 4 horizons
  - FailureModeHead: Multi-class 5 modes
  - Selective enabling for phased training
  - 35 tests passing
- [x] CliffCast main model
  - Full end-to-end model assembly
  - Flexible configuration
  - Attention extraction methods
  - 26 tests passing

### Transect Viewer (COMPLETE)
- [x] Data Dashboard with temporal coverage stats
- [x] Single Transect Inspector with epoch selection
- [x] Temporal Slider view with fixed y-axis for comparison
- [x] Transect Evolution with temporal heatmap (interpolated to common distance grid)
- [x] Cross-Transect View with spatial analysis
- [x] String transect ID support (MOP names like "MOP 595")

### Test Suite (216 tests passing)
- [x] test_transect_extractor.py - 30 tests for extraction + cube format + subsetting + paths
- [x] test_transect_viewer.py - 27 tests for data_loader + validators
- [x] test_utils.py - 8 tests for config utilities
- [x] test_transect_encoder.py - 28 tests for spatio-temporal encoder
- [x] test_environmental_encoder.py - 37 tests for environmental encoders
- [x] test_fusion.py - 25 tests for cross-attention fusion
- [x] test_prediction_heads.py - 35 tests for prediction heads
- [x] test_cliffcast.py - 26 tests for full model
- [x] **100% code coverage** for all model components

### Subset Utility
- [x] `scripts/processing/subset_transects.py` for filtering NPZ by MOP range
- [x] Parse MOP numbers from string IDs (e.g., "MOP 595")
- [x] --beach argument for preset ranges
- [x] --list flag to inspect cube contents

### Cross-Platform Support
- [x] Auto-detect OS (Mac vs Linux)
- [x] Convert paths in survey CSV: `/Volumes/group/...` <-> `/project/group/...`
- [x] --target-os flag to override auto-detection

## In Progress

### Data Collection
- [ ] Process all beaches through extraction pipeline
- [ ] Validate cube files with transect viewer
- [ ] Generate per-beach NPZ files (recommend <500MB each)
- [ ] Download wave data (CDIP MOP system)
- [ ] Process atmospheric data (PRISM + computed features)

## Next Steps

### Phase 3: Training Infrastructure
- [ ] Dataset class for cube format data
  - Load transect NPZ files
  - Integrate wave loader (CDIP)
  - Integrate atmospheric loader
  - Handle temporal alignment
  - Batch collation with padding
- [ ] Loss functions (CliffCastLoss)
  - Risk Index: Smooth L1 loss (weight=1.0)
  - Expected Retreat: Smooth L1 loss (weight=1.0)
  - Collapse Probability: Binary cross-entropy (weight=2.0)
  - Failure Mode: Cross-entropy (weight=0.5, only on failures)
  - Combined weighted loss
- [ ] Training loop with W&B logging
  - Optimizer (AdamW) + scheduler (cosine)
  - Gradient clipping
  - Mixed precision training (AMP)
  - Checkpointing (save top-k)
  - Early stopping
- [ ] Validation on synthetic data
  - Generate synthetic transects with known relationships
  - Verify model learns expected patterns
  - Debug training on small scale

### Phase 4: Evaluation & Metrics
- [ ] Evaluation metrics
  - Risk Index: R², MAE, RMSE
  - Collapse Probability: AUC-ROC, precision-recall, calibration (ECE)
  - Expected Retreat: MAE, RMSE
  - Failure Mode: Accuracy, confusion matrix, per-class F1
- [ ] Baseline comparisons
  - Linear regression baseline
  - Random forest baseline
  - Simple MLP baseline
- [ ] Attention visualization tools
  - Spatial attention heatmaps on cliff profiles
  - Temporal attention over LiDAR epochs
  - Environmental attention over storms/events
  - Interactive visualization app

## Backlog

### Phase 5: Production Deployment
- [ ] Batch inference pipeline for state-wide predictions
- [ ] Model serving API
- [ ] Uncertainty quantification
- [ ] Model interpretation dashboard
- [ ] Transfer learning to other coastlines
- [ ] 3D context enhancement (Option C)

### Data Enhancements
- [ ] Additional environmental features
  - Sea level data
  - Storm surge predictions
  - Groundwater levels
- [ ] Multi-scale modeling (cliff + beach + regional)
- [ ] Incorporate historical failure data
