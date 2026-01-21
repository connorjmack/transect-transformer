# CliffCast Implementation Plan

> **Target Architecture**: See `docs/model_plan.md` for the full technical specification.
> **Data Contracts**: See `docs/data_requirements.md` for schemas and validation rules.

This document is an actionable checklist to reach the susceptibility-based CliffCast classifier described in `model_plan.md`.

---

## Quick Reference

| Resource | Purpose |
|----------|---------|
| `docs/model_plan.md` | Target architecture: 5-class susceptibility classification |
| `docs/data_requirements.md` | Data schemas, validation rules, file formats |
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

## Phase 1: Labeling Infrastructure (CURRENT PRIORITY)

Build the system to efficiently label transect-epoch pairs with erosion mode classes.

### 1.1 Labeling Interface
- [ ] **Create `apps/labeling/` Streamlit app**
  - Side-by-side epoch comparison view
  - Show M3C2 change colored on profile
  - Keyboard shortcuts: S=stable, B=beach, T=toe, R=rockfall, F=failure
  - Progress tracking and session management

- [ ] **Pre-filter candidates using M3C2 thresholds**
  - Identify pairs with significant change (|mean M3C2| > 0.5m or max erosion > 1m)
  - Prioritize large events for manual review
  - Auto-label clearly stable pairs (minimal change)

### 1.2 Labeling Guidelines
- [ ] **Create `docs/labeling_guidelines.md`**
  - Visual examples of each class
  - Decision tree for ambiguous cases
  - Dominance hierarchy: Large failure > Small rockfall > Toe erosion > Beach erosion > Stable

### 1.3 Label Storage
- [ ] **Create label CSV schema**
  - Columns: transect_idx, epoch_before, epoch_after, erosion_class, confidence, labeler, notes
  - Store in `data/labels/erosion_mode_labels.csv`

### 1.4 Quality Control
- [ ] **Double-label 10% of samples**
  - Track inter-rater agreement
  - Resolve disagreements to refine class definitions

---

## Phase 2: Data Pipeline

Create training-ready dataset with proper water year splits.

### 2.1 Sample Generation
- [ ] **Create `src/data/susceptibility_dataset.py`**
  - Load transect cube and labels
  - Build samples: context epochs + M3C2 + environment + label
  - Handle variable context lengths with masking

- [ ] **Create `scripts/processing/prepare_susceptibility_data.py`**
  - Input: unified_cube.npz, erosion_mode_labels.csv, cdip/, atmospheric/
  - Output: susceptibility_train.npz, susceptibility_val.npz, susceptibility_test.npz

### 2.2 Water Year Splits
- [ ] **Implement water year splitting logic**
  - Train: WY2017-WY2023 (Oct 2016 - Sep 2023)
  - Validation: WY2024 (Oct 2023 - Sep 2024)
  - Test: WY2025 (Oct 2024 - Sep 2025)
  - Water year N runs from Oct 1 of year N-1 to Sep 30 of year N

- [ ] **Handle survey date → water year mapping**
  - Survey from July 2023 → WY2023 (training)
  - Survey from Nov 2023 → WY2024 (validation)

### 2.3 Environmental Alignment
- [ ] **Load wave features for each sample**
  - 90 days before current epoch, 6hr intervals (360 timesteps)
  - Match MOP ID to CDIP station

- [ ] **Load atmospheric features for each sample**
  - 90 days before current epoch, daily (90 timesteps)
  - From pre-computed parquet files

### 2.4 Validation
- [ ] **Create `scripts/processing/validate_susceptibility_data.py`**
  - Verify label distribution per split
  - Check for data leakage (same transect in train and test at same epoch)
  - Report class balance statistics

---

## Phase 3: Model Updates

Update architecture for 5-class susceptibility classification.

### 3.1 Classification Head
- [ ] **Create `src/models/susceptibility_head.py`**
  - Input: pooled embedding (B, d_model)
  - Output: 5-class logits (B, 5)
  - Classes: stable, beach_erosion, toe_erosion, small_rockfall, large_failure

- [ ] **Implement risk score derivation**
  ```python
  risk_weights = [0.0, 0.1, 0.4, 0.6, 1.0]
  risk_score = (probs * risk_weights).sum(dim=-1)
  ```

### 3.2 Update CliffCast
- [ ] **Update `src/models/cliffcast.py`**
  - Replace prediction heads with SusceptibilityHead
  - Output: logits, probs, risk_score, predicted_class
  - Add `return_attention` option for interpretability

### 3.3 Uncertainty Quantification
- [ ] **Add MC Dropout for epistemic uncertainty**
  - Keep dropout active during inference
  - Run N forward passes, compute mean and std of predictions

### 3.4 Tests
- [ ] **Update model tests**
  - Test 5-class output shape
  - Test risk score derivation
  - Test attention extraction

---

## Phase 4: Training Infrastructure

Build training loop with asymmetric loss weighting.

### 4.1 Loss Function
- [ ] **Create `src/training/susceptibility_loss.py`**
  - Weighted cross-entropy with class weights: [0.3, 1.0, 2.0, 2.0, 5.0]
  - Label smoothing (0.1) for uncertainty
  - Asymmetric treatment: low weight for stable (weak negative evidence)

### 4.2 Training Script
- [ ] **Create `scripts/train_susceptibility.py`**
  - Load config from YAML
  - Initialize model, optimizer (AdamW), scheduler (cosine)
  - Training loop with gradient clipping (max_norm=1.0)
  - W&B logging: losses, per-class metrics, confusion matrix
  - Checkpointing: save top-k by validation macro F1
  - Early stopping: patience=15 epochs

### 4.3 Data Augmentation
- [ ] **Implement geometric augmentation**
  - Small noise to elevation (±0.1m)
  - Small noise to slope (±1°)

- [ ] **Implement temporal augmentation**
  - Randomly drop one context epoch
  - Train robustness to missing data

### 4.4 Config Files
- [ ] **Create `configs/susceptibility_v1.yaml`**
  - Model hyperparameters per `model_plan.md`
  - Training settings: batch_size=32, lr=1e-4, epochs=100
  - Water year split configuration

---

## Phase 5: Evaluation

Comprehensive evaluation with susceptibility-focused metrics.

### 5.1 Classification Metrics
- [ ] **Create `src/training/metrics.py`**
  - Overall accuracy
  - Per-class precision, recall, F1
  - Macro F1 (balanced across classes)
  - Confusion matrix

### 5.2 Susceptibility Ranking Metrics
- [ ] **Implement ranking evaluation**
  - ROC-AUC for binary (any event vs stable)
  - ROC-AUC for "dangerous" (classes 3-4 vs classes 0-2)
  - Precision@K: of top K highest-risk transects, how many experienced events?

### 5.3 Calibration
- [ ] **Implement calibration metrics**
  - Expected Calibration Error (ECE)
  - Reliability diagrams per class

### 5.4 Evaluation Script
- [ ] **Create `scripts/evaluate_susceptibility.py`**
  - Load checkpoint and test data
  - Compute all metrics
  - Generate confusion matrix visualization
  - Save predictions to CSV for analysis

---

## Phase 6: Deployment Pipeline

Operational risk map generation.

### 6.1 Inference Script
- [ ] **Create `scripts/generate_risk_map.py`**
  - Input: trained model, latest survey cube, environmental data
  - Output: risk map with per-transect predictions

### 6.2 Risk Map Visualization
- [ ] **Create `apps/risk_map_viewer/` Streamlit app**
  - Interactive map colored by risk score
  - Click transect for details (class probs, uncertainty, recent M3C2)
  - Comparison to previous assessment

### 6.3 Output Format
- [ ] **Define risk map output schema**
  - Per-transect: predicted_class, class_probabilities, risk_score, risk_category, model_confidence
  - Spatial: latitude, longitude, mop_id
  - Context: recent_change_m, historical_events

### 6.4 Documentation
- [ ] **Create `docs/risk_map_guide.md`**
  - Interpretation guide for coastal managers
  - Risk category definitions and recommended actions
  - Limitations and uncertainty communication

---

## Phase 7: Refinement

Iterate based on evaluation results.

### 7.1 Class Imbalance Mitigation
- [ ] **Experiment with focal loss**
  - Gamma=2.0 for focusing on hard examples
  - Compare to weighted cross-entropy

- [ ] **Experiment with oversampling**
  - Weighted random sampler for rare classes

### 7.2 Hyperparameter Tuning
- [ ] **Grid search key hyperparameters**
  - d_model: [128, 256, 384]
  - n_heads: [4, 8]
  - n_layers: [2, 3, 4]
  - dropout: [0.1, 0.2]

### 7.3 Spatial Generalization
- [ ] **Leave-one-beach-out cross-validation**
  - Test generalization to unseen beaches
  - Identify beach-specific vs general patterns

### 7.4 Attention Analysis
- [ ] **Create `scripts/visualization/attention_analysis.py`**
  - Extract and visualize spatial attention (which cliff locations matter)
  - Extract and visualize temporal attention (which past epochs matter)
  - Extract and visualize cross-attention (which environmental conditions matter)

---

## Success Criteria

From `model_plan.md` Section "Success Metrics":

### Minimum Viable Performance
| Metric | Target |
|--------|--------|
| Overall Accuracy | > 60% |
| Large Failure Recall | > 70% |
| Binary AUC (any event) | > 0.70 |
| Dangerous AUC (class 3-4) | > 0.75 |

### Target Performance
| Metric | Target |
|--------|--------|
| Overall Accuracy | > 70% |
| Macro F1 | > 0.50 |
| Large Failure F1 | > 0.60 |
| Dangerous AUC | > 0.85 |
| Calibration Error | < 0.10 |

---

## Commands Reference

```bash
# Launch labeling interface (Phase 1)
streamlit run apps/labeling/app.py

# Prepare training data with water year splits (Phase 2)
python scripts/processing/prepare_susceptibility_data.py \
  --cube data/processed/unified_cube.npz \
  --labels data/labels/erosion_mode_labels.csv \
  --cdip-dir data/raw/cdip/ \
  --atmos-dir data/processed/atmospheric/ \
  --output-dir data/processed/

# Validate data (Phase 2)
python scripts/processing/validate_susceptibility_data.py --data-dir data/processed/

# Train model (Phase 4)
python scripts/train_susceptibility.py --config configs/susceptibility_v1.yaml --wandb-project cliffcast

# Evaluate model (Phase 5)
python scripts/evaluate_susceptibility.py \
  --checkpoint checkpoints/best.pt \
  --data data/processed/susceptibility_test.npz \
  --output results/

# Generate risk map (Phase 6)
python scripts/generate_risk_map.py \
  --checkpoint checkpoints/best.pt \
  --cube data/processed/latest_survey.npz \
  --output results/risk_map.csv
```

---

## Notes

- **Phase 1 (labeling) is the critical path** - model training depends on quality labels
- **Start labeling big events first** - these are most important for the model to learn
- **Use water year boundaries** - keeps complete storm seasons intact
- **Test incrementally** - validate each phase before moving on
- Previous approaches archived in `docs/archive/`
