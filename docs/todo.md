# CliffCast Todo List

> **Current Priority**: Phase 1 - Labeling Infrastructure
> **Reference**: See `docs/plan.md` for detailed checklist, `docs/model_plan.md` for target architecture

---

## Current Phase: Labeling Infrastructure

### Labeling Interface
- [ ] Create `apps/labeling/` Streamlit app
  - Side-by-side epoch comparison view
  - Show M3C2 change colored on profile
  - Keyboard shortcuts: S=stable, B=beach, T=toe, R=rockfall, F=failure
  - Progress tracking and session management

### Pre-filtering
- [ ] Identify candidate pairs using M3C2 thresholds
  - |mean M3C2| > 0.5m or max erosion > 1m for manual review
  - Auto-label clearly stable pairs (minimal change)

### Documentation
- [ ] Create `docs/labeling_guidelines.md`
  - Visual examples of each class
  - Decision tree for ambiguous cases
  - Dominance hierarchy: Large failure > Small rockfall > Toe erosion > Beach erosion > Stable

### Storage
- [ ] Create label CSV schema
  - Columns: transect_idx, epoch_before, epoch_after, erosion_class, confidence, labeler, notes
  - Store in `data/labels/erosion_mode_labels.csv`

---

## Decisions to Make

- [ ] Should we limit the transect to be only between the cliff toe and cliff top? Can use CliffDelinea 2.0 for this.
- [ ] Include beach in transect or no? (beach erosion is one of the failure classes)
- [ ] Need transect inputs to be refined (M3C2 distance, horizontal/inland change from previous survey?)

---

## Previously Completed

### Phase 0: Foundation (COMPLETE)
- [x] ShapefileTransectExtractor for LiDAR point extraction
- [x] Cube format conversion (n_transects, T, N, 12)
- [x] Canonical beach/MOP range definitions
- [x] Date parsing from LAS filenames

### Model Architecture v1 (COMPLETE)
- [x] SpatioTemporalTransectEncoder (spatial then temporal attention)
- [x] EnvironmentalEncoder for wave/atmospheric data
- [x] CrossAttentionFusion module
- [x] Initial prediction heads (to be updated to SusceptibilityHead)
- [x] CliffCast main model assembly
- [x] **216 tests passing, 100% code coverage**

### Environmental Data Loaders (COMPLETE)
- [x] CDIPWaveLoader for THREDDS/OPeNDAP wave data access
- [x] WaveLoader with LRU caching
- [x] AtmosphericLoader for PRISM-derived features (24 features)

### Transect Viewer App (COMPLETE)
- [x] Data Dashboard with temporal coverage stats
- [x] Single Transect Inspector with epoch selection
- [x] Temporal Slider view with fixed y-axis for comparison
- [x] Transect Evolution with temporal heatmap
- [x] Cross-Transect View with spatial analysis

### Cliff Delineation Integration (COMPLETE)
- [x] CliffDelineaTool v2.0 integration for toe/top detection
- [x] CliffFeatureAdapter transforms 12 features to 13 for detection model
- [x] Sidecar file output (`*.cliff.npz`) with detection results

---

## Future Phases (After Labeling)

### Phase 2: Data Pipeline
- [ ] Create `src/data/susceptibility_dataset.py`
- [ ] Create `scripts/processing/prepare_susceptibility_data.py`
- [ ] Implement water year splitting logic (WY2017-WY2023 train, WY2024 val, WY2025 test)
- [ ] Create `scripts/processing/validate_susceptibility_data.py`

### Phase 3: Model Updates
- [ ] Create `src/models/susceptibility_head.py` (5-class classification)
- [ ] Update `src/models/cliffcast.py` with SusceptibilityHead
- [ ] Implement risk score derivation from class probabilities
- [ ] Update model tests for 5-class output

### Phase 4: Training Infrastructure
- [ ] Create `src/training/susceptibility_loss.py` (weighted cross-entropy)
- [ ] Create `scripts/train_susceptibility.py`
- [ ] Implement data augmentation (geometric, temporal)
- [ ] Create `configs/susceptibility_v1.yaml`

### Phase 5: Evaluation
- [ ] Create `src/training/metrics.py` (per-class F1, macro F1, calibration)
- [ ] Implement ranking metrics (ROC-AUC for binary, dangerous)
- [ ] Create `scripts/evaluate_susceptibility.py`

### Phase 6: Deployment
- [ ] Create `scripts/generate_risk_map.py`
- [ ] Create `apps/risk_map_viewer/` Streamlit app
- [ ] Create `docs/risk_map_guide.md`

---

## Backlog

### Data Enhancements
- [ ] Additional environmental features (sea level, storm surge, groundwater)
- [ ] Multi-scale modeling (cliff + beach + regional)
- [ ] Incorporate historical failure data

### Model Enhancements
- [ ] MC Dropout for uncertainty quantification
- [ ] Attention visualization tools
- [ ] Transfer learning to other coastlines
