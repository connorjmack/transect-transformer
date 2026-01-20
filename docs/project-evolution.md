# CliffCast Project Evolution

This file tracks the project's trajectory, completed phases, key decisions, and lessons learned.

---

## Completed Phases

### Phase 0: Foundation (Complete)

**Timeline**: Initial development through early 2025

**Deliverables**:
- Transect extraction pipeline (`src/data/shapefile_transect_extractor.py`)
- Cube format: `(n_transects, n_epochs, 128, 12)` with metadata
- 10m transect spacing aligned with MOP monitoring points
- Scripts for batch extraction (`scripts/processing/extract_transects.py`)

**Key Files**: `src/data/`, `scripts/processing/extract_transects.py`

### Phase 1: Model Architecture v1 (Complete)

**Deliverables**:
- SpatioTemporalTransectEncoder with spatial-then-temporal attention
- Distance-based sinusoidal positional encoding (not sequential indices)
- Learned temporal positional encoding for LiDAR epochs
- WaveEncoder and AtmosphericEncoder for environmental forcing
- CrossAttentionFusion module (cliff queries environment)
- Initial prediction heads (RiskIndex, ExpectedRetreat, CollapseProbability, FailureMode)
- Full CliffCast model assembly

**Test Coverage**: 216 tests passing, 100% code coverage

**Key Files**: `src/models/`

### Phase 2: Environmental Data Loaders (Complete)

**Deliverables**:
- CDIPWaveLoader for THREDDS/OPeNDAP wave data access
- WaveLoader with LRU caching and optional derived features
- AtmosphericLoader for PRISM-derived features (24 features)
- 183 of 245 San Diego MOPs downloaded (74.7% coverage)

**Key Files**: `src/data/cdip_wave_loader.py`, `src/data/wave_loader.py`, `src/data/atmos_loader.py`

### Phase 3: Transect Viewer App (Complete)

**Deliverables**:
- Streamlit app for NPZ inspection
- Dashboard, single transect, temporal slider, evolution, cross-transect views
- String transect ID support (MOP names like "MOP 595")

**Key Files**: `apps/transect_viewer/`

### Phase 4: Cliff Delineation Integration (Complete)

**Deliverables**:
- CliffDelineaTool v2.0 integration for toe/top detection
- CliffFeatureAdapter transforms 12 features to 13 for detection model
- Sidecar file output (`*.cliff.npz`) with detection results

**Key Files**: `src/data/cliff_delineation/`

---

## Current Phase: Event-Supervised Model Update

**Goal**: Transition from retreat-based to event-supervised learning using M3C2-derived volume measurements.

**Key Changes Required**:
1. **Cube Format v2**: Add 13th feature (M3C2 distance), coverage_mask, beach_slices, mop_ids
2. **Event Alignment**: Connect M3C2 event CSVs to transect cube
3. **New Prediction Heads**: Replace ExpectedRetreat/FailureMode with VolumeHead/EventClassHead
4. **Hybrid Labeling**: Observed events (confidence=1.0) + derived labels (confidence=0.5)

**Reference**: See `docs/plan.md` for detailed checklist, `docs/model_plan.md` for target architecture

---

## Key Architectural Decisions

### 1. Distance-Based Spatial Encoding
**Decision**: Use actual distance from cliff toe for positional encoding, not sequential point indices.
**Rationale**: Physical distance matters for erosion patterns; sequential indices would lose this information.

### 2. Spatio-Temporal Hierarchy
**Decision**: Spatial attention within each timestep, then temporal attention across timesteps.
**Rationale**: Cliff geometry at each scan is a coherent structure; temporal evolution connects these structures.

### 3. Cube Data Format
**Decision**: Store transects as `(n_transects, T, N, 12)` rather than flat format.
**Rationale**: Enables efficient temporal batching and preserves temporal relationships.

### 4. 10m Transect Spacing for Training
**Decision**: Use 10m spacing (~1,958 transects) for model input, while M3C2 analysis used 1m (~19,600 transects).
**Rationale**: 10m provides sufficient spatial resolution while reducing computational cost; 1m was needed for accurate volume integration.

### 5. Cross-Attention for Environmental Fusion
**Decision**: Cliff embeddings query environmental embeddings (not vice versa).
**Rationale**: Learns "which environmental conditions explain each cliff location's state" - more interpretable.

### 6. Phased Training Strategy
**Decision**: Enable prediction heads incrementally (risk → collapse → volume → event class).
**Rationale**: Easier to debug, ensures encoder learns good representations before adding complex heads.

### 7. Event Classification Thresholds
**Decision**: 4 classes based on volume: stable (<10m³), minor (10-50m³), major (50-200m³), failure (≥200m³).
**Rationale**: Geomorphologically meaningful thresholds based on observed event distribution.

### 8. Confidence-Weighted Loss
**Decision**: Weight observed events at 1.5x, derived labels at 0.5x confidence.
**Rationale**: Direct measurements are more reliable than LiDAR-derived change estimates.

---

## Document Hierarchy

| Document | Purpose |
|----------|---------|
| `docs/model_plan.md` | Target architecture, detailed model design, training pipeline |
| `docs/DATA_REQUIREMENTS.md` | Data schemas, validation rules, file formats |
| `docs/plan.md` | Actionable checklist to reach model_plan.md end state |
| `docs/todo.md` | Current sprint tasks and recent completions |
| `docs/archive/plan_v1_spec.md` | Original detailed spec (2600+ lines, superseded by model_plan.md) |
| `CLAUDE.md` | Codebase overview, commands, conventions |

---

## Lessons Learned

### Data Pipeline
- **LAZ files are 10-100x faster** than LAS for loading; use `--prefer-laz` when available
- **Unified cube mode** handles partial-coverage surveys better than per-beach extraction
- **Coverage mask** is essential for training with surveys that don't cover all transects

### Model Development
- **Test small first**: Debug with `d_model=32, n_layers=1` before full config
- **Synthetic data validation**: Test on data with known relationships before real data
- **Attention visualization early**: Catch encoder bugs during training, not at evaluation

### Environmental Data
- **CDIP fill values**: -999.99 must be filtered to NaN
- **Circular mean for direction**: Can't simple-average wave directions; use sin/cos components
- **Temporal alignment**: Environmental lookback is BEFORE scan date (no future leakage)

---

## Success Metrics (Target)

| Metric | Minimum | Target |
|--------|---------|--------|
| Event Detection AUC | > 0.70 | > 0.85 |
| Volume Log-MAE | < 1.0 | < 0.5 |
| Risk Index Correlation | > 0.50 | > 0.70 |
| Event Class F1 (macro) | > 0.40 | > 0.55 |
| Failure Class F1 | - | > 0.50 |

---

## Notes

- **Blacks beach events** not yet available - will be added when M3C2 analysis completes
- **Event CSV filenames** are CamelCase (e.g., `DelMar_events_sig.csv`); loader handles both cases
- Start with Phases 1-3 (data pipeline) before touching model code
