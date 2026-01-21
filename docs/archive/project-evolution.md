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

## Current Phase: Susceptibility-Based Classification

**Goal**: Build a 5-class susceptibility classifier that answers "What erosion mode is this transect susceptible to?" using manually labeled erosion mode classes.

**Key Components**:
1. **Labeling Infrastructure**: Streamlit app for efficient transect-epoch labeling with M3C2 visualization
2. **5-Class Classification**: Stable, beach erosion, cliff toe erosion, small rockfall, large upper cliff failure
3. **Derived Risk Scores**: Risk computed from class probabilities weighted by consequence
4. **Water Year Splits**: Train WY2017-WY2023, Val WY2024, Test WY2025

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

### 6. Susceptibility Framing
**Decision**: Classify "what erosion mode is this transect susceptible to?" rather than predicting events.
**Rationale**: Sidesteps stochastic trigger prediction; matches how coastal managers use risk information.

### 7. 5-Class Erosion Mode Classification
**Decision**: 5 classes based on physical process: stable, beach erosion, toe erosion, small rockfall, large failure.
**Rationale**: Classes capture distinct geomorphic processes with different management implications; dominance hierarchy for labeling.

### 8. Asymmetric Loss Weighting
**Decision**: Class weights [0.3, 1.0, 2.0, 2.0, 5.0] with label smoothing (0.1).
**Rationale**: Low weight for stable (weak negative evidence), high weight for dangerous events (must not miss).

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

## Notes

- **Blacks beach events** not yet available - will be added when M3C2 analysis completes
- **Event CSV filenames** are CamelCase (e.g., `DelMar_events_sig.csv`); loader handles both cases
- Start with Phases 1-3 (data pipeline) before touching model code
