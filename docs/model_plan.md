# CliffCast Model Plan: Susceptibility-Based Coastal Erosion Classification

## Executive Summary

CliffCast is a transformer-based deep learning model that classifies coastal cliff erosion susceptibility by learning from multi-temporal cliff geometry (LiDAR transects across epochs), environmental forcing (waves, precipitation), and labeled erosion mode classes.

### Core Philosophy: Nowcast Susceptibility, Not Event Forecasting

The model answers: **"What erosion mode is this transect susceptible to?"** rather than **"What will happen in the next X days?"**

This framing:
- Sidesteps the stochasticity problem (failures require unpredictable triggers)
- Matches how coastal managers use risk information (prioritization, not timing)
- Produces interpretable outputs that map directly to management actions

### Key Design Principles

1. **Susceptibility, not prediction**: The model learns what geometric and environmental signatures indicate susceptibility to each erosion mode, not when specific events will occur.

2. **Class-based outputs**: Five erosion modes capture distinct physical processes with different management implications.

3. **Temporal evolution as signal**: Multi-epoch inputs reveal progressive weakening patterns that precede failures—this is informative signal, not noise.

4. **Risk scores derived from physics**: Risk is computed from class probabilities weighted by consequence, not learned as a separate target.

---

## Table of Contents

1. [Erosion Mode Classes](#erosion-mode-classes)
2. [Architecture Overview](#architecture-overview)
3. [Data Specifications](#data-specifications)
4. [Training Strategy](#training-strategy)
5. [Evaluation Strategy](#evaluation-strategy)
6. [Deployment Workflow](#deployment-workflow)
7. [Implementation Phases](#implementation-phases)
8. [Success Metrics](#success-metrics)

---

## Erosion Mode Classes

Five classes capture distinct geomorphic processes and management implications:

| Class | Name | Physical Process | Management Implication |
|-------|------|------------------|------------------------|
| **0** | Stable | No significant change | Low priority monitoring |
| **1** | Beach erosion | Sediment transport, tidal processes | Monitor; may recover naturally |
| **2** | Cliff toe erosion | Wave undercutting at cliff base | Warning sign—precursor to larger failures |
| **3** | Small rockfall | Weathering-driven small failures | Maintenance, signage, restricted access |
| **4** | Large upper cliff failure | Major structural collapse | Critical—setbacks, closures, infrastructure review |

### Class Assignment Rules

- **Mutually exclusive**: Each transect-epoch pair receives one label (dominant process)
- **Dominance hierarchy**: Large failure > Small rockfall > Toe erosion > Beach erosion > Stable
- **Multi-transect events**: Failures spanning multiple 10m transects label all affected transects

### Label Asymmetry

Positive labels (classes 1-4) are strong evidence of susceptibility. A transect that experienced a large failure is definitively susceptible to large failures.

Negative labels (class 0) are weaker evidence. A transect labeled "stable" for one epoch may still be highly susceptible—it just hasn't experienced the right trigger yet. This asymmetry should be reflected in loss weighting.

---

## Architecture Overview

### High-Level Design

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    CLIFFCAST SUSCEPTIBILITY CLASSIFIER                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  INPUTS                                                                       │
│  ──────                                                                       │
│  ┌────────────────────┐   ┌─────────────┐   ┌─────────────┐                  │
│  │  Transect Epochs   │   │    Wave     │   │   Atmos     │                  │
│  │  (T_ctx, 128, 12)  │   │  (360, 4)   │   │   (90, 24)  │                  │
│  │                    │   │             │   │             │                  │
│  │  + M3C2 distances  │   │  90 days    │   │  90 days    │                  │
│  │  from prev epoch   │   │  @ 6hr      │   │  @ daily    │                  │
│  └──────────┬─────────┘   └──────┬──────┘   └──────┬──────┘                  │
│             │                    │                  │                         │
│  ENCODERS   ▼                    │                  │                         │
│  ────────                        │                  │                         │
│  ┌────────────────────┐          │                  │                         │
│  │ Spatio-Temporal    │          │                  │                         │
│  │ Transect Encoder   │          │                  │                         │
│  │                    │          │                  │                         │
│  │ • Spatial attn     │          │                  │                         │
│  │   (cliff geometry) │          │                  │                         │
│  │ • Temporal attn    │          │                  │                         │
│  │   (evolution)      │          │                  │                         │
│  └──────────┬─────────┘          ▼                  ▼                         │
│             │             ┌─────────────┐   ┌─────────────┐                  │
│             │             │    Wave     │   │   Atmos     │                  │
│             │             │   Encoder   │   │   Encoder   │                  │
│             │             └──────┬──────┘   └──────┬──────┘                  │
│             │                    │                  │                         │
│             │                    └────────┬─────────┘                         │
│  FUSION     │                             │                                   │
│  ──────     └─────────────┬───────────────┘                                   │
│                           ▼                                                   │
│                  ┌────────────────────┐                                       │
│                  │  Cross-Attention   │                                       │
│                  │      Fusion        │                                       │
│                  │                    │                                       │
│                  │  Q: cliff state    │                                       │
│                  │  K,V: environment  │                                       │
│                  └──────────┬─────────┘                                       │
│                             │                                                 │
│                             ▼                                                 │
│  OUTPUT        ┌────────────────────────┐                                     │
│  ──────        │  Classification Head   │                                     │
│                │                        │                                     │
│                │  → 5-class logits      │                                     │
│                │  → softmax probs       │                                     │
│                │  → derived risk score  │                                     │
│                └────────────────────────┘                                     │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Model Dimensions

| Parameter | Value | Notes |
|-----------|-------|-------|
| `d_model` | 256 | Hidden dimension |
| `n_heads` | 8 | Attention heads |
| `n_spatial_layers` | 3 | Spatial attention depth |
| `n_temporal_layers` | 2 | Temporal attention depth |
| `n_env_layers` | 2 | Environmental encoder depth |
| `n_fusion_layers` | 2 | Cross-attention depth |
| `n_points` | 128 | Points per transect |
| `n_classes` | 5 | Erosion mode classes |
| `dropout` | 0.1 | Regularization |

### Component Roles

**SpatioTemporalTransectEncoder**
- Learns cliff geometry features (overhangs, notches, slope patterns)
- Captures temporal evolution (progressive steepening, crack development)
- Spatial attention: which cliff locations indicate susceptibility
- Temporal attention: which past epochs show precursor patterns

**Environmental Encoders**
- Encode cumulative environmental exposure
- For susceptibility, environment is context (what has this cliff endured?)
- Less emphasis on trigger timing than in forecast framing

**CrossAttentionFusion**
- Links cliff state to environmental history
- Learns: "this cliff geometry under these environmental conditions → susceptibility"
- Attention weights reveal which environmental factors matter for each cliff

**Classification Head**
- Maps fused representation to 5-class susceptibility
- Risk score derived from probability-weighted consequences

### Risk Score Derivation

Risk is computed from class probabilities, not learned separately:

```
risk_score = Σ (P(class_i) × risk_weight_i)

where risk_weights = [0.0, 0.1, 0.4, 0.6, 1.0]
                      stable, beach, toe, rockfall, large_failure
```

This ensures risk has physical meaning: high risk means high probability of high-consequence classes.

---

## Data Specifications

### Study Sites

| Beach | MOP Range | Transects (10m spacing) | Alongshore Span |
|-------|-----------|-------------------------|-----------------|
| Blacks | 520-567 | ~470 | 4700m |
| Torrey Pines | 567-581 | ~140 | 1400m |
| Del Mar | 595-620 | ~250 | 2500m |
| Solana | 637-666 | ~290 | 2900m |
| San Elijo | 683-708 | ~250 | 2500m |
| Encinitas | 708-764 | ~560 | 5600m |
| **Total** | | **~1960** | **~19,600m** |

### Transect Features

**Per-Point Features (12)**

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | distance_m | Distance along transect from start |
| 1 | elevation_m | Elevation (NAVD88) |
| 2 | slope_deg | Local slope |
| 3 | curvature | Profile curvature (1/m) |
| 4 | roughness | Surface roughness (std) |
| 5 | intensity | LAS intensity [0,1] |
| 6 | red | Red channel [0,1] |
| 7 | green | Green channel [0,1] |
| 8 | blue | Blue channel [0,1] |
| 9 | classification | LAS class code |
| 10 | return_number | Return number |
| 11 | num_returns | Total returns |

**M3C2 Distance (computed separately)**

Change distance from previous epoch, provided as additional input:
- Negative values: material loss (erosion)
- Positive values: material gain (deposition/noise)
- Available for all epochs except the first

**Transect-Level Metadata (12)**

| Index | Field | Description |
|-------|-------|-------------|
| 0 | cliff_height_m | Total cliff height |
| 1 | mean_slope_deg | Average cliff face slope |
| 2 | max_slope_deg | Maximum slope (overhang indicator) |
| 3 | toe_elevation_m | Transect start elevation |
| 4 | top_elevation_m | Transect end elevation |
| 5 | orientation_deg | Transect azimuth from N |
| 6 | transect_length_m | Total transect length |
| 7 | latitude | Transect midpoint Y |
| 8 | longitude | Transect midpoint X |
| 9 | transect_id | Numeric ID |
| 10 | mean_intensity | Mean LAS intensity |
| 11 | dominant_class | Most common LAS class |

### Environmental Features

**Wave Features (CDIP MOP System)**
- Shape: (T_w, 4) where T_w = 360 (90 days @ 6hr)
- Features: hs (wave height), tp (peak period), dp (direction), power (kW/m)
- Day-of-year for seasonality encoding

**Atmospheric Features (PRISM + Derived)**
- Shape: (T_a, 24) where T_a = 90 (90 days @ daily)
- Features: precipitation, temperature, API, wet/dry cycles, VPD, freeze-thaw
- Day-of-year for seasonality encoding

---

## Training Strategy

### Sample Construction

Each training sample represents one transect at one epoch pair:

```
Sample = {
    # Context: geometry evolution up to current epoch
    point_features: (T_ctx, 128, 12),  # Multiple epochs of geometry
    metadata: (T_ctx, 12),              # Per-epoch metadata
    distances: (T_ctx, 128),            # Distance along transect
    m3c2_recent: (128,),                # M3C2 from most recent pair

    # Environment: 90 days before current epoch
    wave_features: (360, 4),
    atmos_features: (90, 24),

    # Label: what erosion mode occurred in next epoch pair
    erosion_class: int,  # 0-4
}
```

### What the Model Learns

For a transect at epoch N with context epochs [N-3, N-2, N-1, N]:

**Temporal evolution signal:**
- Epochs N-3 to N-2: stable geometry
- Epochs N-2 to N-1: slight overhang develops
- Epochs N-1 to N: overhang pronounced, toe notch visible
- M3C2 (N-1 to N): negative values at toe (active erosion)

**Environmental context:**
- Strong winter storms in 90 days before epoch N
- High cumulative wave energy

**Label:**
- What happened between epoch N and N+1
- If large failure occurred → label = 4

The model learns: "This geometric evolution pattern + this environmental exposure → susceptible to large failure"

### Loss Function

Weighted cross-entropy with asymmetric class treatment:

```python
class SusceptibilityLoss(nn.Module):
    def __init__(self):
        # Higher weights for rarer, more important classes
        # Lower weight for "stable" (weak negative evidence)
        class_weights = torch.tensor([0.3, 1.0, 2.0, 2.0, 5.0])
        self.loss_fn = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=0.1,  # Helps with label uncertainty
        )

    def forward(self, logits, targets):
        return self.loss_fn(logits, targets)
```

**Weight rationale:**
- Stable (0.3): Weak evidence; don't overfit to "nothing happened"
- Beach erosion (1.0): Baseline
- Toe erosion (2.0): Important precursor signal
- Small rockfall (2.0): Safety concern
- Large failure (5.0): Critical; must not miss these

### Data Augmentation

**Geometric augmentation:**
- Small noise to elevation (±0.1m)
- Small noise to slope (±1°)
- Simulates LiDAR measurement uncertainty

**Temporal augmentation:**
- Randomly drop one context epoch (model must work with variable context)
- Trains robustness to missing data

**No flipping/rotation:** Transects have meaningful orientation (sea → land)

---

## Evaluation Strategy

### Temporal Holdout (Primary)

Train on early epochs, validate on later epochs:

```
Training:   Epochs 1-6 (2017-2022)
Validation: Epochs 6-7 (2022-2023) - hyperparameter tuning
Test:       Epochs 7-8 (2023-2024) - final evaluation
```

This tests: do susceptibility patterns learned from historical data predict future events?

### Spatial Holdout (Generalization)

Leave-one-beach-out cross-validation:

```
Fold 1: Train on 5 beaches, test on Blacks
Fold 2: Train on 5 beaches, test on Del Mar
...
```

This tests: does the model generalize to new locations?

### Combined Spatial-Temporal (Hardest Test)

```
Train: 5 beaches, epochs 1-6
Test:  Held-out beach, epochs 7-8
```

This tests: can the model predict susceptibility for a new location in a future time period?

### Metrics

**Classification metrics:**
- Accuracy (overall)
- Per-class precision, recall, F1
- Macro F1 (treats all classes equally)
- Weighted F1 (accounts for class imbalance)

**Susceptibility ranking metrics:**
- ROC-AUC for binary (any event vs stable)
- ROC-AUC for "dangerous" (classes 3-4 vs classes 0-2)
- Precision@K: of top K highest-risk transects, how many experienced events?

**Calibration metrics:**
- Do predicted probabilities match observed frequencies?
- E.g., among transects with P(large failure) = 0.3, did ~30% actually fail?

**Class-specific validation:**
- Do transects predicted as "toe erosion susceptible" experience toe erosion more than other modes?
- Confusion matrix analysis

---

## Deployment Workflow

### Operational Cycle

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT WORKFLOW                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. NEW SURVEY ARRIVES (Epoch N+1)                              │
│     └─→ LiDAR data for all transects                            │
│                                                                  │
│  2. COMPUTE M3C2 (Epoch N → N+1)                                │
│     └─→ Change detection between epochs                         │
│     └─→ Provides ground truth for epoch N predictions           │
│                                                                  │
│  3. VALIDATE PREVIOUS PREDICTIONS                               │
│     └─→ Compare epoch N susceptibility map to actual outcomes   │
│     └─→ Track calibration, update confidence                    │
│                                                                  │
│  4. GENERATE NEW SUSCEPTIBILITY MAP (Epoch N+1)                 │
│     └─→ Input: N+1 geometry + M3C2(N→N+1) + environment         │
│     └─→ Output: 5-class probs + risk score per transect         │
│                                                                  │
│  5. DELIVER TO MANAGERS                                         │
│     └─→ Risk map visualization                                  │
│     └─→ High-risk area alerts                                   │
│     └─→ Comparison to previous assessment                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Risk Map Output

For each 10m transect segment:

```python
RiskMapOutput = {
    'transect_id': str,
    'mop_id': int,
    'latitude': float,
    'longitude': float,

    # Classification
    'predicted_class': int,           # Most likely erosion mode (0-4)
    'class_probabilities': [5],       # Full probability distribution

    # Risk
    'risk_score': float,              # Derived from probs, range [0,1]
    'risk_category': str,             # "low", "moderate", "high", "critical"

    # Uncertainty
    'model_confidence': float,        # Epistemic uncertainty

    # Context
    'recent_change_m': float,         # Mean M3C2 from previous epoch
    'historical_events': int,         # Count of past events at this transect
}
```

### Risk Categories

| Risk Score | Category | Management Action |
|------------|----------|-------------------|
| 0.0 - 0.2 | Low | Routine monitoring |
| 0.2 - 0.4 | Moderate | Enhanced monitoring |
| 0.4 - 0.6 | High | Access restrictions, signage |
| 0.6 - 1.0 | Critical | Closures, setback enforcement |

---

## Implementation Phases

### Phase 1: Labeling Infrastructure (Current Priority)

**Goal**: Build system to efficiently label transect-epoch pairs

**Tasks**:
- [ ] Create labeling interface (side-by-side epoch comparison)
- [ ] Pre-filter candidates using M3C2 thresholds
- [ ] Define labeling guidelines with examples
- [ ] Label pilot set (100 samples) for calibration
- [ ] Expand to full dataset

**Deliverables**:
- `scripts/labeling/label_interface.py` - Streamlit labeling app
- `data/labels/erosion_mode_labels.csv` - Labeled samples
- `docs/labeling_guidelines.md` - Class definitions and examples

### Phase 2: Data Pipeline

**Goal**: Create training-ready dataset

**Tasks**:
- [ ] Align labels to transect cube
- [ ] Build sample generator (context epochs + environment + label)
- [ ] Implement train/val/test splits (temporal + spatial)
- [ ] Create PyTorch Dataset class

**Deliverables**:
- `src/data/susceptibility_dataset.py` - Dataset class
- `scripts/processing/prepare_susceptibility_data.py` - Pipeline script
- `data/processed/susceptibility_train.npz` - Training data

### Phase 3: Model Implementation

**Goal**: Implement classifier architecture

**Tasks**:
- [ ] Refactor prediction heads for 5-class output
- [ ] Implement risk score derivation
- [ ] Add uncertainty quantification (MC Dropout)
- [ ] Update CliffCast forward pass
- [ ] Unit tests for all components

**Deliverables**:
- `src/models/susceptibility_head.py` - Classification head
- Updated `src/models/cliffcast.py`
- Tests with 100% coverage

### Phase 4: Training Infrastructure

**Goal**: End-to-end training with proper evaluation

**Tasks**:
- [ ] Implement weighted cross-entropy loss
- [ ] Training loop with logging (W&B)
- [ ] Evaluation script with all metrics
- [ ] Hyperparameter search setup

**Deliverables**:
- `src/training/susceptibility_loss.py` - Loss function
- `scripts/train_susceptibility.py` - Training script
- `scripts/evaluate_susceptibility.py` - Evaluation script

### Phase 5: Deployment Pipeline

**Goal**: Operational risk map generation

**Tasks**:
- [ ] Risk map generation script
- [ ] Visualization (static maps + interactive)
- [ ] Validation tracking system
- [ ] Documentation for coastal managers

**Deliverables**:
- `scripts/generate_risk_map.py` - Inference script
- `apps/risk_map_viewer/` - Visualization app
- `docs/risk_map_guide.md` - User documentation

---

## Success Metrics

### Minimum Viable Performance

| Metric | Target | Notes |
|--------|--------|-------|
| Overall Accuracy | > 60% | 5-class classification |
| Large Failure Recall | > 70% | Must not miss dangerous events |
| Binary AUC (any event) | > 0.70 | Discriminate active vs stable |
| Dangerous AUC (class 3-4) | > 0.75 | Critical for safety |

### Target Performance

| Metric | Target | Notes |
|--------|--------|-------|
| Overall Accuracy | > 70% | |
| Macro F1 | > 0.50 | Balanced across classes |
| Large Failure F1 | > 0.60 | Most important class |
| Dangerous AUC | > 0.85 | |
| Calibration Error | < 0.10 | Probabilities match frequencies |

### Interpretability Goals

- Temporal attention identifies which past epochs show precursor patterns
- Spatial attention highlights critical cliff features (overhangs, notches)
- Cross-attention reveals which environmental factors matter
- Risk maps validated by domain experts

---

## Appendix: Comparison to Previous Plan

| Aspect | Previous (v1) | Current (v2) |
|--------|---------------|--------------|
| **Framing** | Event forecasting | Susceptibility nowcast |
| **Primary output** | Volume regression | 5-class classification |
| **Risk score** | Learned target | Derived from class probs |
| **Temporal scope** | Predict next epoch events | Assess current susceptibility |
| **Label source** | M3C2-derived volumes | Manual class labels |
| **Evaluation** | Volume MAE, event AUC | Classification metrics, ranking |
| **Use case** | "50 m³ will erode" | "High risk of large failure" |

**Why the change:**
1. Volume prediction from 1D transects is fundamentally limited
2. Managers need risk prioritization, not volume estimates
3. Class-based outputs map directly to management actions
4. Susceptibility framing sidesteps stochastic trigger prediction
5. Manual labels capture process type, not just magnitude

---

## Appendix: Configuration Template

```yaml
# configs/susceptibility_v1.yaml

model:
  d_model: 256
  n_heads: 8
  n_spatial_layers: 3
  n_temporal_layers: 2
  n_env_layers: 2
  n_fusion_layers: 2
  n_classes: 5
  dropout: 0.1

  # Risk weights for score derivation
  risk_weights: [0.0, 0.1, 0.4, 0.6, 1.0]

data:
  cube_path: data/processed/unified_cube.npz
  labels_path: data/labels/erosion_mode_labels.csv
  min_context_epochs: 2
  max_context_epochs: 5
  wave_lookback_days: 90
  atmos_lookback_days: 90

training:
  batch_size: 32
  learning_rate: 1e-4
  weight_decay: 1e-5
  max_epochs: 100
  early_stopping_patience: 15
  gradient_clip: 1.0

loss:
  class_weights: [0.3, 1.0, 2.0, 2.0, 5.0]
  label_smoothing: 0.1

evaluation:
  temporal_split:
    train_epochs: [1, 2, 3, 4, 5, 6]
    val_epochs: [6, 7]
    test_epochs: [7, 8]
  spatial_holdout_beach: null  # Set for spatial generalization test
```

---

## References

- `archive/model_plan_v1.md`: Previous event-supervised approach
- `docs/architecture-reference.md`: Component API documentation
- `CLAUDE.md`: Project overview and commands
