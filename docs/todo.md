# CliffCast Todo List

## Recently Completed

### Phase 1: Data Pipeline (COMPLETE)
- [x] ShapefileTransectExtractor for LiDAR point extraction
- [x] Cube format conversion (n_transects, T, N, 12)
- [x] Survey CSV input with --beach argument for MOP filtering
- [x] Date parsing from LAS filenames
- [x] Canonical beach/MOP range definitions

### Transect Viewer (COMPLETE)
- [x] Data Dashboard with temporal coverage stats
- [x] Single Transect Inspector with epoch selection
- [x] Temporal Slider view with fixed y-axis for comparison
- [x] Transect Evolution with temporal heatmap (interpolated to common distance grid)
- [x] Cross-Transect View with spatial analysis
- [x] String transect ID support (MOP names like "MOP 595")

### Test Suite (65 tests passing)
- [x] test_transect_extractor.py - 30 tests for extraction + cube format + subsetting + paths
- [x] test_transect_viewer.py - 27 tests for data_loader + validators
- [x] test_utils.py - 8 tests for config utilities
- [x] Fixed config override handling (dot-notation)

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

## Next Steps

### Phase 2: Model Implementation
- [ ] SpatioTemporalTransectEncoder (spatial then temporal attention)
- [ ] EnvironmentalEncoder for wave/precip data
- [ ] CrossAttentionFusion module
- [ ] Risk Index prediction head (first head to validate architecture)

### Phase 3: Training Infrastructure
- [ ] Dataset class for cube format data
- [ ] Loss functions (CliffCastLoss)
- [ ] Training loop with W&B logging
- [ ] Validation on synthetic data

## Backlog

### Future Enhancements
- [ ] Wave data downloader and preprocessor
- [ ] Precipitation data downloader and preprocessor
- [ ] Additional prediction heads (retreat, collapse, failure mode)
- [ ] Attention visualization tools
- [ ] 3D context enhancement (Option C)
- [ ] Transfer learning to other coastlines
