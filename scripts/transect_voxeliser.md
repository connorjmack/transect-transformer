# placeholder concept for 1D voxelization later on

class TransectVoxelizer:
    """
    Voxelize points along a transect into 1D bins.
    
    This is conceptually similar to 3D voxelization but optimized for
    the transect geometry - we bin along the profile distance axis.
    """
    
    def __init__(
        self,
        bin_size_m: float = 0.5,      # Voxel size along transect
        corridor_width_m: float = 2.0, # Width of extraction corridor
        max_bins: int = 256,           # Maximum sequence length
        min_points_per_bin: int = 3,   # Minimum points to form valid bin
    ):
        self.bin_size = bin_size_m
        self.corridor_width = corridor_width_m
        self.max_bins = max_bins
        self.min_points = min_points_per_bin
    
    def extract(
        self,
        points: np.ndarray,           # (M, 3+) full point cloud
        transect_origin: np.ndarray,  # (3,) start point (at toe)
        transect_direction: np.ndarray,  # (3,) unit vector (shore-normal)
        transect_length: float,       # Total length to extract
    ) -> dict:
        """
        Extract voxelized transect from point cloud.
        
        Returns dict with:
            - bin_centers: (N,) distance along transect for each bin
            - bin_features: (N, F) aggregated features per bin
            - bin_mask: (N,) which bins have valid data
        """
        # 1. Extract points within corridor
        corridor_points = self._extract_corridor(
            points, transect_origin, transect_direction
        )
        
        # 2. Project to transect coordinates
        # distance = projection onto transect direction
        # elevation = z coordinate (or perpendicular distance)
        relative = corridor_points[:, :3] - transect_origin
        distances = np.dot(relative, transect_direction)
        elevations = corridor_points[:, 2]  # Keep original z
        
        # 3. Bin by distance
        n_bins = min(int(transect_length / self.bin_size), self.max_bins)
        bin_edges = np.linspace(0, transect_length, n_bins + 1)
        bin_indices = np.digitize(distances, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # 4. Aggregate features per bin
        bin_features = []
        bin_mask = []
        
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() >= self.min_points:
                bin_pts = corridor_points[mask]
                features = self._aggregate_bin(bin_pts, elevations[mask])
                bin_features.append(features)
                bin_mask.append(True)
            else:
                # Empty or sparse bin - use placeholder
                bin_features.append(np.zeros(self.n_features))
                bin_mask.append(False)
        
        return {
            'bin_centers': (bin_edges[:-1] + bin_edges[1:]) / 2,
            'bin_features': np.stack(bin_features),
            'bin_mask': np.array(bin_mask),
        }
    
    def _aggregate_bin(self, points: np.ndarray, elevations: np.ndarray) -> np.ndarray:
        """
        Aggregate point features within a bin.
        
        This is analogous to the voxel feature aggregation in PTv3.
        """
        return np.array([
            np.mean(elevations),           # Mean elevation
            np.std(elevations),            # Roughness (elevation variance)
            np.max(elevations) - np.min(elevations),  # Height range
            self._compute_slope(points),   # Local slope
            self._compute_curvature(points),  # Local curvature
            len(points),                   # Point density (informative!)
        ])
```

## Why This Approach?

**Captures voxelization benefits:**
- Variable density → aggregated naturally (dense vegetation ≠ overweighted)
- Point density becomes a *feature* (low density might indicate overhang/cave)
- No interpolation artifacts
- Robust to noise (aggregate, don't sample)

**Avoids 3D overhead:**
- 1D sequence, not 3D grid
- No space-filling curve serialization needed
- Much smaller memory footprint

**Enables future 3D extension:**
- If you later want full 3D context (Option C), you can voxelize the corridor in 3D
- The conceptual framework is the same

## Visual Comparison
```
UNIFORM RESAMPLING:
Points:     *  * *    *       * * *  *    *  *
            ↓  ↓ ↓    ↓       ↓ ↓ ↓  ↓    ↓  ↓
Resampled:  •    •    •    •    •    •    •    •  (interpolated)
            
Problem: Sparse region gets interpolated (unreliable)


1D VOXELIZATION (SEGMENT AGGREGATION):
Points:     *  * *    *       * * *  *    *  *
            ↓  ↓ ↓    ↓       ↓ ↓ ↓  ↓    ↓  ↓
Bins:       [•••]  [• ]  [ ]  [••••]  [• ]  [••]
            
Features:   mean, std, slope per bin
Mask:       [1]    [1]   [0]  [1]     [1]   [1]

Benefit: Sparse bin gets masked, dense bins use all points