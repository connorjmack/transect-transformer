"""
Synthetic data generator for testing training pipeline.

Generates fake transect, wave, and atmospheric data with simple known
relationships to verify the model can learn patterns before using real data.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch


class SyntheticDataGenerator:
    """
    Generate synthetic cliff erosion data with controllable relationships.

    Creates transects, wave data, and atmospheric data with simple patterns:
    - Higher waves → higher retreat
    - More precipitation → higher retreat
    - Steeper cliffs → higher collapse probability
    - Progressive temporal changes → detectable with temporal attention

    Args:
        n_samples: Number of transect samples to generate
        n_timesteps: Number of LiDAR epochs (T)
        n_points: Number of points per transect (N)
        n_wave_timesteps: Wave time series length (T_w)
        n_atmos_timesteps: Atmospheric time series length (T_a)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        n_samples: int = 100,
        n_timesteps: int = 5,
        n_points: int = 128,
        n_wave_timesteps: int = 360,
        n_atmos_timesteps: int = 90,
        seed: int = 42,
    ):
        self.n_samples = n_samples
        self.n_timesteps = n_timesteps
        self.n_points = n_points
        self.n_wave_timesteps = n_wave_timesteps
        self.n_atmos_timesteps = n_atmos_timesteps
        self.seed = seed

        np.random.seed(seed)
        torch.manual_seed(seed)

    def generate_transects(self) -> Dict[str, np.ndarray]:
        """
        Generate synthetic transect data in cube format.

        Returns:
            Dictionary with keys:
                - data: (n_samples, T, N, 12) point features
                - metadata: (n_samples, T, 12) transect metadata
                - distances: (n_samples, T, N) distances from toe
                - las_sources: (T,) list of epoch names
                - transect_ids: (n_samples,) transect identifiers
        """
        # Reset seed for reproducibility
        rng = np.random.RandomState(self.seed)

        N = self.n_samples
        T = self.n_timesteps
        P = self.n_points

        # Generate distances (0-50m from cliff toe)
        distances = np.tile(
            np.linspace(0, 50, P)[np.newaxis, np.newaxis, :], (N, T, 1)
        )

        # Generate point features (12 features)
        data = np.zeros((N, T, P, 12))

        for i in range(N):
            # Base cliff shape (different for each transect)
            base_elevation = rng.uniform(5, 30)
            cliff_height = rng.uniform(10, 40)

            for t in range(T):
                # Temporal degradation (cliff erodes over time)
                erosion_factor = 1.0 - 0.05 * t

                # distance_m (feature 0) - already in distances
                data[i, t, :, 0] = distances[i, t, :]

                # elevation_m (feature 1) - cliff profile
                # Simple cliff shape: low at toe, high at top
                norm_dist = distances[i, t, :] / 50.0
                data[i, t, :, 1] = base_elevation + cliff_height * (
                    norm_dist ** 1.5
                ) * erosion_factor

                # slope_deg (feature 2) - steeper near cliff face
                data[i, t, :, 2] = 20 + 40 * (1 - norm_dist) ** 2

                # curvature (feature 3) - higher at cliff edge
                data[i, t, :, 3] = np.exp(-((norm_dist - 0.7) ** 2) / 0.1)

                # roughness (feature 4) - random but temporally consistent
                data[i, t, :, 4] = 0.5 + 0.3 * rng.randn(P)

                # intensity (feature 5) - normalized [0, 1]
                data[i, t, :, 5] = rng.uniform(0.3, 0.9, P)

                # RGB (features 6-8) - rock colors
                data[i, t, :, 6:9] = rng.uniform(0.4, 0.7, (P, 3))

                # classification (feature 9) - cliff=6, vegetation=3
                data[i, t, :, 9] = np.where(norm_dist > 0.8, 3, 6)

                # return_number, num_returns (features 10-11)
                data[i, t, :, 10] = 1
                data[i, t, :, 11] = 1

        # Generate metadata (12 fields per transect per timestep)
        metadata = np.zeros((N, T, 12))
        for i in range(N):
            for t in range(T):
                # Compute from point data
                elevations = data[i, t, :, 1]
                slopes = data[i, t, :, 2]

                metadata[i, t, 0] = elevations.max() - elevations.min()  # cliff_height_m
                metadata[i, t, 1] = slopes.mean()  # mean_slope_deg
                metadata[i, t, 2] = slopes.max()  # max_slope_deg
                metadata[i, t, 3] = elevations.min()  # toe_elevation_m
                metadata[i, t, 4] = elevations.max()  # top_elevation_m
                metadata[i, t, 5] = rng.uniform(0, 360)  # orientation_deg
                metadata[i, t, 6] = 50.0  # transect_length_m
                metadata[i, t, 7] = 32.0 + rng.uniform(-0.5, 0.5)  # latitude
                metadata[i, t, 8] = -117.0 + rng.uniform(-0.5, 0.5)  # longitude
                metadata[i, t, 9] = i  # transect_id
                metadata[i, t, 10] = data[i, t, :, 5].mean()  # mean_intensity
                metadata[i, t, 11] = 6  # dominant_class

        # Generate epoch names
        las_sources = np.array([f"epoch_{t:02d}" for t in range(T)])

        # Generate transect IDs
        transect_ids = np.array([f"MOP {600+i}" for i in range(N)])

        return {
            'data': data,
            'metadata': metadata,
            'distances': distances,
            'las_sources': las_sources,
            'transect_ids': transect_ids,
        }

    def generate_wave_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic wave data.

        Returns:
            Tuple of (features, day_of_year):
                - features: (n_samples, T_w, 4) [hs, tp, dp, power]
                - day_of_year: (n_samples, T_w) day of year [0-365]
        """
        # Reset seed for reproducibility
        rng = np.random.RandomState(self.seed + 1)

        N = self.n_samples
        T_w = self.n_wave_timesteps

        features = np.zeros((N, T_w, 4))
        day_of_year = np.zeros((N, T_w))

        for i in range(N):
            # Generate seasonal wave pattern
            t = np.linspace(0, 90, T_w)  # 90 days
            doy = (np.arange(T_w) * 0.25).astype(int) % 366  # Every 6 hours

            # Significant wave height (hs) - sample-specific base level + seasonal + storms
            base_level = rng.uniform(0.8, 2.0)  # Sample-specific baseline
            seasonal = 0.5 * np.sin(2 * np.pi * t / 90)
            storm_events = rng.poisson(0.05, T_w) * rng.uniform(0, 2, T_w)
            features[i, :, 0] = np.clip(base_level + seasonal + storm_events, 0.5, 8.0)

            # Peak period (tp) - correlated with hs
            features[i, :, 1] = 8 + 4 * (features[i, :, 0] / 4.0)

            # Peak direction (dp) - mostly from west (270 deg), wrapped to [0, 360]
            features[i, :, 2] = (270 + rng.normal(0, 30, T_w)) % 360

            # Wave power - computed from hs and tp
            hs = features[i, :, 0]
            tp = features[i, :, 1]
            features[i, :, 3] = 0.5 * (hs ** 2) * tp  # Simplified power formula

            day_of_year[i, :] = doy

        return features, day_of_year

    def generate_atmospheric_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic atmospheric data.

        Returns:
            Tuple of (features, day_of_year):
                - features: (n_samples, T_a, 24) atmospheric features
                - day_of_year: (n_samples, T_a) day of year [0-365]
        """
        # Reset seed for reproducibility
        rng = np.random.RandomState(self.seed + 2)

        N = self.n_samples
        T_a = self.n_atmos_timesteps

        features = np.zeros((N, T_a, 24))
        day_of_year = np.zeros((N, T_a))

        for i in range(N):
            t = np.arange(T_a)
            doy = (t * 1.0).astype(int) % 366  # Daily

            # Feature 0: precip_mm - random rainfall events
            rain_prob = 0.1 + 0.05 * np.sin(2 * np.pi * t / 90)
            rain_days = rng.rand(T_a) < rain_prob
            features[i, :, 0] = rain_days * rng.exponential(10, T_a)

            # Feature 1: temp_mean_c - seasonal pattern
            features[i, :, 1] = 15 + 5 * np.sin(2 * np.pi * t / 365)

            # Feature 2-3: temp_min_c, temp_max_c
            features[i, :, 2] = features[i, :, 1] - 5
            features[i, :, 3] = features[i, :, 1] + 5

            # Feature 4: dewpoint_c
            features[i, :, 4] = features[i, :, 1] - 3

            # Features 5-8: cumulative precipitation (7d, 30d, 60d, 90d)
            precip = features[i, :, 0]
            for j, window in enumerate([7, 30, 60, 90]):
                features[i, :, 5 + j] = np.convolve(
                    precip, np.ones(window), mode='same'
                ) / window

            # Feature 9: API (Antecedent Precipitation Index)
            features[i, :, 9] = np.cumsum(precip * 0.95 ** np.arange(T_a)[::-1])

            # Feature 10: days_since_rain
            days_since = np.zeros(T_a)
            last_rain = -999
            for j in range(T_a):
                if rain_days[j]:
                    last_rain = j
                days_since[j] = j - last_rain if last_rain >= 0 else 999
            features[i, :, 10] = np.clip(days_since, 0, 30)

            # Feature 11: consecutive_dry_days
            features[i, :, 11] = features[i, :, 10]  # Simplified

            # Feature 12-15: rain_day_flag, intensity_class, max_precip_7d, max_precip_30d
            features[i, :, 12] = rain_days.astype(float)
            features[i, :, 13] = np.clip(precip / 10, 0, 3).astype(int)
            features[i, :, 14] = features[i, :, 5]  # 7d cumulative
            features[i, :, 15] = features[i, :, 6]  # 30d cumulative

            # Features 16-17: wet_dry_cycles
            features[i, :, 16:18] = rng.poisson(2, (T_a, 2))

            # Features 18-19: VPD (vapor pressure deficit)
            features[i, :, 18] = rng.uniform(0.5, 2.5, T_a)
            features[i, :, 19] = features[i, :, 18]  # 7d mean

            # Features 20-23: freeze-thaw cycles
            freeze_threshold = 0
            freeze = features[i, :, 2] < freeze_threshold
            features[i, :, 20] = freeze.astype(float)
            features[i, :, 21] = (
                (features[i, :, 2] > freeze_threshold) & (features[i, :, 2] < 5)
            ).astype(float)
            features[i, :, 22:24] = rng.poisson(1, (T_a, 2))

            day_of_year[i, :] = doy

        return features, day_of_year

    def generate_targets(
        self,
        wave_features: np.ndarray,
        atmos_features: np.ndarray,
        transect_metadata: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Generate synthetic targets with known relationships to inputs.

        Simple relationships:
        - retreat_m = 0.5 + 0.3*mean_hs + 0.2*total_precip + noise
        - risk_index = sigmoid(2*(retreat - 1))
        - collapse_labels = based on cliff height and retreat
        - failure_mode = based on slope and retreat

        Args:
            wave_features: (N, T_w, 4) wave data
            atmos_features: (N, T_a, 24) atmospheric data
            transect_metadata: (N, T, 12) transect metadata

        Returns:
            Dictionary with targets:
                - risk_index: (N,)
                - retreat_m: (N,)
                - collapse_labels: (N, 4) for 4 horizons
                - failure_mode: (N,) class indices
        """
        # Reset seed for reproducibility
        rng = np.random.RandomState(self.seed + 3)

        N = self.n_samples

        # Compute retreat based on wave energy and precipitation
        mean_hs = wave_features[:, :, 0].mean(axis=1)  # Mean significant height
        total_precip = atmos_features[:, :, 0].sum(axis=1)  # Total precipitation

        # Simple linear relationship with noise (stronger correlation)
        retreat_m = (
            0.5 + 0.5 * mean_hs + 0.003 * total_precip + rng.normal(0, 0.15, N)
        )
        retreat_m = np.clip(retreat_m, 0.1, 5.0)

        # Compute risk index from retreat and cliff height
        cliff_height = transect_metadata[:, -1, 0]  # Last timestep cliff height
        height_factor = 1 + 0.1 * (cliff_height - 20) / 20
        weighted_retreat = retreat_m * np.clip(height_factor, 0.5, 1.5)
        risk_index = 1 / (1 + np.exp(-2 * (weighted_retreat - 1)))

        # Compute collapse probabilities (4 horizons)
        # Higher probability for taller cliffs and higher retreat
        collapse_base_prob = (risk_index + retreat_m / 5) / 2
        collapse_labels = np.zeros((N, 4))
        for h in range(4):
            # Probability decreases with horizon
            collapse_labels[:, h] = collapse_base_prob * (0.9 ** h)
            # Convert to binary labels with some randomness
            collapse_labels[:, h] = (
                rng.rand(N) < collapse_labels[:, h]
            ).astype(float)

        # Compute failure mode based on slope and retreat with better distribution
        mean_slope = transect_metadata[:, -1, 1]
        failure_mode = np.zeros(N, dtype=int)

        for i in range(N):
            # Diverse failure mode distribution with reasonable physical relationships
            # Use normalized scores to ensure good distribution
            retreat_score = (retreat_m[i] - retreat_m.min()) / (retreat_m.max() - retreat_m.min() + 1e-6)
            slope_score = (mean_slope[i] - mean_slope.min()) / (mean_slope.max() - mean_slope.min() + 1e-6)

            if retreat_score < 0.2:
                failure_mode[i] = 0  # Stable (low retreat)
            elif slope_score > 0.7:
                failure_mode[i] = 1  # Topple (steep)
            elif slope_score > 0.5 and retreat_score < 0.5:
                failure_mode[i] = 2  # Planar (moderate steep, lower retreat)
            elif retreat_score > 0.7:
                failure_mode[i] = 3  # Rotational (high retreat)
            else:
                failure_mode[i] = 4  # Rockfall (middle range)

        return {
            'risk_index': risk_index.astype(np.float32),
            'retreat_m': retreat_m.astype(np.float32),
            'collapse_labels': collapse_labels.astype(np.float32),
            'failure_mode': failure_mode.astype(np.int64),
        }

    def generate_dataset(self) -> Dict[str, np.ndarray]:
        """
        Generate complete synthetic dataset.

        Returns:
            Dictionary containing all inputs and targets.
        """
        # Generate all components
        transect_data = self.generate_transects()
        wave_features, wave_doy = self.generate_wave_data()
        atmos_features, atmos_doy = self.generate_atmospheric_data()

        # Generate targets with known relationships
        targets = self.generate_targets(
            wave_features, atmos_features, transect_data['metadata']
        )

        # Combine everything
        dataset = {
            # Transect inputs
            'point_features': transect_data['data'],
            'metadata': transect_data['metadata'],
            'distances': transect_data['distances'],
            'las_sources': transect_data['las_sources'],
            'transect_ids': transect_data['transect_ids'],
            # Wave inputs
            'wave_features': wave_features,
            'wave_doy': wave_doy,
            # Atmospheric inputs
            'atmos_features': atmos_features,
            'atmos_doy': atmos_doy,
            # Targets
            **targets,
        }

        return dataset

    def save_dataset(self, output_path: str) -> None:
        """
        Generate and save synthetic dataset to NPZ file.

        Args:
            output_path: Path to save NPZ file
        """
        dataset = self.generate_dataset()

        # Convert to appropriate types for NPZ
        save_dict = {}
        for key, value in dataset.items():
            if isinstance(value, np.ndarray):
                save_dict[key] = value
            else:
                save_dict[key] = np.array(value)

        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **save_dict)

        print(f"Saved synthetic dataset to {output_path}")
        print(f"  Samples: {self.n_samples}")
        print(f"  Timesteps: {self.n_timesteps}")
        print(f"  Points per transect: {self.n_points}")
