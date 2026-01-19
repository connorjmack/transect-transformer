#!/usr/bin/env python3
"""
Tests for Wave Metrics Calculator

Run with: pytest test_wave_metrics.py -v
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

# Import the calculator
from src.data.wave_features import (
    WaveMetricsCalculator,
    WaveMetricsConfig,
    RHO_WATER,
    G
)


@pytest.fixture
def config():
    """Default configuration for tests."""
    return WaveMetricsConfig(
        lookback_days=90,
        resample_hours=6,
        storm_hs_threshold_m=2.0,
        storm_gap_hours=12,
        beach_slope=0.1
    )


@pytest.fixture
def calculator(config):
    """Calculator instance for tests."""
    return WaveMetricsCalculator(config)


@pytest.fixture
def sample_wave_data():
    """Generate sample wave data for testing."""
    # 90 days of hourly data
    n_hours = 90 * 24
    dates = pd.date_range(
        start='2023-09-15',
        periods=n_hours,
        freq='h'
    )
    
    # Realistic wave conditions with some storms
    np.random.seed(42)
    
    # Base conditions: Hs ~ 1m, Tp ~ 10s
    hs = 1.0 + 0.3 * np.random.randn(n_hours)
    tp = 10.0 + 2.0 * np.random.randn(n_hours)
    dp = 270 + 30 * np.random.randn(n_hours)  # Westerly waves
    
    # Add some storm events (Hs > 2m)
    storm_starts = [200, 800, 1500]  # Hour indices
    for start in storm_starts:
        storm_len = np.random.randint(12, 48)
        hs[start:start+storm_len] = 2.5 + 0.5 * np.random.randn(storm_len)
        hs[start:start+storm_len] = np.clip(hs[start:start+storm_len], 2.0, 5.0)
    
    # Ensure positive values
    hs = np.clip(hs, 0.1, None)
    tp = np.clip(tp, 3.0, None)
    dp = dp % 360
    
    df = pd.DataFrame({
        'hs': hs,
        'tp': tp,
        'dp': dp
    }, index=dates)
    
    return df


class TestWavePower:
    """Tests for wave power calculation."""
    
    def test_wave_power_formula(self, calculator):
        """Verify wave power formula is correct."""
        hs = np.array([1.0, 2.0, 3.0])
        tp = np.array([10.0, 10.0, 10.0])
        
        power = calculator.compute_wave_power(hs, tp)
        
        # P = (ρg²/64π) * Hs² * Tp [W/m], converted to kW
        expected = (RHO_WATER * G**2 / (64 * np.pi)) * hs**2 * tp / 1000
        
        np.testing.assert_allclose(power, expected, rtol=1e-6)
    
    def test_wave_power_scaling(self, calculator):
        """Power scales with Hs² and linearly with Tp."""
        hs1 = np.array([1.0])
        hs2 = np.array([2.0])
        tp = np.array([10.0])
        
        power1 = calculator.compute_wave_power(hs1, tp)
        power2 = calculator.compute_wave_power(hs2, tp)
        
        # Doubling Hs should quadruple power
        assert abs(power2[0] / power1[0] - 4.0) < 0.01
    
    def test_wave_power_typical_values(self, calculator):
        """Check that typical values are in reasonable range."""
        # Typical storm: Hs=3m, Tp=12s -> ~50-100 kW/m
        power = calculator.compute_wave_power(
            np.array([3.0]), np.array([12.0])
        )
        assert 30 < power[0] < 150


class TestWaveEnergy:
    """Tests for wave energy calculation."""
    
    def test_wave_energy_formula(self, calculator):
        """Verify energy density formula."""
        hs = np.array([1.0, 2.0])
        
        energy = calculator.compute_wave_energy(hs)
        
        # E = (1/16) * ρ * g * Hs²
        expected = (1/16) * RHO_WATER * G * hs**2
        
        np.testing.assert_allclose(energy, expected, rtol=1e-6)


class TestRunup:
    """Tests for runup calculation."""
    
    def test_runup_positive(self, calculator):
        """Runup should always be positive."""
        hs = np.array([1.0, 2.0, 3.0])
        tp = np.array([10.0, 12.0, 14.0])
        
        runup = calculator.compute_runup_stockdon(hs, tp)
        
        assert np.all(runup > 0)
    
    def test_runup_increases_with_hs(self, calculator):
        """Runup should increase with wave height."""
        tp = np.array([10.0, 10.0, 10.0])
        hs_small = np.array([1.0, 1.0, 1.0])
        hs_large = np.array([3.0, 3.0, 3.0])
        
        runup_small = calculator.compute_runup_stockdon(hs_small, tp)
        runup_large = calculator.compute_runup_stockdon(hs_large, tp)
        
        assert np.all(runup_large > runup_small)
    
    def test_runup_typical_values(self, calculator):
        """Check runup is in reasonable range."""
        # Moderate waves: Hs=2m, Tp=10s -> runup ~1-3m
        runup = calculator.compute_runup_stockdon(
            np.array([2.0]), np.array([10.0]), beach_slope=0.1
        )
        assert 0.5 < runup[0] < 5.0


class TestShoreNormal:
    """Tests for shore-normal component calculation."""
    
    def test_perpendicular_waves_max_impact(self, calculator):
        """Waves perpendicular to cliff have maximum impact."""
        hs = np.array([2.0])
        cliff_orientation = 180  # South-facing cliff
        
        # Waves from south (perpendicular)
        dp_perpendicular = np.array([180.0])
        shore_normal_perp = calculator.compute_shore_normal_component(
            hs, dp_perpendicular, cliff_orientation
        )
        
        # Waves from east (parallel)
        dp_parallel = np.array([270.0])
        shore_normal_parallel = calculator.compute_shore_normal_component(
            hs, dp_parallel, cliff_orientation
        )
        
        # Perpendicular should be ~Hs, parallel should be ~0
        assert shore_normal_perp[0] > 1.9  # Close to Hs
        assert shore_normal_parallel[0] < 0.1  # Close to 0
    
    def test_shore_normal_bounds(self, calculator):
        """Shore-normal component should be <= Hs."""
        hs = np.array([2.0, 3.0, 1.5])
        dp = np.array([0, 90, 180])
        cliff_orientation = 45
        
        shore_normal = calculator.compute_shore_normal_component(
            hs, dp, cliff_orientation
        )
        
        assert np.all(shore_normal <= hs + 0.01)
        assert np.all(shore_normal >= 0)


class TestStormDetection:
    """Tests for storm detection algorithm."""
    
    def test_no_storms(self, calculator):
        """Handle case with no storms."""
        # Low wave heights throughout
        hs = np.array([1.0] * 100)
        time_index = pd.date_range('2023-01-01', periods=100, freq='h')
        
        stats = calculator.detect_storms(hs, time_index)
        
        assert stats['storm_count'] == 0
        assert stats['storm_hours'] == 0
        assert stats['max_storm_duration_hr'] == 0
    
    def test_single_storm(self, calculator):
        """Detect a single storm event."""
        hs = np.array([1.0] * 50 + [3.0] * 24 + [1.0] * 26)  # 100 hours
        time_index = pd.date_range('2023-01-01', periods=100, freq='h')
        
        stats = calculator.detect_storms(hs, time_index)
        
        assert stats['storm_count'] == 1
        assert stats['storm_hours'] == 24
        # Duration is calculated from timestamps, so may be 23-24 depending on edge handling
        assert 22 <= stats['max_storm_duration_hr'] <= 24
    
    def test_multiple_storms(self, calculator):
        """Detect multiple distinct storm events."""
        # Two storms separated by 24 hours (> gap threshold of 12)
        hs = np.array(
            [1.0] * 20 +  # Calm
            [3.0] * 12 +  # Storm 1
            [1.0] * 24 +  # Gap (> 12 hours)
            [2.5] * 18 +  # Storm 2
            [1.0] * 26    # Calm
        )
        time_index = pd.date_range('2023-01-01', periods=len(hs), freq='h')
        
        stats = calculator.detect_storms(hs, time_index)
        
        assert stats['storm_count'] == 2
        assert stats['storm_hours'] == 30  # 12 + 18
    
    def test_time_since_storm(self, calculator):
        """Verify time since last storm calculation."""
        # Storm ends at hour 50, series ends at hour 100
        hs = np.array([3.0] * 50 + [1.0] * 50)
        time_index = pd.date_range('2023-01-01', periods=100, freq='h')
        
        stats = calculator.detect_storms(hs, time_index)
        
        # Should be ~50 hours since storm ended
        assert 48 <= stats['time_since_storm_hr'] <= 52


class TestCircularMean:
    """Tests for circular mean of angles."""
    
    def test_simple_mean(self, calculator):
        """Test simple case where angles are close."""
        angles = np.array([350, 10])  # Should average to ~0
        mean = calculator.compute_circular_mean(angles)
        assert abs(mean) < 5 or abs(mean - 360) < 5
    
    def test_opposite_directions(self, calculator):
        """Test averaging opposite directions."""
        angles = np.array([0, 180])  # Ambiguous case
        mean = calculator.compute_circular_mean(angles)
        # Result could be 90 or 270 depending on implementation
        assert 85 < mean < 95 or 265 < mean < 275
    
    def test_all_same(self, calculator):
        """All same direction should return that direction."""
        angles = np.array([45, 45, 45, 45])
        mean = calculator.compute_circular_mean(angles)
        assert abs(mean - 45) < 0.1


class TestTrend:
    """Tests for trend calculation."""
    
    def test_increasing_trend(self, calculator):
        """Detect positive trend."""
        # Hs increases from 1 to 2 over 24 hours
        values = np.linspace(1, 2, 24)
        time_hours = np.arange(24)
        
        slope = calculator.compute_trend(values, time_hours)
        
        # Slope should be ~1 m/day (1m increase over 24 hours)
        assert 0.9 < slope < 1.1
    
    def test_no_trend(self, calculator):
        """Constant values should have zero trend."""
        values = np.array([1.5] * 48)
        time_hours = np.arange(48)
        
        slope = calculator.compute_trend(values, time_hours)
        
        assert abs(slope) < 0.01
    
    def test_handles_nan(self, calculator):
        """Should handle NaN values gracefully."""
        values = np.array([1.0, np.nan, 2.0, np.nan, 3.0])
        time_hours = np.array([0, 1, 2, 3, 4])
        
        slope = calculator.compute_trend(values, time_hours)
        
        # Should still compute trend from non-NaN values
        assert not np.isnan(slope)


class TestAllMetrics:
    """Integration tests for full metrics computation."""
    
    def test_compute_all_metrics(self, calculator, sample_wave_data):
        """Compute all metrics from sample data."""
        metrics = calculator.compute_all_metrics(sample_wave_data)
        
        # Check all expected metrics are present
        expected_keys = [
            'hs_m', 'tp_s', 'dp_deg', 'power_kw',
            'cumulative_energy_mj', 'cumulative_power_kwh', 'mean_power_kw',
            'max_hs_m', 'hs_p90', 'hs_p99',
            'storm_hours', 'storm_count', 'max_storm_duration_hr',
            'time_since_storm_hr', 'mean_storm_duration_hr',
            'rolling_max_7d_m', 'hs_trend_slope'
        ]
        
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"
            assert not np.isnan(metrics[key]), f"NaN value for {key}"
    
    def test_metrics_reasonable_values(self, calculator, sample_wave_data):
        """Check that computed metrics are in reasonable ranges."""
        metrics = calculator.compute_all_metrics(sample_wave_data)
        
        # Wave height should be positive and reasonable
        assert 0 < metrics['hs_m'] < 10
        assert 0 < metrics['max_hs_m'] < 10
        
        # Period should be positive
        assert metrics['tp_s'] > 0
        
        # Direction should be 0-360
        assert 0 <= metrics['dp_deg'] < 360
        
        # Power should be positive
        assert metrics['power_kw'] > 0
        assert metrics['cumulative_power_kwh'] > 0
        
        # Percentiles should be ordered
        assert metrics['hs_p90'] <= metrics['hs_p99']
        assert metrics['hs_p99'] <= metrics['max_hs_m']
        
        # Storm count should be non-negative
        assert metrics['storm_count'] >= 0
        assert metrics['storm_hours'] >= 0
    
    def test_with_cliff_orientation(self, calculator, sample_wave_data):
        """Test metrics with cliff orientation provided."""
        metrics = calculator.compute_all_metrics(
            sample_wave_data,
            cliff_orientation_deg=270  # West-facing
        )
        
        assert 'shore_normal_hs_m' in metrics
        assert not np.isnan(metrics['shore_normal_hs_m'])
        assert metrics['shore_normal_hs_m'] > 0


class TestTimeseries:
    """Tests for time series feature computation."""
    
    def test_timeseries_shape(self, calculator, sample_wave_data):
        """Check output shape is correct."""
        features, doy = calculator.compute_timeseries_features(sample_wave_data)
        
        # 90 days * 24 hours / 6 hour resample = 360 timesteps
        expected_timesteps = 90 * 24 // 6
        
        assert features.shape[0] == expected_timesteps
        assert features.shape[1] >= 4  # At least hs, tp, dp, power
        assert doy.shape[0] == expected_timesteps
    
    def test_timeseries_dtypes(self, calculator, sample_wave_data):
        """Check output dtypes."""
        features, doy = calculator.compute_timeseries_features(sample_wave_data)
        
        assert features.dtype == np.float32
        assert doy.dtype == np.int32
    
    def test_day_of_year_range(self, calculator, sample_wave_data):
        """Day of year should be 1-366."""
        _, doy = calculator.compute_timeseries_features(sample_wave_data)
        
        assert np.all(doy >= 1)
        assert np.all(doy <= 366)
    
    def test_with_orientation_adds_feature(self, calculator, sample_wave_data):
        """Shore-normal feature added when orientation provided."""
        features_without, _ = calculator.compute_timeseries_features(
            sample_wave_data
        )
        features_with, _ = calculator.compute_timeseries_features(
            sample_wave_data, cliff_orientation_deg=270
        )
        
        assert features_with.shape[1] == features_without.shape[1] + 1


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_dataframe(self, calculator):
        """Handle empty DataFrame."""
        df = pd.DataFrame(columns=['hs', 'tp', 'dp'])
        
        metrics = calculator.compute_all_metrics(df)
        
        # Should return NaN values
        for value in metrics.values():
            assert np.isnan(value) or value == 0
    
    def test_all_nan_values(self, calculator):
        """Handle DataFrame with all NaN."""
        dates = pd.date_range('2023-01-01', periods=100, freq='h')
        df = pd.DataFrame({
            'hs': [np.nan] * 100,
            'tp': [np.nan] * 100,
            'dp': [np.nan] * 100
        }, index=dates)
        
        # Should not raise exception
        metrics = calculator.compute_all_metrics(df)
        
        # Most metrics should be NaN
        assert np.isnan(metrics['hs_m'])
    
    def test_single_timestep(self, calculator):
        """Handle single timestep."""
        df = pd.DataFrame({
            'hs': [2.0],
            'tp': [10.0],
            'dp': [270.0]
        }, index=pd.date_range('2023-01-01', periods=1, freq='h'))
        
        metrics = calculator.compute_all_metrics(df)
        
        assert metrics['hs_m'] == 2.0
        assert metrics['max_hs_m'] == 2.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
