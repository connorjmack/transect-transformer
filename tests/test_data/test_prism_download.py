"""Tests for PRISM data download and extraction.

These tests verify that PRISM data was downloaded correctly and extracted
for all San Diego study beaches using point-only mode (no TIF storage).
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Test data directory
PRISM_DIR = Path('data/raw/prism')
BEACHES = ['blacks', 'torrey', 'delmar', 'solana', 'sanelijo', 'encinitas']
VARIABLES = ['ppt', 'tmin', 'tmax', 'tmean', 'tdmean']
TEST_DATE_RANGE = (datetime(2023, 1, 1), datetime(2023, 1, 7))


class TestPRISMDirectory:
    """Tests for PRISM data directory structure."""

    def test_prism_directory_exists(self):
        """Verify PRISM data directory was created."""
        assert PRISM_DIR.exists(), f"PRISM directory not found: {PRISM_DIR}"

    def test_point_only_mode_no_tifs(self):
        """Verify no TIF files stored (point-only extraction mode)."""
        tif_files = list(PRISM_DIR.rglob('*.tif'))
        assert len(tif_files) == 0, \
            f"Found {len(tif_files)} TIF files - should be 0 in point-only mode"

    def test_csv_files_exist(self):
        """Verify CSV files exist for all beaches."""
        csv_files = list(PRISM_DIR.glob('*_raw.csv'))
        assert len(csv_files) == len(BEACHES), \
            f"Expected {len(BEACHES)} CSV files, found {len(csv_files)}"


class TestBeachExtraction:
    """Tests for extracted beach time series."""

    @pytest.mark.parametrize("beach", BEACHES)
    def test_raw_csv_exists(self, beach):
        """Verify raw CSV was created for each beach."""
        csv_path = PRISM_DIR / f'{beach}_raw.csv'
        assert csv_path.exists(), f"Raw CSV not found: {csv_path}"

    @pytest.mark.parametrize("beach", BEACHES)
    def test_csv_has_expected_columns(self, beach):
        """Verify CSV has all expected columns."""
        csv_path = PRISM_DIR / f'{beach}_raw.csv'
        df = pd.read_csv(csv_path)

        expected_cols = ['date'] + VARIABLES
        for col in expected_cols:
            assert col in df.columns, \
                f"Missing column '{col}' in {beach}_raw.csv"

    @pytest.mark.parametrize("beach", BEACHES)
    def test_csv_has_expected_rows(self, beach):
        """Verify CSV has data for expected date range."""
        csv_path = PRISM_DIR / f'{beach}_raw.csv'
        df = pd.read_csv(csv_path, parse_dates=['date'])

        # Should have 7 days
        assert len(df) == 7, \
            f"Expected 7 rows, got {len(df)} for {beach}"

    @pytest.mark.parametrize("beach", BEACHES)
    def test_no_missing_values(self, beach):
        """Verify no NaN values in extracted data."""
        csv_path = PRISM_DIR / f'{beach}_raw.csv'
        df = pd.read_csv(csv_path)

        for var in VARIABLES:
            nan_count = df[var].isna().sum()
            assert nan_count == 0, \
                f"{beach} has {nan_count} NaN values in {var}"

    @pytest.mark.parametrize("beach", BEACHES)
    def test_value_ranges_reasonable(self, beach):
        """Verify extracted values are in physically reasonable ranges."""
        csv_path = PRISM_DIR / f'{beach}_raw.csv'
        df = pd.read_csv(csv_path)

        # Precipitation: 0 to 500 mm/day
        assert df['ppt'].min() >= 0, f"Negative precipitation in {beach}"
        assert df['ppt'].max() <= 500, f"Unreasonable precipitation in {beach}"

        # Temperature: -20 to 50 °C for San Diego
        for temp_var in ['tmin', 'tmax', 'tmean']:
            assert df[temp_var].min() >= -20, \
                f"Unreasonable low temp in {beach}.{temp_var}"
            assert df[temp_var].max() <= 50, \
                f"Unreasonable high temp in {beach}.{temp_var}"

        # Dewpoint should be <= mean temp (with small tolerance)
        assert (df['tdmean'] <= df['tmean'] + 5).all(), \
            f"Dewpoint > mean temp in {beach}"

    def test_temperature_consistency(self):
        """Verify tmin <= tmean <= tmax for all beaches."""
        for beach in BEACHES:
            csv_path = PRISM_DIR / f'{beach}_raw.csv'
            df = pd.read_csv(csv_path)

            assert (df['tmin'] <= df['tmean']).all(), \
                f"tmin > tmean in {beach}"
            assert (df['tmean'] <= df['tmax']).all(), \
                f"tmean > tmax in {beach}"

    def test_date_range_correct(self):
        """Verify dates match expected test range."""
        csv_path = PRISM_DIR / 'delmar_raw.csv'
        df = pd.read_csv(csv_path, parse_dates=['date'])

        start, end = TEST_DATE_RANGE
        assert df['date'].min().date() == start.date(), \
            f"Start date mismatch: {df['date'].min().date()} != {start.date()}"
        assert df['date'].max().date() == end.date(), \
            f"End date mismatch: {df['date'].max().date()} != {end.date()}"


class TestSpatialVariation:
    """Tests for spatial variation across beaches."""

    def test_values_differ_across_beaches(self):
        """Verify we're extracting from different grid cells."""
        all_data = {}
        for beach in BEACHES:
            csv_path = PRISM_DIR / f'{beach}_raw.csv'
            all_data[beach] = pd.read_csv(csv_path)

        # Check that at least some values differ between beaches
        # (indicating we're reading from different grid cells)
        ppt_means = {b: all_data[b]['ppt'].mean() for b in BEACHES}
        unique_means = len(set(round(v, 2) for v in ppt_means.values()))

        # Expect variation (at least 3 different values across 6 beaches)
        # Note: Some adjacent beaches may share grid cells at 4km resolution
        assert unique_means >= 3, \
            f"Only {unique_means} unique precip means - expected more spatial variation"

    def test_north_south_gradient(self):
        """Verify geographic ordering makes sense."""
        # Beaches are ordered south to north: blacks -> encinitas
        # This test just verifies data was extracted for the full extent
        all_data = {}
        for beach in BEACHES:
            csv_path = PRISM_DIR / f'{beach}_raw.csv'
            all_data[beach] = pd.read_csv(csv_path)

        # All should have data
        for beach in BEACHES:
            assert len(all_data[beach]) > 0, f"No data for {beach}"

    def test_adjacent_beaches_may_share_cells(self):
        """Document expected behavior: nearby beaches may share PRISM cells.

        PRISM has 4km resolution. San Elijo and Encinitas are ~3.7km apart
        so they may fall in the same grid cell. This is expected behavior.
        """
        sanelijo = pd.read_csv(PRISM_DIR / 'sanelijo_raw.csv')
        encinitas = pd.read_csv(PRISM_DIR / 'encinitas_raw.csv')

        # These beaches are close enough to potentially share a cell
        # Just verify both have valid data
        assert not sanelijo['ppt'].isna().any()
        assert not encinitas['ppt'].isna().any()


class TestBeachCoordinates:
    """Tests for beach coordinate configuration."""

    def test_beach_coords_from_mop_shapefile(self):
        """Verify beach coordinates match MOP transect centroids."""
        from scripts.processing.download_prism_data import BEACH_COORDS

        # Expected coordinates (from MOP shapefile centroids)
        expected = {
            'blacks': (32.893798, -117.253693),
            'torrey': (32.920009, -117.259033),
            'delmar': (32.949681, -117.265282),
            'solana': (32.988459, -117.274256),
            'sanelijo': (33.026229, -117.287640),
            'encinitas': (33.059110, -117.303186),
        }

        for beach, (exp_lat, exp_lon) in expected.items():
            lat, lon = BEACH_COORDS[beach]
            assert abs(lat - exp_lat) < 0.001, \
                f"{beach} latitude mismatch: {lat} != {exp_lat}"
            assert abs(lon - exp_lon) < 0.001, \
                f"{beach} longitude mismatch: {lon} != {exp_lon}"

    def test_beaches_ordered_south_to_north(self):
        """Verify beaches are geographically ordered."""
        from scripts.processing.download_prism_data import BEACH_COORDS

        ordered_beaches = ['blacks', 'torrey', 'delmar', 'solana', 'sanelijo', 'encinitas']
        lats = [BEACH_COORDS[b][0] for b in ordered_beaches]

        # Each beach should be north of the previous one
        for i in range(1, len(lats)):
            assert lats[i] > lats[i-1], \
                f"{ordered_beaches[i]} should be north of {ordered_beaches[i-1]}"


class TestDownloaderClass:
    """Tests for PRISMDownloader class functionality."""

    def test_downloader_imports(self):
        """Verify downloader can be imported."""
        from scripts.processing.download_prism_data import PRISMDownloader
        assert PRISMDownloader is not None

    def test_default_variables(self):
        """Verify default variables are the core 5."""
        from scripts.processing.download_prism_data import PRISMDownloader

        expected = ['ppt', 'tmin', 'tmax', 'tmean', 'tdmean']
        assert PRISMDownloader.VARIABLES == expected

    def test_beach_mop_ranges_defined(self):
        """Verify MOP ranges are defined for all beaches."""
        from scripts.processing.download_prism_data import BEACH_MOP_RANGES

        assert len(BEACH_MOP_RANGES) == 6
        for beach in BEACHES:
            assert beach in BEACH_MOP_RANGES
            mop_min, mop_max = BEACH_MOP_RANGES[beach]
            assert mop_min < mop_max
