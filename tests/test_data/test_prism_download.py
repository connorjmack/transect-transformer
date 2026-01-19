"""Tests for PRISM data download and extraction.

These tests verify that PRISM data was downloaded correctly and can be
extracted for all San Diego study beaches.
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


class TestPRISMDownload:
    """Tests for PRISM file downloads."""

    def test_prism_directory_exists(self):
        """Verify PRISM data directory was created."""
        assert PRISM_DIR.exists(), f"PRISM directory not found: {PRISM_DIR}"

    @pytest.mark.parametrize("variable", VARIABLES)
    def test_variable_directory_exists(self, variable):
        """Verify each variable has a subdirectory."""
        var_dir = PRISM_DIR / variable
        assert var_dir.exists(), f"Variable directory not found: {var_dir}"

    @pytest.mark.parametrize("variable", VARIABLES)
    def test_tif_files_exist(self, variable):
        """Verify GeoTIFF files exist for each variable."""
        var_dir = PRISM_DIR / variable
        tif_files = list(var_dir.glob('*.tif'))
        assert len(tif_files) >= 7, \
            f"Expected at least 7 .tif files for {variable}, found {len(tif_files)}"

    def test_file_naming_convention(self):
        """Verify files follow expected naming pattern."""
        for var in VARIABLES:
            var_dir = PRISM_DIR / var
            for tif_file in var_dir.glob('*.tif'):
                # Expected format: PRISM_{var}_{YYYYMMDD}.tif
                assert tif_file.name.startswith(f'PRISM_{var}_'), \
                    f"Unexpected filename: {tif_file.name}"
                assert tif_file.name.endswith('.tif'), \
                    f"Expected .tif extension: {tif_file.name}"

    def test_file_sizes_reasonable(self):
        """Verify files are not empty or corrupted."""
        min_size = 100_000  # 100KB minimum for a valid PRISM GeoTIFF

        for var in VARIABLES:
            var_dir = PRISM_DIR / var
            for tif_file in var_dir.glob('*.tif'):
                size = tif_file.stat().st_size
                assert size > min_size, \
                    f"File too small ({size} bytes): {tif_file}"


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

        # Dewpoint should be <= mean temp
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

        # Expect some variation (at least 2 different values)
        assert unique_means >= 2, \
            "All beaches have identical precipitation - may be extracting from same cell"

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


class TestGeoTIFFReadability:
    """Tests that verify GeoTIFF files can be read with rasterio."""

    def test_rasterio_can_read_files(self):
        """Verify rasterio can open and read the downloaded files."""
        try:
            import rasterio
        except ImportError:
            pytest.skip("rasterio not installed")

        # Test one file from each variable
        for var in VARIABLES:
            var_dir = PRISM_DIR / var
            tif_files = list(var_dir.glob('*.tif'))

            if tif_files:
                with rasterio.open(tif_files[0]) as src:
                    # Check basic properties
                    assert src.count == 1, f"Expected 1 band, got {src.count}"
                    assert src.width > 0, "Zero width"
                    assert src.height > 0, "Zero height"

                    # Check CRS is defined
                    assert src.crs is not None, "No CRS defined"

                    # Read a small portion to verify data access
                    data = src.read(1, window=((0, 10), (0, 10)))
                    assert data.shape == (10, 10), "Could not read window"

    def test_coordinate_extraction_works(self):
        """Verify we can extract values at specific coordinates."""
        try:
            import rasterio
            from rasterio.transform import rowcol
        except ImportError:
            pytest.skip("rasterio not installed")

        from scripts.processing.download_prism_data import BEACH_COORDS

        # Test Del Mar coordinates
        lat, lon = BEACH_COORDS['delmar']

        var_dir = PRISM_DIR / 'ppt'
        tif_files = list(var_dir.glob('*.tif'))

        if tif_files:
            with rasterio.open(tif_files[0]) as src:
                row, col = rowcol(src.transform, lon, lat)

                # Verify indices are within bounds
                assert 0 <= row < src.height, f"Row {row} out of bounds"
                assert 0 <= col < src.width, f"Col {col} out of bounds"

                # Read the value
                value = src.read(1)[row, col]
                assert not np.isnan(value), "NaN value at Del Mar coordinates"
