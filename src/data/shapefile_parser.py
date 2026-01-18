"""Shapefile parser for extracting transect lines.

Parses Shapefiles containing shore-normal transect lines and converts them
to numpy arrays for transect extraction.
"""

from pathlib import Path
from typing import Dict, List, Union

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


class ShapefileParser:
    """Parse shapefiles containing transect LineStrings.

    Extracts shore-normal transect lines from shapefiles. Assumes the
    shapefile is already in the target CRS (e.g., UTM Zone 11N).

    Example:
        >>> parser = ShapefileParser()
        >>> transects = parser.parse("DelMarTransects.shp")
        >>> print(f"Extracted {len(transects['origins'])} transect lines")
    """

    def __init__(self, target_crs: str = "EPSG:26911"):
        """Initialize shapefile parser.

        Args:
            target_crs: Target coordinate reference system (default: UTM 11N)
        """
        self.target_crs = target_crs

        # Try to import geopandas
        try:
            import geopandas as gpd
            self.gpd = gpd
            self.has_geopandas = True
        except ImportError:
            logger.error(
                "geopandas not installed. Install with 'pip install geopandas' "
                "to use shapefile parsing."
            )
            self.has_geopandas = False

    def parse(self, shp_path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """Parse shapefile and extract transect lines.

        Args:
            shp_path: Path to shapefile

        Returns:
            Dictionary containing:
                - 'origins': (N, 3) array of transect start points (x, y, z)
                - 'endpoints': (N, 3) array of transect end points (x, y, z)
                - 'normals': (N, 2) array of shore-normal unit vectors
                - 'names': list of N transect names
                - 'lengths': (N,) array of transect lengths in meters

        Raises:
            FileNotFoundError: If shapefile doesn't exist
            ValueError: If shapefile contains no valid transects
            ImportError: If geopandas is not installed
        """
        if not self.has_geopandas:
            raise ImportError("geopandas is required for shapefile parsing")

        shp_path = Path(shp_path)
        if not shp_path.exists():
            raise FileNotFoundError(f"Shapefile not found: {shp_path}")

        logger.info(f"Parsing shapefile: {shp_path}")

        # Read shapefile
        gdf = self.gpd.read_file(shp_path)

        # Reproject to target CRS if needed
        if gdf.crs is not None and gdf.crs.to_string() != self.target_crs:
            logger.info(f"Reprojecting from {gdf.crs} to {self.target_crs}")
            gdf = gdf.to_crs(self.target_crs)
        elif gdf.crs is None:
            logger.warning(f"Shapefile has no CRS, assuming {self.target_crs}")

        origins = []
        endpoints = []
        names = []

        for idx, row in gdf.iterrows():
            geom = row.geometry

            # Skip if not a LineString
            if geom.geom_type != 'LineString':
                logger.warning(f"Skipping row {idx}: not a LineString ({geom.geom_type})")
                continue

            # Get coordinates
            coords = list(geom.coords)
            if len(coords) < 2:
                logger.warning(f"Skipping row {idx}: insufficient coordinates")
                continue

            # Extract start and end points
            start = coords[0]
            end = coords[-1]

            # Get name from attributes (try common field names)
            name = None
            for field in ['name', 'Name', 'NAME', 'id', 'ID', 'FID']:
                if field in row:
                    name = str(row[field])
                    break
            if name is None:
                name = f"Transect_{idx}"

            # Ensure 3D coordinates (add z=0 if missing)
            start_3d = list(start) + [0.0] * (3 - len(start))
            end_3d = list(end) + [0.0] * (3 - len(end))

            origins.append(start_3d)
            endpoints.append(end_3d)
            names.append(name)

        if len(origins) == 0:
            raise ValueError(f"No valid transects found in {shp_path}")

        origins = np.array(origins)
        endpoints = np.array(endpoints)

        # Compute shore-normal directions
        vectors = endpoints - origins
        lengths = np.linalg.norm(vectors[:, :2], axis=1)
        normals = vectors[:, :2] / (lengths[:, None] + 1e-8)

        logger.info(f"Extracted {len(origins)} transects from shapefile")
        logger.info(f"  Length range: [{lengths.min():.1f}, {lengths.max():.1f}]m")

        return {
            'origins': origins,
            'endpoints': endpoints,
            'normals': normals,
            'names': names,
            'lengths': lengths,
        }

    def save_transects(
        self,
        transects: Dict[str, np.ndarray],
        output_path: Union[str, Path],
    ) -> None:
        """Save parsed transects to numpy file.

        Args:
            transects: Dictionary from parse()
            output_path: Output .npz file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert names list to numpy array
        save_dict = {
            'origins': transects['origins'],
            'endpoints': transects['endpoints'],
            'normals': transects['normals'],
            'names': np.array(transects['names']),
            'lengths': transects['lengths'],
        }

        np.savez_compressed(output_path, **save_dict)
        logger.info(f"Saved {len(transects['origins'])} transects to {output_path}")
