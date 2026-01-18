"""KML parser for extracting transect lines.

Parses KML files containing shore-normal transect lines (e.g., MOPs survey lines)
and converts them to numpy arrays for transect extraction.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union
import xml.etree.ElementTree as ET

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


class KMLParser:
    """Parse KML files containing transect LineStrings.

    Extracts shore-normal transect lines from KML files and converts
    coordinates from lat/lon to local projected coordinates.

    Args:
        utm_zone: UTM zone for coordinate projection (e.g., 11 for San Diego)
        hemisphere: 'N' or 'S' for northern/southern hemisphere

    Example:
        >>> parser = KMLParser(utm_zone=11, hemisphere='N')
        >>> transects = parser.parse("MOPs-SD.kml")
        >>> print(f"Extracted {len(transects['origins'])} transect lines")
    """

    def __init__(self, utm_zone: int = 11, hemisphere: str = 'N'):
        self.utm_zone = utm_zone
        self.hemisphere = hemisphere

        # Try to import pyproj for coordinate transformation
        try:
            from pyproj import Transformer
            self.transformer = Transformer.from_crs(
                "EPSG:4326",  # WGS84 lat/lon
                f"EPSG:326{utm_zone}" if hemisphere == 'N' else f"EPSG:327{utm_zone}",  # UTM
                always_xy=True
            )
            self.has_pyproj = True
        except ImportError:
            logger.warning(
                "pyproj not installed. Install with 'pip install pyproj' for "
                "proper coordinate transformation. Using approximate conversion."
            )
            self.has_pyproj = False

    def parse(self, kml_path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """Parse KML file and extract transect lines.

        Args:
            kml_path: Path to KML file

        Returns:
            Dictionary containing:
                - 'origins': (N, 3) array of transect start points (x, y, z)
                - 'endpoints': (N, 3) array of transect end points (x, y, z)
                - 'normals': (N, 2) array of shore-normal unit vectors
                - 'names': list of N transect names
                - 'lengths': (N,) array of transect lengths in meters

        Raises:
            FileNotFoundError: If KML file doesn't exist
            ValueError: If KML contains no valid transects
        """
        kml_path = Path(kml_path)
        if not kml_path.exists():
            raise FileNotFoundError(f"KML file not found: {kml_path}")

        logger.info(f"Parsing KML file: {kml_path}")

        # Parse XML
        tree = ET.parse(kml_path)
        root = tree.getroot()

        # Handle KML namespace
        namespace = {'kml': 'http://www.opengis.net/kml/2.2'}

        # Extract all Placemarks with LineStrings
        placemarks = root.findall('.//kml:Placemark', namespace)

        origins = []
        endpoints = []
        names = []

        for placemark in placemarks:
            # Get name
            name_elem = placemark.find('kml:name', namespace)
            name = name_elem.text if name_elem is not None else "Unnamed"

            # Get LineString coordinates
            linestring = placemark.find('.//kml:LineString', namespace)
            if linestring is None:
                continue

            coords_elem = linestring.find('kml:coordinates', namespace)
            if coords_elem is None or coords_elem.text is None:
                continue

            # Parse coordinates (format: "lon,lat,elev lon,lat,elev")
            coords_text = coords_elem.text.strip()
            coord_pairs = coords_text.split()

            if len(coord_pairs) < 2:
                logger.warning(f"Skipping {name}: insufficient coordinates")
                continue

            # Parse start and end points
            start_coords = [float(x) for x in coord_pairs[0].split(',')]
            end_coords = [float(x) for x in coord_pairs[-1].split(',')]

            # Convert from lon, lat, elev to x, y, z
            if self.has_pyproj:
                start_x, start_y = self.transformer.transform(start_coords[0], start_coords[1])
                end_x, end_y = self.transformer.transform(end_coords[0], end_coords[1])
            else:
                # Approximate conversion (only for testing)
                start_x, start_y = self._approximate_transform(start_coords[0], start_coords[1])
                end_x, end_y = self._approximate_transform(end_coords[0], end_coords[1])

            start_z = start_coords[2] if len(start_coords) > 2 else 0.0
            end_z = end_coords[2] if len(end_coords) > 2 else 0.0

            origins.append([start_x, start_y, start_z])
            endpoints.append([end_x, end_y, end_z])
            names.append(name)

        if len(origins) == 0:
            raise ValueError(f"No valid transects found in {kml_path}")

        origins = np.array(origins)
        endpoints = np.array(endpoints)

        # Compute shore-normal directions
        vectors = endpoints - origins
        lengths = np.linalg.norm(vectors[:, :2], axis=1)
        normals = vectors[:, :2] / (lengths[:, None] + 1e-8)

        logger.info(f"Extracted {len(origins)} transects from KML")

        return {
            'origins': origins,
            'endpoints': endpoints,
            'normals': normals,
            'names': names,
            'lengths': lengths,
        }

    def _approximate_transform(self, lon: float, lat: float) -> Tuple[float, float]:
        """Approximate lat/lon to UTM conversion (for testing only).

        Uses simple equirectangular projection. Not accurate for large areas.
        """
        # Constants for approximate conversion
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        # Reference point (approximate center of San Diego coast)
        ref_lon = -117.25
        ref_lat = 32.95
        ref_lon_rad = np.radians(ref_lon)
        ref_lat_rad = np.radians(ref_lat)

        # Meters per degree
        meters_per_deg_lat = 111132.92 - 559.82 * np.cos(2 * lat_rad) + 1.175 * np.cos(4 * lat_rad)
        meters_per_deg_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad)

        # Convert to local coordinates
        x = (lon - ref_lon) * meters_per_deg_lon
        y = (lat - ref_lat) * meters_per_deg_lat

        return x, y

    @staticmethod
    def extend_origins_inland(
        transects: Dict[str, np.ndarray],
        extension_m: float = 100.0,
        pattern: str = "MOP"
    ) -> Dict[str, np.ndarray]:
        """Extend transect origins inland (opposite to normal direction).

        This is useful when KML transect lines start on the beach and point
        seaward, but we want to sample inland cliff features. The extension
        moves the origin point inland along the reverse of the normal vector.

        Args:
            transects: Dictionary from parse()
            extension_m: Distance to extend inland (meters)
            pattern: Only extend transects with names containing this pattern.
                     Set to None to extend all transects.

        Returns:
            Modified transects dictionary with extended origins
        """
        transects = transects.copy()
        origins = transects['origins'].copy()
        normals = transects['normals']
        names = transects['names']

        n_extended = 0
        for i, name in enumerate(names):
            # Check if this transect matches the pattern
            if pattern is None or pattern.upper() in str(name).upper():
                # Move origin inland (opposite to normal direction)
                # Normal points from origin to endpoint (beach to ocean)
                # So -normal points inland (ocean to beach to cliff)
                origins[i, 0] -= extension_m * normals[i, 0]
                origins[i, 1] -= extension_m * normals[i, 1]
                n_extended += 1

        if n_extended > 0:
            logger.info(f"Extended {n_extended} transects inland by {extension_m}m")

        transects['origins'] = origins
        return transects

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
