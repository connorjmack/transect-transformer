#!/usr/bin/env python
"""Create a publication-quality study site map from a shapefile.

Outputs a landscape map with north on the left (rotated 90° CCW), adds a satellite
basemap, legend, north arrow, and scale bar, and writes to results/figures by default.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.patches import Rectangle
from pyproj import CRS
from shapely import affinity
from shapely.geometry import box
from scipy.ndimage import rotate as nd_rotate
from sklearn.decomposition import PCA

# Define study site regions by MOP ID ranges (inclusive)
MOP_RANGES = {
    "DelMar": [595, 620],
    "Solana": [637, 666],
    "Encinitas": [708, 764],
    "SanElijo": [683, 708],
    "Torrey": [567, 581],
    "Blacks": [520, 567],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a publication-quality map for a study site shapefile. "
            "Adds satellite imagery, a legend, north arrow, and scale bar."
        )
    )
    parser.add_argument(
        "shapefile",
        type=Path,
        help="Path to the shapefile (.shp) describing the study site.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/site_map.png"),
        help="Where to write the figure (png, pdf, etc.).",
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=0.05,
        help="Fractional padding around the geometry bounds for plotting.",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Resolution for the saved figure."
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title placed above the map.",
    )
    parser.add_argument(
        "--no-basemap",
        action="store_true",
        help="Skip adding satellite imagery (requires contextily if enabled).",
    )
    parser.add_argument(
        "--basemap-zoom",
        type=int,
        default=None,
        help="Optional contextily zoom level (higher = more detail). Auto-calculated if omitted.",
    )
    return parser.parse_args()


def utm_crs_for_geometries(gdf: gpd.GeoDataFrame) -> CRS:
    """Choose a suitable projected CRS (UTM) based on the data centroid."""
    centroid_lonlat = gdf.to_crs(4326).geometry.union_all().centroid
    lon, lat = centroid_lonlat.x, centroid_lonlat.y
    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


def choose_projected_crs(gdf: gpd.GeoDataFrame) -> CRS:
    """Return a projected CRS; convert geographic data to UTM."""
    if gdf.crs is None:
        raise ValueError("Input shapefile lacks a defined CRS.")
    crs = CRS(gdf.crs)
    if crs.is_geographic:
        return utm_crs_for_geometries(gdf)
    return crs


def padded_bounds(
    gdf: gpd.GeoDataFrame, buffer_fraction: float
) -> Tuple[float, float, float, float]:
    minx, miny, maxx, maxy = gdf.total_bounds
    width, height = maxx - minx, maxy - miny
    xpad = width * buffer_fraction if width > 0 else 1.0
    ypad = height * buffer_fraction if height > 0 else 1.0
    return minx - xpad, maxx + xpad, miny - ypad, maxy + ypad


def rotate_bounds(
    bounds: Tuple[float, float, float, float], angle_deg: float, origin: Tuple[float, float]
) -> Tuple[float, float, float, float]:
    """Rotate rectangular bounds around an origin and return new bounds."""
    minx, maxx, miny, maxy = bounds
    rotated = affinity.rotate(box(minx, miny, maxx, maxy), angle_deg, origin=origin)
    r_minx, r_miny, r_maxx, r_maxy = rotated.bounds
    return r_minx, r_maxx, r_miny, r_maxy


def rotate_geometries(gdf: gpd.GeoDataFrame, angle_deg: float, origin: Tuple[float, float]) -> gpd.GeoDataFrame:
    """Rotate all geometries around an origin by a fixed angle (degrees)."""
    rotated = gdf.copy()
    rotated["geometry"] = rotated.geometry.apply(lambda geom: affinity.rotate(geom, angle_deg, origin=origin))
    return rotated


def calculate_optimal_rotation(gdf: gpd.GeoDataFrame) -> float:
    """Calculate rotation to make coastline horizontal with land on top."""
    coords = []
    vectors = []
    for geom in gdf.geometry:
        if geom.geom_type == 'LineString':
            pts = list(geom.coords)
            coords.extend(pts)
            if len(pts) >= 2:
                # Vector from start to end (assuming land to sea)
                vectors.append(np.array(pts[-1]) - np.array(pts[0]))
        elif geom.geom_type == 'MultiLineString':
            for part in geom.geoms:
                pts = list(part.coords)
                coords.extend(pts)
                if len(pts) >= 2:
                    vectors.append(np.array(pts[-1]) - np.array(pts[0]))
                
    if not coords:
        return 0.0
        
    coords = np.array(coords)
    pca = PCA(n_components=2).fit(coords)
    angle_deg = np.degrees(np.arctan2(pca.components_[0][1], pca.components_[0][0]))
    
    rotation = -angle_deg
    
    # Ensure water is at the bottom. 
    # Rotate the average transect vector and check if its Y component is negative.
    if vectors:
        avg_v = np.mean(vectors, axis=0)
        rad = np.radians(rotation)
        # Rotation matrix (standard CCW)
        rot_m = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        rotated_v = rot_m @ avg_v
        if rotated_v[1] > 0:
            rotation += 180
            
    return rotation


def add_north_arrow(ax: plt.Axes, rotation_angle: float, xy=(0.08, 0.92)) -> None:
    """Draw a publication-quality North arrow pointing to True North."""
    arrow_angle_deg = 90 + rotation_angle
    arrow_angle_rad = np.radians(arrow_angle_deg)
    
    cx, cy = xy
    size = 0.04
    width = size * 0.6
    
    cos_a = np.cos(arrow_angle_rad)
    sin_a = np.sin(arrow_angle_rad)
    
    def transform(u, v):
        aspect_ratio = 12 / 7
        return (cx + rx / aspect_ratio, cy + ry) if 'rx' in locals() else (cx + (u * cos_a - v * sin_a) / aspect_ratio, cy + (u * sin_a + v * cos_a))

    tip = transform(size, 0)
    tail_center = transform(-size * 0.2, 0)
    wing_left = transform(-size * 0.1, width)
    wing_right = transform(-size * 0.1, -width)
    
    ax.add_patch(plt.Polygon([tip, tail_center, wing_left], transform=ax.transAxes, facecolor="black", edgecolor="black", linewidth=1, zorder=20))
    ax.add_patch(plt.Polygon([tip, tail_center, wing_right], transform=ax.transAxes, facecolor="white", edgecolor="black", linewidth=1, zorder=20))
    
    label_pos = transform(size + 0.03, 0)
    ax.text(label_pos[0], label_pos[1], "N", ha="center", va="center", transform=ax.transAxes, fontsize=14, fontweight="bold", color="black", zorder=20).set_path_effects([pe.withStroke(linewidth=3, foreground="white", alpha=0.8)])


def nice_scale_length(map_width_m: float) -> float:
    """Pick a rounded scale length (in meters) based on map width."""
    target = map_width_m / 6
    candidates = [1, 2, 5]
    exponent = 0
    while target >= 10:
        target /= 10
        exponent += 1
    while target < 1:
        target *= 10
        exponent -= 1
    scaled_candidates = [c * (10**exponent) for c in candidates]
    return min(scaled_candidates, key=lambda c: abs(c - map_width_m / 6))


def add_scale_bar(ax: plt.Axes, location=(0.92, 0.05)) -> None:
    """Add a publication-quality alternating scale bar in the bottom right."""
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    map_width = x_max - x_min
    bar_length_m = nice_scale_length(map_width)
    bar_height = (y_max - y_min) * 0.015
    anchor_x = x_min + (x_max - x_min) * location[0]
    anchor_y = y_min + (y_max - y_min) * location[1]
    right_x = anchor_x
    start_x = right_x - bar_length_m
    mid_x = start_x + bar_length_m / 2
    
    ax.add_patch(Rectangle((start_x, anchor_y), bar_length_m / 2, bar_height, facecolor="black", edgecolor="black", linewidth=1, zorder=20))
    ax.add_patch(Rectangle((mid_x, anchor_y), bar_length_m / 2, bar_height, facecolor="white", edgecolor="black", linewidth=1, zorder=20))
    ax.add_patch(Rectangle((start_x, anchor_y), bar_length_m, bar_height, facecolor="none", edgecolor="black", linewidth=1, zorder=21))
    
    for x_pos, txt in [(start_x, "0"), (mid_x, f"{bar_length_m/2:.0f}"), (right_x, f"{bar_length_m:.0f}")]:
        ax.text(x_pos, anchor_y - bar_height * 0.5, txt, ha="center", va="top", fontsize=9, fontweight="bold", zorder=22).set_path_effects([pe.withStroke(linewidth=2, foreground="white", alpha=0.8)])
        
    ax.text(right_x + bar_length_m * 0.05, anchor_y + bar_height/2, "m", ha="left", va="center", fontsize=10, fontweight="bold", zorder=22).set_path_effects([pe.withStroke(linewidth=2, foreground="white", alpha=0.8)])


def calculate_zoom_level(minx: float, miny: float, maxx: float, maxy: float, max_pixels: int = 16000) -> int:
    """Calculate zoom level to fit bounds within max_pixels for Web Mercator (EPSG:3857)."""
    width = maxx - minx
    height = maxy - miny
    longest_side = max(width, height)

    if longest_side <= 0:
        return 1

    # Resolution at zoom 0 for EPSG:3857 (meters/pixel)
    initial_resolution = 2 * math.pi * 6378137 / 256
    
    # 2^z = (initial_resolution * max_pixels) / longest_side
    zoom_float = math.log2((initial_resolution * max_pixels) / longest_side)
    
    # Clamp to reasonable Web Mercator limits (e.g., 0-19)
    zoom = int(round(zoom_float))
    return max(0, min(zoom, 19))

def add_satellite_basemap(
    ax: plt.Axes,
    rotated_view_bounds: Tuple[float, float, float, float],
    rotation_center: Tuple[float, float],
    rotation_angle: float,
    zoom: int | None,
) -> bool:
    """Fetch and place a satellite basemap that fills the rotated frame."""
    try:
        import contextily as ctx
    except ImportError:
        raise ImportError("contextily is required for the basemap.")

    # rotated_view_bounds: (minx, maxx, miny, maxy) in the rotated coordinate system.
    # To find the coverage needed in EPSG:3857, we rotate these corners back.
    rminx, rmaxx, rminy, rmaxy = rotated_view_bounds
    corners = [
        (rminx, rminy), (rmaxx, rminy), (rmaxx, rmaxy), (rminx, rmaxy)
    ]
    
    # Back-rotate corners around rotation_center by -rotation_angle
    rad = np.radians(-rotation_angle)
    rot_m = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    cx, cy = rotation_center
    
    back_corners = []
    for px, py in corners:
        # Move to origin, rotate, move back
        nx, ny = rot_m @ np.array([px - cx, py - cy])
        back_corners.append((nx + cx, ny + cy))
        
    back_corners = np.array(back_corners)
    minx, miny = back_corners.min(axis=0)
    maxx, maxy = back_corners.max(axis=0)
    
    # Add a small fetch buffer to avoid edge artifacts
    pad = max(maxx - minx, maxy - miny) * 0.1
    minx, maxx, miny, maxy = minx - pad, maxx + pad, miny - pad, maxy + pad

    if zoom is None:
        try:
            zoom_guess = calculate_zoom_level(minx, miny, maxx, maxy, max_pixels=16000)
        except Exception:
            zoom_guess = None
    else:
        zoom_guess = zoom

    img, ext = ctx.bounds2img(minx, miny, maxx, maxy, source=ctx.providers.Esri.WorldImagery, ll=False)
    actual_bounds = (ext[0], ext[1], ext[2], ext[3])

    if rotation_angle != 0:
        img_rot = nd_rotate(img, rotation_angle, reshape=True, order=1, mode='constant', cval=255)
        extent = rotate_bounds(actual_bounds, rotation_angle, origin=rotation_center)
    else:
        img_rot = img
        extent = actual_bounds

    ax.imshow(img_rot, extent=extent, origin="upper", zorder=0, alpha=0.9)
    return True


def plot_study_site(
    shapefile: Path,
    output: Path,
    buffer_fraction: float,
    dpi: int,
    title: str | None,
    use_basemap: bool,
    basemap_zoom: int | None,
) -> None:
    gdf = gpd.read_file(shapefile)
    if gdf.empty:
        raise ValueError(f"No features found in {shapefile}")

    target_crs = CRS.from_epsg(3857) if use_basemap else choose_projected_crs(gdf)
    gdf_proj = gdf.to_crs(target_crs)
    
    if "label" in gdf_proj.columns:
        try:
            gdf_proj["mop_id"] = gdf_proj["label"].apply(lambda x: int(str(x).split(" ")[1]))
        except Exception:
            pass

    # Calculate optimal rotation
    rotation_angle = calculate_optimal_rotation(gdf_proj)
    
    # Calculate center from unrotated data
    b = gdf_proj.total_bounds
    rotation_center = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

    # Rotate geometries
    gdf_rot = rotate_geometries(gdf_proj, rotation_angle, origin=rotation_center)
    
    # Calculate view limits from rotated data
    rminx, rminy, rmaxx, rmaxy = gdf_rot.total_bounds
    rw, rh = rmaxx - rminx, rmaxy - rminy
    view_bounds = (
        rminx - rw * buffer_fraction,
        rmaxx + rw * buffer_fraction,
        rminy - rh * buffer_fraction,
        rmaxy + rh * buffer_fraction
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    if use_basemap:
        add_satellite_basemap(ax, view_bounds, rotation_center, rotation_angle, zoom=basemap_zoom)

    geom_types = set(gdf_rot.geom_type.str.lower())
    if any("line" in g for g in geom_types):
        gdf_rot.plot(ax=ax, color="#1f3c5b", linewidth=1.6)
    else:
        gdf_rot.plot(ax=ax, color="#1f3c5b", markersize=30, alpha=0.9)
    
    if "mop_id" in gdf_rot.columns:
        annotate_regions(ax, gdf_rot)

    ax.set_xlim(view_bounds[0], view_bounds[1])
    ax.set_ylim(view_bounds[2], view_bounds[3])
    ax.set_xticks([])
    ax.set_yticks([])
    
    if title:
        ax.set_title(title, fontsize=12, pad=10)

    add_north_arrow(ax, rotation_angle, xy=(0.08, 0.92))
    add_scale_bar(ax, location=(0.92, 0.05))
    ax.set_aspect("equal")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def annotate_regions(ax: plt.Axes, gdf: gpd.GeoDataFrame) -> None:
    """Annotate regions and delineate boundaries based on MOP_RANGES."""
    for name, (start_id, end_id) in MOP_RANGES.items():
        # Filter for the region using mop_id
        region = gdf[(gdf["mop_id"] >= start_id) & (gdf["mop_id"] <= end_id)]
        if region.empty:
            continue

        # Delineate boundaries (start and end transects)
        # Note: We look for the exact start/end ID in the data. If missing, we pick the closest available extremum.
        boundaries = region[region["mop_id"].isin([start_id, end_id])]
        if not boundaries.empty:
            boundaries.plot(ax=ax, color="black", linewidth=2.5, zorder=5)

        # Place label at the centroid of the MIDDLE transect to avoid overlapping/skewed centroids
        target_mid = (start_id + end_id) / 2
        
        # Find the transect closest to the middle ID
        # Calculate distance to target_mid
        region = region.copy() # Avoid SettingWithCopyWarning
        region['dist_to_mid'] = (region['mop_id'] - target_mid).abs()
        
        # Select the transect with min distance
        mid_transect_row = region.loc[region['dist_to_mid'].idxmin()]
        mid_geom = mid_transect_row.geometry
        
        centroid = mid_geom.centroid
        
        # Add text with halo for visibility
        txt = ax.text(
            centroid.x,
            centroid.y,
            name,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="black",
            zorder=10
        )
        txt.set_path_effects([pe.withStroke(linewidth=3, foreground="white", alpha=0.8)])


def main() -> None:
    args = parse_args()
    plot_study_site(
        shapefile=args.shapefile,
        output=args.output,
        buffer_fraction=args.buffer,
        dpi=args.dpi,
        title=args.title,
        use_basemap=not args.no_basemap,
        basemap_zoom=args.basemap_zoom,
    )
    print(f"Saved map to {args.output}")


if __name__ == "__main__":
    main()
