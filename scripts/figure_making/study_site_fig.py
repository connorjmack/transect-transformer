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
        default=Path("results/figures/study_site_map.png"),
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


def add_north_arrow(ax: plt.Axes, xy=(0.92, 0.12)) -> None:
    """Draw a publication-quality North arrow pointing Left (North)."""
    # Create a simple compass arrow using polygon coordinates
    # Arrow points Left (since map is rotated 90 deg CCW)
    # Origin at xy (center of the arrow base)
    
    # Scale of the arrow
    size = 0.04
    aspect = 0.6  # Width relative to length
    
    # Coordinates for a stylized arrow pointing LEFT
    # Tip at (-size, 0), Base at (0, 0), Wings at (0, +/- width)
    # Actually, let's do a classic "N" arrow.
    # Because North is Left, the arrow tip is at x - size
    
    x, y = xy
    length = size
    width = size * aspect
    
    # Vertices relative to (x, y) in axes coordinates? 
    # It's easier to plot in axes fraction.
    
    # Main arrow shaft (pointing left)
    # We'll draw two halves: top half (black) and bottom half (white/outlined) for 3D effect
    
    # Tip (Left), Tail (Right)
    # Tip: (x - length, y)
    # Tail Center: (x + length * 0.2, y)
    # Top Wing: (x, y + width)
    # Bottom Wing: (x, y - width)
    
    tip = (x - length, y)
    tail_center = (x + length * 0.2, y)
    top_wing = (x + length * 0.1, y + width)
    bottom_wing = (x + length * 0.1, y - width)
    
    # Upper half (Black)
    ax.add_patch(
        plt.Polygon(
            [tip, tail_center, top_wing],
            transform=ax.transAxes,
            facecolor="black",
            edgecolor="black",
            linewidth=1,
            zorder=20
        )
    )
    
    # Lower half (White with black edge)
    ax.add_patch(
        plt.Polygon(
            [tip, tail_center, bottom_wing],
            transform=ax.transAxes,
            facecolor="white",
            edgecolor="black",
            linewidth=1,
            zorder=20
        )
    )
    
    # N Label (to the left of the tip)
    ax.text(
        x - length - 0.02,
        y,
        "N",
        ha="right",
        va="center",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        color="black",
        zorder=20
    ).set_path_effects([pe.withStroke(linewidth=3, foreground="white", alpha=0.8)])


def nice_scale_length(map_width_m: float) -> float:
    """Pick a rounded scale length (in meters) based on map width."""
    target = map_width_m / 6  # Target ~1/6th of map width
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
    
    # Determine scale bar length
    bar_length_m = nice_scale_length(map_width)
    
    # Height of the bar
    bar_height = (y_max - y_min) * 0.015
    
    # Location is in axes fraction, convert to data coordinates
    # But for precise sizing in meters, we construct it in data coords
    # Anchor: Center of the scale bar block
    anchor_x = x_min + (x_max - x_min) * location[0]
    anchor_y = y_min + (y_max - y_min) * location[1]
    
    # We'll center the bar horizontally around anchor_x? 
    # No, let's align the *right* side to anchor_x to keep it in the corner.
    right_x = anchor_x
    start_x = right_x - bar_length_m
    
    # 2 segments (or 4 for finer grain)
    # Let's do 2 major segments: 0 to 50%, 50% to 100%
    mid_x = start_x + bar_length_m / 2
    
    # Segment 1 (Left, Black)
    rect1 = Rectangle(
        (start_x, anchor_y),
        bar_length_m / 2,
        bar_height,
        facecolor="black",
        edgecolor="black",
        linewidth=1,
        zorder=20
    )
    # Segment 2 (Right, White)
    rect2 = Rectangle(
        (mid_x, anchor_y),
        bar_length_m / 2,
        bar_height,
        facecolor="white",
        edgecolor="black",
        linewidth=1,
        zorder=20
    )
    
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    
    # Border around the whole thing
    rect_border = Rectangle(
        (start_x, anchor_y),
        bar_length_m,
        bar_height,
        facecolor="none",
        edgecolor="black",
        linewidth=1,
        zorder=21
    )
    ax.add_patch(rect_border)
    
    # Labels
    # 0, Mid, Max
    label_y = anchor_y - bar_height * 0.5 # Below bar
    
    for x_pos, txt in [(start_x, "0"), (mid_x, f"{bar_length_m/2:.0f}"), (right_x, f"{bar_length_m:.0f}")]:
        t = ax.text(
            x_pos,
            label_y,
            txt,
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            zorder=22
        )
        t.set_path_effects([pe.withStroke(linewidth=2, foreground="white", alpha=0.8)])
        
    # Units
    unit_txt = "m" if bar_length_m < 1000 else "km"
    # If km, format labels? No, keeping it simple meters for now or simple logic
    if bar_length_m >= 1000:
        # Re-label for km
        km_val = bar_length_m / 1000
        # Update text would be complex here, assuming meters for typical beach transects (~10km)
        # If the map is huge, we might need logic.
        pass

    ax.text(
        right_x + bar_length_m * 0.05,
        anchor_y + bar_height/2,
        "m",
        ha="left",
        va="center",
        fontsize=10,
        fontweight="bold",
        zorder=22
    ).set_path_effects([pe.withStroke(linewidth=2, foreground="white", alpha=0.8)])


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
    bounds: Tuple[float, float, float, float],
    rotation_center: Tuple[float, float],
    rotation_angle: float,
    zoom: int | None,
) -> bool:
    """Fetch and place a satellite basemap rotated to match the plot."""
    try:
        import contextily as ctx
    except ImportError:
        raise ImportError(
            "contextily is required for the basemap. Install with `pip install contextily` "
            "or rerun with --no-basemap."
        )

    minx, maxx, miny, maxy = bounds
    # contextily expects bounds in Web Mercator if ll=False.
    if zoom is None:
        try:
            # Use custom zoom calculation with high pixel target (16000) for sharp basemaps
            zoom_guess = calculate_zoom_level(minx, miny, maxx, maxy, max_pixels=16000)
            print(f"Auto-selected basemap zoom level: {zoom_guess} (based on max_pixels=16000)")
        except Exception as e:
            print(f"Could not auto-calculate zoom: {e}")
            zoom_guess = None
    else:
        zoom_guess = zoom

    bounds_kwargs = dict(source=ctx.providers.Esri.WorldImagery, ll=False)
    if zoom_guess is not None:
        bounds_kwargs["zoom"] = int(zoom_guess)

    img, ext = ctx.bounds2img(minx, miny, maxx, maxy, **bounds_kwargs)
    actual_bounds = (ext[0], ext[1], ext[2], ext[3])

    if rotation_angle % 90 != 0:
        warnings.warn("Basemap rotation supports 90° increments; skipping rotation.", stacklevel=2)
        k = 0
    else:
        k = int(rotation_angle / 90) % 4

    if k:
        img = np.rot90(img, k=k)
        extent = rotate_bounds(actual_bounds, rotation_angle, origin=rotation_center)
    else:
        extent = actual_bounds

    ax.imshow(
        img, extent=(extent[0], extent[1], extent[2], extent[3]), origin="upper", zorder=0, alpha=0.9
    )
    return True


ROTATION_DEG = 90  # counter-clockwise; north -> left, south -> right


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

    # Web Mercator keeps satellite tiles simple; when basemap is disabled fall back to a local projected CRS.
    target_crs = CRS.from_epsg(3857) if use_basemap else choose_projected_crs(gdf)
    gdf_proj = gdf.to_crs(target_crs)
    
    # Extract MOP ID from label (e.g., "MOP 520" -> 520) for correct region filtering
    if "label" in gdf_proj.columns:
        try:
            gdf_proj["mop_id"] = gdf_proj["label"].apply(lambda x: int(str(x).split(" ")[1]))
        except (ValueError, IndexError):
            warnings.warn("Could not parse 'mop_id' from 'label' column. Region annotation may fail.", stacklevel=2)

    minx, maxx, miny, maxy = padded_bounds(gdf_proj, buffer_fraction)
    rotation_center = ((minx + maxx) / 2, (miny + maxy) / 2)

    gdf_rot = rotate_geometries(gdf_proj, ROTATION_DEG, origin=rotation_center)
    rotated_bounds = rotate_bounds((minx, maxx, miny, maxy), ROTATION_DEG, origin=rotation_center)

    fig, ax = plt.subplots(figsize=(12, 7))

    if use_basemap:
        add_satellite_basemap(
            ax, (minx, maxx, miny, maxy), rotation_center, ROTATION_DEG, zoom=basemap_zoom
        )

    geom_types = set(gdf_rot.geom_type.str.lower())
    if any("polygon" in g for g in geom_types):
        gdf_rot.plot(ax=ax, facecolor="#6aa2c9", edgecolor="#1f3c5b", linewidth=0.9, alpha=0.45)
    elif any("line" in g for g in geom_types):
        gdf_rot.plot(ax=ax, color="#1f3c5b", linewidth=1.6)
    else:
        gdf_rot.plot(ax=ax, color="#1f3c5b", markersize=30, alpha=0.9)
    
    # Annotate specific regions if mop_id is present
    if "mop_id" in gdf_rot.columns:
        annotate_regions(ax, gdf_rot)

    ax.set_xlim(rotated_bounds[0], rotated_bounds[1])
    ax.set_ylim(rotated_bounds[2], rotated_bounds[3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    if title:
        ax.set_title(title, fontsize=12, pad=10)

    add_north_arrow(ax)
    add_scale_bar(ax)
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
