
import geopandas as gpd
import numpy as np
from sklearn.decomposition import PCA

def calculate_coastline_orientation(shapefile_path):
    gdf = gpd.read_file(shapefile_path)
    # Project to UTM/meters
    gdf = gdf.to_crs(gdf.estimate_utm_crs())
    
    # Get all coordinates
    coords = []
    for geom in gdf.geometry:
        if geom.type == 'LineString':
            coords.extend(list(geom.coords))
        elif geom.type == 'MultiLineString':
            for part in geom.geoms:
                coords.extend(list(part.coords))
                
    coords = np.array(coords)
    
    # PCA to find primary axis
    pca = PCA(n_components=2)
    pca.fit(coords)
    
    # First component is the direction of maximum variance (the coastline)
    v1 = pca.components_[0]
    angle_rad = np.arctan2(v1[1], v1[0])
    angle_deg = np.degrees(angle_rad)
    
    print(f"Primary axis vector: {v1}")
    print(f"Angle of coastline (deg): {angle_deg}")
    
    # We want this angle to be 0 (horizontal).
    # So we rotate by -angle_deg.
    rotation_needed = -angle_deg
    print(f"Rotation needed to make horizontal: {rotation_needed}")
    
    return rotation_needed

if __name__ == "__main__":
    calculate_coastline_orientation("data/mops/transects_10m/transect_lines.shp")
