
import geopandas as gpd
import pandas as pd
from pathlib import Path

MOP_RANGES = {
    "DelMar": [595, 620],
    "Solana": [637, 666],
    "Encinitas": [708, 764],
    "SanElijo": [683, 708],
    "Torrey": [567, 581],
    "Blacks": [520, 567],
}

def analyze_ranges(shapefile_path):
    gdf = gpd.read_file(shapefile_path)
    print(f"Total rows: {len(gdf)}")
    print(f"tr_id range: {gdf['tr_id'].min()} to {gdf['tr_id'].max()}")
    
    print("\nRegion Analysis:")
    for name, (start_id, end_id) in MOP_RANGES.items():
        region = gdf[(gdf["tr_id"] >= start_id) & (gdf["tr_id"] <= end_id)]
        count = len(region)
        print(f"Region: {name} ({start_id}-{end_id})")
        print(f"  Count: {count}")
        if count > 0:
            min_found = region['tr_id'].min()
            max_found = region['tr_id'].max()
            print(f"  Found IDs: {min_found} to {max_found}")
            
    
            
    # Check for duplicates
            
    if gdf['tr_id'].nunique() != len(gdf):
            
        print(f"\nWARNING: tr_id is not unique! Unique IDs: {gdf['tr_id'].nunique()}, Total rows: {len(gdf)}")
            
        
            
        # Check a sample duplicate
            
        counts = gdf['tr_id'].value_counts()
            
        dup_id = counts.index[0]
            
        print(f"Sample duplicate ID: {dup_id} (count: {counts[dup_id]})")
            
        
            
                dups = gdf[gdf['tr_id'] == dup_id]
            
        
            
                print("Geometries and Labels for sample duplicate:")
            
        
            
                for idx, row in dups.iterrows():
            
        
            
                    print(f"  {idx}: Label='{row['label']}', Geom={row.geometry}")
            
        
            
                    
            
        
            
            print(f"Geometry types: {gdf.geom_type.unique()}")
            

            
if __name__ == "__main__":    analyze_ranges("data/mops/transects_10m/transect_lines.shp")

