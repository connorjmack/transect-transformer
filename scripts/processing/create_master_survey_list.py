#!/usr/bin/env python3
"""Combine all survey lists and remove duplicates to create master_list.csv"""

import pandas as pd
from pathlib import Path


def main():
    survey_lists_dir = Path("data/raw")

    # Find all CSV files except master_list.csv
    csv_files = [f for f in survey_lists_dir.glob("*.csv") if f.name != "master_list.csv"]

    print(f"Found {len(csv_files)} survey list files:")
    for f in csv_files:
        print(f"  - {f.name}")

    # Read and combine all CSV files
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        print(f"\n{csv_file.name}: {len(df)} rows")
        dfs.append(df)

    # Concatenate all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined total: {len(combined_df)} rows")

    # Remove duplicates based on 'path' column
    master_df = combined_df.drop_duplicates(subset=['path'], keep='first')
    print(f"After removing duplicates: {len(master_df)} unique paths")
    print(f"Removed {len(combined_df) - len(master_df)} duplicate entries")

    # Sort by date for consistency
    master_df = master_df.sort_values('date').reset_index(drop=True)

    # Save to master_list.csv
    output_path = survey_lists_dir / "master_list.csv"
    master_df.to_csv(output_path, index=False)
    print(f"\nSaved master list to: {output_path}")
    print(f"Total unique surveys: {len(master_df)}")


if __name__ == "__main__":
    main()
