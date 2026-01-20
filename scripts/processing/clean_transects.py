#!/usr/bin/env python3
"""Clean and process raw transects for model training.

This script transforms raw extracted transects into cliff-centered training data:
1. Crops to a window around the detected cliff (toe - 10m to top + 5m)
2. Resamples to 128 points
3. Drops uninformative features (RGB, return_number, num_returns)
4. Recomputes slope and curvature on the resampled data

Requires cliff delineation sidecar files (*.cliff.npz) to exist alongside
the raw transect files. Run detect_cliffs.py first if needed.

Feature Changes:
    Input (12 features):
        distance_m, elevation_m, slope_deg, curvature, roughness,
        intensity, red, green, blue, classification, return_number, num_returns

    Output (7 features):
        distance_m, elevation_m, slope_deg, curvature, roughness,
        intensity, classification

    M3C2 distance is added later as the 8th feature during cube building.

Usage:
    # Process single file
    python scripts/processing/clean_transects.py \\
        --input data/raw/transects/delmar_20230115.npz \\
        --cliff data/raw/transects/delmar_20230115.cliff.npz \\
        --output data/processed/transects/delmar_20230115.npz

    # Process entire directory
    python scripts/processing/clean_transects.py \\
        --input-dir data/raw/transects/ \\
        --output-dir data/processed/transects/

    # Custom window parameters
    python scripts/processing/clean_transects.py \\
        --input-dir data/raw/transects/ \\
        --output-dir data/processed/transects/ \\
        --toe-buffer 10 \\
        --top-buffer 5 \\
        --n-points 128
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.transect_processor import TransectProcessor, process_directory
from src.utils.logging import setup_logger

logger = setup_logger(__name__, level="INFO")


def print_header():
    """Print clean header with timestamp."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 70)
    print("TRANSECT CLEANING / PROCESSING")
    print("=" * 70)
    print(f"Started: {now}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Clean and process raw transects for model training. "
                    "Crops to cliff window, resamples, and drops uninformative features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process single file
    python scripts/processing/clean_transects.py \\
        --input data/raw/transects/delmar_20230115.npz \\
        --output data/processed/transects/delmar_20230115.npz

    # Process directory (cliff sidecars auto-discovered)
    python scripts/processing/clean_transects.py \\
        --input-dir data/raw/transects/ \\
        --output-dir data/processed/transects/

    # Custom window: 15m seaward of toe, 10m landward of top
    python scripts/processing/clean_transects.py \\
        --input-dir data/raw/transects/ \\
        --output-dir data/processed/transects/ \\
        --toe-buffer 15 \\
        --top-buffer 10
        """
    )

    # Input options (mutually exclusive: single file or directory)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", "-i",
        type=Path,
        help="Single raw transect NPZ file to process"
    )
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing raw transect NPZ files"
    )

    # Output options
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output path for single file processing"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for batch processing"
    )

    # Cliff delineation file (for single file mode)
    parser.add_argument(
        "--cliff", "-c",
        type=Path,
        help="Cliff delineation sidecar file (*.cliff.npz). "
             "If not specified, assumes same name as input with .cliff.npz extension."
    )

    # Processing parameters
    parser.add_argument(
        "--n-points",
        type=int,
        default=128,
        help="Number of output points per transect (default: 128)"
    )
    parser.add_argument(
        "--toe-buffer",
        type=float,
        default=10.0,
        help="Distance seaward of cliff toe to include, in meters (default: 10.0)"
    )
    parser.add_argument(
        "--top-buffer",
        type=float,
        default=5.0,
        help="Distance landward of cliff top to include, in meters (default: 5.0)"
    )
    parser.add_argument(
        "--min-cliff-width",
        type=float,
        default=5.0,
        help="Minimum cliff width in meters; smaller uses fallback (default: 5.0)"
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Skip transects without valid cliff detection (default: use full transect)"
    )

    # Batch processing options
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.npz",
        help="Glob pattern for finding raw transect files (default: *.npz)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist in output directory"
    )

    args = parser.parse_args()

    # Print header
    print_header()

    # Validate arguments
    if args.input and not args.output:
        parser.error("--output is required when using --input")
    if args.input_dir and not args.output_dir:
        parser.error("--output-dir is required when using --input-dir")

    # Print configuration
    print("Configuration:")
    print(f"  Output points: {args.n_points}")
    print(f"  Window: [toe - {args.toe_buffer}m] to [top + {args.top_buffer}m]")
    print(f"  Min cliff width: {args.min_cliff_width}m")
    print(f"  Fallback to full transect: {not args.no_fallback}")
    print()

    # Process single file
    if args.input:
        if not args.input.exists():
            logger.error(f"Input file not found: {args.input}")
            return 1

        # Find cliff delineation file
        cliff_path = args.cliff
        if cliff_path is None:
            # Try common naming conventions
            cliff_path = args.input.with_suffix('.cliff.npz')
            if not cliff_path.exists():
                cliff_path = args.input.parent / f"{args.input.stem}.cliff.npz"

        if not cliff_path.exists():
            logger.error(
                f"Cliff delineation file not found. "
                f"Run detect_cliffs.py first or specify --cliff"
            )
            return 1

        print(f"Input: {args.input}")
        print(f"Cliff: {cliff_path}")
        print(f"Output: {args.output}")
        print()

        # Initialize processor
        processor = TransectProcessor(
            n_output_points=args.n_points,
            toe_buffer_m=args.toe_buffer,
            top_buffer_m=args.top_buffer,
            min_cliff_width_m=args.min_cliff_width,
            fallback_to_full=not args.no_fallback,
        )

        # Process
        try:
            stats = processor.process(args.input, cliff_path, args.output)
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # Process directory
    else:
        if not args.input_dir.exists():
            logger.error(f"Input directory not found: {args.input_dir}")
            return 1

        print(f"Input directory: {args.input_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Pattern: {args.pattern}")
        print(f"Skip existing: {args.skip_existing}")
        print()

        # Process directory
        try:
            stats = process_directory(
                raw_dir=args.input_dir,
                output_dir=args.output_dir,
                n_output_points=args.n_points,
                toe_buffer_m=args.toe_buffer,
                top_buffer_m=args.top_buffer,
                min_cliff_width_m=args.min_cliff_width,
                fallback_to_full=not args.no_fallback,
                pattern=args.pattern,
                skip_existing=args.skip_existing,
            )
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # Final summary
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 70)
    print("✓ PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Completed: {now}")

    if args.output:
        print(f"Output: {args.output}")
        if args.output.exists():
            print(f"Size: {args.output.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        print(f"Output directory: {args.output_dir}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
