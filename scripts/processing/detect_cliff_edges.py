#!/usr/bin/env python3
"""
CLI script for running cliff edge detection on NPZ files.

Detects cliff toe and cliff top positions using the CliffDelineaTool v2.0
CNN-BiLSTM model. Results can be saved as:
  - Sidecar file (*.cliff.npz) - detection results only (default)
  - Merged file (*_with_cliffs.npz) - full copy with detection results integrated (--merge)

Prerequisites:
    Install CliffDelineaTool as an editable package:
    ```
    pip install -e /path/to/CliffDelineaTool_2.0/v2
    ```

Usage:
    # Basic usage - creates sidecar file (delmar.cliff.npz)
    python scripts/processing/detect_cliff_edges.py \\
        --input data/processed/delmar.npz \\
        --checkpoint /path/to/best_model.pth

    # Merge mode - creates new file with cliff data integrated (delmar_with_cliffs.npz)
    python scripts/processing/detect_cliff_edges.py \\
        --input data/processed/delmar.npz \\
        --checkpoint /path/to/best_model.pth \\
        --merge

    # Using environment variables
    export CLIFF_DELINEA_CHECKPOINT=/path/to/best_model.pth
    python scripts/processing/detect_cliff_edges.py --input data/processed/delmar.npz --merge
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.cliff_delineation import detect_cliff_edges, detect_and_merge, load_cliff_results
from src.data.cliff_delineation.detector import get_cliff_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Detect cliff toe/top edges in transect NPZ files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Detect cliffs in a single file
    python detect_cliff_edges.py -i data/processed/delmar.npz -c /path/to/best_model.pth

    # Use environment variable for checkpoint
    export CLIFF_DELINEA_CHECKPOINT=/path/to/best_model.pth
    python detect_cliff_edges.py -i data/processed/delmar.npz

    # Process with custom output path
    python detect_cliff_edges.py -i input.npz -o results/output.cliff.npz -c checkpoint.pth
        """,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input NPZ file (cube format from extract_transects.py)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for results (default: <input>.cliff.npz)",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default=os.environ.get("CLIFF_DELINEA_CHECKPOINT"),
        help="Path to CliffDelineaTool checkpoint (or set CLIFF_DELINEA_CHECKPOINT env var)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for cliff detection (default: 0.5)",
    )
    parser.add_argument(
        "--n-vert",
        type=int,
        default=20,
        help="Window size for local slope calculation (default: 20, must match training)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for inference (default: auto)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress bar",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Only show metrics from existing .cliff.npz file (no detection)",
    )
    parser.add_argument(
        "--merge",
        "-m",
        action="store_true",
        help="Create merged NPZ file with cliff results integrated (default: sidecar file)",
    )

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Metrics-only mode: just load and display existing results
    if args.metrics_only:
        cliff_path = input_path.with_suffix(".cliff.npz")
        if not cliff_path.exists():
            print(f"Error: Cliff results not found: {cliff_path}")
            print("Run detection first without --metrics-only")
            sys.exit(1)

        print(f"Loading results from: {cliff_path}")
        results = load_cliff_results(cliff_path)
        metrics = get_cliff_metrics(results)

        print("\n" + "=" * 50)
        print("Cliff Detection Metrics")
        print("=" * 50)
        print(f"Valid transect-epochs:  {metrics['n_valid']}")
        print(f"Cliffs detected:        {metrics['n_detected']}")
        print(f"Detection rate:         {metrics['detection_rate']:.1%}")
        print(f"Mean cliff width:       {metrics['mean_cliff_width_m']:.1f} m")
        print(f"Std cliff width:        {metrics['std_cliff_width_m']:.1f} m")
        print(f"Min cliff width:        {metrics['min_cliff_width_m']:.1f} m")
        print(f"Max cliff width:        {metrics['max_cliff_width_m']:.1f} m")
        print(f"Mean toe confidence:    {metrics['mean_toe_confidence']:.3f}")
        print(f"Mean top confidence:    {metrics['mean_top_confidence']:.3f}")
        print("=" * 50)
        return

    # Validate checkpoint
    if args.checkpoint is None:
        print("Error: Checkpoint path required.")
        print("Either use --checkpoint or set CLIFF_DELINEA_CHECKPOINT environment variable.")
        print("\nExample:")
        print("  export CLIFF_DELINEA_CHECKPOINT=/path/to/CliffDelineaTool_2.0/v2/scripts/experiments/runs_all_aois/checkpoints/best_model.pth")
        sys.exit(1)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Print banner
    mode_str = "MERGE" if args.merge else "SIDECAR"
    print("=" * 70)
    print("Cliff Edge Detection using CliffDelineaTool v2.0")
    print("=" * 70)
    print(f"Input:       {args.input}")
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Confidence:  {args.confidence}")
    print(f"n_vert:      {args.n_vert}")
    print(f"Device:      {args.device}")
    print(f"Mode:        {mode_str}")
    print("=" * 70)

    # Run detection
    try:
        if args.merge:
            # Merge mode: create new NPZ with cliff data integrated
            output_path = detect_and_merge(
                npz_path=args.input,
                checkpoint_path=args.checkpoint,
                output_path=args.output,
                confidence_threshold=args.confidence,
                n_vert=args.n_vert,
                device=args.device,
                show_progress=not args.quiet,
            )
            # Load results from merged file for metrics
            import numpy as np
            merged_data = np.load(output_path, allow_pickle=True)
            results = {key: merged_data[key] for key in merged_data.keys()}
        else:
            # Sidecar mode: create separate cliff results file
            results = detect_cliff_edges(
                npz_path=args.input,
                checkpoint_path=args.checkpoint,
                output_path=args.output,
                confidence_threshold=args.confidence,
                n_vert=args.n_vert,
                device=args.device,
                show_progress=not args.quiet,
            )
    except ImportError as e:
        print(f"\nError: {e}")
        print("\nMake sure CliffDelineaTool is installed:")
        print("  pip install -e /path/to/CliffDelineaTool_2.0/v2")
        sys.exit(1)

    # Print metrics
    metrics = get_cliff_metrics(results)
    print("\n" + "-" * 50)
    print("Summary Metrics")
    print("-" * 50)
    print(f"Detection rate:      {metrics['detection_rate']:.1%}")
    print(f"Mean cliff width:    {metrics['mean_cliff_width_m']:.1f} m")
    print(f"Mean toe confidence: {metrics['mean_toe_confidence']:.3f}")
    print(f"Mean top confidence: {metrics['mean_top_confidence']:.3f}")
    print("-" * 50)


if __name__ == "__main__":
    main()
