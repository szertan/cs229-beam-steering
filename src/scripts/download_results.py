#!/usr/bin/env python3
# =============================================================================
# Download Training Results from S3
# =============================================================================
"""
Utility script to download training results from AWS S3 to your local machine.

Usage:
    # List all available results
    python src/scripts/download_results.py --list
    
    # Download specific training run
    python src/scripts/download_results.py --prefix 90deg_20251214_120000
    
    # Download all results
    python src/scripts/download_results.py --all
    
    # Download to custom directory
    python src/scripts/download_results.py --prefix 90deg_20251214_120000 --output ./my_results
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training import download_from_s3, list_s3_results


def main():
    parser = argparse.ArgumentParser(
        description='Download ES training results from S3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  List available results:
    python src/scripts/download_results.py --list
    
  Download a specific training run:
    python src/scripts/download_results.py --prefix 90deg_20251214_120000
    
  Download all results:
    python src/scripts/download_results.py --all
    
  Download to custom directory:
    python src/scripts/download_results.py --prefix 90deg_20251214_120000 --output ./my_results
"""
    )
    
    parser.add_argument('--list', '-l', action='store_true',
                        help='List all available training results in S3')
    parser.add_argument('--prefix', '-p', type=str,
                        help='S3 prefix (training run name) to download')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Download all training results')
    parser.add_argument('--output', '-o', type=str, default='results/aws_outputs',
                        help='Local output directory (default: results/aws_outputs)')
    parser.add_argument('--bucket', '-b', type=str, default='cs229-beam-steering-results',
                        help='S3 bucket name (default: cs229-beam-steering-results)')
    
    args = parser.parse_args()
    
    # Default action: list results
    if not args.list and not args.prefix and not args.all:
        args.list = True
    
    if args.list:
        results = list_s3_results(args.bucket, verbose=True)
        if results:
            print("\nTo download a specific result, run:")
            print(f"  python src/scripts/download_results.py --prefix {results[0]}")
        return
    
    if args.all:
        results = list_s3_results(args.bucket, verbose=False)
        if not results:
            print("No results found in S3 bucket.")
            return
        
        print(f"Downloading {len(results)} training runs...")
        for prefix in results:
            local_dir = Path(args.output) / prefix
            download_from_s3(args.bucket, prefix, str(local_dir), verbose=True)
        
        print(f"\n✅ All results downloaded to: {args.output}")
        return
    
    if args.prefix:
        local_dir = Path(args.output) / args.prefix
        success = download_from_s3(args.bucket, args.prefix, str(local_dir), verbose=True)
        
        if success:
            print(f"\n📁 Results downloaded to: {local_dir}")
            print("\nKey files:")
            print(f"  - Final model: {local_dir}/checkpoint_*/best_rho.npy")
            print(f"  - Metadata:    {local_dir}/checkpoint_*/metadata.json")
            print(f"  - Top-10:      {local_dir}/checkpoint_*/top_k_configs/")
            print(f"  - Field map:   {local_dir}/checkpoint_*/best_Ez.npy")
            
            # Show how to load the model
            print("\n📊 To load the trained model in Python:")
            print(f"""
    import numpy as np
    
    # Load the optimized rod configuration (8x8 array)
    best_rho = np.load('{local_dir}/checkpoint_00999/best_rho.npy')
    
    # Each value is the normalized plasma frequency (0-1) for that rod
    print(f"Rod configuration shape: {{best_rho.shape}}")
    print(f"Values range: [{{best_rho.min():.4f}}, {{best_rho.max():.4f}}]")
""")


if __name__ == '__main__':
    main()
