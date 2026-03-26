#!/usr/bin/env python3
"""ALBart Database Setup — builds everything from scratch.

Runs the full pipeline to create a new ALBart database from your Spotify
top tracks.  Steps:

  1. Pull top tracks from Spotify, download previews + art, compute CLAP
     embeddings, build FAISS indices.
  2. Build 2D UMAP projection (for the UMAP map view).
  3. Build 5D UMAP projection (for the neighborhood 3D view).
  4. Build Voronoi cluster labels (optional, requires ANTHROPIC_API_KEY).
  5. Build CLAP text label embeddings (optional, for future use).

Prerequisites:
  - Set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET environment variables
    (from https://developer.spotify.com/dashboard)
  - Optionally set ANTHROPIC_API_KEY for Voronoi cluster labeling
  - Install dependencies: pip install -r requirements.txt

Usage:
    python setup_database.py              # full setup
    python setup_database.py --skip-voronoi --skip-text-labels
    python setup_database.py --force      # rebuild everything from scratch
"""

import argparse
import os
import subprocess
import sys
import time


def run_step(name: str, cmd: list[str], required: bool = True) -> bool:
    """Run a pipeline step, printing status."""
    print(f"\n{'='*60}")
    print(f"  Step: {name}")
    print(f"{'='*60}\n")

    t0 = time.monotonic()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.monotonic() - t0

    if result.returncode == 0:
        print(f"\n  ✓ {name} completed ({elapsed:.1f}s)")
        return True
    else:
        label = "FAILED" if required else "skipped (optional)"
        print(f"\n  ✗ {name} {label} (exit code {result.returncode})")
        if required:
            print("    Stopping setup. Fix the issue and re-run.")
            sys.exit(1)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="ALBart Database Setup — build everything from scratch"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild everything even if outputs exist",
    )
    parser.add_argument(
        "--skip-voronoi", action="store_true",
        help="Skip Voronoi cluster labeling (requires ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--skip-text-labels", action="store_true",
        help="Skip CLAP text label embeddings",
    )
    args = parser.parse_args()

    force = ["--force"] if args.force else []

    # Check Spotify credentials
    if not os.environ.get("SPOTIPY_CLIENT_ID"):
        print("ERROR: SPOTIPY_CLIENT_ID not set.")
        print("Get credentials from https://developer.spotify.com/dashboard")
        print("Then: export SPOTIPY_CLIENT_ID='...'")
        print("      export SPOTIPY_CLIENT_SECRET='...'")
        sys.exit(1)

    print("ALBart Database Setup")
    print("=" * 60)

    # Step 1: Main pipeline (required)
    run_step(
        "Pipeline: Spotify pull → download → embed → FAISS",
        [sys.executable, "-m", "albart.pipeline.run_pipeline"] + force,
    )

    # Step 2: 2D UMAP (required for map view)
    run_step(
        "2D UMAP projection (map view)",
        [sys.executable, "tools/build_umap.py"] + force,
    )

    # Step 3: 5D UMAP (required for neighborhood view)
    run_step(
        "5D UMAP projection (neighborhood view)",
        [sys.executable, "tools/build_umap_5d.py"] + force,
    )

    # Step 4: Voronoi labels (optional)
    if not args.skip_voronoi:
        if os.environ.get("ANTHROPIC_API_KEY"):
            run_step(
                "Voronoi cluster labels (Claude API)",
                [sys.executable, "tools/build_voronoi.py"] + force,
                required=False,
            )
        else:
            print("\n  Skipping Voronoi labels (ANTHROPIC_API_KEY not set)")
    else:
        print("\n  Skipping Voronoi labels (--skip-voronoi)")

    # Step 5: Text labels (optional)
    if not args.skip_text_labels:
        run_step(
            "CLAP text label embeddings",
            [sys.executable, "tools/build_text_labels.py"] + force,
            required=False,
        )
    else:
        print("\n  Skipping text labels (--skip-text-labels)")

    print(f"\n{'='*60}")
    print("  Setup complete!")
    print(f"{'='*60}")
    print("\nTo run ALBart:")
    print("  python -m albart.listener                    # audio recognition + LED")
    print("  python -m albart.mapview --mode half --view neighborhood  # 3D visualization")
    print("  python -m albart.dj --mode exact             # automated DJ")


if __name__ == "__main__":
    main()
