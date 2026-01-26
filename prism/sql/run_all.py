"""
prism/sql/run_all.py

CLI wrapper for SQL pipeline.

CANONICAL RULE: This is PURE PLUMBING.
All logic lives in SQL files. This just sequences execution.

Usage:
    python run_all.py /path/to/observations.parquet
    python run_all.py /path/to/observations.parquet ./custom_outputs/
    python run_all.py /path/to/observations.parquet ./outputs/ --primitives /path/to/primitives.parquet
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from prism.sql.orchestrator import SQLOrchestrator

DEFAULT_OUTPUT_DIR = Path(__file__).parent / 'outputs'


def main():
    """CLI entry point. PURE: just parses args and calls orchestrator."""

    if len(sys.argv) < 2:
        print(__doc__)
        print("\nStages:")
        from prism.sql.orchestrator import STAGES
        for name, stage_class in STAGES:
            print(f"  {name}: {stage_class.__doc__.strip().split(chr(10))[0] if stage_class.__doc__ else ''}")
        sys.exit(1)

    # Parse arguments
    observations_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else str(DEFAULT_OUTPUT_DIR)

    # Check for --primitives flag
    primitives_path = None
    if '--primitives' in sys.argv:
        idx = sys.argv.index('--primitives')
        if idx + 1 < len(sys.argv):
            primitives_path = sys.argv[idx + 1]

    # Validate input exists
    if not Path(observations_path).exists():
        print(f"ERROR: Input file not found: {observations_path}")
        sys.exit(1)

    # Run pipeline
    orchestrator = SQLOrchestrator()

    try:
        result = orchestrator.run_pipeline(
            observations_path=observations_path,
            output_dir=output_dir,
            primitives_path=primitives_path,
        )

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Status: {result['status']}")
        print(f"Input rows: {result['input_rows']:,}")
        print(f"Output dir: {result['output_dir']}")
        print(f"Files: {len(result['files'])}")

    except Exception as e:
        print(f"\nFATAL: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
