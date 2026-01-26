"""
prism/sql/validate_outputs.py

Post-hoc validation that all outputs are complete and consistent.

Usage:
    python validate_outputs.py
    python validate_outputs.py ./custom_outputs/
"""

import duckdb
from pathlib import Path
import json
import sys

SQL_DIR = Path(__file__).parent
DEFAULT_OUTPUT_DIR = SQL_DIR / 'outputs'

EXPECTED_OUTPUTS = {
    'calculus.parquet': {
        'required_columns': ['entity_id', 'signal_id', 'I', 'y', 'dy', 'd2y', 'kappa'],
        'no_all_null': ['entity_id', 'signal_id', 'I', 'y'],
    },
    'signal_class.parquet': {
        'required_columns': ['entity_id', 'signal_id', 'signal_class'],
        'no_all_null': ['entity_id', 'signal_id'],
    },
    'signal_typology.parquet': {
        'required_columns': ['entity_id', 'signal_id', 'hurst_rs', 'sample_entropy'],
        'no_all_null': ['entity_id', 'signal_id'],
    },
    'behavioral_geometry.parquet': {
        'required_columns': ['entity_id', 'signal_a', 'signal_b', 'correlation'],
        'no_all_null': ['entity_id', 'signal_a', 'signal_b'],
    },
    'dynamical_systems.parquet': {
        'required_columns': ['entity_id', 'signal_id', 'regime_id'],
        'no_all_null': ['entity_id', 'signal_id'],
    },
    'causal_mechanics.parquet': {
        'required_columns': ['entity_id', 'source_signal', 'target_signal'],
        'no_all_null': ['entity_id', 'source_signal', 'target_signal'],
    },
}


def validate(output_dir: str = None, verbose: bool = True) -> bool:
    """
    Validate all output parquets.

    Returns True if all validations pass, False otherwise.
    """
    output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR

    if not output_dir.exists():
        print(f"ERROR: Output directory does not exist: {output_dir}")
        return False

    manifest_path = output_dir / 'manifest.json'

    if not manifest_path.exists():
        print("WARNING: No manifest.json - pipeline may not have completed")
        manifest = None
    else:
        manifest = json.loads(manifest_path.read_text())
        if verbose:
            print(f"Manifest generated: {manifest.get('generated_at', 'unknown')}")
            print(f"Input rows: {manifest.get('input_rows', 'unknown')}")

    conn = duckdb.connect(':memory:')
    errors = []
    warnings = []

    if verbose:
        print(f"\n{'='*60}")
        print("VALIDATING OUTPUTS")
        print('='*60)

    for filename, checks in EXPECTED_OUTPUTS.items():
        path = output_dir / filename
        if verbose:
            print(f"\n{filename}:")

        # Check file exists
        if not path.exists():
            errors.append(f"MISSING: {filename}")
            if verbose:
                print(f"  MISSING")
            continue

        # Check file size
        size = path.stat().st_size
        if size == 0:
            errors.append(f"EMPTY FILE: {filename}")
            if verbose:
                print(f"  EMPTY FILE (0 bytes)")
            continue

        if verbose:
            print(f"  Size: {size:,} bytes")

        # Check row count
        try:
            actual_rows = conn.execute(f"SELECT COUNT(*) FROM '{path}'").fetchone()[0]
            if verbose:
                print(f"  Rows: {actual_rows:,}")

            if actual_rows == 0:
                errors.append(f"ZERO ROWS: {filename}")

            # Check against manifest
            if manifest and filename in manifest.get('files', {}):
                expected_rows = manifest['files'][filename]['rows']
                if actual_rows != expected_rows:
                    errors.append(f"ROW MISMATCH: {filename} - manifest says {expected_rows}, file has {actual_rows}")
        except Exception as e:
            errors.append(f"UNREADABLE: {filename} - {e}")
            continue

        # Check required columns exist
        try:
            df = conn.execute(f"SELECT * FROM '{path}' LIMIT 0").fetchdf()
            actual_columns = set(df.columns)

            for col in checks.get('required_columns', []):
                if col not in actual_columns:
                    errors.append(f"MISSING COLUMN: {filename}.{col}")

            if verbose:
                print(f"  Columns: {len(actual_columns)}")
        except Exception as e:
            errors.append(f"SCHEMA ERROR: {filename} - {e}")
            continue

        # Check for all-null columns
        for col in checks.get('no_all_null', []):
            if col in actual_columns:
                try:
                    null_count = conn.execute(
                        f"SELECT COUNT(*) FROM '{path}' WHERE \"{col}\" IS NULL"
                    ).fetchone()[0]
                    if null_count == actual_rows and actual_rows > 0:
                        errors.append(f"ALL NULL: {filename}.{col}")
                except Exception:
                    pass

    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print('='*60)

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"  {e}")

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  {w}")

    if not errors:
        print("\nAll outputs validated successfully")
        return True
    else:
        print(f"\nValidation FAILED with {len(errors)} errors")
        return False


def validate_single(output_dir: str, filename: str) -> bool:
    """Validate a single output file."""
    output_dir = Path(output_dir)
    path = output_dir / filename

    if not path.exists():
        print(f"ERROR: File does not exist: {path}")
        return False

    conn = duckdb.connect(':memory:')

    try:
        rows = conn.execute(f"SELECT COUNT(*) FROM '{path}'").fetchone()[0]
        df = conn.execute(f"SELECT * FROM '{path}' LIMIT 5").fetchdf()

        print(f"File: {filename}")
        print(f"Rows: {rows:,}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nSample:")
        print(df)

        return rows > 0
    except Exception as e:
        print(f"ERROR: {e}")
        return False


if __name__ == '__main__':
    output_dir = sys.argv[1] if len(sys.argv) > 1 else None

    if len(sys.argv) > 2:
        # Validate single file
        success = validate_single(output_dir, sys.argv[2])
    else:
        # Validate all
        success = validate(output_dir)

    sys.exit(0 if success else 1)
