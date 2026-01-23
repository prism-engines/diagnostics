"""
orchestrator.py - Ørthon Pipeline Orchestrator

Single entry point for comprehensive analysis.
Runs: load -> vector -> geometry -> state -> cohort discovery -> report

Usage:
    python -m prism.entry_points.orchestrator --input data.csv
    python -m prism.entry_points.orchestrator --input data.csv --stages vector,geometry
    python -m prism.entry_points.orchestrator --input data.csv --report html
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import polars as pl

from prism.modules.vector import compute_vector_features
from prism.modules.geometry import compute_geometry_features
from prism.modules.state import compute_state_features
from prism.modules.discovery import discover_cohorts, compute_cohort_summary


@dataclass
class PipelineResult:
    """Results from a pipeline run."""
    outputs: Dict[str, Path]
    metadata: Dict[str, Any]
    started_at: datetime
    completed_at: datetime

    @property
    def duration_seconds(self) -> float:
        return (self.completed_at - self.started_at).total_seconds()

    def summary(self) -> str:
        lines = [
            "Ørthon Pipeline Results",
            "=" * 40,
            f"Duration: {self.duration_seconds:.1f}s",
            f"Input: {self.metadata.get('input', 'unknown')}",
            "",
            "Outputs:",
        ]
        for stage, path in self.outputs.items():
            if path.exists():
                size_kb = path.stat().st_size / 1024
                lines.append(f"  {stage}: {path.name} ({size_kb:.1f} KB)")
            else:
                lines.append(f"  {stage}: {path.name} (not found)")
        return "\n".join(lines)


def run(
    input_path: str | Path,
    output_dir: str | Path = "./results",
    entity_col: str = "entity_id",
    time_col: str = "timestamp",
    signal_col: str = "signal_id",
    value_col: str = "value",
    stages: Optional[List[str]] = None,
    report_format: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> PipelineResult:
    """
    Run comprehensive Ørthon pipeline.

    Args:
        input_path: Input data (csv, parquet, txt)
        output_dir: Directory for outputs
        entity_col: Entity identifier column
        time_col: Time/cycle column
        signal_col: Signal identifier column
        value_col: Value column
        stages: Stages to run. Default: all
                Options: ['vector', 'geometry', 'state', 'cohort']
        report_format: Generate report in this format ("html" or "parquet")
        config_overrides: Override config values

    Returns:
        PipelineResult with paths to all outputs
    """
    started_at = datetime.now()
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stages = ['vector', 'geometry', 'state', 'cohort']
    stages = stages or all_stages
    stages = [s for s in all_stages if s in stages]

    outputs = {}

    # Load data
    print(f"Loading {input_path}...")
    df = _load_data(input_path)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Detect column types
    entity_col, time_col, signal_col, value_col = _detect_columns(
        df, entity_col, time_col, signal_col, value_col
    )
    print(f"  Entity: {entity_col}, Time: {time_col}, Signal: {signal_col}, Value: {value_col}")

    current_df = df

    # Vector
    if 'vector' in stages:
        print("\nComputing vector features...")
        vector_df = compute_vector_features(
            df=current_df,
            entity_col=entity_col,
            time_col=time_col,
            signal_col=signal_col,
            value_col=value_col,
        )
        vector_path = output_dir / "vector.parquet"
        vector_df.write_parquet(vector_path)
        outputs['vector'] = vector_path
        current_df = vector_df
        print(f"  -> {vector_path} ({len(vector_df):,} rows)")

    # Geometry
    if 'geometry' in stages:
        print("\nComputing geometry features...")
        geometry_df = compute_geometry_features(
            df=current_df,
            entity_col=entity_col,
            time_col=time_col,
        )
        geometry_path = output_dir / "geometry.parquet"
        geometry_df.write_parquet(geometry_path)
        outputs['geometry'] = geometry_path
        current_df = geometry_df
        print(f"  -> {geometry_path} ({len(geometry_df):,} rows)")

    # State
    if 'state' in stages:
        print("\nComputing state features...")
        state_df = compute_state_features(
            df=current_df,
            entity_col=entity_col,
            time_col=time_col,
        )
        state_path = output_dir / "state.parquet"
        state_df.write_parquet(state_path)
        outputs['state'] = state_path
        current_df = state_df
        print(f"  -> {state_path} ({len(state_df):,} rows)")

    # Cohort discovery
    if 'cohort' in stages:
        print("\nDiscovering cohorts...")
        cohort_df = discover_cohorts(
            df=current_df,
            entity_col=entity_col,
        )
        cohort_path = output_dir / "cohort.parquet"
        cohort_df.write_parquet(cohort_path)
        outputs['cohort'] = cohort_path
        print(f"  -> {cohort_path} ({len(cohort_df):,} entities)")

        # Print cohort summary
        cohort_counts = cohort_df.group_by('cohort').len()
        for row in cohort_counts.iter_rows(named=True):
            print(f"     Cohort {row['cohort']}: {row['len']} entities")

    # Report
    if report_format:
        print(f"\nGenerating {report_format} report...")
        report_path = _generate_report(
            outputs=outputs,
            output_dir=output_dir,
            format=report_format,
            entity_col=entity_col,
            input_path=input_path,
        )
        outputs['report'] = report_path
        print(f"  -> {report_path}")

    completed_at = datetime.now()

    result = PipelineResult(
        outputs=outputs,
        metadata={
            'input': str(input_path),
            'entity_col': entity_col,
            'time_col': time_col,
            'stages': stages,
        },
        started_at=started_at,
        completed_at=completed_at,
    )

    print(f"\nComplete in {result.duration_seconds:.1f}s")
    return result


def _load_data(path: Path) -> pl.DataFrame:
    """Load data from various formats."""
    suffix = path.suffix.lower()
    if suffix == '.parquet':
        return pl.read_parquet(path)
    elif suffix == '.csv':
        return pl.read_csv(path)
    elif suffix == '.txt':
        return pl.read_csv(path, separator='\t')
    else:
        raise ValueError(f"Unsupported format: {suffix}")


def _detect_columns(
    df: pl.DataFrame,
    entity_col: str,
    time_col: str,
    signal_col: str,
    value_col: str,
) -> tuple:
    """Detect or validate column names."""
    cols = df.columns

    # Try to find entity column
    if entity_col not in cols:
        candidates = ['entity_id', 'unit_id', 'unit', 'id', 'machine_id', 'asset_id']
        for c in candidates:
            if c in cols:
                entity_col = c
                break

    # Try to find time column
    if time_col not in cols:
        candidates = ['timestamp', 'time', 'cycle', 't', 'datetime', 'date']
        for c in candidates:
            if c in cols:
                time_col = c
                break

    # Try to find signal column
    if signal_col not in cols:
        candidates = ['signal_id', 'signal', 'sensor', 'sensor_id', 'feature', 'variable']
        for c in candidates:
            if c in cols:
                signal_col = c
                break

    # Try to find value column
    if value_col not in cols:
        candidates = ['value', 'reading', 'measurement', 'y']
        for c in candidates:
            if c in cols:
                value_col = c
                break

    return entity_col, time_col, signal_col, value_col


def _generate_report(
    outputs: Dict[str, Path],
    output_dir: Path,
    format: str,
    entity_col: str,
    input_path: Path,
) -> Path:
    """Generate summary report."""
    if format == 'html':
        return _generate_html_report(outputs, output_dir, entity_col, input_path)
    elif format == 'parquet':
        return _generate_parquet_report(outputs, output_dir, entity_col)
    else:
        raise ValueError(f"Unknown format: {format}")


def _generate_html_report(
    outputs: Dict[str, Path],
    output_dir: Path,
    entity_col: str,
    input_path: Path,
) -> Path:
    """Generate HTML summary report."""
    report_path = output_dir / "report.html"

    # Gather stats from each output
    stats = {}
    for stage, path in outputs.items():
        if path.exists() and path.suffix == '.parquet':
            df = pl.read_parquet(path)
            stats[stage] = {
                'rows': len(df),
                'columns': len(df.columns),
                'col_names': df.columns[:20],
            }

    # Get cohort info if available
    cohort_info = ""
    if 'cohort' in outputs and outputs['cohort'].exists():
        cohort_df = pl.read_parquet(outputs['cohort'])
        cohort_counts = cohort_df.group_by('cohort').len().sort('cohort')
        cohort_info = "<h2>Cohort Summary</h2><table>"
        cohort_info += "<tr><th>Cohort</th><th>Entities</th></tr>"
        for row in cohort_counts.iter_rows(named=True):
            cohort_info += f"<tr><td>{row['cohort']}</td><td>{row['len']}</td></tr>"
        cohort_info += "</table>"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Ørthon Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        h1 {{ color: #333; }}
        .tagline {{ color: #666; font-style: italic; }}
        .stage {{ background-color: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Ørthon Analysis Report</h1>
    <p class="tagline">geometry leads — ørthon</p>
    <p><em>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</em></p>
    <p><strong>Input:</strong> {input_path.name}</p>

    <h2>Pipeline Outputs</h2>
    <table>
        <tr><th>Stage</th><th>Rows</th><th>Columns</th><th>File</th></tr>
"""

    for stage, path in outputs.items():
        if stage in stats:
            s = stats[stage]
            html += f"<tr><td>{stage}</td><td>{s['rows']:,}</td><td>{s['columns']}</td><td>{path.name}</td></tr>\n"

    html += f"""
    </table>

    {cohort_info}

    <h2>Column Details</h2>
"""

    for stage, s in stats.items():
        html += f"""
    <div class="stage">
        <h3>{stage.title()}</h3>
        <p><strong>Columns:</strong> {', '.join(s['col_names'])}</p>
    </div>
"""

    html += """
</body>
</html>
"""

    report_path.write_text(html)
    return report_path


def _generate_parquet_report(
    outputs: Dict[str, Path],
    output_dir: Path,
    entity_col: str,
) -> Path:
    """Generate Parquet summary (combined features)."""
    report_path = output_dir / "summary.parquet"

    # Combine all outputs into one summary
    dfs = []
    for stage, path in outputs.items():
        if path.exists() and path.suffix == '.parquet':
            df = pl.read_parquet(path)
            # Add stage column
            df = df.with_columns(pl.lit(stage).alias('_stage'))
            dfs.append(df)

    if dfs:
        # This is a simple concat - in practice you'd want a proper join
        combined = pl.concat(dfs, how='diagonal')
        combined.write_parquet(report_path)

    return report_path


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ørthon Pipeline - Comprehensive behavioral geometry analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m prism.entry_points.orchestrator --input data.csv
    python -m prism.entry_points.orchestrator --input data.parquet --stages vector,geometry
    python -m prism.entry_points.orchestrator --input data.csv --report html
    python -m prism.entry_points.orchestrator --input data.csv --output results/
"""
    )

    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input data file (csv, parquet, txt)")
    parser.add_argument("--output", "-o", type=str, default="./results",
                        help="Output directory (default: ./results)")
    parser.add_argument("--entity-col", type=str, default="entity_id",
                        help="Entity identifier column (default: entity_id)")
    parser.add_argument("--time-col", type=str, default="timestamp",
                        help="Time/cycle column (default: timestamp)")
    parser.add_argument("--signal-col", type=str, default="signal_id",
                        help="Signal identifier column (default: signal_id)")
    parser.add_argument("--value-col", type=str, default="value",
                        help="Value column (default: value)")
    parser.add_argument("--stages", type=str, default=None,
                        help="Comma-separated stages: vector,geometry,state,cohort")
    parser.add_argument("--report", type=str, choices=["html", "parquet"],
                        help="Generate report in this format")

    args = parser.parse_args()

    stages = args.stages.split(",") if args.stages else None

    run(
        input_path=args.input,
        output_dir=args.output,
        entity_col=args.entity_col,
        time_col=args.time_col,
        signal_col=args.signal_col,
        value_col=args.value_col,
        stages=stages,
        report_format=args.report,
    )


if __name__ == "__main__":
    main()
