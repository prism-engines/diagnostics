"""
Orthon Experimental Comparison Report Generator

Generates automated scientific reports comparing experimental entities
against control cohorts. Pure computation - no AI required.

Outputs quantified structural differences, temporal dynamics,
and templated scientific prose ready for publication.
"""

import polars as pl
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from scipy import stats


@dataclass
class PairwiseDivergence:
    """Divergence metrics for a signal pair."""
    signal_a: str
    signal_b: str
    corr_control: float
    corr_experiment: float
    delta: float
    pct_diff: float           # Percentage difference from control
    outside_range: bool       # Is experiment outside control's typical range?
    n_control: int            # Sample size context
    n_experiment: int


@dataclass
class TemporalDivergence:
    """Temporal divergence metrics for a signal."""
    signal: str
    slope_control: float
    slope_experiment: float
    ratio: float
    divergence_onset_cycle: Optional[int]


@dataclass
class ComparisonResult:
    """Complete comparison results."""
    experiment_ids: list[str]
    control_ids: list[str]
    structural_divergences: list[PairwiseDivergence]
    temporal_divergences: list[TemporalDivergence]
    signal_type_effects: dict[str, float]
    divergence_onset: Optional[int]
    primary_effect_cluster: Optional[str]


def compare_cohorts(
    experiment_ids: list[str],
    control_ids: list[str],
    geometry_path: Path | str,
    state_path: Path | str,
    vector_path: Optional[Path | str] = None,
    significance_threshold: float = 0.05,
) -> ComparisonResult:
    """
    Compare experimental entities against control cohort.

    Parameters
    ----------
    experiment_ids : List of experimental entity IDs
    control_ids : List of control entity IDs
    geometry_path : Path to geometry.parquet
    state_path : Path to state.parquet
    vector_path : Optional path to vector.parquet for signal-level analysis
    significance_threshold : Unused, kept for API compatibility

    Returns
    -------
    ComparisonResult with all divergence metrics
    """
    geometry = pl.read_parquet(geometry_path)
    state = pl.read_parquet(state_path)

    # Structural comparison
    structural = _compute_structural_divergence(
        geometry, experiment_ids, control_ids, significance_threshold
    )

    # Temporal comparison
    temporal = _compute_temporal_divergence(
        state, experiment_ids, control_ids
    )

    # Signal type effects (if vector data available)
    signal_effects = {}
    if vector_path:
        signal_effects = _compute_signal_type_effects(
            pl.read_parquet(vector_path), experiment_ids, control_ids
        )

    # Find divergence onset
    divergence_onset = _find_divergence_onset(state, experiment_ids, control_ids)

    # Identify primary effect cluster
    primary_cluster = _identify_primary_cluster(structural, signal_effects)

    return ComparisonResult(
        experiment_ids=experiment_ids,
        control_ids=control_ids,
        structural_divergences=structural,
        temporal_divergences=temporal,
        signal_type_effects=signal_effects,
        divergence_onset=divergence_onset,
        primary_effect_cluster=primary_cluster,
    )


def _compute_structural_divergence(
    geometry: pl.DataFrame,
    experiment_ids: list[str],
    control_ids: list[str],
    significance_threshold: float,
) -> list[PairwiseDivergence]:
    """Compute correlation differences for all signal pairs."""
    divergences = []

    # Get correlation columns (if stored in geometry)
    # Otherwise compute from raw correlations
    corr_cols = [c for c in geometry.columns if 'corr' in c.lower() or 'mi' in c.lower()]

    if not corr_cols:
        # Use available geometry metrics as proxy
        corr_cols = ['distance_mean', 'mi_mean', 'clustering_silhouette',
                     'pca_var_1', 'pca_effective_dim', 'mst_total_weight']
        corr_cols = [c for c in corr_cols if c in geometry.columns]

    for col in corr_cols:
        # Get values for control and experiment
        control_vals = geometry.filter(
            pl.col('entity_id').is_in(control_ids)
        ).select(col).drop_nulls().to_series().to_numpy()

        exp_vals = geometry.filter(
            pl.col('entity_id').is_in(experiment_ids)
        ).select(col).drop_nulls().to_series().to_numpy()

        if len(control_vals) == 0 or len(exp_vals) == 0:
            continue

        control_mean = np.mean(control_vals)
        control_std = np.std(control_vals) if len(control_vals) > 1 else 0.0
        exp_mean = np.mean(exp_vals)
        delta = exp_mean - control_mean

        # Percentage difference from control mean
        pct_diff = (delta / abs(control_mean) * 100) if control_mean != 0 else 0.0

        # Is experiment outside control's typical range? (beyond 2 std devs)
        if control_std > 0:
            z_score = abs(delta) / control_std
            outside_range = z_score > 2.0
        else:
            # If no variance in control, any difference is notable
            outside_range = abs(delta) > 0.01

        divergences.append(PairwiseDivergence(
            signal_a=col,
            signal_b="aggregate",
            corr_control=control_mean,
            corr_experiment=exp_mean,
            delta=delta,
            pct_diff=pct_diff,
            outside_range=outside_range,
            n_control=len(control_vals),
            n_experiment=len(exp_vals),
        ))

    # Sort by absolute delta
    divergences.sort(key=lambda x: abs(x.delta), reverse=True)
    return divergences


def _compute_temporal_divergence(
    state: pl.DataFrame,
    experiment_ids: list[str],
    control_ids: list[str],
) -> list[TemporalDivergence]:
    """Compute trajectory differences (hd_slope, velocity, etc.)."""
    divergences = []

    # Find slope/velocity columns
    slope_cols = [c for c in state.columns if 'slope' in c.lower() or 'velocity' in c.lower()]

    if not slope_cols:
        slope_cols = ['speed', 'acceleration_magnitude']
        slope_cols = [c for c in slope_cols if c in state.columns]

    for col in slope_cols:
        control_vals = state.filter(
            pl.col('entity_id').is_in(control_ids)
        ).select(col).drop_nulls().to_series().to_numpy()

        exp_vals = state.filter(
            pl.col('entity_id').is_in(experiment_ids)
        ).select(col).drop_nulls().to_series().to_numpy()

        if len(control_vals) == 0 or len(exp_vals) == 0:
            continue

        control_mean = np.mean(control_vals)
        exp_mean = np.mean(exp_vals)

        ratio = exp_mean / control_mean if control_mean != 0 else np.inf

        divergences.append(TemporalDivergence(
            signal=col,
            slope_control=control_mean,
            slope_experiment=exp_mean,
            ratio=ratio,
            divergence_onset_cycle=None,  # Computed separately
        ))

    return divergences


def _compute_signal_type_effects(
    vector: pl.DataFrame,
    experiment_ids: list[str],
    control_ids: list[str],
) -> dict[str, float]:
    """Compute which signal types show largest divergence."""
    effects = {}

    if 'signal_type' not in vector.columns:
        return effects

    for sig_type in vector['signal_type'].unique().to_list():
        type_data = vector.filter(pl.col('signal_type') == sig_type)

        control_vals = type_data.filter(
            pl.col('entity_id').is_in(control_ids)
        ).select('value').drop_nulls().to_series().to_numpy()

        exp_vals = type_data.filter(
            pl.col('entity_id').is_in(experiment_ids)
        ).select('value').drop_nulls().to_series().to_numpy()

        if len(control_vals) > 0 and len(exp_vals) > 0:
            # Cohen's d effect size
            pooled_std = np.sqrt((np.var(control_vals) + np.var(exp_vals)) / 2)
            if pooled_std > 0:
                effects[sig_type] = (np.mean(exp_vals) - np.mean(control_vals)) / pooled_std

    return effects


def _find_divergence_onset(
    state: pl.DataFrame,
    experiment_ids: list[str],
    control_ids: list[str],
) -> Optional[int]:
    """Find the first cycle where experiment diverges from control."""
    if 'timestamp' not in state.columns:
        return None

    # Get time-ordered data
    control_ts = state.filter(
        pl.col('entity_id').is_in(control_ids)
    ).sort('timestamp')

    exp_ts = state.filter(
        pl.col('entity_id').is_in(experiment_ids)
    ).sort('timestamp')

    if control_ts.height == 0 or exp_ts.height == 0:
        return None

    # Simple change-point: first timestamp where exp > 2*std of control
    metric_col = 'speed' if 'speed' in state.columns else state.columns[-1]

    control_mean = control_ts.select(metric_col).mean().item()
    control_std = control_ts.select(metric_col).std().item()

    if control_std is None or control_std == 0:
        return None

    threshold = control_mean + 2 * control_std

    divergent = exp_ts.filter(pl.col(metric_col) > threshold)

    if divergent.height > 0:
        return int(divergent['timestamp'].min())

    return None


def _identify_primary_cluster(
    structural: list[PairwiseDivergence],
    signal_effects: dict[str, float],
) -> Optional[str]:
    """Identify which signal cluster shows primary effect."""
    if signal_effects:
        # Return cluster with largest absolute effect
        return max(signal_effects.keys(), key=lambda k: abs(signal_effects[k]))

    if structural:
        # Return signal from largest divergence
        return structural[0].signal_a

    return None


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(
    result: ComparisonResult,
    output_format: str = "markdown",
    include_figures: bool = False,
) -> str:
    """
    Generate scientific comparison report from results.

    Parameters
    ----------
    result : ComparisonResult from compare_cohorts()
    output_format : "markdown", "latex", or "html"
    include_figures : Whether to include figure placeholders

    Returns
    -------
    Formatted report string
    """
    if output_format == "markdown":
        return _generate_markdown_report(result, include_figures)
    elif output_format == "latex":
        return _generate_latex_report(result, include_figures)
    elif output_format == "html":
        return _generate_html_report(result, include_figures)
    else:
        raise ValueError(f"Unknown format: {output_format}")


def _generate_markdown_report(result: ComparisonResult, include_figures: bool) -> str:
    """Generate Markdown formatted report."""

    exp_str = ", ".join(result.experiment_ids)
    ctrl_str = ", ".join(result.control_ids)

    report = f"""# Experimental Comparison Report
## {exp_str} vs Control Cohort ({ctrl_str})
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Orthon Automated Analysis

---

### Structural Divergence

| Metric | Control | Experiment | Difference | Notable? |
|--------|---------|------------|------------|----------|
"""

    for div in result.structural_divergences[:10]:  # Top 10
        pct_str = f"{div.pct_diff:+.1f}%"
        notable = "Yes" if div.outside_range else "No"
        report += f"| {div.signal_a} | {div.corr_control:.3f} | {div.corr_experiment:.3f} | {pct_str} | {notable} |\n"

    if result.primary_effect_cluster:
        report += f"\n**Primary effect:** {result.primary_effect_cluster} signals\n"

    report += """
---

### Temporal Dynamics

| Metric | Control | Experiment | Ratio |
|--------|---------|------------|-------|
"""

    for div in result.temporal_divergences:
        report += f"| {div.signal} | {div.slope_control:.4f} | {div.slope_experiment:.4f} | {div.ratio:.2f}x |\n"

    if result.divergence_onset:
        report += f"\n**Divergence onset:** Cycle {result.divergence_onset}\n"

    # Mechanistic summary (templated prose)
    report += """
---

### Mechanistic Summary

"""
    report += _generate_mechanistic_summary(result)

    if include_figures:
        report += """
---

### Figures

![Structural Divergence Heatmap](figures/structural_divergence.png)
*Figure 1: Correlation difference between experimental and control cohorts.*

![Temporal Trajectory Comparison](figures/temporal_trajectories.png)
*Figure 2: Degradation trajectories showing divergence onset.*
"""

    return report


def _generate_mechanistic_summary(result: ComparisonResult) -> str:
    """Generate templated scientific prose summary."""

    exp_str = ", ".join(result.experiment_ids)

    # Find notable structural differences
    notable_structural = [d for d in result.structural_divergences if d.outside_range]

    # Find notable temporal differences
    notable_temporal = [d for d in result.temporal_divergences if abs(d.ratio - 1.0) > 0.1]

    summary = f"The experimental {'entity' if len(result.experiment_ids) == 1 else 'entities'} ({exp_str}) exhibit"

    if notable_structural:
        top = notable_structural[0]
        direction = "lower" if top.delta < 0 else "higher"
        summary += f" {abs(top.pct_diff):.0f}% {direction} {top.signal_a} than control"

    if notable_temporal:
        top_temp = notable_temporal[0]
        if notable_structural:
            summary += ", with"
        summary += f" coherence loss velocity on {top_temp.signal} signals at {top_temp.ratio:.2f}x the control rate"

    if result.divergence_onset:
        summary += f". Divergence onset occurs at cycle {result.divergence_onset}, preceding state transition"

    summary += ".\n"

    # Add hypothesis if patterns suggest one
    if result.primary_effect_cluster:
        summary += f"\n**Suggested mechanism:** Primary effect localized to {result.primary_effect_cluster} signal cluster, "
        summary += "suggesting differential pathway compared to control population.\n"

    return summary


def _generate_latex_report(result: ComparisonResult, include_figures: bool) -> str:
    """Generate LaTeX formatted report."""

    exp_str = ", ".join(result.experiment_ids)
    ctrl_str = ", ".join(result.control_ids)

    report = f"""\\documentclass{{article}}
\\usepackage{{booktabs}}
\\usepackage{{graphicx}}

\\title{{Experimental Comparison Report: {exp_str} vs Control}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle

\\section{{Structural Divergence}}

\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{lrrrl}}
\\toprule
Metric & Control & Experiment & Difference & Notable \\\\
\\midrule
"""

    for div in result.structural_divergences[:10]:
        pct_str = f"{div.pct_diff:+.1f}\\%"
        notable = "Yes" if div.outside_range else "No"
        report += f"{div.signal_a} & {div.corr_control:.3f} & {div.corr_experiment:.3f} & {pct_str} & {notable} \\\\\n"

    report += """\\bottomrule
\\end{tabular}
\\caption{Structural divergence between experimental and control cohorts.}
\\end{table}

\\section{Temporal Dynamics}

\\begin{table}[h]
\\centering
\\begin{tabular}{lrrr}
\\toprule
Metric & Control & Experiment & Ratio \\\\
\\midrule
"""

    for div in result.temporal_divergences:
        report += f"{div.signal} & {div.slope_control:.4f} & {div.slope_experiment:.4f} & {div.ratio:.2f}$\\times$ \\\\\n"

    report += """\\bottomrule
\\end{tabular}
\\caption{Temporal dynamics comparison.}
\\end{table}

\\section{Mechanistic Summary}

"""
    report += _generate_mechanistic_summary(result).replace("**", "\\textbf{").replace("\n", "}\n")

    report += """
\\end{document}
"""
    return report


def _md_to_html(text: str) -> str:
    """Convert basic markdown to HTML."""
    import re
    # Convert **bold** to <strong>bold</strong>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Convert newlines to <br>
    text = text.replace('\n', '<br>')
    return text


def _generate_html_report(result: ComparisonResult, include_figures: bool) -> str:
    """Generate HTML formatted report."""

    exp_str = ", ".join(result.experiment_ids)
    ctrl_str = ", ".join(result.control_ids)

    report = f"""<!DOCTYPE html>
<html>
<head>
    <title>Experimental Comparison: {exp_str} vs Control</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
        th {{ background-color: #4CAF50; color: white; }}
        .notable {{ font-weight: bold; color: #d32f2f; }}
        h1, h2, h3 {{ color: #333; }}
    </style>
</head>
<body>
    <h1>Experimental Comparison Report</h1>
    <h2>{exp_str} vs Control Cohort ({ctrl_str})</h2>
    <p><em>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Orthon Automated Analysis</em></p>

    <h3>Structural Divergence</h3>
    <table>
        <tr><th>Metric</th><th>Control</th><th>Experiment</th><th>Difference</th><th>Notable?</th></tr>
"""

    for div in result.structural_divergences[:10]:
        cls = ' class="notable"' if div.outside_range else ''
        pct_str = f"{div.pct_diff:+.1f}%"
        notable = "Yes" if div.outside_range else "No"
        report += f'        <tr{cls}><td>{div.signal_a}</td><td>{div.corr_control:.3f}</td><td>{div.corr_experiment:.3f}</td><td>{pct_str}</td><td>{notable}</td></tr>\n'

    report += """    </table>

    <h3>Temporal Dynamics</h3>
    <table>
        <tr><th>Metric</th><th>Control</th><th>Experiment</th><th>Ratio</th></tr>
"""

    for div in result.temporal_divergences:
        report += f'        <tr><td>{div.signal}</td><td>{div.slope_control:.4f}</td><td>{div.slope_experiment:.4f}</td><td>{div.ratio:.2f}x</td></tr>\n'

    report += f"""    </table>

    <h3>Mechanistic Summary</h3>
    <p>{_md_to_html(_generate_mechanistic_summary(result))}</p>

</body>
</html>
"""
    return report


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """Command-line interface for experimental comparison."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate scientific comparison report between experimental and control cohorts"
    )
    parser.add_argument("--experiment", required=True, nargs="+", help="Experimental entity IDs")
    parser.add_argument("--control", required=True, nargs="+", help="Control entity IDs")
    parser.add_argument("--geometry", required=True, help="Path to geometry.parquet")
    parser.add_argument("--state", required=True, help="Path to state.parquet")
    parser.add_argument("--vector", help="Optional path to vector.parquet")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--format", choices=["markdown", "latex", "html"], default="markdown")
    parser.add_argument("--figures", action="store_true", help="Include figure placeholders")

    args = parser.parse_args()

    print(f"Comparing {args.experiment} vs {args.control}...")

    result = compare_cohorts(
        experiment_ids=args.experiment,
        control_ids=args.control,
        geometry_path=args.geometry,
        state_path=args.state,
        vector_path=args.vector,
    )

    report = generate_report(result, output_format=args.format, include_figures=args.figures)

    with open(args.output, 'w') as f:
        f.write(report)

    print(f"Report saved to {args.output}")

    # Print summary
    n_notable = len([d for d in result.structural_divergences if d.outside_range])
    print(f"\nSummary:")
    print(f"  Notable structural divergences: {n_notable}")
    print(f"  Divergence onset: {result.divergence_onset or 'Not detected'}")
    print(f"  Primary effect cluster: {result.primary_effect_cluster or 'Unknown'}")


if __name__ == "__main__":
    main()
