# PRISM Reports

SQL-first reports that run identically in MotherDuck UI and Python CLI.

## Quick Start

### List available reports
```bash
python scripts/reports.py --list
```

### Run a report (local DB)
```bash
python scripts/reports.py --report 00_catalog_health
```

### Run against MotherDuck
```bash
python scripts/reports.py --db md:prism --report 00_catalog_health
```

### Export to CSV
```bash
python scripts/reports.py --report 20_univariate_signal topology \
    --indicator BEARING_01 --metric hurst_exponent \
    --start 2020-01-01 --end 2024-12-31 \
    --format csv --out sensor_hurst.csv
```

### Export to Parquet
```bash
python scripts/reports.py --report 32_geometry_3d_export \
    --cohort sensors \
    --format parquet --out geometry_3d.parquet
```

### Print as Markdown (for docs/issues)
```bash
python scripts/reports.py --report 70_leadership_rankings \
    --cohort sensors --state_time 2024-12-31 \
    --format markdown
```

### Show SQL without executing
```bash
python scripts/reports.py --report 60_regime_timeline --sql
```

## Report Categories

| Range | Category | Description |
|-------|----------|-------------|
| 00-09 | Health | Catalog health, coverage gaps |
| 10-19 | Raw | Raw observation summaries |
| 20-29 | Univariate | Single-indicator signal topology |
| 30-39 | Geometry | Cohort structure snapshots |
| 40-49 | State | Temporal dynamics |
| 50-59 | Coupling | Univariate vs state comparison |
| 60-69 | Regime | Regime timeline, transitions |
| 70-79 | Leadership | Granger/TE rankings, networks |
| 80-89 | Outliers | LOF alerts, outlier history |
| 90-99 | Audit | Engine runs, drift detection |

## MotherDuck UI Usage

Copy any SQL file content into a MotherDuck notebook cell. Replace `{{placeholders}}` with actual values:

```sql
-- Example: 20_univariate_signal topology.sql
-- Replace {{indicator_id}} with 'BEARING_01', etc.

SELECT
    window_end AS date,
    metric_value AS value
FROM results.univariate
WHERE indicator_id = 'BEARING_01'  -- was {{indicator_id}}
  AND metric_name = 'hurst_exponent'  -- was {{metric_name}}
  AND window_end BETWEEN '2020-01-01' AND '2024-12-31'
ORDER BY window_end;
```

## Adding New Reports

1. Create `sql/reports/XX_name.sql`
2. Add description as first comment line
3. Use `{{placeholder}}` for parameters
4. Test: `python scripts/reports.py --report XX_name --sql`
5. Run: `python scripts/reports.py --report XX_name`

## Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--indicator` | BEARING_01 | Indicator ID |
| `--cohort` | default | Cohort ID |
| `--metric` | hurst_exponent | Metric name |
| `--engine` | hurst | Engine name |
| `--domain` | % | Domain filter (% = all) |
| `--start` | 1970-01-01 | Start date |
| `--end` | 2099-12-31 | End date |
| `--limit` | 100 | Row limit |
