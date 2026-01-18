"""
ChemKED Database Fetcher for PRISM
==================================

Parses ChemKED YAML files from the pr-omethe-us/ChemKED-database repository
and converts ignition delay time data to PRISM format.

Data Source:
    GitHub: https://github.com/pr-omethe-us/ChemKED-database
    License: CC BY 4.0

Reference:
    Weber, B. W., & Niemeyer, K. E. (2018). ChemKED: A human- and machine-readable
    data standard for chemical kinetics experiments. International Journal of
    Chemical Kinetics, 50(3), 135-148. https://doi.org/10.1002/kin.21142

Data Format:
    - Ignition delay times (τ) vs temperature (T)
    - Arrhenius relationship: τ = A × exp(Ea/RT)
    - Multiple fuels: n-heptane, butanol isomers, toluene, pentane/NOx

PRISM Validation:
    Can PRISM metrics detect the Arrhenius temperature dependence?
    Hypothesis: log(τ) vs 1/T should show linear relationship
"""

import yaml
import polars as pl
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import re


def parse_value_with_unit(value_str: str) -> tuple[float, str]:
    """Parse a value string like '336 us' or '1186.5 kelvin'."""
    if isinstance(value_str, (int, float)):
        return float(value_str), ''

    match = re.match(r'([\d.eE+-]+)\s*(\w*)', str(value_str))
    if match:
        return float(match.group(1)), match.group(2)
    return float(value_str), ''


def convert_to_si(value: float, unit: str) -> float:
    """Convert to SI units (seconds, Kelvin, Pascal)."""
    unit = unit.lower()

    # Time units -> seconds
    if unit in ['us', 'μs', 'microsecond', 'microseconds']:
        return value * 1e-6
    elif unit in ['ms', 'millisecond', 'milliseconds']:
        return value * 1e-3
    elif unit in ['s', 'second', 'seconds', '']:
        return value

    # Pressure units -> Pascal
    elif unit in ['bar']:
        return value * 1e5
    elif unit in ['atm']:
        return value * 101325
    elif unit in ['pa', 'pascal']:
        return value
    elif unit in ['mpa']:
        return value * 1e6

    # Temperature -> Kelvin (assume already Kelvin)
    elif unit in ['kelvin', 'k']:
        return value

    return value


def parse_chemked_file(filepath: Path) -> list[dict]:
    """Parse a single ChemKED YAML file and extract datapoints."""
    with open(filepath, 'r') as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"  Error parsing {filepath}: {e}")
            return []

    if not data or 'datapoints' not in data:
        return []

    # Extract metadata
    reference = data.get('reference', {})
    doi = reference.get('doi', '')
    year = reference.get('year', '')
    authors = reference.get('authors', [])
    first_author = authors[0].get('name', 'Unknown') if authors else 'Unknown'

    experiment_type = data.get('experiment-type', '')
    apparatus = data.get('apparatus', {})
    apparatus_kind = apparatus.get('kind', '')

    # Extract fuel from composition
    common = data.get('common-properties', {})
    composition = common.get('composition', {})
    species_list = composition.get('species', [])

    fuel_name = 'unknown'
    for species in species_list:
        name = species.get('species-name', '')
        # Skip O2, N2, Ar, etc.
        if name not in ['O2', 'N2', 'Ar', 'He', 'CO2', 'H2O']:
            fuel_name = name
            break

    # Parse datapoints
    rows = []
    for i, dp in enumerate(data.get('datapoints', [])):
        row = {
            'source_file': filepath.name,
            'fuel': fuel_name,
            'experiment_type': experiment_type,
            'apparatus': apparatus_kind,
            'doi': doi,
            'year': year,
            'first_author': first_author,
            'datapoint_idx': i,
        }

        # Temperature
        temp_list = dp.get('temperature', [])
        if temp_list:
            val, unit = parse_value_with_unit(temp_list[0])
            row['temperature_K'] = convert_to_si(val, unit) if unit else val

        # Pressure
        press_list = dp.get('pressure', [])
        if press_list:
            val, unit = parse_value_with_unit(press_list[0])
            row['pressure_Pa'] = convert_to_si(val, unit)

        # Ignition delay
        delay_list = dp.get('ignition-delay', [])
        if delay_list:
            val, unit = parse_value_with_unit(delay_list[0])
            row['ignition_delay_s'] = convert_to_si(val, unit)

        # Equivalence ratio
        row['equivalence_ratio'] = dp.get('equivalence-ratio', None)

        rows.append(row)

    return rows


def fetch_chemked_data(chemked_dir: Path) -> pl.DataFrame:
    """Fetch all ChemKED data from directory."""
    all_rows = []

    yaml_files = list(chemked_dir.rglob('*.yaml'))
    print(f"Found {len(yaml_files)} YAML files")

    for filepath in yaml_files:
        rows = parse_chemked_file(filepath)
        all_rows.extend(rows)

    if not all_rows:
        return pl.DataFrame()

    df = pl.DataFrame(all_rows)
    print(f"Parsed {len(df)} datapoints")

    return df


def convert_to_prism_format(df: pl.DataFrame, output_dir: Path):
    """Convert ChemKED data to PRISM parquet format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'raw').mkdir(exist_ok=True)
    (output_dir / 'config').mkdir(exist_ok=True)
    (output_dir / 'vector').mkdir(exist_ok=True)

    # Filter for valid data
    df = df.filter(
        pl.col('temperature_K').is_not_null() &
        pl.col('ignition_delay_s').is_not_null() &
        (pl.col('ignition_delay_s') > 0)
    )

    print(f"Valid datapoints after filtering: {len(df)}")

    # Group by fuel and create indicators
    # Each fuel/pressure/phi combination becomes an indicator
    # The "signal topology" is ignition delay vs temperature (sorted by 1/T)

    obs_rows = []
    ind_rows = []
    base_date = datetime(2020, 1, 1)

    # Group by fuel, pressure range, and equivalence ratio
    groups = df.group_by(['fuel', 'equivalence_ratio']).agg([
        pl.col('temperature_K'),
        pl.col('ignition_delay_s'),
        pl.col('pressure_Pa'),
        pl.col('doi').first(),
        pl.col('first_author').first(),
        pl.col('year').first(),
    ])

    for row in groups.iter_rows(named=True):
        fuel = row['fuel']
        phi = row['equivalence_ratio']
        temps = row['temperature_K']
        delays = row['ignition_delay_s']
        pressures = row['pressure_Pa']

        if len(temps) < 5:  # Need enough points
            continue

        # Check for temperature variation
        if len(set(temps)) < 3:  # Need at least 3 different temperatures
            continue

        # Create indicator ID
        phi_str = f"phi{phi}" if phi else "phi_unknown"
        indicator_id = f"chemked_{fuel}_{phi_str}"

        # Sort by 1/T (Arrhenius plot order)
        sorted_idx = np.argsort([1/t for t in temps])

        for i, idx in enumerate(sorted_idx):
            obs_rows.append({
                'indicator_id': indicator_id,
                'obs_date': base_date + timedelta(seconds=i),  # Pseudo-time
                'value': float(delays[idx]),  # Ignition delay
                'temperature_K': float(temps[idx]),
                'pressure_Pa': float(pressures[idx]) if pressures[idx] else None,
            })

        # Calculate Arrhenius parameters for ground truth
        inv_T = np.array([1/t for t in temps])
        log_tau = np.log(np.array(delays))

        # Linear fit: log(τ) = log(A) + Ea/(R*T)
        # slope = Ea/R, intercept = log(A)
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(inv_T, log_tau)

        R = 8.314  # J/(mol·K)
        Ea = slope * R  # Activation energy in J/mol
        A = np.exp(intercept)  # Pre-exponential factor

        ind_rows.append({
            'indicator_id': indicator_id,
            'name': f"{fuel} ignition delay (φ={phi})",
            'fuel': fuel,
            'equivalence_ratio': phi,
            'n_points': len(temps),
            'doi': row['doi'],
            'first_author': row['first_author'],
            'year': row['year'],
            # Ground truth Arrhenius parameters
            'activation_energy_kJ_mol': Ea / 1000,
            'pre_exponential_factor': A,
            'arrhenius_r_squared': r_value**2,
        })

    # Write observations
    obs_df = pl.DataFrame(obs_rows)
    obs_df.write_parquet(output_dir / 'raw' / 'observations.parquet')
    print(f"Wrote {len(obs_df)} observations")

    # Write indicators
    ind_df = pl.DataFrame(ind_rows)
    ind_df.write_parquet(output_dir / 'raw' / 'indicators.parquet')
    print(f"Wrote {len(ind_df)} indicators")

    # Write cohort config
    pl.DataFrame([
        {'cohort_id': 'chemked_ignition', 'name': 'ChemKED Ignition Delays', 'domain': 'combustion'}
    ]).write_parquet(output_dir / 'config' / 'cohorts.parquet')

    pl.DataFrame([
        {'cohort_id': 'chemked_ignition', 'indicator_id': r['indicator_id']}
        for r in ind_rows
    ]).write_parquet(output_dir / 'config' / 'cohort_members.parquet')

    # Summary statistics
    print("\n" + "="*60)
    print("ChemKED Data Summary")
    print("="*60)
    print(f"Total indicators: {len(ind_df)}")
    print(f"Total observations: {len(obs_df)}")
    print(f"\nFuels: {ind_df['fuel'].unique().to_list()}")
    print(f"\nArrhenius fit quality (R²):")
    print(ind_df.select(['indicator_id', 'arrhenius_r_squared']).sort('arrhenius_r_squared', descending=True))

    return obs_df, ind_df


def main():
    import sys

    chemked_dir = Path('/Users/jasonrudder/prism-mac/data/chemked')
    output_dir = Path('/Users/jasonrudder/prism-mac/data/chemked_prism')

    if len(sys.argv) > 1:
        chemked_dir = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])

    print("="*60)
    print("ChemKED Data Fetcher for PRISM")
    print("="*60)
    print(f"Source: {chemked_dir}")
    print(f"Output: {output_dir}")
    print()

    # Fetch raw data
    df = fetch_chemked_data(chemked_dir)

    if df.is_empty():
        print("No data found!")
        return

    # Convert to PRISM format
    convert_to_prism_format(df, output_dir)

    print("\n" + "="*60)
    print("DONE")
    print("="*60)


if __name__ == '__main__':
    main()
