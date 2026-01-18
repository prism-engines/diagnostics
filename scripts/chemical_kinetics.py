"""
Chemical Kinetics Simulator for PRISM Validation
=================================================

Generates concentration signal topology with known rate equations to test:
- Can PRISM recover rate behavior from concentration dynamics?
- Do PRISM metrics correlate with rate constants?
- Can PRISM detect reaction order?

Reaction Types:
    1. First-order:     A → B,        d[A]/dt = -k[A]
    2. Second-order:    A + A → B,    d[A]/dt = -k[A]²
    3. Reversible:      A ⇌ B,        d[A]/dt = -k₁[A] + k₂[B]
    4. Consecutive:     A → B → C,    d[A]/dt = -k₁[A], d[B]/dt = k₁[A] - k₂[B]
    5. Oscillating:     Brusselator,  Limit cycle behavior

Ground Truth: Arrhenius equation k = A × exp(-Ea/RT)
"""

import numpy as np
from scipy.integrate import odeint
import polars as pl
from pathlib import Path
from datetime import datetime, timedelta


# Physical constants
R = 8.314  # Gas constant J/(mol·K)


def arrhenius(A: float, Ea: float, T: float) -> float:
    """
    Arrhenius equation for rate constant.

    k = A × exp(-Ea/RT)

    Args:
        A: Pre-exponential factor (s⁻¹ or M⁻¹s⁻¹)
        Ea: Activation energy (J/mol)
        T: Temperature (K)

    Returns:
        Rate constant k
    """
    return A * np.exp(-Ea / (R * T))


# =============================================================================
# REACTION MODELS
# =============================================================================

def first_order(y, t, k):
    """
    First-order reaction: A → B

    d[A]/dt = -k[A]
    Analytical: [A] = [A]₀ exp(-kt)
    """
    A = y[0]
    return [-k * A]


def second_order(y, t, k):
    """
    Second-order reaction: A + A → B

    d[A]/dt = -k[A]²
    Analytical: 1/[A] = 1/[A]₀ + kt
    """
    A = y[0]
    return [-k * A**2]


def reversible(y, t, k1, k2):
    """
    Reversible reaction: A ⇌ B

    d[A]/dt = -k₁[A] + k₂[B]
    d[B]/dt = k₁[A] - k₂[B]
    """
    A, B = y
    dAdt = -k1 * A + k2 * B
    dBdt = k1 * A - k2 * B
    return [dAdt, dBdt]


def consecutive(y, t, k1, k2):
    """
    Consecutive reactions: A → B → C

    d[A]/dt = -k₁[A]
    d[B]/dt = k₁[A] - k₂[B]
    d[C]/dt = k₂[B]
    """
    A, B, C = y
    dAdt = -k1 * A
    dBdt = k1 * A - k2 * B
    dCdt = k2 * B
    return [dAdt, dBdt, dCdt]


def brusselator(y, t, a, b):
    """
    Brusselator: Model oscillating chemical system

    A → X           (rate a)
    2X + Y → 3X     (rate 1)
    B + X → Y + D   (rate b)
    X → E           (rate 1)

    Simplified:
    dX/dt = a - (b+1)X + X²Y
    dY/dt = bX - X²Y

    Oscillates when b > 1 + a²
    """
    X, Y = y
    dXdt = a - (b + 1) * X + X**2 * Y
    dYdt = b * X - X**2 * Y
    return [dXdt, dYdt]


def oregonator(y, t, q, f, eps):
    """
    Oregonator: Model for Belousov-Zhabotinsky reaction

    Exhibits complex oscillations and chaos.
    """
    X, Y, Z = y
    dXdt = (q * Y - X * Y + X * (1 - X)) / eps
    dYdt = (-q * Y - X * Y + f * Z)
    dZdt = X - Z
    return [dXdt, dYdt, dZdt]


# =============================================================================
# TRAJECTORY GENERATION
# =============================================================================

def generate_first_order_trajectories(temperatures, A=1e13, Ea=50000, A0=1.0, t_max=10, n_points=1000):
    """
    Generate first-order reaction trajectories at different temperatures.

    Tests: Can PRISM detect rate constant from concentration decay?
    """
    trajectories = {}
    t = np.linspace(0, t_max, n_points)

    for T in temperatures:
        k = arrhenius(A, Ea, T)
        half_life = np.log(2) / k

        # Analytical solution
        conc = A0 * np.exp(-k * t)

        # Also integrate numerically for validation
        solution = odeint(first_order, [A0], t, args=(k,))
        conc_numerical = solution[:, 0]

        trajectories[f'first_order_{T}K'] = {
            't': t,
            'concentration': conc,
            'concentration_numerical': conc_numerical,
            'rate_constant': k,
            'half_life': half_life,
            'temperature': T,
            'order': 1,
            'reaction': 'A → B',
        }

        print(f"  T={T}K: k={k:.4e} s⁻¹, t½={half_life:.4f} s")

    return trajectories


def generate_second_order_trajectories(temperatures, A=1e10, Ea=40000, A0=1.0, t_max=10, n_points=1000):
    """
    Generate second-order reaction trajectories.
    """
    trajectories = {}
    t = np.linspace(0, t_max, n_points)

    for T in temperatures:
        k = arrhenius(A, Ea, T)
        half_life = 1 / (k * A0)

        # Analytical solution
        conc = A0 / (1 + k * A0 * t)

        # Numerical
        solution = odeint(second_order, [A0], t, args=(k,))

        trajectories[f'second_order_{T}K'] = {
            't': t,
            'concentration': conc,
            'concentration_numerical': solution[:, 0],
            'rate_constant': k,
            'half_life': half_life,
            'temperature': T,
            'order': 2,
            'reaction': 'A + A → B',
        }

        print(f"  T={T}K: k={k:.4e} M⁻¹s⁻¹, t½={half_life:.4f} s")

    return trajectories


def generate_oscillating_trajectories(params_list, t_max=100, n_points=5000):
    """
    Generate oscillating reaction trajectories (Brusselator).

    Tests: Can PRISM detect limit cycles and oscillation frequency?
    """
    trajectories = {}
    t = np.linspace(0, t_max, n_points)

    for i, (a, b) in enumerate(params_list):
        y0 = [1.0, 1.0]  # Initial concentrations

        solution = odeint(brusselator, y0, t, args=(a, b))
        X, Y = solution[:, 0], solution[:, 1]

        # Check if oscillating (b > 1 + a²)
        is_oscillating = b > 1 + a**2

        # Estimate period if oscillating
        if is_oscillating and len(X) > 100:
            # Find peaks
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(X, distance=20)
            if len(peaks) > 2:
                period = np.mean(np.diff(t[peaks]))
            else:
                period = np.nan
        else:
            period = np.nan

        label = 'oscillating' if is_oscillating else 'stable'

        trajectories[f'brusselator_{label}_{i}'] = {
            't': t,
            'X': X,
            'Y': Y,
            'a': a,
            'b': b,
            'is_oscillating': is_oscillating,
            'period': period,
            'reaction': 'Brusselator',
        }

        print(f"  a={a}, b={b}: {label}" + (f", period={period:.2f}" if not np.isnan(period) else ""))

    return trajectories


def generate_consecutive_trajectories(k1_values, k2=0.5, A0=1.0, t_max=20, n_points=1000):
    """
    Generate consecutive reaction trajectories: A → B → C

    Tests: Can PRISM detect intermediate species dynamics?
    """
    trajectories = {}
    t = np.linspace(0, t_max, n_points)

    for k1 in k1_values:
        y0 = [A0, 0.0, 0.0]  # [A, B, C]

        solution = odeint(consecutive, y0, t, args=(k1, k2))
        A, B, C = solution[:, 0], solution[:, 1], solution[:, 2]

        # Time of maximum B concentration
        t_max_B = t[np.argmax(B)]
        max_B = np.max(B)

        trajectories[f'consecutive_k1={k1}'] = {
            't': t,
            'A': A,
            'B': B,
            'C': C,
            'k1': k1,
            'k2': k2,
            't_max_B': t_max_B,
            'max_B': max_B,
            'reaction': 'A → B → C',
        }

        print(f"  k1={k1}, k2={k2}: t_max_B={t_max_B:.2f}, max_B={max_B:.3f}")

    return trajectories


# =============================================================================
# SAVE TO PRISM FORMAT
# =============================================================================

def save_to_prism_format(all_trajectories, data_dir):
    """Save all trajectories in PRISM parquet format."""
    data_dir = Path(data_dir)

    # Create directory structure
    (data_dir / 'raw').mkdir(parents=True, exist_ok=True)
    (data_dir / 'config').mkdir(parents=True, exist_ok=True)
    (data_dir / 'vector').mkdir(parents=True, exist_ok=True)

    obs_rows = []
    ind_rows = []
    base_date = datetime(2020, 1, 1)

    for traj_name, traj in all_trajectories.items():
        t = traj['t']
        n = len(t)

        # Determine which concentration variables exist
        conc_vars = []
        if 'concentration' in traj:
            conc_vars.append(('concentration', traj['concentration']))
        if 'X' in traj:
            conc_vars.append(('X', traj['X']))
            conc_vars.append(('Y', traj['Y']))
        if 'A' in traj and 'B' in traj:
            conc_vars.append(('A', traj['A']))
            conc_vars.append(('B', traj['B']))
            if 'C' in traj:
                conc_vars.append(('C', traj['C']))

        for var_name, values in conc_vars:
            indicator_id = f"{traj_name}_{var_name}"

            for i, val in enumerate(values):
                obs_rows.append({
                    'indicator_id': indicator_id,
                    'obs_date': base_date + timedelta(seconds=float(t[i])),
                    'value': float(val),
                })

            ind_rows.append({
                'indicator_id': indicator_id,
                'name': f"{traj_name} - {var_name}",
                'description': f"Concentration of {var_name}",
                'source': 'simulation',
                'domain': 'chemistry',
                'reaction': traj.get('reaction', 'unknown'),
                'order': traj.get('order'),
                'rate_constant': traj.get('rate_constant'),
                'temperature': traj.get('temperature'),
            })

    # Write parquet files
    obs_df = pl.DataFrame(obs_rows)
    obs_df.write_parquet(data_dir / 'raw' / 'observations.parquet')
    print(f"Wrote {len(obs_rows)} observations")

    ind_df = pl.DataFrame(ind_rows)
    ind_df.write_parquet(data_dir / 'raw' / 'indicators.parquet')
    print(f"Wrote {len(ind_rows)} indicators")

    # Cohort config
    pl.DataFrame([
        {'cohort_id': 'chemical_kinetics', 'name': 'Chemical Kinetics', 'domain': 'chemistry'},
    ]).write_parquet(data_dir / 'config' / 'cohorts.parquet')

    member_rows = [{'cohort_id': 'chemical_kinetics', 'indicator_id': r['indicator_id']} for r in ind_rows]
    pl.DataFrame(member_rows).write_parquet(data_dir / 'config' / 'cohort_members.parquet')

    # Summary
    summary_rows = []
    for traj_name, traj in all_trajectories.items():
        summary_rows.append({
            'trajectory': traj_name,
            'reaction': traj.get('reaction', 'unknown'),
            'order': traj.get('order'),
            'rate_constant': traj.get('rate_constant'),
            'half_life': traj.get('half_life'),
            'temperature': traj.get('temperature'),
            'is_oscillating': traj.get('is_oscillating'),
            'period': traj.get('period'),
        })

    summary_df = pl.DataFrame(summary_rows)
    summary_df.write_parquet(data_dir / 'raw' / 'trajectory_summary.parquet')

    print(f"\nTrajectory Summary:")
    print(summary_df)

    return summary_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    import sys

    data_dir = Path(__file__).parent.parent / 'data' / 'chemical_kinetics'

    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])

    print("=" * 70)
    print("CHEMICAL KINETICS DATA GENERATION")
    print("=" * 70)
    print(f"Output directory: {data_dir}")
    print()

    all_trajectories = {}

    # 1. First-order reactions at different temperatures
    print("\n[1] First-Order Reactions (A → B)")
    print("-" * 50)
    temperatures = [300, 350, 400, 450, 500]  # Kelvin
    first_order_trajs = generate_first_order_trajectories(temperatures)
    all_trajectories.update(first_order_trajs)

    # 2. Second-order reactions
    print("\n[2] Second-Order Reactions (A + A → B)")
    print("-" * 50)
    second_order_trajs = generate_second_order_trajectories(temperatures)
    all_trajectories.update(second_order_trajs)

    # 3. Oscillating reactions (Brusselator)
    print("\n[3] Oscillating Reactions (Brusselator)")
    print("-" * 50)
    # (a, b) pairs - oscillates when b > 1 + a²
    params = [
        (1.0, 1.5),   # Stable (b < 1 + a²)
        (1.0, 2.5),   # Oscillating
        (1.0, 3.0),   # Stronger oscillation
        (0.5, 2.0),   # Oscillating (b > 1 + 0.25)
    ]
    oscillating_trajs = generate_oscillating_trajectories(params)
    all_trajectories.update(oscillating_trajs)

    # 4. Consecutive reactions
    print("\n[4] Consecutive Reactions (A → B → C)")
    print("-" * 50)
    k1_values = [0.1, 0.5, 1.0, 2.0]
    consecutive_trajs = generate_consecutive_trajectories(k1_values)
    all_trajectories.update(consecutive_trajs)

    # Save
    print("\n" + "=" * 70)
    print("SAVING TO PRISM FORMAT")
    print("=" * 70)
    summary = save_to_prism_format(all_trajectories, data_dir)

    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nTotal trajectories: {len(all_trajectories)}")
    print(f"Data saved to: {data_dir}")


if __name__ == '__main__':
    main()
