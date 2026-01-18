"""
Double Pendulum Simulator for PRISM Validation
===============================================

Generates signal topology at multiple energy levels to test:
- Chaos transition detection (Lyapunov crosses zero)
- Energy conservation (field potential vs true energy)
- Regime detection at chaos onset

Physics:
    L = ½(m₁+m₂)l₁²θ̇₁² + ½m₂l₂²θ̇₂² + m₂l₁l₂θ̇₁θ̇₂cos(θ₁-θ₂)
        - (m₁+m₂)gl₁cosθ₁ - m₂gl₂cosθ₂

The system transitions from regular to chaotic motion as energy increases.
"""

import numpy as np
from scipy.integrate import odeint
import polars as pl
from pathlib import Path
from datetime import datetime, timedelta
import os


# Physical parameters (normalized units)
G = 9.81  # gravity
L1, L2 = 1.0, 1.0  # pendulum lengths
M1, M2 = 1.0, 1.0  # masses


def derivatives(state, t, L1, L2, M1, M2, G):
    """
    Compute derivatives for the double pendulum.

    State: [θ₁, ω₁, θ₂, ω₂] where ω = dθ/dt
    """
    theta1, omega1, theta2, omega2 = state

    delta = theta2 - theta1

    # Denominators
    den1 = (M1 + M2) * L1 - M2 * L1 * np.cos(delta) * np.cos(delta)
    den2 = (L2 / L1) * den1

    # θ₁ acceleration
    num1 = (M2 * L1 * omega1**2 * np.sin(delta) * np.cos(delta) +
            M2 * G * np.sin(theta2) * np.cos(delta) +
            M2 * L2 * omega2**2 * np.sin(delta) -
            (M1 + M2) * G * np.sin(theta1))

    domega1 = num1 / den1

    # θ₂ acceleration
    num2 = (-M2 * L2 * omega2**2 * np.sin(delta) * np.cos(delta) +
            (M1 + M2) * G * np.sin(theta1) * np.cos(delta) -
            (M1 + M2) * L1 * omega1**2 * np.sin(delta) -
            (M1 + M2) * G * np.sin(theta2))

    domega2 = num2 / den2

    return [omega1, domega1, omega2, domega2]


def compute_energy(state, L1, L2, M1, M2, G):
    """
    Compute total mechanical energy E = T + V.

    Should be conserved (constant) for the ideal system.
    """
    theta1, omega1, theta2, omega2 = state.T

    # Kinetic energy
    T = (0.5 * (M1 + M2) * L1**2 * omega1**2 +
         0.5 * M2 * L2**2 * omega2**2 +
         M2 * L1 * L2 * omega1 * omega2 * np.cos(theta1 - theta2))

    # Potential energy (relative to hanging position)
    V = (-(M1 + M2) * G * L1 * np.cos(theta1) -
         M2 * G * L2 * np.cos(theta2))

    return T + V


def compute_cartesian(state, L1, L2):
    """Convert angles to Cartesian coordinates for visualization."""
    theta1, omega1, theta2, omega2 = state.T

    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)

    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)

    return x1, y1, x2, y2


def generate_trajectory(initial_angle_deg, n_steps=10000, dt=0.01):
    """
    Generate a double pendulum trajectory.

    Args:
        initial_angle_deg: Initial angle for both pendulums (degrees)
                          Higher angles = more energy = more chaos
        n_steps: Number of time steps
        dt: Time step

    Returns:
        dict with signal topology and energy
    """
    # Convert to radians
    theta0 = np.radians(initial_angle_deg)

    # Initial state: both pendulums at angle, at rest
    state0 = [theta0, 0.0, theta0, 0.0]

    # Time array
    t = np.linspace(0, n_steps * dt, n_steps)

    # Integrate
    solution = odeint(derivatives, state0, t, args=(L1, L2, M1, M2, G))

    # Extract components
    theta1 = solution[:, 0]
    omega1 = solution[:, 1]
    theta2 = solution[:, 2]
    omega2 = solution[:, 3]

    # Compute energy (should be conserved)
    energy = compute_energy(solution, L1, L2, M1, M2, G)

    # Cartesian coordinates
    x1, y1, x2, y2 = compute_cartesian(solution, L1, L2)

    return {
        't': t,
        'theta1': theta1,
        'omega1': omega1,
        'theta2': theta2,
        'omega2': omega2,
        'x1': x1,
        'y1': y1,
        'x2': x2,
        'y2': y2,
        'energy': energy,
        'initial_angle': initial_angle_deg,
        'mean_energy': energy.mean(),
    }


def generate_multi_energy_dataset():
    """
    Generate trajectories at multiple energy levels.

    Energy levels (via initial angle):
    - 10°: Very low energy, nearly linear (regular)
    - 30°: Low energy, weakly nonlinear (regular)
    - 60°: Moderate energy, onset of chaos
    - 90°: High energy, chaotic
    - 120°: Very high energy, strongly chaotic
    - 150°: Near-maximum energy, fully chaotic
    """
    angles = [10, 30, 60, 90, 120, 150]
    regime_labels = ['regular', 'regular', 'transition', 'chaotic', 'chaotic', 'chaotic']

    trajectories = {}

    for angle, regime in zip(angles, regime_labels):
        print(f"Generating trajectory: {angle}° ({regime})")
        traj = generate_trajectory(angle, n_steps=10000, dt=0.01)
        traj['regime'] = regime
        trajectories[f'dp_{angle}deg'] = traj

    return trajectories


def save_to_prism_format(trajectories, data_dir):
    """
    Save trajectories in PRISM parquet format.
    """
    data_dir = Path(data_dir)

    # Create directory structure
    (data_dir / 'raw').mkdir(parents=True, exist_ok=True)
    (data_dir / 'config').mkdir(parents=True, exist_ok=True)
    (data_dir / 'vector').mkdir(parents=True, exist_ok=True)
    (data_dir / 'geometry').mkdir(parents=True, exist_ok=True)

    # Build observations dataframe
    obs_rows = []
    base_date = datetime(2020, 1, 1)

    for traj_name, traj in trajectories.items():
        n = len(traj['t'])
        angle = traj['initial_angle']

        # Create indicators for each variable
        variables = ['theta1', 'omega1', 'theta2', 'omega2', 'x2', 'y2', 'energy']

        for var in variables:
            indicator_id = f"{traj_name}_{var}"
            values = traj[var]

            for i, val in enumerate(values):
                obs_rows.append({
                    'indicator_id': indicator_id,
                    'obs_date': base_date + timedelta(hours=i),
                    'value': float(val),
                })

    # Create observations parquet
    obs_df = pl.DataFrame(obs_rows)
    obs_df.write_parquet(data_dir / 'raw' / 'observations.parquet')
    print(f"Wrote {len(obs_rows)} observations")

    # Create indicators metadata
    ind_rows = []
    for traj_name, traj in trajectories.items():
        angle = traj['initial_angle']
        regime = traj['regime']

        variables = ['theta1', 'omega1', 'theta2', 'omega2', 'x2', 'y2', 'energy']
        descriptions = [
            'Angle of first pendulum (rad)',
            'Angular velocity of first pendulum (rad/s)',
            'Angle of second pendulum (rad)',
            'Angular velocity of second pendulum (rad/s)',
            'X position of second mass',
            'Y position of second mass',
            'Total mechanical energy (T+V)',
        ]

        for var, desc in zip(variables, descriptions):
            ind_rows.append({
                'indicator_id': f"{traj_name}_{var}",
                'name': f"Double Pendulum {angle}° - {var}",
                'description': desc,
                'source': 'simulation',
                'domain': 'physics',
                'regime': regime,
                'initial_angle': angle,
                'energy_level': traj['mean_energy'],
            })

    ind_df = pl.DataFrame(ind_rows)
    ind_df.write_parquet(data_dir / 'raw' / 'indicators.parquet')
    print(f"Wrote {len(ind_rows)} indicators")

    # Create cohort configuration
    cohort_rows = [
        {'cohort_id': 'double_pendulum', 'name': 'Double Pendulum', 'domain': 'physics'},
    ]
    pl.DataFrame(cohort_rows).write_parquet(data_dir / 'config' / 'cohorts.parquet')

    # Create cohort members
    member_rows = [{'cohort_id': 'double_pendulum', 'indicator_id': r['indicator_id']}
                   for r in ind_rows]
    pl.DataFrame(member_rows).write_parquet(data_dir / 'config' / 'cohort_members.parquet')

    print(f"Created cohort with {len(member_rows)} members")

    # Save trajectory summary
    summary_rows = []
    for traj_name, traj in trajectories.items():
        summary_rows.append({
            'trajectory': traj_name,
            'initial_angle_deg': traj['initial_angle'],
            'regime': traj['regime'],
            'mean_energy': traj['mean_energy'],
            'energy_std': np.std(traj['energy']),
            'energy_conservation': np.std(traj['energy']) / np.abs(traj['mean_energy']),
        })

    summary_df = pl.DataFrame(summary_rows)
    summary_df.write_parquet(data_dir / 'raw' / 'trajectory_summary.parquet')

    print("\nTrajectory Summary:")
    print(summary_df)

    return summary_df


if __name__ == '__main__':
    import sys

    # Default data directory
    data_dir = Path(__file__).parent.parent / 'data' / 'double_pendulum'

    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])

    print("=" * 60)
    print("DOUBLE PENDULUM DATA GENERATION")
    print("=" * 60)
    print(f"Output directory: {data_dir}")
    print()

    # Generate trajectories
    trajectories = generate_multi_energy_dataset()

    print()

    # Save to PRISM format
    summary = save_to_prism_format(trajectories, data_dir)

    print()
    print("=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nEnergy conservation check (std/mean):")
    for row in summary.iter_rows(named=True):
        status = "✓ GOOD" if row['energy_conservation'] < 0.01 else "⚠ DRIFT"
        print(f"  {row['trajectory']}: {row['energy_conservation']:.2e} {status}")
