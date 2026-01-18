"""
The Well Dataset Fetcher for PRISM
==================================

Downloads and processes physics simulation data from PolymathicAI's "The Well"
for PRISM validation.

Data Source:
    GitHub: https://github.com/PolymathicAI/the_well
    HuggingFace: https://huggingface.co/collections/polymathic-ai/the-well
    Paper: NeurIPS 2024 Datasets and Benchmarks Track

Gray-Scott Reaction-Diffusion:
    - 6 pattern regimes (Gliders, Bubbles, Maze, Worms, Spirals, Spots)
    - 200 trajectories per regime
    - 1001 timesteps of 128x128 images
    - Two chemical species (A and B)

Physics:
    ∂A/∂t = δ_A ΔA - AB² + f(1-A)
    ∂B/∂t = δ_B ΔB + AB² - (f+k)B

PRISM Test:
    Can PRISM distinguish the 6 pattern regimes from spatially-averaged signal topology?
"""

import numpy as np
import polars as pl
from pathlib import Path
from datetime import datetime, timedelta
import h5py
from huggingface_hub import hf_hub_download
import os


def download_gray_scott_via_thewell(output_dir: Path, n_trajectories: int = 30):
    """
    Download Gray-Scott trajectories using the-well package.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'raw').mkdir(exist_ok=True)
    (output_dir / 'config').mkdir(exist_ok=True)
    (output_dir / 'vector').mkdir(exist_ok=True)

    print("Loading Gray-Scott data via the-well package...")
    print(f"Requesting {n_trajectories} trajectories total")

    try:
        from the_well.data import WellDataset
        from torch.utils.data import DataLoader

        # Load dataset streaming from HF
        dataset = WellDataset(
            well_base_path="hf://datasets/polymathic-ai/",
            well_dataset_name="gray_scott_reaction_diffusion",
            well_split_name="train",
        )

        print(f"Dataset loaded. Length: {len(dataset)}")

        # Dataset is organized: each sample is ONE timestep
        # Total: 960,000 = 1200 trajectories × 800 timesteps
        # Trajectories per regime: 200
        n_total_traj = 1200
        n_timesteps = len(dataset) // n_total_traj  # 800
        print(f"Inferred: {n_total_traj} trajectories × {n_timesteps} timesteps")

        obs_rows = []
        ind_rows = []
        base_date = datetime(2020, 1, 1)

        regime_names = ['gliders', 'bubbles', 'maze', 'worms', 'spirals', 'spots']

        # Sample evenly from each regime
        trajectories_per_regime = max(1, n_trajectories // 6)

        for regime_idx, regime in enumerate(regime_names):
            traj_start = regime_idx * 200  # First trajectory in this regime

            for traj_offset in range(trajectories_per_regime):
                traj_idx = traj_start + traj_offset
                if traj_idx >= n_total_traj:
                    break

                print(f"  Processing trajectory {traj_idx}: regime={regime}")

                # Collect all timesteps for this trajectory
                # Each sample: input_fields shape (1, 128, 128, 2) = (t, h, w, channels)
                spatial_means_A = []
                spatial_means_B = []

                start_sample = traj_idx * n_timesteps
                end_sample = start_sample + n_timesteps

                # Subsample timesteps (every 10th)
                for sample_idx in range(start_sample, end_sample, 10):
                    sample = dataset[sample_idx]
                    # input_fields: (1, 128, 128, 2) - last dim is species A, B
                    fields = sample['input_fields'].numpy()  # (1, h, w, 2)
                    spatial_means_A.append(float(fields[0, :, :, 0].mean()))
                    spatial_means_B.append(float(fields[0, :, :, 1].mean()))

                # Create indicators
                for species, values in [('A', spatial_means_A), ('B', spatial_means_B)]:
                    indicator_id = f"gs_{regime}_{traj_offset}_{species}_mean"

                    for t_idx, val in enumerate(values):
                        obs_rows.append({
                            'indicator_id': indicator_id,
                            'obs_date': base_date + timedelta(seconds=t_idx),
                            'value': val,
                        })

                    ind_rows.append({
                        'indicator_id': indicator_id,
                        'regime': regime,
                        'trajectory': traj_offset,
                        'species': species,
                        'statistic': 'spatial_mean',
                        'n_points': len(values),
                    })

        # Write data
        if obs_rows:
            obs_df = pl.DataFrame(obs_rows)
            obs_df.write_parquet(output_dir / 'raw' / 'observations.parquet')
            print(f"Wrote {len(obs_df)} observations")

            ind_df = pl.DataFrame(ind_rows)
            ind_df.write_parquet(output_dir / 'raw' / 'indicators.parquet')
            print(f"Wrote {len(ind_df)} indicators")

            # Cohort config
            pl.DataFrame([
                {'cohort_id': 'gray_scott', 'name': 'Gray-Scott Reaction-Diffusion', 'domain': 'pde'}
            ]).write_parquet(output_dir / 'config' / 'cohorts.parquet')

            pl.DataFrame([
                {'cohort_id': 'gray_scott', 'indicator_id': r['indicator_id']}
                for r in ind_rows
            ]).write_parquet(output_dir / 'config' / 'cohort_members.parquet')

            return obs_df, ind_df

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

    return None, None


def extract_spatial_statistics(data_array: np.ndarray) -> dict:
    """
    Extract spatially-averaged statistics from a 2D concentration field.

    Args:
        data_array: Shape (n_timesteps, height, width) concentration field

    Returns:
        Signal of spatial statistics
    """
    n_t = data_array.shape[0]

    stats = {
        'mean': np.zeros(n_t),
        'std': np.zeros(n_t),
        'max': np.zeros(n_t),
        'min': np.zeros(n_t),
        'gradient_mean': np.zeros(n_t),
    }

    for t in range(n_t):
        frame = data_array[t]
        stats['mean'][t] = np.mean(frame)
        stats['std'][t] = np.std(frame)
        stats['max'][t] = np.max(frame)
        stats['min'][t] = np.min(frame)

        # Spatial gradient magnitude
        grad_x = np.diff(frame, axis=0)
        grad_y = np.diff(frame, axis=1)
        grad_mag = np.sqrt(grad_x[:-1, :]**2 + grad_y[:, :-1]**2)
        stats['gradient_mean'][t] = np.mean(grad_mag)

    return stats


def process_gray_scott_hdf5(h5_path: str, output_dir: Path, max_per_regime: int = 10):
    """
    Process Gray-Scott HDF5 file and extract signal topology for PRISM.
    """
    print(f"Processing: {h5_path}")

    (output_dir / 'raw').mkdir(parents=True, exist_ok=True)
    (output_dir / 'config').mkdir(parents=True, exist_ok=True)
    (output_dir / 'vector').mkdir(parents=True, exist_ok=True)

    obs_rows = []
    ind_rows = []
    base_date = datetime(2020, 1, 1)

    with h5py.File(h5_path, 'r') as f:
        print(f"Keys in file: {list(f.keys())}")

        # The Well uses 'fields' for the data
        if 'fields' in f:
            fields = f['fields']
            print(f"Fields shape: {fields.shape}")
            # Shape is typically (n_trajectories, n_timesteps, n_channels, height, width)

            n_traj = min(fields.shape[0], max_per_regime * 6)

            for traj_idx in range(n_traj):
                # Determine regime from trajectory index (200 per regime)
                regime_idx = traj_idx // 200
                regime_names = ['gliders', 'bubbles', 'maze', 'worms', 'spirals', 'spots']
                regime = regime_names[regime_idx] if regime_idx < 6 else 'unknown'

                traj_in_regime = traj_idx % 200
                if traj_in_regime >= max_per_regime:
                    continue

                print(f"  Processing trajectory {traj_idx}: regime={regime}")

                # Get data: (n_timesteps, n_channels, height, width)
                traj_data = fields[traj_idx]
                n_timesteps = traj_data.shape[0]

                # Species A (channel 0) and B (channel 1)
                species_A = traj_data[:, 0, :, :]  # (n_t, h, w)
                species_B = traj_data[:, 1, :, :]

                # Extract spatial statistics as signal topology
                stats_A = extract_spatial_statistics(species_A)
                stats_B = extract_spatial_statistics(species_B)

                # Create indicator for each statistic
                for species, stats in [('A', stats_A), ('B', stats_B)]:
                    for stat_name, values in stats.items():
                        indicator_id = f"gs_{regime}_{traj_in_regime}_{species}_{stat_name}"

                        # Subsample if too many timesteps
                        step = max(1, len(values) // 500)
                        for i, val in enumerate(values[::step]):
                            obs_rows.append({
                                'indicator_id': indicator_id,
                                'obs_date': base_date + timedelta(seconds=i),
                                'value': float(val),
                            })

                        ind_rows.append({
                            'indicator_id': indicator_id,
                            'regime': regime,
                            'trajectory': traj_in_regime,
                            'species': species,
                            'statistic': stat_name,
                            'n_points': len(values[::step]),
                        })
        else:
            # Try alternative structure
            print(f"Available keys: {list(f.keys())}")
            for key in f.keys():
                print(f"  {key}: {type(f[key])}")
                if hasattr(f[key], 'shape'):
                    print(f"    Shape: {f[key].shape}")

    if obs_rows:
        # Write parquet files
        obs_df = pl.DataFrame(obs_rows)
        obs_df.write_parquet(output_dir / 'raw' / 'observations.parquet')
        print(f"Wrote {len(obs_df)} observations")

        ind_df = pl.DataFrame(ind_rows)
        ind_df.write_parquet(output_dir / 'raw' / 'indicators.parquet')
        print(f"Wrote {len(ind_df)} indicators")

        # Cohort config
        pl.DataFrame([
            {'cohort_id': 'gray_scott', 'name': 'Gray-Scott Reaction-Diffusion', 'domain': 'pde'}
        ]).write_parquet(output_dir / 'config' / 'cohorts.parquet')

        pl.DataFrame([
            {'cohort_id': 'gray_scott', 'indicator_id': r['indicator_id']}
            for r in ind_rows
        ]).write_parquet(output_dir / 'config' / 'cohort_members.parquet')

        # Summary
        print("\n" + "="*60)
        print("Gray-Scott Data Summary")
        print("="*60)
        regime_counts = ind_df.group_by('regime').len()
        print(regime_counts)

        return obs_df, ind_df

    return None, None


def main():
    import sys

    output_dir = Path('/Users/jasonrudder/prism-mac/data/the_well')
    n_trajectories = 30  # 5 per regime × 6 regimes

    if len(sys.argv) > 1:
        n_trajectories = int(sys.argv[1])

    print("="*60)
    print("The Well Dataset Fetcher for PRISM")
    print("="*60)
    print(f"Output: {output_dir}")
    print(f"Trajectories: {n_trajectories}")
    print()

    # Use the-well package to download/stream data
    obs_df, ind_df = download_gray_scott_via_thewell(output_dir, n_trajectories)

    if obs_df is not None:
        print("\n" + "="*60)
        print("Download complete!")
        print("="*60)
        print(f"Observations: {len(obs_df)}")
        print(f"Indicators: {len(ind_df)}")

        # Summary by regime
        print("\nRegime summary:")
        print(ind_df.group_by('regime').len())
    else:
        print("\nFailed to download data.")
        print("Try: pip install the-well --upgrade")


if __name__ == '__main__':
    main()
