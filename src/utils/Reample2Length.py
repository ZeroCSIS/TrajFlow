import pandas as pd
import numpy as np
import os
import glob
import argparse


def resample_trajectory(traj, target_length):
    """Resample a trajectory to the target length"""
    # Simple linear interpolation
    current_length = traj.shape[0]
    if current_length == target_length:
        return traj

    # Create indices for interpolation
    orig_indices = np.linspace(0, current_length - 1, current_length)
    new_indices = np.linspace(0, current_length - 1, target_length)

    # Interpolate each dimension
    resampled = np.zeros((target_length, traj.shape[1]))
    for dim in range(traj.shape[1]):
        resampled[:, dim] = np.interp(new_indices, orig_indices, traj[:, dim])

    return resampled


def resample_csv_in_folder(folder_path, target_length, output_suffix='_resampled', replace_files=False):
    """
    Resample all CSV files in the specified folder to target trajectory length

    Args:
        folder_path: Path to folder containing CSV files
        target_length: Target length for trajectory resampling
        output_suffix: Suffix to add to output files (default: '_resampled')
        replace_files: If True, replace original files instead of creating new ones
    """
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return

    for csv_file in csv_files:
        print(f"Processing: {csv_file}")

        # Read CSV
        df = pd.read_csv(csv_file)

        # Check what columns are available
        print(f"  Available columns: {df.columns.tolist()}")

        # Group by trajectory ID and resample each trajectory
        resampled_data = []
        for uid, group in df.groupby('uid'):
            # Skip if trajectory is too short
            if len(group) < 2:
                print(f"  Skipping trajectory {uid} - too short ({len(group)} points)")
                continue

            # Extract trajectory coordinates (longitude, latitude, time)
            coords_to_resample = []

            # Add longitude and latitude
            if 'longitude' in group.columns and 'latitude' in group.columns:
                coords_to_resample.extend(['longitude', 'latitude'])
            else:
                print(f"  Error: longitude/latitude columns not found")
                continue

            # Add time if it exists
            if 'time' in group.columns:
                # Convert time to numeric for interpolation
                time_numeric = pd.to_datetime(group['time']).astype(np.int64)
                traj_data = np.column_stack([
                    group['longitude'].values,
                    group['latitude'].values,
                    time_numeric.values
                ])
                coords_to_resample.append('time')
            else:
                traj_data = group[['longitude', 'latitude']].values

            # Resample trajectory
            resampled_traj = resample_trajectory(traj_data, target_length)

            # Create new dataframe for this trajectory
            new_group_data = {
                'uid': [uid] * target_length,
                'longitude': resampled_traj[:, 0],
                'latitude': resampled_traj[:, 1],
                'step': range(target_length)
            }

            # Add resampled time if it was included
            if 'time' in coords_to_resample:
                # Convert back from numeric to datetime
                new_group_data['time'] = pd.to_datetime(resampled_traj[:, 2].astype(np.int64))

            new_group = pd.DataFrame(new_group_data)

            # Copy other columns from first row if they exist
            other_cols = [col for col in group.columns if col not in ['uid', 'longitude', 'latitude', 'step', 'time']]
            for col in other_cols:
                new_group[col] = group[col].iloc[0]

            resampled_data.append(new_group)

        if not resampled_data:
            print(f"  No valid trajectories found in {csv_file}")
            continue

        # Combine all trajectories
        resampled_df = pd.concat(resampled_data, ignore_index=True)

        # Determine output path
        if replace_files:
            output_path = csv_file
        else:
            base_name = os.path.splitext(os.path.basename(csv_file))[0]
            output_path = os.path.join(folder_path, f"{base_name}{output_suffix}.csv")

        # Save resampled CSV
        resampled_df.to_csv(output_path, index=False)

        if replace_files:
            print(f"  Replaced: {output_path}")
        else:
            print(f"  Saved: {output_path}")
        print(f"  Original: {len(df)} rows, Resampled: {len(resampled_df)} rows")

def main():
    # parser = argparse.ArgumentParser(description='Resample CSV trajectories to target length')
    # parser.add_argument('folder_path', help='Path to folder containing CSV files')
    # parser.add_argument('target_length', type=int, default=120, help='Target length for trajectory resampling')
    # parser.add_argument('--suffix', default='_resampled', help='Output file suffix (default: _resampled)')
    # parser.add_argument('--replace', action='store_true', help='Replace original files instead of creating new ones')
    #
    # args = parser.parse_args()
    args = argparse.Namespace(
        folder_path='./generate_results_full_clean/04-25-12-44-58/generation_5000_real',
        target_length=120,
        suffix='_resampled',
        replace=False
    )

    if not os.path.exists(args.folder_path):
        print(f"Error: Folder {args.folder_path} does not exist")
        return

    print(f"Resampling CSV files in: {args.folder_path}")
    print(f"Target length: {args.target_length}")
    if args.replace:
        print("Mode: Replace original files")
    else:
        print(f"Mode: Create new files with suffix: {args.suffix}")

    resample_csv_in_folder(args.folder_path, args.target_length, args.suffix, args.replace)
    print("Resampling complete!")


if __name__ == "__main__":
    main()