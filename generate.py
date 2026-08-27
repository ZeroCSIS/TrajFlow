import argparse
import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.transforms import para2point_batch
from src.eval.metrics import compute_baseline_metrics
from src.models.networks import (
    CNN,
    MLP,
    BiLSTMVelocity,
    ConditionalVelocityModel,
    TransformerVelocity,
)
from src.models.trajectory_gan import ConditionalTrajectoryGAN, TrajectoryGAN
from src.models.trajectory_vae import ConditionalTrajectoryVAE, TrajectoryVAE
from src.utils.reproducibility import seed_everything
from src.utils.visualization import visualize_density_comparison, visualize_trajectories

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.dataset import FlowMatchingDataset
from src.eval.inference import FlowMatchingInference

# Device is resolved in main() from CLI/config.
device = torch.device("cpu")


# Field name mapping
switcher = {
    0: 'departure_time',
    1: 'total_dis',
    2: 'total_time',
    3: 'total_len',
    4: 'avg_dis',
    5: 'avg_speed',
    6: 'starting_location',
    7: 'ending_location'
}


def resolve_device(device_str):
    """Resolve requested device safely with fallback to available CUDA/CPU."""
    if not torch.cuda.is_available():
        return torch.device("cpu")

    requested = device_str or "cuda:0"
    if requested == "cpu":
        return torch.device("cpu")

    if requested.startswith("cuda:"):
        try:
            idx = int(requested.split(":", 1)[1])
        except (IndexError, ValueError):
            idx = 0
        if idx >= torch.cuda.device_count():
            print(f"Requested device {requested} not available; fallback to cuda:0")
            return torch.device("cuda:0")
        return torch.device(requested)

    if requested == "cuda":
        return torch.device("cuda:0")

    # Unknown string: fall back to CPU for safety.
    print(f"Unknown device '{requested}'; fallback to cpu")
    return torch.device("cpu")


def model_input_dim(config):
    """Return the same trajectory representation width used by train.py."""
    if config['data']['parametrized']:
        points = config['data']['parametrized_M']
    else:
        points = config['data']['trajectory_length']
    return int(points) * 2


def find_config_by_timestamp(exp_savename_str):
    """Find config file by timestamp in folder name"""
    # Search standard output root used in the open-source package.
    base_paths = ['./outputs']
    matching_folders = []

    for base_path in base_paths:
        for root, dirs, files in os.walk(base_path):
            for folder in dirs:
                if exp_savename_str in folder:
                    matching_folders.append(os.path.join(root, folder))
    if not matching_folders:
        return None, None, None
    config_path = os.path.join(matching_folders[0], 'config.yaml')

    model_path = os.path.join(os.path.dirname(matching_folders[0]), 'models', os.path.basename(matching_folders[0]))

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return matching_folders[0], model_path, config

    return None, None, None

def generate_trajectories(model, all_gt_data, all_head, lengths, traj_mean, traj_std,
                          cond_mean, cond_std, config, batch_size, num_batches,
                          result_dir, dataset, condition_mode="real", generate_num=10, save_dir='./test_output'):
    """Generate trajectories using flow matching model"""
    os.makedirs(save_dir, exist_ok=True)

    # IF SAVE TRAJECTORIES BEFORE DENORMALIZATION
    SAVE_RAW_TRAJS = True
    if SAVE_RAW_TRAJS:
        all_raw_sol_np = []
        all_raw_ground_truth_np = []

    # Setup dataloader with indices - use passed batch_size parameter for memory efficiency
    # batch_size parameter allows us to use smaller batches during inference than training
    if config['data']['parametrized'] == True:
        M = config['data']['parametrized_M']
    else:
        M = config['data']['trajectory_length']
    traj_length = config['data']['trajectory_length']

    # Initialize collectors
    all_sol_np = []
    all_ground_truth_np = []
    all_indices = []
    all_conditions = []
    raw_gt_trajs = []  # For storing raw ground truth trajectories

    indices = list(range(len(dataset)))
    random_state = np.random.RandomState(int(config.get('project', {}).get('seed', 42)))
    selected_indices = random_state.choice(indices, size=min(generate_num, len(indices)), replace=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=selected_indices)

    # Initialize inference model
    inference = FlowMatchingInference(config=config,
                                      model=model,
                                      dataset=dataset,
                                      save_dir=save_dir,
                                      device=device)

    # Use DDPM sampling steps for DDPM checkpoints, otherwise use inference settings.
    if config.get('ddpm', {}).get('enabled', False):
        n_steps = config['ddpm']['ddim_steps']
    else:
        n_steps = config['inference']['num_steps']
    solve_method = config['inference']['sampling_method']

    # Check if od_finer is enabled
    od_finer_enabled = config['data'].get('od_finer', False)

    # Process all batches
    for batch_idx, batch_data in enumerate(dataloader):
# ======================================================For Generate===================================================#
        # Handle conditional vs unconditional data
        if config['condition']['enabled']:
            x_1_sample, condition_sample = batch_data
            x_1_sample = x_1_sample.to(device)
            # change the condition as the the condition_mode
            transmode_switcher = {'WALK': 0, 'CAR': 1, 'BUS': 2, 'TRAIN': 3, 'BIKE': 4}
            if condition_mode == "real":
                pass
            else:
                condition_sample[:, 8] = transmode_switcher[condition_mode]
            condition_sample = condition_sample.to(device)
            current_batch_size = x_1_sample.size(0)

            # Calculate correct indices for this batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + current_batch_size, len(indices))
            batch_indices = selected_indices[start_idx:end_idx].tolist()
            all_indices.extend(batch_indices)

            # Extract raw ground truth for these indices
            for idx in batch_indices:
                if idx < len(all_gt_data):
                    raw_traj = np.array(all_gt_data[idx], copy=True)
                    if len(raw_traj) != M:
                        raw_traj = resample_trajectory(raw_traj, config['data']['trajectory_length'])
                    if not dataset.grid_metadata.get('coordinates_are_raw', False):
                        for k in range(2):
                            raw_traj[:, k] = raw_traj[:, k] * traj_std[k] + traj_mean[k]
                    raw_gt_trajs.append(raw_traj)

            print(
                f"Processing batch {batch_idx + 1}/{(generate_num + batch_size - 1) // batch_size}, size: {current_batch_size}")

            # Sample with condition using full input
            sol = inference.sample(n_samples=current_batch_size, n_steps=n_steps,
                                   method=solve_method, condition=condition_sample)

            # Clear GPU cache after sampling to free memory
            torch.cuda.empty_cache()
        else:
            x_1_sample = batch_data.to(device)
            condition_sample = None
            current_batch_size = x_1_sample.size(0)
            print(
                f"Processing batch {batch_idx + 1}/{(generate_num + batch_size - 1) // batch_size}, size: {current_batch_size}")
            # Sample without condition
            sol = inference.sample(n_samples=current_batch_size, n_steps=n_steps, method=solve_method)

            # Clear GPU cache after sampling to free memory
            torch.cuda.empty_cache()

        # Convert to numpy for visualization
        sol_np = [s.detach().cpu().numpy() for s in sol][-1]
        ground_truth_np = x_1_sample.detach().cpu().numpy()
#=======================================================================================================================#
        # Handle the parameterized trajectories
        if od_finer_enabled:
            # Extract trajectory segments and OD finer parameters
            traj_dim = M * 2

            # For generated trajectories
            sol_traj = sol_np[:, :traj_dim]
            sol_od_finer = sol_np[:, traj_dim:]  # Use GT OD finer parameters

            # For ground truth
            gt_traj = ground_truth_np[:, :traj_dim]
            gt_od_finer = ground_truth_np[:, traj_dim:]

            if config['data']['parametrized']:
                para_method = config['data'].get('para_method', 'rdp_k')
                print("Converting parameterized representation to point sequence...")
                # Convert only trajectory points
                sol_traj = para2point_batch(sol_traj, traj_length, para_method)
                gt_traj = para2point_batch(gt_traj, traj_length, para_method)
            else:
                pass

            # Store OD finer parameters for denormalization
            sol_np = sol_traj
            ground_truth_np = gt_traj
            od_finer_params_sol = sol_od_finer
            od_finer_params_gt = gt_od_finer
        else:
            if config['data']['parametrized']:
                para_method = config['data'].get('para_method', 'rdp_k')
                print("Converting parameterized representation to point sequence...")
                # Convert only trajectory points
                sol_np = para2point_batch(sol_np, traj_length, para_method)
                ground_truth_np = para2point_batch(ground_truth_np, traj_length, para_method)
            else:
                pass
            od_finer_params_sol = None
            od_finer_params_gt = None

        if SAVE_RAW_TRAJS and config['data']['norm1by1']:
            raw_sol_np = np.reshape(sol_np, (sol_np.shape[0], -1))
            raw_ground_truth_np = np.reshape(ground_truth_np, (ground_truth_np.shape[0], -1))

        # Denormalize if needed
        if (config['condition']['enabled']
                and config['data']['norm1by1']
                and config['visualization'].get('norm1by12origialvis', False)):
            condition_sample_np = condition_sample.detach().cpu().numpy()
            print(f"Denormalizing data for batch {batch_idx + 1}...")

            if od_finer_enabled:
                sol_np = inference.denormalize_trajectories([sol_np], condition_sample_np,
                                                            dataset, od_finer_params=od_finer_params_sol)[0]
                ground_truth_np = inference.denormalize_trajectories([ground_truth_np], condition_sample_np,
                                                                     dataset, od_finer_params=od_finer_params_gt)[0]
            else:
                sol_np = inference.denormalize_trajectories([sol_np], condition_sample_np, dataset)[0]
                ground_truth_np = \
                inference.denormalize_trajectories([ground_truth_np], condition_sample_np, dataset)[0]
        else:
            sol_np = np.reshape(sol_np, (sol_np.shape[0], traj_length, 2))
            ground_truth_np = np.reshape(ground_truth_np, (ground_truth_np.shape[0], traj_length, 2))
            #  multiply by std and add mean
            for k in range(2):
                sol_np[:, :, k] = sol_np[:, :, k] * traj_std[k] + traj_mean[k]
                ground_truth_np[:, :, k] = ground_truth_np[:, :, k] * traj_std[k] + traj_mean[k]

        # Collect results
        all_sol_np.extend(sol_np)
        all_ground_truth_np.extend(ground_truth_np)
        if SAVE_RAW_TRAJS and config['data']['norm1by1']:
            all_raw_sol_np.extend(raw_sol_np)
            all_raw_ground_truth_np.extend(raw_ground_truth_np)

        # Clean up batch tensors to free memory
        del sol, x_1_sample
        if config['condition']['enabled']:
            del condition_sample
        torch.cuda.empty_cache()

    # Extract corresponding condition info if available
    total_cond_info = []
    if config['condition']['enabled'] and all_indices:
        for idx in all_indices:
            if idx < len(all_head):
                total_cond_info.append(all_head[idx])

    # Convert final sample (last timestep) trajectories to the expected format
    total_gen_trajs = [x.reshape(traj_length, 2) for x in all_sol_np]
    total_gt_trajs = [x.reshape(traj_length, 2) for x in all_ground_truth_np]
    if SAVE_RAW_TRAJS:
        total_raw_gt_trajs = [x.reshape(traj_length, 2) for x in all_raw_ground_truth_np]
        total_raw_gen_trajs = [x.reshape(traj_length, 2) for x in all_raw_sol_np]

    # Visualize the results using imported visualization functions
    visualize_trajectories(total_gen_trajs, total_gt_trajs,
                           config['data']['trajectory_length'],
                           parametrized=False, save_folder=save_dir)
    visualize_density_comparison(total_gen_trajs, total_gt_trajs,
                                 config['data']['trajectory_length'],
                                 save_dir)

    # Save trajectories separately
    gen_df = save_trajectories_to_csv(
        trajs=total_gen_trajs,
        cond_info=total_cond_info,
        cond_std=dataset.cond_std,
        cond_mean=dataset.cond_mean,
        save_dir=save_dir,
        traj_type="generated"
    )

    gt_df = save_trajectories_to_csv(
        trajs=total_gt_trajs,
        cond_info=total_cond_info,
        cond_std=dataset.cond_std,
        cond_mean=dataset.cond_mean,
        save_dir=save_dir,
        traj_type="ground_truth"
    )

    # Also save raw ground truth if available
    if raw_gt_trajs:
        raw_gt_df = save_trajectories_to_csv(
            trajs=raw_gt_trajs,
            cond_info=total_cond_info,
            cond_std=dataset.cond_std,
            cond_mean=dataset.cond_mean,
            save_dir=save_dir,
            traj_type="raw_ground_truth"
        )

    if SAVE_RAW_TRAJS:
        # Save raw generated trajectories to CSV
        total_raw_gen_df = save_trajectories_to_csv(
            trajs=total_raw_gen_trajs,
            cond_info=total_cond_info,
            cond_std=dataset.cond_std,
            cond_mean=dataset.cond_mean,
            save_dir=save_dir,
            traj_type="generated_before_denormalization"
        )
        # Save raw ground truth trajectories to CSV
        total_raw_gt_df = save_trajectories_to_csv(
            trajs=total_raw_gt_trajs,
            cond_info=total_cond_info,
            cond_std=dataset.cond_std,
            cond_mean=dataset.cond_mean,
            save_dir=save_dir,
            traj_type="ground_truth_before_denormalization"
        )

    evaluation_config = config.get('evaluation', {})
    if evaluation_config.get('enabled', False):
        metrics = compute_baseline_metrics(
            np.asarray(total_gen_trajs),
            np.asarray(total_gt_trajs),
            grid_metadata=dataset.grid_metadata,
            max_pairs=int(evaluation_config.get('max_curve_pairs', 200)),
        )
        metrics_path = os.path.join(save_dir, 'baseline_metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as stream:
            json.dump(metrics, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        print(f"Baseline metrics saved to {metrics_path}")

    return total_gen_trajs, total_gt_trajs, total_cond_info

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


def save_trajectories_to_csv(trajs, cond_info, cond_std, cond_mean, save_dir, traj_type="generated"):
    """Save trajectories to CSV file, processing one type at a time

    Args:
        trajs: Generated or ground truth trajectories
        cond_info: Condition information
        cond_std: Standard deviation for condition normalization
        cond_mean: Mean for condition normalization
        save_dir: Directory to save the CSV files
        traj_type: Either "generated" or "ground_truth"
    """
    cond_info_list = []
    cond_info = np.array(cond_info)

    # Convert conditions to expected format
    for i in range(cond_info.shape[0]):
        # First column is departure_time, columns 1-5 need denormalization
        head_vals = []
        head_vals.append(cond_info[i, 0])
        for j in range(1, 6):
            head_vals.append(cond_info[i, j] * cond_std[j - 1] + cond_mean[j - 1])

        # Combine into dictionary
        temp_dict = {}
        # Convert departure_time numeric value to time string
        temp_time = head_vals[0] * 0.0833333
        hour = int(temp_time)
        minute = int((temp_time - hour) * 60)
        second = int(((temp_time - hour) * 60 - minute) * 60)
        temp_dict['departure_time'] = f"2024-04-01 {hour:02d}:{minute:02d}:{second:02d}"

        for j in range(1, 6):
            temp_dict[switcher[j]] = head_vals[j]
        cond_info_list.append(temp_dict)

    rows = []
    # Convert trajectories to expected format
    for i, (traj, cond) in enumerate(zip(trajs, cond_info_list)):
        for j in range(len(traj)):
            rows.append({
                "uid": i,
                "departure_time": cond["departure_time"],
                "total_dis": cond["total_dis"],
                "total_time": cond["total_time"],
                "total_len": cond["total_len"],
                "avg_dis": cond["avg_dis"],
                "avg_speed": cond["avg_speed"],
                "time": pd.to_datetime(cond["departure_time"]) + pd.Timedelta(minutes=j),
                "latitude": traj[j, 0],
                "longitude": traj[j, 1]
            })

    df = pd.DataFrame(rows)

    df.to_csv(os.path.join(save_dir, f"{traj_type}_trajectories.csv"), index=False)

    return df

def visualize_flow_field(model, config, save_dir, device):
    """Visualize the vector field of the flow model"""
    # Set up grid for visualization
    grid_size = 20
    x_range = (-3, 3)
    y_range = (-3, 3)
    x_grid = np.linspace(x_range[0], x_range[1], grid_size)
    y_grid = np.linspace(y_range[0], y_range[1], grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Create dummy condition
    if hasattr(config, 'data'):
        traj_length = config.data.trajectory_length
    else:
        traj_length = 30  # Default value

    dummy_condition = torch.zeros(1, 8).to(device)  # Adjust size based on your condition dim

    # Visualize vector field at different timesteps
    timesteps = [0.0, 0.25, 0.5, 0.75, 0.99]

    for t_val in timesteps:
        plt.figure(figsize=(10, 8))

        # Initialize velocity field
        U = np.zeros((grid_size, grid_size))
        V = np.zeros((grid_size, grid_size))

        # Compute vector field at each grid point
        with torch.no_grad():
            for i in range(grid_size):
                for j in range(grid_size):
                    # Create a single point trajectory
                    point = torch.zeros(1, 2, traj_length).to(device)
                    point[0, 0, :] = Y[i, j]  # Set y-value
                    point[0, 1, :] = X[i, j]  # Set x-value

                    # Get velocity at this point
                    t = torch.tensor([t_val]).to(device)
                    velocity = model(point, t, c=dummy_condition)

                    # Extract velocity vector (average across trajectory)
                    U[i, j] = velocity[0, 1, 0].cpu().item()  # x-velocity
                    V[i, j] = velocity[0, 0, 0].cpu().item()  # y-velocity

        # Normalize velocities for better visualization
        magnitude = np.sqrt(U ** 2 + V ** 2)
        max_mag = np.max(magnitude) if np.max(magnitude) > 0 else 1
        U = U / max_mag
        V = V / max_mag

        # Plot the vector field
        plt.quiver(X, Y, U, V, magnitude, cmap='viridis', scale=25)
        plt.colorbar(label='Velocity Magnitude')
        plt.title(f"Flow Vector Field at t={t_val:.2f}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig(os.path.join(save_dir, f"vector_field_t{t_val:.2f}.png"))
        plt.close()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Flow Matching Trajectory Generation")
    parser.add_argument("--exp_savename_str", type=str, default="run_20251117_014039", help="Timestamp for model selection")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cuda:0 or cpu")
    parser.add_argument("--num_batches", type=int, default=10, help="Number of batches to generate")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size for generation (default 32 for memory efficiency, reduce to 16 or 8 if OOM occurs)")
    parser.add_argument("--generate_num", type=int, default=1000, help="Total num for generation")
    parser.add_argument("--placeholder", type=str, default='real', help="Use placeholder conditions")
    parser.add_argument("--USE_GIVEN_STEPS", type=bool, default=False, help="If using the number of given sampling steps parameters")
    parser.add_argument("--steps", type=int, default=10, help="Number of sampling steps")
    parser.add_argument("--method", type=str, default="euler", choices=["euler", "em", "rk4"],
                        help="Integration method: euler, euler-maruyama, or runge-kutta4")
    parser.add_argument("--generate_results_dir", type=str, default="generate_results_baseline_full_clean", help="generate results directory")
    args = parser.parse_args()

    # args.exp_savename_str can be used to select a run folder by timestamp.
    args.output_dir = './%s/%s'%(args.generate_results_dir,args.exp_savename_str)
    # Condition mode based on argument
    condition_mode = args.placeholder
    print(f"Using {condition_mode} conditions for generation")

    # Get config and model path
    if args.config is None:
        config_dir, model_dir, config_dict = find_config_by_timestamp(args.exp_savename_str)
        if config_dict is None:
            print(f"Could not find config with timestamp {args.exp_savename_str}")
            return
        config = config_dict
    else:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = config_dict
        model_dir = os.path.dirname(args.config)

    seed = int(config.get('project', {}).get('seed', 42))
    seed_everything(
        seed,
        deterministic=bool(config.get('project', {}).get('deterministic', True)),
    )

    global device
    configured_device = config.get("training", {}).get("device", "cuda:0")
    device = resolve_device(args.device or configured_device)
    print(f"Using device: {device}")

    # Respect CLI method override for flow matching sampling.
    method_map = {
        "euler": "em",
        "em": "em",
        "rk4": "em",  # kept for backward CLI compatibility
    }
    if args.method in method_map:
        effective_method = method_map[args.method]
        config.setdefault("inference", {})
        config["inference"]["sampling_method"] = effective_method
        if args.method != effective_method:
            print(f"Method '{args.method}' is mapped to '{effective_method}' in this release.")
    else:
        effective_method = config.get("inference", {}).get("sampling_method", "em")

    # Find best model if checkpoint not specified
    if args.checkpoint is None:
        # get the best model named best_model.pt
        checkpoint_path = os.path.join(model_dir, 'best_model.pt')
        print(f"Using model checkpoint: {checkpoint_path}")
    else:
        # Support three forms:
        # 1) full file path to a .pt checkpoint
        # 2) filename inside model_dir (e.g., best_model.pt)
        # 3) epoch token (legacy): checkpoint_epoch_<token>.pt
        if os.path.isfile(args.checkpoint):
            checkpoint_path = args.checkpoint
        else:
            candidate_file = os.path.join(model_dir, args.checkpoint)
            candidate_epoch = os.path.join(model_dir, f'checkpoint_epoch_{args.checkpoint}.pt')
            if os.path.isfile(candidate_file):
                checkpoint_path = candidate_file
            elif os.path.isfile(candidate_epoch):
                checkpoint_path = candidate_epoch
            else:
                print(f"Checkpoint not found: {args.checkpoint}")
                return -1

    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = os.path.join(args.output_dir, f"generation_{args.generate_num}_{condition_mode}_test")
    if args.USE_GIVEN_STEPS:
        result_dir += f"_steps_{args.steps}"
    else:
        pass
    os.makedirs(result_dir, exist_ok=True)

    # Save config for reference
    print('!!!args.USE_GIVEN_STEPS %s' % args.USE_GIVEN_STEPS)
    if args.USE_GIVEN_STEPS:
        config_dict['ddpm']['ddim_steps'] = args.steps
        # config_dict['inference']['num_steps'] = args.steps
        print(f"Using given sampling steps: {args.steps}")
    else:
        if config_dict.get('ddpm', {}).get('enabled', False):
            configured_steps = config_dict['ddpm']['ddim_steps']
        else:
            configured_steps = config_dict['inference']['num_steps']
        print(f"Using config sampling steps: {configured_steps}")

    with open(os.path.join(result_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config_dict, f)

    # Evaluation uses the test split selected by FlowMatchingDataset. Reusing
    # the dataset-owned arrays prevents accidental indexing into the full file.
    dataset = FlowMatchingDataset(config_dict, mode='test')
    all_head = dataset.all_head
    traj_mean = dataset.traj_mean
    traj_std = dataset.traj_std
    lengths = dataset.lengths
    cond_mean = dataset.cond_mean
    cond_std = dataset.cond_std
    all_gt_data = getattr(dataset, 'traj_segments_before_stdize', dataset.traj_segments)

    # Create model
    input_dim = model_input_dim(config)
    hidden_dim = config['model']['hidden_dim']
    condition_dim = dataset.location_dim

    print(f"Model parameters: input_dim={input_dim}, hidden_dim={hidden_dim}, condition_dim={condition_dim}")

    # Check if this is a baseline model from config
    if config.get('baseline', {}).get('enabled', True):
        model_type = config.get('baseline', {}).get('type', 'flow_matching')
    else:
        model_type = config.get('model', {}).get('type', 'error')

    if model_type == 'gan':
        print("Creating baseline GAN model...")
        # Create baseline GAN model using existing classes
        if config['condition']['enabled']:
            model = ConditionalTrajectoryGAN(config, dataset)
            print("Created conditional baseline GAN model")
        else:
            model = TrajectoryGAN(config)
            print("Created baseline GAN model")

        # Load checkpoint for baseline
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
        print("Baseline GAN model loaded successfully")

    elif model_type == 'vae':
        print("Creating baseline VAE model...")
        # Import VAE model
        # Create baseline VAE model using existing classes
        if config['condition']['enabled']:
            model = ConditionalTrajectoryVAE(config, dataset)
            print("Created conditional baseline VAE model")
        else:
            model = TrajectoryVAE(config)
            print("Created baseline VAE model")
        # Load checkpoint for baseline
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        print("Baseline VAE model loaded successfully")
    else:
        print("Creating flow matching model...")
        # Existing flow matching model creation code
        if config['model']['type'] == 'mlp' or config['model']['type'] == 'unet':
            if config['condition']['enabled']:
                model = ConditionalVelocityModel(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    condition_dim=condition_dim,
                    embedding_dim=hidden_dim,
                    dropout_prob=config['flow_matching']['dropout_prob'],
                    config=config,
                    dataset=dataset
                )
            else:
                model = MLP(input_dim=input_dim, hidden_dim=hidden_dim)
        elif config['model']['type'] == 'cnn':
            model = CNN(input_dim=2, hidden_dim=hidden_dim)
        elif config['model']['type'] == 'transformer':
            model = TransformerVelocity(input_dim=2, hidden_dim=hidden_dim)
        elif config['model']['type'] == 'bilstm':
            model = BiLSTMVelocity(input_dim=2, hidden_dim=hidden_dim)
        else:
            raise ValueError(f"Unknown model type: {config['model']['type']}")

        # Load checkpoint for flow matching
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
        print("Flow matching model loaded successfully")

    if config.get('training', {}).get('data_parallel', False):
        if device.type == 'cuda' and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).to(device)
            model.eval()
            print(f"Using DataParallel across {torch.cuda.device_count()} visible GPUs")
        else:
            print("data_parallel requested, but fewer than two CUDA devices are visible")

    # Print memory usage information
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        print(f"Inference batch size: {args.batch_size} (reduce if OOM occurs)")

    # Load given parameters if specified
    # Get sampling parameters from config
    if args.USE_GIVEN_STEPS:
        config['inference']['num_steps'] = args.steps
        config['ddpm']['ddim_steps'] = args.steps
        print(f"Using given sampling steps: {args.steps}")
    else:
        pass
        print(f"Using config sampling steps")

    # Generate trajectories
    gen_trajs, gt_trajs, cond_info = generate_trajectories(
        model=model,
        all_gt_data=all_gt_data,
        all_head=all_head,
        lengths=lengths,
        traj_mean=traj_mean,
        traj_std=traj_std,
        cond_mean=cond_mean,
        cond_std=cond_std,
        config=config,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        result_dir=result_dir,
        dataset=dataset,
        condition_mode=condition_mode,
        generate_num = args.generate_num,
        save_dir=result_dir
    )

    print(f"Generation complete. Results saved to {result_dir}")
    print(f"Generated {len(gen_trajs)} trajectories using {args.steps} steps with {effective_method} method")
if __name__ == "__main__":
    main()
