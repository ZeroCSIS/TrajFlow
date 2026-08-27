import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.data.transforms import para2point

def visualize_velocity_field(model, device, save_folder, resolution=20, range_min=-3, range_max=3):
    """
    Visualize the velocity field of the model

    Args:
        model: Model to visualize
        device: Device to run model on
        save_folder: Folder to save visualization
        resolution: Number of points per dimension
        range_min: Minimum value of grid
        range_max: Maximum value of grid
    """
    # Create a grid
    x = np.linspace(range_min, range_max, resolution)
    y = np.linspace(range_min, range_max, resolution)
    X, Y = np.meshgrid(x, y)

    # Evaluate model on grid
    positions = np.stack([X.flatten(), Y.flatten()], axis=1)
    positions_tensor = torch.tensor(positions, dtype=torch.float32).to(device)

    # Get velocity at different time steps
    plt.figure(figsize=(15, 10))

    for t_idx, t_val in enumerate([0.0, 0.33, 0.66, 1.0]):
        t = torch.ones(len(positions_tensor)).to(device) * t_val
        with torch.no_grad():
            velocity = model(positions_tensor, t).cpu().numpy()

        vx = velocity[:, 0].reshape(resolution, resolution)
        vy = velocity[:, 1].reshape(resolution, resolution)

        plt.subplot(2, 2, t_idx+1)
        plt.quiver(X, Y, vx, vy, scale=25)
        plt.title(f"Velocity Field at t={t_val:.2f}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(range_min, range_max)
        plt.ylim(range_min, range_max)
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'velocity_field.png'), dpi=300)
    plt.close()

def visualize_density_comparison(sol, ground_truth, M, save_folder,**kwargs):
    """
    Visualize density comparison between generated samples and ground truth

    Parameters:
    - sol: Generated samples from flow matching, shape [timesteps, n_samples, n_dim]
      Where n_dim = M*2 (flattened DCT coefficients)
    - ground_truth: Ground truth DCT coefficients, shape [n_samples, n_dim]
    - M: Number of points per trajectory
    - save_folder: Directory to save visualization results
    """
    if isinstance(sol, list):
        sample_count = len(sol)
        sol = np.array(sol)
    elif isinstance(sol, np.ndarray):
        sample_count = sol.shape[0]
    else:
        raise ValueError("Invalid input type for sol")
    if isinstance(ground_truth, list):
        ground_truth = np.array(ground_truth)
    else:
        pass
    # get trajectory length from **kwargs, if not provided, use default 120
    traj_length = kwargs.get('traj_length', 120)

    if 'parametrized' in kwargs:
        parametrized = kwargs['parametrized']
    else:
        parametrized = False

    # Convert parametric representation back to points
    # IMPORTANT: Reshape the flattened coefficients to [M, 2] before passing to para2point
    if parametrized:
        generated_trajectories = np.array([
            para2point(sol[i].reshape(M, 2), N_new=traj_length, method='dct')
            for i in range(sample_count)
        ])  # Shape: [sample_count, M, 2]

        gt_trajectories = np.array([
            para2point(ground_truth[i].reshape(M, 2), N_new=traj_length, method='dct')
            for i in range(sample_count)
        ])  # Shape: [sample_count, M, 2]
    else:
        generated_trajectories = sol.reshape(sample_count, M, 2)
        gt_trajectories = ground_truth.reshape(sample_count, M, 2)

    # Flatten all points for density visualization
    gen_points = generated_trajectories.reshape(-1, 2)  # Shape: [sample_count*M, 2]
    gt_points = gt_trajectories.reshape(-1, 2)  # Shape: [sample_count*M, 2]

    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Calculate plot range
    y_min = min(gt_points[:, 0].min(), gen_points[:, 0].min()) - 1
    y_max = max(gt_points[:, 0].max(), gen_points[:, 0].max()) + 1
    x_min = min(gt_points[:, 1].min(), gen_points[:, 1].min()) - 1
    x_max = max(gt_points[:, 1].max(), gen_points[:, 1].max()) + 1

    # Plot density for ground truth
    h1 = ax1.hist2d(gt_points[:, 1], gt_points[:, 0], bins=100,
                   range=((x_min, x_max), (y_min, y_max)),
                   cmap='Blues', density=True)
    ax1.set_title("Ground Truth Density")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.grid(True, alpha=0.3)
    plt.colorbar(h1[3], ax=ax1, label='Density')

    # Plot density for generated samples
    h2 = ax2.hist2d(gen_points[:, 1], gen_points[:, 0], bins=100,
                   range=((x_min, x_max), (y_min, y_max)),
                   cmap='Reds', density=True)
    ax2.set_title("Generated Density")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.grid(True, alpha=0.3)
    plt.colorbar(h2[3], ax=ax2, label='Density')

    # Save plot
    plt.tight_layout()
    plt.savefig(f"{save_folder}/density_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def visualize_trajectories(sol, ground_truth, M, parametrized,save_folder,**kwargs):
    """
    Visualize generated trajectories and ground truth trajectories.

    Parameters:
    - sol: Generated samples (flattened DCT coefficients), shape [n_samples, n_samples, n_dim]
    - ground_truth: Ground truth flattened DCT coefficients, shape [n_samples, n_dim]
    - M: Number of points per trajectory
    - save_folder: Directory to save visualization results
    - parametrized: Whether the data is parametrized or not
    """
    # Get the final generated samples (last timestep)
    # Accept either a list of samples or a NumPy array.
    if isinstance(sol, list):
        sample_count = len(sol)
        sol = np.array(sol)
    elif isinstance(sol, np.ndarray):
        sample_count = sol.shape[0]
    else:
        raise ValueError("Invalid input type for sol")
    if isinstance(ground_truth, list):
        ground_truth = np.array(ground_truth)
    else:
        pass
    # get trajectory length from **kwargs, if not provided, use default 120
    traj_length = kwargs.get('traj_length', 120)
    representation_note = kwargs.get('representation_note')
    point_label = f"{M} plotted points"
    if representation_note:
        point_label += f"; {representation_note}"

    max_trajs_to_plot = min(300, sample_count)

    # Convert parametric representation back to points
    print("Converting parametric representations to points...")
    if parametrized:
        para_method = 'rdp_k'  # Default method
        generated_trajectories = []
        for i in range(sample_count):
            sample_data = sol[i].reshape(M, 2)
            # Build the parameter dictionary expected by the reconstruction helper.
            if para_method == 'rdp_k':
                para_dict = {
                    'method': 'rdp_k',
                    'simplified_points': sample_data,
                    'K_actual': M,
                    'K_target': M
                }
            elif para_method == 'dct_deviation':
                # dct_deviation uses explicit endpoint and coefficient fields.
                para_dict = {
                    'method': 'dct_deviation',
                    'P_start': sample_data[0],
                    'P_end': sample_data[-1],
                    'coeffs': sample_data[1:-1].flatten(),
                    'N_uniform': M,
                    'M_coeffs': M - 2
                }
            else:
                para_dict = sample_data

            points = para2point(para_dict, N_new=traj_length, method_override=para_method)
            if points is None:
                print(f"Warning: failed to reconstruct sample {i}; using zeros instead")
                points = np.zeros((M, 2))
            generated_trajectories.append(points)

        # Apply the same reconstruction logic to the ground-truth samples.
        gt_trajectories = []
        for i in range(len(ground_truth)):
            sample_data = ground_truth[i].reshape(M, 2)

            if para_method == 'rdp_k':
                para_dict = {
                    'method': 'rdp_k',
                    'simplified_points': sample_data,
                    'K_actual': M,
                    'K_target': M
                }
            elif para_method == 'dct_deviation':
                para_dict = {
                    'method': 'dct_deviation',
                    'P_start': sample_data[0],
                    'P_end': sample_data[-1],
                    'coeffs': sample_data[1:-1].flatten(),
                    'N_uniform': M,
                    'M_coeffs': M - 2
                }
            else:
                para_dict = sample_data

            points = para2point(para_dict, N_new=traj_length, method_override=para_method)
            if points is None:
                print(f"Warning: failed to reconstruct ground-truth sample {i}; using zeros instead")
                points = np.zeros((M, 2))
            gt_trajectories.append(points)

        generated_trajectories = np.array(generated_trajectories)
        gt_trajectories = np.array(gt_trajectories)
    else:
        print("Using direct point representations...")
        generated_trajectories = sol.reshape(sample_count, M, 2)
        gt_trajectories = ground_truth.reshape(sample_count, M, 2)

    # --- Plot generated trajectories ---
    plt.figure(figsize=(12, 10))
    for i in range(max_trajs_to_plot):
        # Each trajectory is a sequence of M points
        traj = generated_trajectories[i]  # Shape: [M, 2]
        plt.plot(traj[:, 1], traj[:, 0], '-', color='blue', linewidth=1, alpha=0.1)

    plt.title(f"Generated Trajectories ({point_label})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{save_folder}/generated_trajectories.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot ground truth trajectories ---
    plt.figure(figsize=(12, 10))
    for i in range(max_trajs_to_plot):
        traj = gt_trajectories[i]  # Shape: [M, 2]
        plt.plot(traj[:, 1], traj[:, 0], '-', color='red', linewidth=1, alpha=0.1)

    plt.title(f"Ground Truth Trajectories ({point_label})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{save_folder}/ground_truth_trajectories.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot comparison of generated vs ground truth ---
    plt.figure(figsize=(12, 10))
    for i in range(max_trajs_to_plot):
        # Generated trajectories in blue
        gen_traj = generated_trajectories[i]
        plt.plot(gen_traj[:, 1], gen_traj[:, 0], '-', color='blue', linewidth=1, alpha=0.1)

        # Ground truth trajectories in red
        gt_traj = gt_trajectories[i]
        plt.plot(gt_traj[:, 1], gt_traj[:, 0], '-', color='red', linewidth=1, alpha=0.1)

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='blue', lw=2),
                  Line2D([0], [0], color='red', lw=2)]
    plt.legend(custom_lines, ['Generated', 'Ground Truth'])
    plt.title(
        f"Generated (blue) vs Ground Truth (red) Trajectories ({point_label})"
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{save_folder}/trajectory_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


def _visualize_conditional_samples(self, samples, month, cfg_scale, save_dir):
    """Visualize conditional samples"""
    import matplotlib.pyplot as plt

    # Get final samples and reshape to trajectory format
    final_samples = samples[-1].reshape(-1, self.traj_length, 2)

    # Plot trajectories
    plt.figure(figsize=(12, 10))
    for i in range(min(20, len(final_samples))):
        traj = final_samples[i]
        plt.plot(traj[:, 0], traj[:, 1], '-', linewidth=1, alpha=0.7)
        plt.plot(traj[0, 0], traj[0, 1], 'o', markersize=5)  # Start point

    plt.title(f"Month {month} Trajectories (CFG Scale: {cfg_scale})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "trajectories.png"), dpi=300)
    plt.close()
