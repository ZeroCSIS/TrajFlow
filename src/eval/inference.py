import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.models.wrappers import WrappedModel
from src.utils.visualization import visualize_trajectories, visualize_density_comparison
import jismesh.utils as ju
import math

_BASE32_ALPHABET = '0123456789bcdefghjkmnpqrstuvwxyz'
_BASE32_MAP = {c: i for i, c in enumerate(_BASE32_ALPHABET)}
_BASE32_BITS = [16, 8, 4, 2, 1]


def _geohash_int_to_str(value: int, precision: int) -> str:
    """Convert an integer geohash representation back into a string."""
    if value < 0:
        raise ValueError("Geohash integer must be non-negative")
    chars = []
    if value == 0:
        chars.append('0')
    while value > 0:
        chars.append(_BASE32_ALPHABET[value % 32])
        value //= 32
    chars = list(reversed(chars))
    if precision > len(chars):
        chars = ['0'] * (precision - len(chars)) + chars
    elif precision < len(chars):
        chars = chars[-precision:]
    return ''.join(chars)


def _geohash_bounds(geohash: str):
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]
    even = True
    for char in geohash:
        bits = _BASE32_MAP[char]
        for mask in _BASE32_BITS:
            if even:
                mid = (lon_range[0] + lon_range[1]) / 2
                if bits & mask:
                    lon_range[0] = mid
                else:
                    lon_range[1] = mid
            else:
                mid = (lat_range[0] + lat_range[1]) / 2
                if bits & mask:
                    lat_range[0] = mid
                else:
                    lat_range[1] = mid
            even = not even
    return lat_range, lon_range


def _geohash_point_from_multiplier(cell_id: int, lat_mult: float, lon_mult: float,
                                   precision: int) -> np.ndarray:
    gh_str = _geohash_int_to_str(int(cell_id), precision)
    lat_range, lon_range = _geohash_bounds(gh_str)
    lat = lat_range[0] + lat_mult * (lat_range[1] - lat_range[0])
    lon = lon_range[0] + lon_mult * (lon_range[1] - lon_range[0])
    return np.array([lat, lon])
class ConditionedVelocityModelWrapper(torch.nn.Module):
    """Wrapper around velocity model to inject month condition during inference.

    Implements classifier-free guidance according to the formula:
    u ← (1-w)*u_null + w*u_cond

    where:
    - u_null is the velocity with condition dropped
    - u_cond is the velocity with condition intact
    - w is the cfg_scale (default=1.0, which means no guidance)
    """

    def __init__(self, velocity_model, condition, cfg_scale=1.0,**kwargs):
        super().__init__()
        self.velocity_model = velocity_model
        self.condition = condition
        self.cfg_scale = cfg_scale
        # if extra kwargs are provided, initialize them
        self.mapping_dict = kwargs.get('mapping_dict', None)
        self.norm1by1_viz = kwargs.get('norm1by1_viz', False)

    def forward(self, x, t, c = None, **kwargs):
        """Forward pass with classifier-free guidance.

        Args:
            x: Input tensor (batch_size, ...)
            t: Time tensor (batch_size, ) or ()

        Returns:
            Predicted velocity with CFG applied if cfg_scale > 1.0
        """
        # For cfg_scale = 1.0, just use the regular conditioned model (no guidance)
        if self.cfg_scale == 1.0:
            cond = c if c is not None else self.condition
            return self.velocity_model(x, t, c=cond, **kwargs)

        # For cfg_scale > 1.0, compute both conditional and unconditional predictions
        batch_size = x.shape[0]
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size)

        # Duplicate inputs for conditional and unconditional forward passes
        # This allows computing both in parallel in a single forward pass
        x_doubled = torch.cat([x, x], dim=0)  # shape: [batch_size*2, ...]
        t_doubled = torch.cat([t, t], dim=0)  # shape: [batch_size*2]
        cond = c if c is not None else self.condition
        c_doubled = torch.cat([cond, cond], dim=0)

        # Create force_drop_ids with zeros for first half (conditional)
        # and ones for second half (unconditional)
        force_drop_ids = torch.cat([
            torch.zeros(batch_size, dtype=torch.long, device=x.device),  # keep condition
            torch.ones(batch_size, dtype=torch.long, device=x.device)    # drop condition
        ], dim=0)

        # Single forward pass with doubled batch
        v_doubled = self.velocity_model(x_doubled, t_doubled, c=c_doubled, force_drop_ids=force_drop_ids, **kwargs)

        # Split the results
        v_cond, v_null = v_doubled.chunk(2, dim=0)  # Each shape: [batch_size, ...]

        # Apply classifier-free guidance formula: u ← (1-w)*u_null + w*u_cond
        guided_velocity = (1 - self.cfg_scale) * v_null + self.cfg_scale * v_cond

        return guided_velocity

class FlowMatchingInference:
    def __init__(self, config, model, dataset, save_dir, device):
        """
        Inference for Flow Matching

        Args:
            config: Configuration dictionary
            model: Trained model
            dataset: Dataset for evaluation
            save_dir: Directory to save results
            device: Device to run inference on
        """
        self.config = config
        self.model = model.to(device)
        self.model = self.model.eval()
        self.dataset = dataset
        self.save_dir = save_dir
        self.device = device
        self.model_type = self._get_model_type(config)

        if config['data']['parametrized']:
            self.M = config['data']['parametrized_M']
        else:
            self.M = config['data']['trajectory_length']

        # Setup dataloader
        batch_size = config['data']['batch_size']
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Prepare model for sampling
        self.wrapped_model = WrappedModel(self.model)

        # Add support for conditional flow
        self.conditional = config.get('condition', {}).get('enabled', False)
        if self.conditional:
            self.condition_dim = dataset.location_dim
            self.cfg_scale = config.get('condition', {}).get('cfg_scale', 1.0)

    def sample(self, n_samples, n_steps=None, method='em', condition=None):
        """Sample trajectories based on model type"""
        if self.model_type == 'baseline':
            return self._sample_baseline(n_samples, condition)
        elif self.model_type == 'flow_matching':
            return self._sample_flow_matching(n_samples, n_steps, method, condition)
        elif self.model_type == 'ddpm':
            return self._sample_ddpm(n_samples, n_steps, condition)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _get_model_type(self, config):
        """Determine model type from config"""
        if config.get('baseline', {}).get('enabled', False):
            return 'baseline'
        elif config['flow_matching']['enabled']:
            return 'flow_matching'
        elif config['ddpm']['enabled']:
            return 'ddpm'
        else:
            raise ValueError("No model type specified")

    # def _sample_ddpm(self, n_samples, condition=None):
    #     """DDPM sampling using DDIM"""
    #
    #     n_steps = self.config['ddpm']['ddim_steps']
    #
    #     # Create beta schedule
    #     num_timesteps = self.config['ddpm']['num_diffusion_timesteps']
    #     beta_start = self.config['ddpm']['beta_start']
    #     beta_end = self.config['ddpm']['beta_end']
    #     betas = torch.linspace(beta_start, beta_end, num_timesteps, device=self.device)
    #     alphas = 1 - betas
    #     alphas_cumprod = torch.cumprod(alphas, dim=0)
    #
    #     # Start from pure noise
    #     # Generate initial random points (noise)
    #     if self.config['data']['od_finer']:
    #         x_init = torch.randn((n_samples, self.M*2+4), dtype=torch.float32, device=self.device)
    #     else:
    #         x_init = torch.randn((n_samples,self.M,2), dtype=torch.float32, device=self.device)
    #     x = x_init.clone()
    #
    #     # Create sampling timesteps (reverse order)
    #     timesteps = torch.linspace(num_timesteps - 1, 0, n_steps, dtype=torch.long, device=self.device)
    #
    #     samples = [x.clone()]
    #
    #     for i, t in enumerate(timesteps):
    #         t_batch = t.repeat(n_samples).to(self.device)
    #
    #         with torch.no_grad():
    #             # Predict noise using the same neural network
    #             predicted_noise = self.model(x, t_batch.float() / num_timesteps, condition)
    #
    #             # DDIM sampling step
    #             alpha_t = alphas_cumprod[t]
    #             alpha_prev = alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=self.device)
    #
    #             # Predicted x_0
    #             # reshape predicted_noise to match x shape
    #             if predicted_noise.shape != x.shape:
    #                 predicted_noise = predicted_noise.reshape(x.shape)
    #             pred_x0 = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
    #
    #             # Direction to x_t
    #             if t > 0:
    #                 direction = torch.sqrt(1 - alpha_prev) * predicted_noise
    #                 x = torch.sqrt(alpha_prev) * pred_x0 + direction
    #             else:
    #                 x = pred_x0
    #
    #             samples.append(x.clone())
    #
    #     return samples

    def _sample_baseline(self, n_samples, condition=None):
        """Sample from baseline models"""
        if hasattr(self.model, 'generate'):
            # Handle conditional vs unconditional generation
            if condition is not None:
                generated = self.model.generate(n_samples, condition, device=self.device)
            else:
                generated = self.model.generate(n_samples, device=self.device)

            # Check if generated is already a tensor
            if isinstance(generated, torch.Tensor):
                generated_tensor = generated.to(self.device)
            else:
                # Convert from numpy if needed
                generated_tensor = torch.from_numpy(generated).float().to(self.device)

            return [generated_tensor]  # Return as list for consistency
        else:
            raise NotImplementedError("Baseline model must implement generate method")

    def _sample_ddpm(self, n_samples,n_steps, condition=None):
        """DDPM sampling using DDIM"""

        # Create beta schedule
        num_timesteps = self.config['ddpm']['num_diffusion_timesteps']
        beta_start = self.config['ddpm']['beta_start']
        beta_end = self.config['ddpm']['beta_end']
        betas = torch.linspace(beta_start, beta_end, num_timesteps, device=self.device)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Start from pure noise - match the training data shape
        if self.config['data']['od_finer']:
            x = torch.randn((n_samples, self.M * 2 + 4), dtype=torch.float32, device=self.device)
        else:
            # Flatten the shape to match training format
            x = torch.randn((n_samples, self.M * 2), dtype=torch.float32, device=self.device)

        # Create sampling timesteps (reverse order)
        timesteps = torch.linspace(num_timesteps - 1, 0, n_steps, dtype=torch.long, device=self.device)

        samples = [x.clone()]

        for i, t in enumerate(timesteps):
            t_batch = t.repeat(n_samples).to(self.device)

            with torch.no_grad():
                # Predict noise using the same neural network
                # Make sure time is normalized the same way as in training
                t_normalized = t_batch.float() / num_timesteps
                predicted_noise = self.model(x, t_normalized, condition)

                # Ensure predicted_noise matches x shape
                if predicted_noise.shape != x.shape:
                    predicted_noise = predicted_noise.reshape(x.shape)

                # DDIM sampling step
                alpha_t = alphas_cumprod[t]
                alpha_prev = alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=self.device)

                # Predicted x_0
                pred_x0 = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)

                # Direction to x_t
                if t > 0:
                    direction = torch.sqrt(1 - alpha_prev) * predicted_noise
                    x = torch.sqrt(alpha_prev) * pred_x0 + direction
                else:
                    x = pred_x0

                samples.append(x.clone())

        return samples

    def _sample_flow_matching(self, n_samples, n_steps=10, method='em', condition=None, cfg_scale=None):
        """
        Sample from the flow matching model

        Args:
            n_samples: Number of samples to generate
            n_steps: Number of steps for numerical integration
            method: Sampling method ('em' for Euler-Maruyama, 'rejection' for rejection sampling)
            condition: Optional condition tensor [n_samples, condition_dim] or [1, condition_dim]
            cfg_scale: Optional classifier-free guidance scale (defaults to config value)

        Returns:
            samples: Sampled trajectories (only initial and final states to save memory)
        """

        # Generate initial random points (noise)
        if self.config['data']['od_finer']:
            x_init = torch.randn((n_samples, self.M*2+4), dtype=torch.float32, device=self.device)
        else:
            x_init = torch.randn((n_samples,self.M,2), dtype=torch.float32, device=self.device)

        # Create model for sampling
        sampling_model = self.wrapped_model

        # Apply condition if provided and model is conditional
        if self.conditional and condition is not None:
            # Use default cfg_scale if not provided
            if cfg_scale is None:
                cfg_scale = self.cfg_scale

            # Ensure condition is on the correct device
            condition = condition.to(self.device)

            # Expand condition if needed (for batching)
            if condition.size(0) == 1 and n_samples > 1:
                condition = condition.expand(n_samples, -1)

            # Create conditional wrapper
            sampling_model = ConditionedVelocityModelWrapper(
                sampling_model, condition, cfg_scale=cfg_scale,
                mapping_dict=self.dataset.grid_mapping_dict,
                norm1by1_viz=self.config['visualization']['norm1by12origialvis'])

        # Create time grid for numerical integration
        time_grid = torch.linspace(0, 1, n_steps).to(self.device)

        # Set step size based on number of steps
        step_size = 1.0 / (n_steps - 1)

        # Sample using the wrapped model
        # MEMORY OPTIMIZATION: Only store initial state, not all intermediate steps
        x_t = x_init.clone()

        # Numerical integration
        if method == 'em':  # Euler-Maruyama method
            for t_idx in range(n_steps - 1):
                t = time_grid[t_idx] * torch.ones(n_samples, 1, device=self.device)
                # Get velocity from the model
                with torch.no_grad():  # Disable gradient computation for memory efficiency
                    v_t = sampling_model(x_t, t, c = condition)
                # Update using Euler method: x_{t+1} = x_t + v_t * step_size
                # let the shape of vt be the same as xt
                if v_t.shape != x_t.shape:
                    v_t = v_t.reshape(x_t.shape)
                x_t = x_t + v_t * step_size

                # Clear unnecessary tensors to free memory
                del v_t
                if (t_idx + 1) % 10 == 0:  # Periodic cleanup
                    torch.cuda.empty_cache()
        elif method == 'rejection':
            raise NotImplementedError("Rejection sampling not implemented yet")
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        # Return only initial and final states to save memory
        return [x_init, x_t]

    def denormalize_trajectories(self, trajectory_list,onehoted_condition_sample,dataset,od_finer_params=None):
        """Denormalize trajectories using mapping_dict for visualization."""

        denormalized_list = []

        if dataset.condition_type == 'od':
            pass
        elif dataset.condition_type == 'full':
            onehoted_condition_sample = onehoted_condition_sample[:, 6:8]  # Only take the first two columns (lat/lon)

        # Change the condition_sample to original od based on dataset
        grid_mapping_dict = dataset.cr_sample_grid_mapping_dict
        condition_sample = np.zeros((len(onehoted_condition_sample),2))

        for i in range(condition_sample.shape[0]):
            for j in range(condition_sample.shape[1]):
                condition_sample[i][j] = grid_mapping_dict[onehoted_condition_sample[i][j]].astype(int)

        # else:
        #     raise ValueError("Unknown encoding format. Use 'onehot' or 'embedding'.")

        # calculate the OD by lat/lon multipliers if given
        if od_finer_params is None:
            # Default: set all multipliers to 0.5 (center of grid cell)
            lat_mult = np.ones((condition_sample.shape[0], condition_sample.shape[1])) * 0.5
            lon_mult = np.ones((condition_sample.shape[0], condition_sample.shape[1])) * 0.5
        else:
            # Extract the lat/lon multipliers from the given od_finer_params
            # Format: each batch item has [o_lat_mult, o_lon_mult, d_lat_mult, d_lon_mult]
            batch_size = condition_sample.shape[0]
            lat_mult = np.zeros((batch_size, 2))  # For origin and destination
            lon_mult = np.zeros((batch_size, 2))  # For origin and destination

            for i in range(batch_size):
                if i < len(od_finer_params):
                    # Origin multipliers (lat, lon)
                    lat_mult[i, 0] = od_finer_params[i][0]
                    lon_mult[i, 0] = od_finer_params[i][1]

                    # Destination multipliers (lat, lon)
                    lat_mult[i, 1] = od_finer_params[i][2]
                    lon_mult[i, 1] = od_finer_params[i][3]
                else:
                    # Default values if parameters not available
                    lat_mult[i, :] = 0.5
                    lon_mult[i, :] = 0.5

        # Get the offset from the od's grid
        condition_sample_lonlat = np.zeros((len(onehoted_condition_sample), 2,2))
        grid_encoding = getattr(dataset, 'grid_encoding', 'jismesh')
        grid_meta = getattr(dataset, 'grid_metadata', {})
        geohash_precision = grid_meta.get('geohash_precision', 6)
        for i in range(condition_sample.shape[0]):
            for j in range(condition_sample.shape[1]):
                cell_value = int(condition_sample[i][j])
                if grid_encoding == 'geohash':
                    condition_sample_lonlat[i, j, :] = _geohash_point_from_multiplier(
                        cell_value, lat_mult[i][j], lon_mult[i][j], geohash_precision)
                else:
                    condition_sample_lonlat[i, j, :] = ju.to_meshpoint(
                        cell_value, lat_mult[i][j], lon_mult[i][j])

        for trajectories in trajectory_list:
            # Reshape to (n_samples, M, 2) format
            batch_size = trajectories.shape[0]
            traj_length = self.config['data']['trajectory_length']
            trajectories_reshaped = trajectories.reshape(batch_size, traj_length, 2)
            # Create output array
            denorm_trajectories = np.zeros_like(trajectories_reshaped)

            DENORM_METHOD = 'MixStrategy'  # Choose denormalization method
            if DENORM_METHOD == 'Affine':
                # 各向异性缩放：分别对 x 和 y 计算缩放因子，保证 OD 严格对齐
                for i in range(batch_size):
                    # Get normalized trajectory (copy 避免 in-place 修改)
                    traj = trajectories_reshaped[i].copy()
                    # 归一化轨迹起点和终点
                    origin_norm = traj[0]  # 第一个点
                    dest_norm = traj[-1]  # 最后一个点

                    # 目标 OD 坐标（严格给定的矩形）
                    origin_target = condition_sample_lonlat[i, 0]
                    dest_target = condition_sample_lonlat[i, 1]

                    # 计算 x 和 y 的缩放因子
                    norm_range_x = dest_norm[0] - origin_norm[0]
                    norm_range_y = dest_norm[1] - origin_norm[1]
                    target_range_x = dest_target[0] - origin_target[0]
                    target_range_y = dest_target[1] - origin_target[1]

                    epsilon = 1e-6
                    if abs(norm_range_x) < epsilon:
                        scale_x = 1.0
                    else:
                        scale_x = target_range_x / norm_range_x

                    if abs(norm_range_y) < epsilon:
                        scale_y = 1.0
                    else:
                        scale_y = target_range_y / norm_range_y

                    # 对轨迹上每个点应用转换
                    for j in range(traj_length):
                        denorm_trajectories[i, j, 0] = (traj[j, 0] - origin_norm[0]) * scale_x + origin_target[0]
                        denorm_trajectories[i, j, 1] = (traj[j, 1] - origin_norm[1]) * scale_y + origin_target[1]

                # # Plot comparison before and after denormalization
                # import matplotlib.pyplot as plt
                # for i in range(min(5, batch_size)):  # Plot first 5 samples
                #     fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                # 
                #     # Plot original normalized trajectory
                #     ax[0].plot(trajectories_reshaped[i, :, 0], trajectories_reshaped[i, :, 1], 'b-', linewidth=2)
                #     ax[0].plot(trajectories_reshaped[i, 0, 0], trajectories_reshaped[i, 0, 1], 'go', markersize=8, label='起点')
                #     ax[0].plot(trajectories_reshaped[i, -1, 0], trajectories_reshaped[i, -1, 1], 'ro', markersize=8, label='终点')
                #     ax[0].set_title('归一化轨迹（原始）')
                #     ax[0].legend()
                #     ax[0].grid(True)
                #     ax[0].axis('equal')
                # 
                #     # Plot denormalized trajectory
                #     ax[1].plot(denorm_trajectories[i, :, 0], denorm_trajectories[i, :, 1], 'b-', linewidth=2)
                #     ax[1].plot(condition_sample_lonlat[i, 0, 0], condition_sample_lonlat[i, 0, 1], 'go', markersize=8, label='目标起点')
                #     ax[1].plot(condition_sample_lonlat[i, 1, 0], condition_sample_lonlat[i, 1, 1], 'ro', markersize=8, label='目标终点')
                #     ax[1].plot(denorm_trajectories[i, 0, 0], denorm_trajectories[i, 0, 1], 'gx', markersize=8, label='实际起点')
                #     ax[1].plot(denorm_trajectories[i, -1, 0], denorm_trajectories[i, -1, 1], 'rx', markersize=8, label='实际终点')
                #     ax[1].set_title('反归一化轨迹')
                #     ax[1].legend()
                #     ax[1].grid(True)
                #     ax[1].axis('equal')
                # 
                #     plt.tight_layout()
                #     plt.show()
                #     plt.close(fig)


            elif DENORM_METHOD == 'UniformPreserveAspect':
                epsilon = 1e-8 # Small number to avoid division by zero

                for i in range(batch_size):
                    traj_norm = trajectories_reshaped[i] # Shape (traj_length, 2)

                    if traj_norm.shape[0] < 2: # 至少需要两个点来确定起终点
                        denorm_trajectories[i] = traj_norm # 或者返回空/特定值？
                        continue

                    # Normalized start and end points
                    origin_norm = traj_norm[0]
                    dest_norm = traj_norm[-1]

                    # Target original start and end points
                    origin_target = condition_sample_lonlat[i, 0]
                    dest_target = condition_sample_lonlat[i, 1]

                    # Calculate difference vectors
                    delta_norm = dest_norm - origin_norm
                    delta_target = dest_target - origin_target

                    # Calculate norms (magnitudes)
                    norm_delta_norm = np.linalg.norm(delta_norm)
                    norm_delta_target = np.linalg.norm(delta_target)

                    # --- Determine the uniform scale factor 's' ---
                    if norm_delta_norm > epsilon:
                        # Normal case: calculate scale factor
                        s = norm_delta_target / norm_delta_norm
                    else:
                        # Degenerate case: Normalized start and end points are coincident.
                        if norm_delta_target <= epsilon:
                            # Target start/end are also coincident. No scaling needed relative to start/end diff.
                            # We still need *some* scale if the trajectory wasn't just a single point originally.
                            # What was the fallback scale during standardization? We don't know it here!
                            # Best guess: Assume scale is 1.0, meaning the fallback std dev was also ~1.0 or trajectory was single point.
                            s = 1.0
                            # print(f"Debug: Both normalized and target start/end are coincident for batch item {i}. Using scale=1.0.")
                        else:
                            # Normalized points are same, but target points differ. Problematic case.
                            print(f"Warning: Normalized start/end points are identical for batch item {i}, but target points differ. Cannot accurately recover scale. Using scale=1.0.")
                            s = 1.0 # Default scaling, likely inaccurate.

                    # --- Calculate the CORRECT offset based on the midpoint centering ---
                    # The transformation is: traj_target = traj_norm * s + center_point_target
                    # where center_point_target = (origin_target + dest_target) / 2.0
                    center_point_target = (origin_target + dest_target) / 2.0

                    # --- 1. Apply the initial inverse transformation ---
                    initial_denorm_traj = traj_norm * s + center_point_target

                    # --- 2. Calculate endpoint errors ---
                    if traj_length > 0: # Ensure there are points to correct
                        current_start = initial_denorm_traj[0]
                        current_end = initial_denorm_traj[-1]

                        start_error_vector = origin_target - current_start
                        end_error_vector = dest_target - current_end

                        # --- 3. Create interpolation weights ---
                        # linspace is inclusive, so it generates traj_length points from 1 down to 0
                        w = np.linspace(1, 0, traj_length)

                        # --- 4. Apply interpolated correction ---
                        # Use np.newaxis to make w broadcast correctly with (traj_length, 2) arrays
                        # correction = w[:, np.newaxis] * start_error_vector + (1 - w)[:, np.newaxis] * end_error_vector
                        # More efficient calculation:
                        correction = start_error_vector + (end_error_vector - start_error_vector) * (1 - w)[:, np.newaxis]


                        final_denorm_traj = initial_denorm_traj + correction
                    else:
                        # If trajectory is empty, keep it empty
                        final_denorm_traj = initial_denorm_traj

                    # Assign the final, corrected trajectory to the output array
                    denorm_trajectories[i] = final_denorm_traj

                    # --- Optional: Final Sanity Check ---
                    # calculated_origin_final = denorm_trajectories[i, 0]
                    # calculated_dest_final = denorm_trajectories[i, -1]
                    # origin_error_final = np.linalg.norm(calculated_origin_final - origin_target)
                    # dest_error_final = np.linalg.norm(calculated_dest_final - dest_target)
                    # if origin_error_final > epsilon or dest_error_final > epsilon: # Check with smaller tolerance now
                    #    print(f"!!! Critical Error: Endpoint forcing failed for batch item {i} !!!")
                    #    print(f"  Final Origin Error: {origin_error_final:.2e}")
                    #    print(f"  Final Dest Error:   {dest_error_final:.2e}")

            elif DENORM_METHOD == 'Affine_ExplicitParams':
            # --------------------------------------------------------------------
            # BEGINNING OF 'Affine_ExplicitParams' method logic (INLINED)
            # --------------------------------------------------------------------
                epsilon = 1e-7  # 用于比较浮点数是否接近零的阈值

                for i in range(batch_size):  # 遍历批次中的每条轨迹
                    traj_norm = trajectories_reshaped[i]  # 当前归一化轨迹 (L, 2)

                    if traj_norm.shape[0] == 0:
                        continue

                    origin_norm = traj_norm[0]  # 归一化起点
                    dest_norm = traj_norm[-1] if traj_length > 1 else origin_norm  # 归一化终点

                    origin_target = condition_sample_lonlat[i, 0]  # 原始目标起点 (lon, lat)
                    dest_target = condition_sample_lonlat[i, 1]  # 原始目标终点 (lon, lat)

                    # --- X维度参数 Ax, Bx 计算 (target_x = Ax * norm_x + Bx) ---
                    norm_val1_x = origin_norm[0]
                    target_val1_x = origin_target[0]
                    norm_val2_x = dest_norm[0]
                    target_val2_x = dest_target[0]

                    norm_range_x = norm_val2_x - norm_val1_x
                    target_range_x = target_val2_x - target_val1_x

                    Ax = 0.0
                    Bx = target_val1_x  # 如果 norm_range_x 为0, 所有点映射到 target_val1_x

                    if abs(norm_range_x) < epsilon:
                        # Ax 保持为 0.0
                        # Bx 保持为 target_val1_x
                        if abs(target_range_x) >= epsilon:
                            print(f"警告 (样本 {i}, 维度 X, Affine_ExplicitParams): "
                                  f"归一化X范围约等于0，但目标X范围 ({target_range_x:.2f}) 非0。"
                                  f"所有反归一化后的X坐标将被设为目标起点X ({target_val1_x:.2f})。"
                                  f"因此，目标终点X ({target_val2_x:.2f}) 将不会被此变换精确匹配。")
                    else:
                        # 正常情况
                        Ax = target_range_x / norm_range_x
                        Bx = target_val1_x - Ax * norm_val1_x

                    # --- Y维度参数 Ay, By 计算 (target_y = Ay * norm_y + By) ---
                    norm_val1_y = origin_norm[1]
                    target_val1_y = origin_target[1]
                    norm_val2_y = dest_norm[1]
                    target_val2_y = dest_target[1]

                    norm_range_y = norm_val2_y - norm_val1_y
                    target_range_y = target_val2_y - target_val1_y

                    Ay = 0.0
                    By = target_val1_y  # 如果 norm_range_y 为0, 所有点映射到 target_val1_y

                    if abs(norm_range_y) < epsilon:
                        # Ay 保持为 0.0
                        # By 保持为 target_val1_y
                        if abs(target_range_y) >= epsilon:
                            print(f"警告 (样本 {i}, 维度 Y, Affine_ExplicitParams): "
                                  f"归一化Y范围约等于0，但目标Y范围 ({target_range_y:.2f}) 非0。"
                                  f"所有反归一化后的Y坐标将被设为目标起点Y ({target_val1_y:.2f})。"
                                  f"因此，目标终点Y ({target_val2_y:.2f}) 将不会被此变换精确匹配。")
                    else:
                        # 正常情况
                        Ay = target_range_y / norm_range_y
                        By = target_val1_y - Ay * norm_val1_y

                    # 应用变换到当前轨迹的每个点
                    for j in range(traj_length):
                        norm_px = traj_norm[j, 0]
                        norm_py = traj_norm[j, 1]

                        # 使用 denorm_trajectories 进行赋值
                        denorm_trajectories[i, j, 0] = Ax * norm_px + Bx
                        denorm_trajectories[i, j, 1] = Ay * norm_py + By
            # --------------------------------------------------------------------
                # END OF 'Affine_ExplicitParams' method logic (INLINED)
                # --------------------------------------------------------------------

            elif DENORM_METHOD == 'SimilarityWithEndpointCorrection':
                epsilon = 1e-7  # 用于比较浮点数是否接近零的阈值

                for i in range(batch_size):  # 遍历批次中的每条轨迹
                    traj_norm = trajectories_reshaped[i]

                    if traj_norm.shape[0] == 0: continue

                    origin_norm = traj_norm[0]
                    dest_norm = traj_norm[-1] if traj_length > 1 else origin_norm

                    origin_target = condition_sample_lonlat[i, 0]
                    dest_target = condition_sample_lonlat[i, 1]

                    # --- 步骤 1: 进行最优的初始变换 (完整的相似变换) ---
                    V_norm = dest_norm - origin_norm
                    V_target = dest_target - origin_target

                    len_norm = np.linalg.norm(V_norm)
                    len_target = np.linalg.norm(V_target)

                    scale_s = 1.0
                    cos_theta = 1.0
                    sin_theta = 0.0

                    if len_norm > epsilon:
                        scale_s = len_target / len_norm
                        angle_norm = np.arctan2(V_norm[1], V_norm[0])
                        angle_target = np.arctan2(V_target[1], V_target[0])
                        theta = angle_target - angle_norm
                        cos_theta = np.cos(theta)
                        sin_theta = np.sin(theta)
                    else:
                        if len_target > epsilon:
                            print(f"警告 (样本 {i}, SimilarityWithEndpointCorrection): "
                                  f"归一化轨迹O/D重合，但目标O/D不重合。轨迹将塌缩至目标起点。")

                    # 计算初始变换后的轨迹
                    initial_denorm_traj = np.zeros_like(traj_norm)
                    for j in range(traj_length):
                        point_norm = traj_norm[j]
                        p_centered_x = point_norm[0] - origin_norm[0]
                        p_centered_y = point_norm[1] - origin_norm[1]
                        p_scaled_x = p_centered_x * scale_s
                        p_scaled_y = p_centered_y * scale_s
                        p_rotated_x = p_scaled_x * cos_theta - p_scaled_y * sin_theta
                        p_rotated_y = p_scaled_x * sin_theta + p_scaled_y * cos_theta
                        initial_denorm_traj[j, 0] = p_rotated_x + origin_target[0]
                        initial_denorm_traj[j, 1] = p_rotated_y + origin_target[1]

                    # --- 步骤 2: 进行强制端点修正 (消除浮点误差，保证100%对齐) ---
                    if traj_length > 1:
                        current_start = initial_denorm_traj[0]
                        current_end = initial_denorm_traj[-1]

                        start_error_vector = origin_target - current_start
                        end_error_vector = dest_target - current_end

                        # 创建从1到0的线性插值权重
                        w = np.linspace(1, 0, traj_length)

                        # 将误差向量线性地分配到轨迹上的所有点
                        # correction[j] = w[j] * start_error_vector + (1 - w[j]) * end_error_vector
                        correction = w[:, np.newaxis] * start_error_vector + (1 - w)[:, np.newaxis] * end_error_vector

                        # 将修正量加到初始变换后的轨迹上
                        denorm_trajectories[i] = initial_denorm_traj + correction
                    else:  # 如果轨迹只有一个点，直接使用初始变换的结果
                        denorm_trajectories[i] = initial_denorm_traj

            elif DENORM_METHOD == 'HybridSimilarity':  # 新增的混合策略（高级）方法
                epsilon = 1e-7
                # 关键参数：旋转阈值。如果计算出的旋转角度绝对值小于此值，则忽略旋转。
                # 这个值可能需要根据您的数据特性进行微调。5.0度是一个合理的初始值。
                rotation_threshold_degrees = 5.0
                rotation_threshold_rad = np.deg2rad(rotation_threshold_degrees)

                for i in range(batch_size):
                    traj_norm = trajectories_reshaped[i]

                    if traj_norm.shape[0] == 0: continue

                    origin_norm = traj_norm[0]
                    dest_norm = traj_norm[-1] if traj_length > 1 else origin_norm

                    origin_target = condition_sample_lonlat[i, 0]
                    dest_target = condition_sample_lonlat[i, 1]

                    # --- 步骤 1: 计算相似变换所需的缩放因子 s 和旋转角度 theta ---
                    V_norm = dest_norm - origin_norm
                    V_target = dest_target - origin_target

                    len_norm = np.linalg.norm(V_norm)
                    len_target = np.linalg.norm(V_target)

                    scale_s = 1.0
                    theta = 0.0

                    if len_norm > epsilon:
                        scale_s = len_target / len_norm
                        angle_norm = np.arctan2(V_norm[1], V_norm[0])
                        angle_target = np.arctan2(V_target[1], V_target[0])
                        theta = angle_target - angle_norm
                    else:  # 归一化轨迹的O/D点重合
                        if len_target > epsilon:
                            print(
                                f"警告 (样本 {i}, HybridSimilarity): 归一化轨迹O/D重合，但目标O/D不重合。轨迹将塌缩至目标起点。")

                    # --- 步骤 2: 根据阈值判断是否执行旋转 ---
                    if abs(theta) < rotation_threshold_rad:
                        # 角度过小，视为噪声，不执行旋转
                        # if len_norm > epsilon: # 仅用于调试，可移除
                        #     print(f"调试 (样本 {i}): 旋转角 {np.rad2deg(theta):.2f}° < 阈值 {rotation_threshold_degrees}°，已忽略旋转。")
                        cos_theta = 1.0
                        sin_theta = 0.0
                    else:
                        # 角度显著，执行旋转
                        # if len_norm > epsilon: # 仅用于调试，可移除
                        #     print(f"调试 (样本 {i}): 旋转角 {np.rad2deg(theta):.2f}° >= 阈值 {rotation_threshold_degrees}°，执行旋转。")
                        cos_theta = np.cos(theta)
                        sin_theta = np.sin(theta)

                    # --- 步骤 3: 执行初始变换（统一缩放 + 条件性旋转 + 平移）---
                    initial_denorm_traj = np.zeros_like(traj_norm)
                    for j in range(traj_length):
                        point_norm = traj_norm[j]
                        # 平移至原点
                        p_centered_x = point_norm[0] - origin_norm[0]
                        p_centered_y = point_norm[1] - origin_norm[1]
                        # 缩放
                        p_scaled_x = p_centered_x * scale_s
                        p_scaled_y = p_centered_y * scale_s
                        # (条件性)旋转
                        p_rotated_x = p_scaled_x * cos_theta - p_scaled_y * sin_theta
                        p_rotated_y = p_scaled_x * sin_theta + p_scaled_y * cos_theta
                        # 平移至目标起点
                        initial_denorm_traj[j, 0] = p_rotated_x + origin_target[0]
                        initial_denorm_traj[j, 1] = p_rotated_y + origin_target[1]

                    # --- 步骤 4: 应用强制端点修正，保证100%对齐 ---
                    if traj_length > 1:
                        current_start = initial_denorm_traj[0]
                        current_end = initial_denorm_traj[-1]

                        start_error = origin_target - current_start
                        end_error = dest_target - current_end

                        # 只有当误差不可忽略时才进行修正，以提高效率
                        if np.linalg.norm(start_error) > epsilon or np.linalg.norm(end_error) > epsilon:
                            w = np.linspace(1, 0, traj_length)
                            correction = w[:, np.newaxis] * start_error + (1 - w)[:, np.newaxis] * end_error
                            denorm_trajectories[i] = initial_denorm_traj + correction
                        else:  # 误差可忽略，无需修正
                            denorm_trajectories[i] = initial_denorm_traj
                    else:  # 轨迹只有一个点
                        denorm_trajectories[i] = initial_denorm_traj

            elif DENORM_METHOD == 'MixStrategy':  # 新增的智能混合策略
                epsilon = 1e-7
                # 关键超参数：用于判断是否发生极端变形的各向异性比率阈值
                anisotropy_threshold = 10.0
                # 用于混合策略中 'HybridSimilarity' 部分的旋转阈值
                rotation_threshold_degrees = 5.0
                rotation_threshold_rad = np.deg2rad(rotation_threshold_degrees)

                for i in range(batch_size):
                    traj_norm = trajectories_reshaped[i]
                    if traj_norm.shape[0] == 0: continue

                    origin_norm = traj_norm[0]
                    dest_norm = traj_norm[-1] if traj_length > 1 else origin_norm
                    origin_target = condition_sample_lonlat[i, 0]
                    dest_target = condition_sample_lonlat[i, 1]

                    # --- 步骤 1: 预计算 'Affine_ExplicitParams' 的缩放因子以评估风险 ---
                    norm_range_x = dest_norm[0] - origin_norm[0]
                    target_range_x = dest_target[0] - origin_target[0]
                    norm_range_y = dest_norm[1] - origin_norm[1]
                    target_range_y = dest_target[1] - origin_target[1]

                    # 计算缩放比例的绝对值，为避免除零，给分母加一个极小值
                    scale_x_abs = abs(target_range_x / (norm_range_x + epsilon)) if abs(
                        norm_range_x) > epsilon else float('inf')
                    scale_y_abs = abs(target_range_y / (norm_range_y + epsilon)) if abs(
                        norm_range_y) > epsilon else float('inf')

                    # --- 步骤 2: 判断是否会发生严重扭曲 ---
                    is_distorted = False
                    # 如果一个轴的归一化范围为零但目标范围不为零，这是一个明确的扭曲信号
                    if (abs(norm_range_x) < epsilon and abs(target_range_x) >= epsilon) or \
                            (abs(norm_range_y) < epsilon and abs(target_range_y) >= epsilon):
                        is_distorted = True
                    # 如果两个轴的缩放比例相差过大
                    elif min(scale_x_abs, scale_y_abs) > 0 and max(scale_x_abs, scale_y_abs) / min(scale_x_abs,
                                                                                                   scale_y_abs) > anisotropy_threshold:
                        is_distorted = True

                    # --- 步骤 3: 根据判断结果选择并应用相应的方法 ---
                    if is_distorted:
                        # **执行备用策略: 'HybridSimilarity'**
                        # (此方法能保持形状，避免扭曲)
                        # if i % 10 == 0: # 可选的调试信息
                        #     print(f"调试 (样本 {i}): 检测到高风险扭曲，切换到 HybridSimilarity 方法。")

                        V_norm = dest_norm - origin_norm
                        V_target = dest_target - origin_target
                        len_norm = np.linalg.norm(V_norm)
                        len_target = np.linalg.norm(V_target)

                        scale_s = 1.0
                        theta = 0.0
                        if len_norm > epsilon:
                            scale_s = len_target / len_norm
                            angle_norm = np.arctan2(V_norm[1], V_norm[0])
                            angle_target = np.arctan2(V_target[1], V_target[0])
                            theta = angle_target - angle_norm

                        if abs(theta) < rotation_threshold_rad:
                            cos_theta, sin_theta = 1.0, 0.0
                        else:
                            cos_theta, sin_theta = np.cos(theta), np.sin(theta)

                        initial_denorm_traj = np.zeros_like(traj_norm)
                        for j in range(traj_length):
                            p_centered = traj_norm[j] - origin_norm
                            p_scaled_x = p_centered[0] * scale_s
                            p_scaled_y = p_centered[1] * scale_s
                            p_rotated_x = p_scaled_x * cos_theta - p_scaled_y * sin_theta
                            p_rotated_y = p_scaled_x * sin_theta + p_scaled_y * cos_theta
                            initial_denorm_traj[j, 0] = p_rotated_x + origin_target[0]
                            initial_denorm_traj[j, 1] = p_rotated_y + origin_target[1]

                        if traj_length > 1:
                            start_error = origin_target - initial_denorm_traj[0]
                            end_error = dest_target - initial_denorm_traj[-1]
                            if np.linalg.norm(start_error) > epsilon or np.linalg.norm(end_error) > epsilon:
                                w = np.linspace(1, 0, traj_length)
                                correction = w[:, np.newaxis] * start_error + (1 - w)[:, np.newaxis] * end_error
                                denorm_trajectories[i] = initial_denorm_traj + correction
                            else:
                                denorm_trajectories[i] = initial_denorm_traj
                        else:
                            denorm_trajectories[i] = initial_denorm_traj

                    else:
                        # **执行默认策略: 'Affine_ExplicitParams'**
                        # (因为风险低，且您认为它在一般情况下表现更好)
                        # if i % 10 == 0: # 可选的调试信息
                        #     print(f"调试 (样本 {i}): 低风险，使用 Affine_ExplicitParams 方法。")

                        # 直接使用预计算的范围来计算Ax, Bx, Ay, By
                        if abs(norm_range_x) < epsilon:
                            Ax, Bx = 0.0, origin_target[0]
                        else:
                            Ax = target_range_x / norm_range_x
                            Bx = origin_target[0] - Ax * origin_norm[0]

                        if abs(norm_range_y) < epsilon:
                            Ay, By = 0.0, origin_target[1]
                        else:
                            Ay = target_range_y / norm_range_y
                            By = origin_target[1] - Ay * origin_norm[1]

                        for j in range(traj_length):
                            denorm_trajectories[i, j, 0] = Ax * traj_norm[j, 0] + Bx
                            denorm_trajectories[i, j, 1] = Ay * traj_norm[j, 1] + By


            else:
                raise ValueError(f"Unknown denormalization method: {DENORM_METHOD}")
            # Reshape back to original format
            denormalized_list.append(denorm_trajectories.reshape(batch_size, traj_length * 2))

        return denormalized_list

    # def denormalize_trajectories_v1(self, trajectory_list, onehoted_condition_sample, dataset):
    #     """
    #     Denormalize trajectories using mapping_dict for visualization.
    #
    #     Args:
    #         trajectory_list: List of trajectory tensors to denormalize
    #         onehoted_condition_sample: One-hot encoded condition samples
    #         dataset: Dataset containing mapping dictionaries
    #
    #     Returns:
    #         denormalized_list: List of denormalized trajectories with preserved shapes
    #     """
    #     denormalized_list = []
    #
    #     # Get mappings from dataset
    #     onehot_od_dict = dataset.onehot_od_dict
    #     grid_mapping_dict = dataset.grid_mapping_dict
    #
    #     # Convert one-hot condition samples to coordinates
    #     condition_sample = np.zeros((len(onehoted_condition_sample), 2))
    #     for i in range(len(condition_sample)):
    #         condition_sample[i] = onehot_od_dict[np.where(onehoted_condition_sample[i] == 1)[0][0]]
    #
    #     # Convert to grid coordinates
    #     for i in range(condition_sample.shape[0]):
    #         for j in range(condition_sample.shape[1]):
    #             condition_sample[i][j] = grid_mapping_dict[condition_sample[i][j]].astype(int)
    #
    #     # Convert grid coordinates to lat/lon points
    #     condition_sample_lonlat = np.zeros((len(onehoted_condition_sample), 2, 2))
    #     for i in range(condition_sample.shape[0]):
    #         for j in range(condition_sample.shape[1]):
    #             condition_sample_lonlat[i, j, :] = ju.to_meshpoint(int(condition_sample[i][j]), 0.5, 0.5)
    #
    #     for trajectories in trajectory_list:
    #         # Reshape to (n_samples, M, 2) format
    #         batch_size = trajectories.shape[0]
    #         trajectories_reshaped = trajectories.reshape(batch_size, self.M, 2)
    #
    #         # Initialize output array
    #         denorm_trajectories = np.zeros_like(trajectories_reshaped)
    #
    #         # Process each trajectory in the batch
    #         for i in range(batch_size):
    #             # Get normalized trajectory
    #             traj = trajectories_reshaped[i].copy()
    #
    #             # Get origin and destination points
    #             origin_norm = traj[0]  # First point
    #             dest_norm = traj[-1]  # Last point
    #
    #             # Get target coordinates
    #             origin_target = condition_sample_lonlat[i, 0]
    #             dest_target = condition_sample_lonlat[i, 1]
    #
    #             # Calculate vectors
    #             norm_vector = dest_norm - origin_norm
    #             target_vector = dest_target - origin_target
    #
    #             # Calculate length of vectors
    #             norm_length = np.linalg.norm(norm_vector)
    #             target_length = np.linalg.norm(target_vector)
    #
    #             # Handle zero-length cases
    #             epsilon = 1e-6
    #             if norm_length < epsilon:
    #                 # If normalized trajectory is just a point, place it at the midpoint
    #                 for j in range(self.M):
    #                     ratio = j / (self.M - 1) if self.M > 1 else 0.5
    #                     denorm_trajectories[i, j] = origin_target + ratio * target_vector
    #                 continue
    #
    #             # Calculate uniform scale to preserve shape
    #             scale = target_length / norm_length
    #
    #             # Calculate relative positions for each point along the trajectory
    #             for j in range(self.M):
    #                 # Calculate relative vector from origin
    #                 rel_vector = traj[j] - origin_norm
    #
    #                 # Project this vector onto the direction of the normalized path
    #                 if norm_length > epsilon:
    #                     norm_dir = norm_vector / norm_length
    #                     proj_length = np.dot(rel_vector, norm_dir)
    #                     # Projection along the path direction
    #                     proj_vector = proj_length * norm_dir
    #                     # Perpendicular component (preserves shape)
    #                     perp_vector = rel_vector - proj_vector
    #
    #                     # Scale projection component to match target length
    #                     if target_length > epsilon:
    #                         target_dir = target_vector / target_length
    #                         scaled_proj = (proj_length / norm_length) * target_length * target_dir
    #                         # Apply scaled projection + perpendicular component (preserves shape)
    #                         denorm_trajectories[i, j] = origin_target + scaled_proj + scale * perp_vector
    #                     else:
    #                         # If target is a point, use simple interpolation
    #                         denorm_trajectories[i, j] = origin_target
    #                 else:
    #                     # If original path is a point, interpolate
    #                     ratio = j / (self.M - 1) if self.M > 1 else 0.5
    #                     denorm_trajectories[i, j] = origin_target + ratio * target_vector
    #
    #         # Reshape back to original format
    #         denormalized_list.append(denorm_trajectories.reshape(batch_size, self.M * 2))
    #
    #     return denormalized_list

    def evaluate(self):
        """
        Evaluate the model on the test dataset and generate samples

        Returns:
            metrics: Evaluation metrics
            samples: Generated samples
        """
        # Prepare directory for results
        results_dir = os.path.join(self.save_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)

        # Sampling parameters from config
        n_samples = self.config['inference']['n_samples']
        n_steps = self.config['inference']['sampling_steps']
        method = self.config['inference']['sampling_method']

        # Generate samples
        print(f"Generating {n_samples} samples using {method} method with {n_steps} steps...")
        sol = self.sample(n_samples, n_steps, method)

        # Get ground truth samples
        ground_truth = next(iter(self.dataloader))[:n_samples].to(self.device)
        if isinstance(ground_truth, tuple) and self.conditional:
            ground_truth = ground_truth[0]  # Only use data, not condition

        # Convert to numpy for visualization
        sol_np = [s.detach().cpu().numpy() for s in sol]
        ground_truth_np = ground_truth.detach().cpu().numpy()

        # Reconstrut normlized trajectories based on mapping_dict
        if (self.config['visualization']['norm1by12origialvis'] and
                self.dataset.mapping_dict is not None):
            print("Denormalizing data for visualization...")
            sol_np = self.denormalize_trajectories(sol_np)
            ground_truth_np = self.denormalize_trajectories([ground_truth_np])[0]

        # Visualize results
        print("Generating visualizations...")
        visualize_trajectories(sol_np, ground_truth_np, self.M, False, results_dir)
        visualize_density_comparison(sol_np, ground_truth_np, self.M, results_dir)

        # Compute evaluation metrics
        metrics = self.compute_metrics(sol[-1], ground_truth)

        # Save metrics
        self.save_metrics(metrics, results_dir)

        # If conditional, evaluate with different conditions
        if self.conditional:
            self.evaluate_with_conditions()

        return metrics, sol

    def compute_metrics(self, generated_samples, ground_truth):
        """
        Compute evaluation metrics

        Args:
            generated_samples: Generated samples
            ground_truth: Ground truth samples

        Returns:
            metrics: Dictionary of evaluation metrics
        """
        # Convert to numpy
        gen = generated_samples.detach().cpu().numpy()
        gt = ground_truth.detach().cpu().numpy()

        # Compute basic statistics
        mean_error = np.mean(np.abs(gen - gt))
        std_error = np.std(np.abs(gen - gt))

        # Compute distribution statistics
        gen_mean = np.mean(gen, axis=0)
        gt_mean = np.mean(gt, axis=0)

        gen_std = np.std(gen, axis=0)
        gt_std = np.std(gt, axis=0)

        mean_diff = np.mean(np.abs(gen_mean - gt_mean))
        std_diff = np.mean(np.abs(gen_std - gt_std))

        # Assemble metrics
        metrics = {
            'mean_absolute_error': float(mean_error),
            'std_absolute_error': float(std_error),
            'mean_difference': float(mean_diff),
            'std_difference': float(std_diff)
        }

        return metrics

    def save_metrics(self, metrics, save_dir):
        """
        Save metrics to file

        Args:
            metrics: Dictionary of metrics
            save_dir: Directory to save metrics
        """
        import json

        metrics_path = os.path.join(save_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        print(f"Metrics saved to {metrics_path}")

        # Also print metrics to console
        print("\nEvaluation Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.6f}")
