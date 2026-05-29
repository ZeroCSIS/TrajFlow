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
        # Initialize any extra keyword attributes on the manager instance.
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
                condition_sample[i][j] = int(grid_mapping_dict[onehoted_condition_sample[i][j]])

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
                # Apply anisotropic scaling so both endpoints match exactly.
                for i in range(batch_size):
                    # Work on a copy of the normalized trajectory to avoid in-place edits.
                    traj = trajectories_reshaped[i].copy()
                    # Normalized trajectory endpoints.
                    origin_norm = traj[0]
                    dest_norm = traj[-1]

                    # Target origin/destination coordinates.
                    origin_target = condition_sample_lonlat[i, 0]
                    dest_target = condition_sample_lonlat[i, 1]

                    # Compute scale factors for the x and y dimensions.
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

                    # Apply the transform to every point in the trajectory.
                    for j in range(traj_length):
                        denorm_trajectories[i, j, 0] = (traj[j, 0] - origin_norm[0]) * scale_x + origin_target[0]
                        denorm_trajectories[i, j, 1] = (traj[j, 1] - origin_norm[1]) * scale_y + origin_target[1]



            elif DENORM_METHOD == 'UniformPreserveAspect':
                epsilon = 1e-8 # Small number to avoid division by zero

                for i in range(batch_size):
                    traj_norm = trajectories_reshaped[i] # Shape (traj_length, 2)

                    if traj_norm.shape[0] < 2:  # Need at least two points to define the endpoints.
                        denorm_trajectories[i] = traj_norm
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
                        # More efficient calculation:
                        correction = start_error_vector + (end_error_vector - start_error_vector) * (1 - w)[:, np.newaxis]


                        final_denorm_traj = initial_denorm_traj + correction
                    else:
                        # If trajectory is empty, keep it empty
                        final_denorm_traj = initial_denorm_traj

                    # Assign the final, corrected trajectory to the output array
                    denorm_trajectories[i] = final_denorm_traj


            elif DENORM_METHOD == 'Affine_ExplicitParams':
            # --------------------------------------------------------------------
            # BEGINNING OF 'Affine_ExplicitParams' method logic (INLINED)
            # --------------------------------------------------------------------
                epsilon = 1e-7  # Threshold for near-zero floating-point comparisons.

                for i in range(batch_size):
                    traj_norm = trajectories_reshaped[i]  # Current normalized trajectory with shape (L, 2).

                    if traj_norm.shape[0] == 0:
                        continue

                    origin_norm = traj_norm[0]
                    dest_norm = traj_norm[-1] if traj_length > 1 else origin_norm

                    origin_target = condition_sample_lonlat[i, 0]  # Target origin (lon, lat).
                    dest_target = condition_sample_lonlat[i, 1]  # Target destination (lon, lat).

                    # --- Solve Ax and Bx in target_x = Ax * norm_x + Bx. ---
                    norm_val1_x = origin_norm[0]
                    target_val1_x = origin_target[0]
                    norm_val2_x = dest_norm[0]
                    target_val2_x = dest_target[0]

                    norm_range_x = norm_val2_x - norm_val1_x
                    target_range_x = target_val2_x - target_val1_x

                    Ax = 0.0
                    Bx = target_val1_x  # If norm_range_x is zero, map all points to target_val1_x.

                    if abs(norm_range_x) < epsilon:
                        # Keep Ax at 0.0.
                        # Keep Bx at target_val1_x.
                        if abs(target_range_x) >= epsilon:
                            print(f"Warning (sample {i}, X dimension, Affine_ExplicitParams): "
                                  f"normalized X range is near zero, but the target X range ({target_range_x:.2f}) is not. "
                                  f"All denormalized X coordinates will be set to the target origin X ({target_val1_x:.2f}). "
                                  f"The target destination X ({target_val2_x:.2f}) will therefore not be matched exactly by this transform.")
                    else:
                        # Regular case.
                        Ax = target_range_x / norm_range_x
                        Bx = target_val1_x - Ax * norm_val1_x

                    # --- Solve Ay and By in target_y = Ay * norm_y + By. ---
                    norm_val1_y = origin_norm[1]
                    target_val1_y = origin_target[1]
                    norm_val2_y = dest_norm[1]
                    target_val2_y = dest_target[1]

                    norm_range_y = norm_val2_y - norm_val1_y
                    target_range_y = target_val2_y - target_val1_y

                    Ay = 0.0
                    By = target_val1_y  # If norm_range_y is zero, map all points to target_val1_y.

                    if abs(norm_range_y) < epsilon:
                        # Keep Ay at 0.0.
                        # Keep By at target_val1_y.
                        if abs(target_range_y) >= epsilon:
                            print(f"Warning (sample {i}, Y dimension, Affine_ExplicitParams): "
                                  f"normalized Y range is near zero, but the target Y range ({target_range_y:.2f}) is not. "
                                  f"All denormalized Y coordinates will be set to the target origin Y ({target_val1_y:.2f}). "
                                  f"The target destination Y ({target_val2_y:.2f}) will therefore not be matched exactly by this transform.")
                    else:
                        # Regular case.
                        Ay = target_range_y / norm_range_y
                        By = target_val1_y - Ay * norm_val1_y

                    # Apply the transform to every point in the current trajectory.
                    for j in range(traj_length):
                        norm_px = traj_norm[j, 0]
                        norm_py = traj_norm[j, 1]

                        # Write results into denorm_trajectories.
                        denorm_trajectories[i, j, 0] = Ax * norm_px + Bx
                        denorm_trajectories[i, j, 1] = Ay * norm_py + By
            # --------------------------------------------------------------------
                # END OF 'Affine_ExplicitParams' method logic (INLINED)
                # --------------------------------------------------------------------

            elif DENORM_METHOD == 'SimilarityWithEndpointCorrection':
                epsilon = 1e-7  # Threshold for near-zero floating-point comparisons.

                for i in range(batch_size):
                    traj_norm = trajectories_reshaped[i]

                    if traj_norm.shape[0] == 0: continue

                    origin_norm = traj_norm[0]
                    dest_norm = traj_norm[-1] if traj_length > 1 else origin_norm

                    origin_target = condition_sample_lonlat[i, 0]
                    dest_target = condition_sample_lonlat[i, 1]

                    # --- Step 1: apply the initial full similarity transform. ---
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
                            print(f"Warning (sample {i}, SimilarityWithEndpointCorrection): "
                                  f"normalized O/D points coincide, but target O/D points do not. The trajectory will collapse to the target origin.")

                    # Compute the trajectory after the initial transform.
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

                    # --- Step 2: enforce endpoint correction to eliminate floating-point drift. ---
                    if traj_length > 1:
                        current_start = initial_denorm_traj[0]
                        current_end = initial_denorm_traj[-1]

                        start_error_vector = origin_target - current_start
                        end_error_vector = dest_target - current_end

                        # Create linearly interpolated weights from 1 to 0.
                        w = np.linspace(1, 0, traj_length)

                        # Distribute the endpoint error vectors across the full trajectory.
                        # correction[j] = w[j] * start_error_vector + (1 - w[j]) * end_error_vector
                        correction = w[:, np.newaxis] * start_error_vector + (1 - w)[:, np.newaxis] * end_error_vector

                        # Apply the correction to the initial transformed trajectory.
                        denorm_trajectories[i] = initial_denorm_traj + correction
                    else:  # For single-point trajectories, keep the initial transform result.
                        denorm_trajectories[i] = initial_denorm_traj

            elif DENORM_METHOD == 'HybridSimilarity':  # Higher-fidelity fallback strategy.
                epsilon = 1e-7
                # Ignore rotation when the angle stays below this threshold.
                # 5 degrees is a reasonable starting point and can be tuned per dataset.
                rotation_threshold_degrees = 5.0
                rotation_threshold_rad = np.deg2rad(rotation_threshold_degrees)

                for i in range(batch_size):
                    traj_norm = trajectories_reshaped[i]

                    if traj_norm.shape[0] == 0: continue

                    origin_norm = traj_norm[0]
                    dest_norm = traj_norm[-1] if traj_length > 1 else origin_norm

                    origin_target = condition_sample_lonlat[i, 0]
                    dest_target = condition_sample_lonlat[i, 1]

                    # --- Step 1: compute the scale factor s and rotation angle theta. ---
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
                    else:  # The normalized O/D points coincide.
                        if len_target > epsilon:
                            print(
                                f"Warning (sample {i}, HybridSimilarity): normalized O/D points coincide, but target O/D points do not. The trajectory will collapse to the target origin.")

                    # --- Step 2: decide whether the rotation should be applied. ---
                    if abs(theta) < rotation_threshold_rad:
                        # Treat very small angles as noise and skip rotation.
                        cos_theta = 1.0
                        sin_theta = 0.0
                    else:
                        # Apply rotation when the angle is significant.
                        cos_theta = np.cos(theta)
                        sin_theta = np.sin(theta)

                    # --- Step 3: apply uniform scaling, optional rotation, and translation. ---
                    initial_denorm_traj = np.zeros_like(traj_norm)
                    for j in range(traj_length):
                        point_norm = traj_norm[j]
                        # Translate to the origin.
                        p_centered_x = point_norm[0] - origin_norm[0]
                        p_centered_y = point_norm[1] - origin_norm[1]
                        # Scale.
                        p_scaled_x = p_centered_x * scale_s
                        p_scaled_y = p_centered_y * scale_s
                        # Rotate when required.
                        p_rotated_x = p_scaled_x * cos_theta - p_scaled_y * sin_theta
                        p_rotated_y = p_scaled_x * sin_theta + p_scaled_y * cos_theta
                        # Translate to the target origin.
                        initial_denorm_traj[j, 0] = p_rotated_x + origin_target[0]
                        initial_denorm_traj[j, 1] = p_rotated_y + origin_target[1]

                    # --- Step 4: enforce endpoint correction. ---
                    if traj_length > 1:
                        current_start = initial_denorm_traj[0]
                        current_end = initial_denorm_traj[-1]

                        start_error = origin_target - current_start
                        end_error = dest_target - current_end

                        # Only correct when the residual error is non-negligible.
                        if np.linalg.norm(start_error) > epsilon or np.linalg.norm(end_error) > epsilon:
                            w = np.linspace(1, 0, traj_length)
                            correction = w[:, np.newaxis] * start_error + (1 - w)[:, np.newaxis] * end_error
                            denorm_trajectories[i] = initial_denorm_traj + correction
                        else:  # Residual error is negligible; keep the initial result.
                            denorm_trajectories[i] = initial_denorm_traj
                    else:  # Single-point trajectory.
                        denorm_trajectories[i] = initial_denorm_traj

            elif DENORM_METHOD == 'MixStrategy':  # Adaptive blend between affine and similarity transforms.
                epsilon = 1e-7
                # Threshold used to flag extreme anisotropic distortion.
                anisotropy_threshold = 10.0
                # Rotation threshold reused by the HybridSimilarity branch.
                rotation_threshold_degrees = 5.0
                rotation_threshold_rad = np.deg2rad(rotation_threshold_degrees)

                for i in range(batch_size):
                    traj_norm = trajectories_reshaped[i]
                    if traj_norm.shape[0] == 0: continue

                    origin_norm = traj_norm[0]
                    dest_norm = traj_norm[-1] if traj_length > 1 else origin_norm
                    origin_target = condition_sample_lonlat[i, 0]
                    dest_target = condition_sample_lonlat[i, 1]

                    # --- Step 1: precompute affine scale factors to assess distortion risk. ---
                    norm_range_x = dest_norm[0] - origin_norm[0]
                    target_range_x = dest_target[0] - origin_target[0]
                    norm_range_y = dest_norm[1] - origin_norm[1]
                    target_range_y = dest_target[1] - origin_target[1]

                    # Compute absolute scale ratios with epsilon-protected denominators.
                    scale_x_abs = abs(target_range_x / (norm_range_x + epsilon)) if abs(
                        norm_range_x) > epsilon else float('inf')
                    scale_y_abs = abs(target_range_y / (norm_range_y + epsilon)) if abs(
                        norm_range_y) > epsilon else float('inf')

                    # --- Step 2: decide whether severe distortion is likely. ---
                    is_distorted = False
                    # A zero normalized range with a non-zero target range is a clear distortion signal.
                    if (abs(norm_range_x) < epsilon and abs(target_range_x) >= epsilon) or \
                            (abs(norm_range_y) < epsilon and abs(target_range_y) >= epsilon):
                        is_distorted = True
                    # Large scale-ratio mismatch between axes also indicates distortion.
                    elif min(scale_x_abs, scale_y_abs) > 0 and max(scale_x_abs, scale_y_abs) / min(scale_x_abs,
                                                                                                   scale_y_abs) > anisotropy_threshold:
                        is_distorted = True

                    # --- Step 3: select and apply the appropriate transform. ---
                    if is_distorted:
                        # Fallback to HybridSimilarity when distortion risk is high.
                        # This preserves shape better under extreme geometry changes.

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
                        # Use Affine_ExplicitParams when distortion risk is low.
                        # This is the default path for the typical low-risk case.

                        # Reuse the precomputed ranges to derive Ax, Bx, Ay, and By.
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
