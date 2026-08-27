import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import src.utils.jismesh_v2.jismesh.utils as ju_v2
from src.data.transforms import point2para


def map_two_columns_to_shared_range(input_array):
    # Flatten the array to get all integers in one list
    all_integers = input_array.flatten()

    # Get unique integers and create a mapping dictionary
    unique_integers = np.unique(all_integers)
    max_unique_length = len(unique_integers)
    mapping_dict = {num: i for i, num in enumerate(unique_integers)}

    # Map the integers in both columns to the new range
    mapped_array = np.vectorize(mapping_dict.get)(input_array)

    return mapped_array, mapping_dict, max_unique_length


def standardize_traj_start_end_scaling(traj, epsilon=1e-8):
    """
    Standardizes a trajectory by centering on the start-end midpoint
    and scaling uniformly based on the start-end distance. Preserves aspect ratio.
    Uses max standard deviation as fallback scaling if start/end points are coincident.

    Args:
        traj (np.ndarray): Trajectory array of shape (N, 2), expected float dtype.
        epsilon (float): Threshold for considering distances/std devs near zero.

    Returns:
        np.ndarray: Standardized trajectory array. Returns original shape with zeros
                    if input is empty, or centered if only one point.
    """
    N = traj.shape[0]

    # Handle empty or single-point trajectories
    if N == 0:
        # Return an empty array with the correct shape (N, 2)
        return np.empty((0, 2), dtype=traj.dtype)
    if N == 1:
        # Only one point, cannot scale. Return centered at origin.
        # Result is [[0., 0.]]
        return traj - traj[0]

    start = traj[0]
    end = traj[-1]

    # --- 1. Calculate primary scale factor: start-end distance ---
    vector_se = end - start
    s = np.linalg.norm(vector_se)

    # --- 2. Determine final scale factor, using fallback if s is too small ---
    use_fallback = (s < epsilon)

    if use_fallback:
        # Fallback: Use max standard deviation across dimensions
        try:
            # Ensure calculation is robust even if N=1 (though handled above)
            if N > 1:
                std_devs = traj.std(axis=0)
                s_fallback = np.max(std_devs)
                # Handle case where all points are identical (zero std dev)
                if s_fallback < epsilon:
                    s_final = 1.0  # No scaling needed if points are identical
                else:
                    s_final = s_fallback
            else:  # Should not happen due to N==1 check, but for safety
                s_final = 1.0
        except Exception:  # Catch potential errors with std calculation
            s_final = 1.0  # Safest fallback is no scaling
    else:
        s_final = s

    # --- 3. Center the trajectory on the midpoint of the start-end segment ---
    # This helps distribute points more evenly around the origin than centering on start point
    center_point = (start + end) / 2.0
    centered_traj = traj - center_point

    # --- 4. Apply uniform scaling ---
    # Avoid division by zero explicitly, although s_final should be >= 1.0 if fallback logic is correct
    if s_final < epsilon:
        # This case implies all points were identical AND start==end
        # centered_traj should already be all zeros. Return it.
        standardized_traj = centered_traj
    else:
        standardized_traj = centered_traj / s_final

    return standardized_traj


class FlowMatchingDataset(Dataset):
    def __init__(self, config, mode='train'):
        """
        Flow Matching dataset implementation

        Args:
            config: Configuration dictionary
            mode: 'train' or 'eval'
        """
        super().__init__()
        if mode not in {'train', 'val', 'test', 'eval', 'all'}:
            raise ValueError("mode must be one of train, val, test, eval, or all")
        self.mode = 'test' if mode == 'eval' else mode
        self.traj_length = config['data']['trajectory_length']
        self.traj_dim = self.traj_length * 2
        self.requested_sample_count = config['data']['sample_count']
        self.dataset_size = self.requested_sample_count
        self.dataset_type = config['data']['dataset_type']
        self.output_dir = config['project']['output_dir']
        self.split_indices_path = None
        self.parameterization_workers = config['data'].get('parameterization_workers')

        if self.dataset_type in {"didi", "open_source", "bwtraj"}:
            self._load_trajectory_data(config)
        elif self.dataset_type == "naive_circle":
            self._create_synthetic_data()
        else:
            raise ValueError(
                f"Unknown dataset type: {self.dataset_type}. "
                "Use 'didi' (or legacy 'bwtraj') for open-source data."
            )

        # Add support for conditional flow
        self.conditional = config.get('condition', {}).get('enabled', False)
        if self.conditional:
            self.condition_type = config['condition']['condition_type']
            self._prepare_conditions()

        if config['data']['parametrized']:
            # Reuse cached parameterized trajectories when available.
            if self.split_indices_path is None:
                cache_dir = os.path.dirname(self.input_folder)
                cache_suffix = ''
            else:
                cache_dir = self.input_folder
                cache_suffix = f"_{self.mode}"
            processed_data_path = os.path.join(
                cache_dir,
                f"processed_coeffs_{config['data']['region']}_"
                f"{config['data']['parametrized_method']}_"
                f"{config['data']['parametrized_M']}{cache_suffix}.npy"
            )
            cache_valid = False
            if os.path.exists(processed_data_path):
                print(f"Loading pre-processed coefficients from {processed_data_path}")
                self.coffs = np.load(processed_data_path)
                if len(self.coffs) == self.dataset_size:
                    self.traj_segments = self.coffs
                    cache_valid = True
                else:
                    print(
                        f"Ignoring coefficient cache with {len(self.coffs)} samples; "
                        f"current {self.mode} split has {self.dataset_size}"
                    )
                    self._convert_to_coefficients(
                        para_M=config['data']['parametrized_M'],
                        method=config['data']['parametrized_method'],
                    )
            else:
                self._convert_to_coefficients(para_M=config['data']['parametrized_M'],
                                              method=config['data']['parametrized_method'])
            # Cache complete split coefficients next to the converted dataset.
            if self.requested_sample_count == -1:
                if not cache_valid:
                    print(f"Saving processed coefficients to {processed_data_path}")
                    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
                    np.save(processed_data_path, self.coffs)
            else:
                print(f"Skipping saving coefficients (partial dataset with {self.dataset_size} samples)")
        else:
            pass

    def _prepare_conditions(self):
           """Prepare conditions based on configuration"""
           if self.condition_type == 'od':
               self.conditions = self.all_head.copy()
               # Because the sampled data is not the same as the original data, we need to record the OD mapping dict
               # Map the last two columns (o,d) to a shared range
               self._prepare_location_ids()
               self.conditions = self.conditions[:, 6:8]
           elif self.condition_type == 'full':
               self.conditions = self.all_head.copy()
               # Because the sampled data is not the same as the original data, we need to record the OD mapping dict
               # Map the last two columns (o,d) to a shared range
               self._prepare_location_ids()
           else:
               # Default empty conditions
               self.conditions = np.zeros((self.dataset_size, self.condition_dim))
           # Convert to tensor
           self.conditions = torch.FloatTensor(self.conditions)

    def _prepare_location_ids(self):
        """Prepare a stable location vocabulary, or retain the legacy local remap."""
        ids_are_mapped = self.grid_metadata.get('condition_ids_are_mapped', False)
        fixed_location_dim = self.grid_metadata.get('fixed_location_dim')
        if ids_are_mapped and fixed_location_dim is not None:
            location_ids = self.conditions[:, 6:8].astype(np.int64)
            if location_ids.size and (
                location_ids.min() < 0 or location_ids.max() >= int(fixed_location_dim)
            ):
                raise ValueError("Condition location id is outside fixed_location_dim")
            self.conditions[:, 6:8] = location_ids
            self.location_dim = int(fixed_location_dim)
            observed_ids = np.unique(location_ids)
            self.onehot_mapping_dict = {int(value): int(value) for value in observed_ids}
            self.cr_sample_grid_mapping_dict = dict(self.grid_mapping_dict)
            return

        mapped, mapping_dict, max_unique_length = map_two_columns_to_shared_range(
            self.conditions[:, 6:8]
        )
        self.conditions[:, 6:8] = mapped
        self.onehot_mapping_dict = mapping_dict
        self.location_dim = max_unique_length
        self.cr_sample_grid_mapping_dict = {
            value: self.grid_mapping_dict[key] for key, value in mapping_dict.items()
        }

    def _load_trajectory_data(self, config):
        """Load trajectory data from open-source dataset folders."""
        from data_utils import PrepareDataset

        # Define dataset folders
        PROJ_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        FOLDERS = {
            'Chengdu': f'{PROJ_PATH}/data/DiDiTaxi_Chengdu_traj',
            'XiAn': f'{PROJ_PATH}/data/DiDiTaxi_XiAn_traj',
        }
        custom_folder = config['data'].get('dataset_folder', '')
        if custom_folder:
            if os.path.isabs(custom_folder):
                self.input_folder = custom_folder
            else:
                self.input_folder = os.path.join(PROJ_PATH, custom_folder)
        else:
            region = config['data']['region']
            if region not in FOLDERS:
                raise ValueError(f"Unsupported open-source region: {region}. Use Chengdu or XiAn.")
            self.input_folder = FOLDERS[region]
            if not os.path.exists(self.input_folder):
                import glob

                fallback_candidates = glob.glob(f"{self.input_folder}*")
                fallback_candidates = [p for p in fallback_candidates if os.path.isdir(p)]
                if fallback_candidates:
                    self.input_folder = sorted(fallback_candidates)[0]
        self.grid_encoding = 'jismesh'
        self.grid_metadata = {}
        (self.all_head, self.traj_mean, self.traj_std, self.lengths,
         self.cond_mean, self.cond_std, self.traj_segments, self.grid_mapping_dict) = PrepareDataset.loadExistingData(
            self.input_folder, resample_length=self.traj_length)

        meta_path = os.path.join(self.input_folder, 'grid_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                try:
                    self.grid_metadata = json.load(f)
                except json.JSONDecodeError:
                    self.grid_metadata = {}
            self.grid_encoding = self.grid_metadata.get('encoding', 'jismesh')
        elif config['data']['region'] in ('Chengdu', 'XiAn'):
            precision = config['data'].get('geohash_precision', 6)
            self.grid_encoding = 'geohash'
            self.grid_metadata = {'encoding': 'geohash', 'geohash_precision': precision}

        split_filename = config['data'].get('split_file', 'split_indices.npz')
        candidate_split_path = os.path.join(self.input_folder, split_filename)
        self.split_indices_path = candidate_split_path if os.path.exists(candidate_split_path) else None
        available_count = len(self.traj_segments)
        if len(self.all_head) != available_count:
            raise ValueError("conditions.pkl and traj_segments.pkl have different sample counts")
        if self.split_indices_path is not None and self.mode != 'all':
            with np.load(self.split_indices_path) as split_file:
                if self.mode not in split_file.files:
                    raise ValueError(
                        f"Split file {self.split_indices_path} does not contain '{self.mode}'"
                    )
                selected_indices = np.asarray(split_file[self.mode], dtype=np.int64)
            if selected_indices.size and (
                selected_indices.min() < 0 or selected_indices.max() >= available_count
            ):
                raise ValueError(f"Split '{self.mode}' contains an out-of-range sample index")
        else:
            selected_indices = np.arange(available_count, dtype=np.int64)

        if self.requested_sample_count != -1:
            selected_indices = selected_indices[:self.requested_sample_count]
        if selected_indices.size == 0:
            raise ValueError(f"The '{self.mode}' dataset split is empty")
        self.sample_indices = selected_indices
        self.traj_segments = np.asarray(self.traj_segments)[selected_indices]
        self.all_head = np.asarray(self.all_head)[selected_indices]
        self.lengths = np.asarray(self.lengths)[selected_indices]
        self.dataset_size = len(selected_indices)

        # get the odfiner in config, if not, set as False
        self.od_finer = config['data'].get('od_finer', False)
        if self.od_finer:
            # Get the inner location within OD mesh
            self.all_od_finer_paras = np.zeros((self.dataset_size, 4))  # [lon, lat, lon, lat]
            for i in range(self.dataset_size):
                # Get the first and last points of the trajectory
                start_point = self.traj_segments[i][0]  # First point coordinates [x, y]
                end_point = self.traj_segments[i][-1]   # Last point coordinates [x, y]
                # Convert to actual coordinates by * traj_std + traj_mean
                start_point = start_point * self.traj_std + self.traj_mean
                end_point = end_point * self.traj_std + self.traj_mean
                # Calculate relative position within origin and destination grids
                meshcode, o_lat_mult, o_lon_mult = ju_v2.to_meshcode(start_point[0], start_point[1], 3, return_multipliers=True)
                meshcode, d_lat_mult, d_lon_mult = ju_v2.to_meshcode(end_point[0], end_point[1], 3, return_multipliers=True)
                # Store relative positions (between 0.0 and 1.0)
                self.all_od_finer_paras[i] = [o_lat_mult, o_lon_mult,d_lat_mult, d_lon_mult]  # Or store both O&D if needed
        else:
            pass

        # Normalize trajectories
        if config['data']['norm1by1']:
            # Store the original data before standardization
            self.traj_segments_before_stdize = self.traj_segments.copy()
            # Standardize data
            self.standardize_trajectories_preserve_aspect()
        else:
            pass

    def _standardize_trajectories(self):
        """Standardize trajectory data"""
        # Apply standardization for each trajectory
        for i in range(self.dataset_size):
            self.traj_segments[i] = (self.traj_segments[i] -
                                     self.traj_segments[i].mean(axis=0)) / self.traj_segments[i].std(axis=0)

    def _standardize_trajectories_v3(self):
        """
        Applies the start-end distance based standardization to all trajectories.
        This method modifies self.traj_segments in place.
        """
        print("Applying Start-End Distance Standardization...")
        standardized_segments = []
        for i in range(self.dataset_size):
            # Apply the new standardization method to each trajectory
            standardized = standardize_traj_start_end_scaling(self.traj_segments[i])
            standardized_segments.append(standardized)
        # Replace original trajectories with the standardized versions
        self.traj_segments = standardized_segments
        print("Standardization complete.")

    def standardize_trajectories_preserve_aspect(self):
        """
        Standardize trajectory data by centering and scaling while preserving aspect ratio.

        Args:
            traj_segments (list of np.ndarray): List of trajectories, where each trajectory
                                                is a numpy array of shape (N, 2).

        Returns:
            list of np.ndarray: Standardized trajectories with mean=0, std≈1, and aspect ratio preserved.
        """
        standardized_segments = []

        for traj in self.traj_segments:
            # Step 1: Center the trajectory (mean = 0)
            mean = traj.mean(axis=0)  # (2,)
            centered_traj = traj - mean

            # Step 2: Calculate the overall standard deviation (preserve aspect ratio)
            # Overall std is based on the distance of points from the center
            std = np.sqrt(np.mean(np.sum(centered_traj ** 2, axis=1)))  # Scalar

            # Handle cases where std is very small to avoid division by zero
            epsilon = 1e-8
            if std < epsilon:
                std = 1.0  # If trajectory is degenerate (all points are the same)

            # Step 3: Scale the trajectory
            standardized_traj = centered_traj / std

            # Append to the results
            standardized_segments.append(standardized_traj)

        self.traj_segments = standardized_segments


    def _create_synthetic_data(self):
        """Create synthetic circular data"""
        samples = []
        for i in range(self.traj_length):
            angle = torch.rand(self.dataset_size) * 2 * np.pi
            radius = 0.8 + 0.2 * torch.randn(self.dataset_size).abs()
            offset = i * 0.5
            x = radius * torch.cos(angle) + offset
            y = radius * torch.sin(angle) + offset
            samples.append(torch.stack([x, y], dim=1))

        points_data = torch.stack(samples, dim=1)  # Shape: [dataset_size, M, 2]

        # Convert to DCT coefficients
        dct_coeffs = []
        for i in range(self.dataset_size):
            coeff = point2para(points_data[i].numpy(), method='dct')
            dct_coeffs.append(coeff)

        self.data = torch.FloatTensor(np.array(dct_coeffs))  # Shape: [dataset_size, M, 2]

    def _process_trajectory(self, traj, para_M, method):
        """Process a single trajectory - worker function for multiprocessing"""
        if method == 'rdp_k':
            para_dict = point2para(traj, K=para_M, method=method)
            if para_dict is not None:
                return para_dict['simplified_points']
            else:
                return np.zeros((para_M, 2))  # Fallback if parameterization fails
        elif method == 'rdp_k_withod':
            # Special case for methods that need start/end point info
            para_dict = {
                'simplified_points': traj,
                'start_point': traj[0],
                'end_point': traj[-1]
            }
            para_dict = point2para(traj, K=para_M, method=method, **para_dict)
            if para_dict is not None:
                return para_dict['simplified_points']
            else:
                return np.zeros((para_M, 2))
        else:
            print(f"Warning: Using {method} for parameterization")
            return point2para(traj, method=method)

    def _convert_to_coefficients(self, para_M=10, method='dct'):
        """Convert trajectory points to coefficients using multiprocessing for improved performance"""
        import multiprocessing as mp
        from functools import partial

        # One worker is intentionally supported for notebooks, stdin-driven
        # checks, and platforms whose multiprocessing start method is spawn.
        if self.parameterization_workers is None:
            num_processes = max(1, mp.cpu_count() // 2)
        else:
            num_processes = max(1, int(self.parameterization_workers))
        num_processes = min(num_processes, self.dataset_size)
        print(f"Converting trajectories using {num_processes} worker(s)...")

        # Create partial function with fixed parameters
        process_func = partial(self._process_trajectory, para_M=para_M, method=method)

        coeffs = np.zeros((self.dataset_size, para_M, 2))
        if num_processes == 1:
            results = list(tqdm(
                map(process_func, self.traj_segments),
                total=self.dataset_size,
                desc=f"Parameterizing with {method}"
            ))
        else:
            chunksize = max(1, min(1000, self.dataset_size // (num_processes * 4)))
            with mp.Pool(processes=num_processes) as pool:
                results = list(tqdm(
                    pool.imap(process_func, self.traj_segments, chunksize=chunksize),
                    total=self.dataset_size,
                    desc=f"Parameterizing with {method}"
                ))

        # Collect results
        for i, result in enumerate(results):
            if result is not None:
                coeffs[i] = result
            else:
                print(f"Warning: Parameterization failed for trajectory {i}")
        self.coffs = coeffs
        self.traj_segments = self.coffs

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
                    if not isinstance(idx, torch.Tensor):
                        pass  # idx = idx.item()

                    if self.conditional:
                        if hasattr(self, 'od_finer') and self.od_finer:
                            # Concatenate trajectory segments and od_finer parameters
                            traj_segment = torch.Tensor(self.traj_segments[idx])
                            od_finer_params = torch.Tensor(self.all_od_finer_paras[idx])
                            # Combine trajectory with OD finer parameters
                            combined_data = torch.cat([traj_segment.view(-1), od_finer_params], dim=0)
                            # Return combined data and conditions
                            return combined_data, torch.Tensor(self.conditions[idx])
                        else:
                            # Return just trajectory and conditions
                            return torch.Tensor(self.traj_segments[idx]), torch.Tensor(self.conditions[idx])
                    else:
                        return torch.Tensor(self.traj_segments[idx])  # Flatten to [M*2] for model input
