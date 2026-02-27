import json
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from src.data.transforms import point2para, para2point
import jismesh.utils as ju
import src.utils.jismesh_v2.jismesh.utils as ju_v2
from tqdm import tqdm

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
        # print(f"Debug: Start-end distance near zero ({s:.2e}). Using fallback scale: {s_final:.2e}") # Optional debug info
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
        self.traj_length = config['data']['trajectory_length']
        self.traj_dim = self.traj_length * 2
        self.dataset_size = config['data']['sample_count']
        self.dataset_type = config['data']['dataset_type']
        self.output_dir = config['project']['output_dir']

        if self.dataset_type == "bwtraj":
            self._load_trajectory_data(config)
        elif self.dataset_type == "naive_circle":
            self._create_synthetic_data()
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

        # Add support for conditional flow
        self.conditional = config.get('condition', {}).get('enabled', False)
        if self.conditional:
            self.condition_type = config['condition']['condition_type']
            self._prepare_conditions()

        # visualization for test
        if config['data']['parametrized']:
            # parameterize example_points
            # self._visualize_example_parameterization(user_sample_id=25,
            #                                          para_M=config['data']['parametrized_M'],
            #                                          method = config['data']['parametrized_method'])
            # Convert to DCT coefficients

            # detect if exist the processed data, if not, then process
            processed_data_path = os.path.join(
                os.path.dirname(self.input_folder),
                f"processed_coeffs_{config['data']['region']}_"
                f"{config['data']['parametrized_method']}_"
                f"{config['data']['parametrized_M']}.npy"
            )
            if os.path.exists(processed_data_path):
                print(f"Loading pre-processed coefficients from {processed_data_path}")
                self.coffs = np.load(processed_data_path)
                self.traj_segments = self.coffs[:self.dataset_size]
            else:
                self._convert_to_coefficients(para_M=config['data']['parametrized_M'],
                                              method=config['data']['parametrized_method'])
            # if full dataset, saved the processed self.coffs to input dataset dir
            if config['data']['sample_count'] == -1:
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
               self.conditions = self.all_head
               # Because the sampled data is not the same as the original data, we need to record the OD mapping dict
               # Map the last two columns (o,d) to a shared range
               self.conditions[:, 6:8], self.onehot_mapping_dict, max_unique_length = (
                   map_two_columns_to_shared_range(self.conditions[:, 6:8]))
               self.conditions = self.conditions[:, 6:8]
               self.location_dim = max_unique_length
               # revalue the value of the grid_mapping_dict
               # replace the key with the value of the onehot_mapping_dict
               # drop the items that are not in the onehot_mapping_dict
               cr_sample_grid_mapping_dict = {}
               for key, value in self.onehot_mapping_dict.items():
                   cr_sample_grid_mapping_dict[value] = self.grid_mapping_dict[key]
               self.cr_sample_grid_mapping_dict = cr_sample_grid_mapping_dict
           elif self.condition_type == 'full':
               self.conditions = self.all_head
               # Because the sampled data is not the same as the original data, we need to record the OD mapping dict
               # Map the last two columns (o,d) to a shared range
               self.conditions[:, 6:8], self.onehot_mapping_dict, max_unique_length = (
                   map_two_columns_to_shared_range(self.conditions[:, 6:8]))
               self.location_dim = max_unique_length
               # revalue the value of the grid_mapping_dict
               # replace the key with the value of the onehot_mapping_dict
               # drop the items that are not in the onehot_mapping_dict
               cr_sample_grid_mapping_dict = {}
               for key, value in self.onehot_mapping_dict.items():
                   cr_sample_grid_mapping_dict[value] = self.grid_mapping_dict[key]
               self.cr_sample_grid_mapping_dict = cr_sample_grid_mapping_dict
           else:
               # Default empty conditions
               self.conditions = np.zeros((self.dataset_size, self.condition_dim))
           # Convert to tensor
           self.conditions = torch.FloatTensor(self.conditions)

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

        # Limit dataset size based on available data
        if self.dataset_size == -1:
            self.dataset_size = len(self.traj_segments)
        else:
            self.dataset_size = min(self.dataset_size, len(self.traj_segments))
        self.traj_segments = self.traj_segments[:self.dataset_size]
        self.all_head = self.all_head[:self.dataset_size]

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

    # def _convert_to_coefficients(self, para_M=10, method='dct'):
    #     """Convert trajectory points to DCT coefficients"""
    #     coeffs = np.zeros((self.dataset_size, para_M, 2))
    #     for i in tqdm(range(self.dataset_size)):
    #         if method == 'rdp_k':
    #             coeff = point2para(self.traj_segments[i],K=para_M, method=method)
    #             coeff = coeff['simplified_points']
    #         else:
    #             print(f"Warning: Using {method} for parameterization")
    #         coeffs[i] = coeff
    #     self.traj_segments = coeffs

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

        # Determine number of processes (use half of available cores)
        num_processes = max(1, mp.cpu_count()//2)
        print(f"Converting trajectories using {num_processes} processes...")

        # Create partial function with fixed parameters
        process_func = partial(self._process_trajectory, para_M=para_M, method=method)

        # Process in batches using multiprocessing Pool
        coeffs = np.zeros((self.dataset_size, para_M, 2))
        with mp.Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(process_func, self.traj_segments, chunksize=1000),
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

    def _visualize_example_parameterization(self, user_sample_id, para_M=10, method='dct'):
        """可视化参数化示例"""
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        vis_path = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(vis_path, exist_ok=True)
        # Visualize example
        points = self.traj_segments[user_sample_id]
        raw_points = self.traj_segments_before_stdize[user_sample_id]
        example_condition = self.all_head[user_sample_id]
        # get the raw points in real lat/lon
        raw_points = raw_points * self.traj_std + self.traj_mean
        # get the odfiner from the start and end points of raw_points
        _, o_lat_mult, o_lon_mult = ju_v2.to_meshcode(raw_points[0][0], raw_points[0][1], 3, return_multipliers=True)
        _, d_lat_mult, d_lon_mult = ju_v2.to_meshcode(raw_points[-1][0], raw_points[-1][1], 3, return_multipliers=True)
        lat_mult = np.array([[o_lat_mult, d_lat_mult]])
        lon_mult = np.array([[o_lon_mult, d_lon_mult]])

        # 根据方法设置适当的参数
        kwargs = {}
        if method == 'dct':
            kwargs['DCT_M'] = para_M
            method_label = f"DCT (M={para_M})"
        elif method == 'dct_deviation':
            kwargs['K'] = para_M  # 对于dct_deviation，使用K参数
            method_label = f"DCT偏差 (K={para_M})"
        elif method in ['anchor', 'spline_lsq']:
            kwargs['P'] = para_M  # 对于这些方法使用P参数
            method_label = f"{method} (P={para_M})"
        else:
            kwargs['K'] = para_M  # 默认情况
            method_label = f"{method} (K={para_M})"

        # 转换为参数表示
        para = point2para(points, method=method, **kwargs)

        if para is None:
            print(f"警告：方法{method}的参数化失败")
            return

        # 从参数重建点序列
        reconstructed_points = para2point(para, N_new=self.traj_length)

        # denormalize
        def denormalize_trajectories(self, trajectory_list, onehoted_condition_sample):
            """Denormalize trajectories using mapping_dict for visualization."""
            # if not self.norm1by1_viz or self.mapping_dict is None:
            #     return trajectory_list

            denormalized_list = []

            # Change the condition_sample to original od based on dataset
            grid_mapping_dict = self.grid_mapping_dict
            condition_sample = np.zeros((len(onehoted_condition_sample), 2))

            ENCODING_FORMAT = 'embedding'
            if ENCODING_FORMAT == 'onehot':
                onehot_od_dict = self.onehot_od_dict
                # Convert condition_sample from grid_mapping_dict
                for i in range(len(condition_sample)):
                    condition_sample[i] = onehot_od_dict[np.where(onehoted_condition_sample[i] == 1)[0][0]]
                # Convert od_sample to grid od, for each value of condition_sample_list
                # let every element in condition_sample to grid_mapping_dict[condition_sample]
                for i in range(condition_sample.shape[0]):
                    for j in range(condition_sample.shape[1]):
                        condition_sample[i][j] = grid_mapping_dict[condition_sample[i][j]].astype(int)
            elif ENCODING_FORMAT == 'embedding':
                # For embedding format, we need to:
                # 1. Get the inverse of onehot_mapping_dict to map embeddings back to indices
                # 2. Use those indices to look up the original encoded OD values
                onehot_mapping_dict = self.onehot_mapping_dict
                inverse_mapping = {v: k for k, v in onehot_mapping_dict.items()}
                # Convert od_sample to grid od, for each value of condition_sample_list
                # let every element in condition_sample to grid_mapping_dict[condition_sample]
                for i in range(condition_sample.shape[0]):
                    for j in range(condition_sample.shape[1]):
                        condition_sample[i][j] = inverse_mapping[onehoted_condition_sample[i][j]]
                        condition_sample[i][j] = grid_mapping_dict[condition_sample[i][j]].astype(int)
            else:
                raise ValueError("Unknown encoding format. Use 'onehot' or 'embedding'.")

            # Get the offset from the od's grid
            condition_sample_lonlat = np.zeros((len(onehoted_condition_sample), 2, 2))
            for i in range(condition_sample.shape[0]):
                for j in range(condition_sample.shape[1]):
                    condition_sample_lonlat[i, j, :] = ju.to_meshpoint(int(condition_sample[i][j]),
                                                                       lat_mult[i][j],
                                                                       lon_mult[i][j])

            for trajectories in trajectory_list:
                # Reshape to (n_samples, M, 2) format
                batch_size = trajectories.shape[0]
                traj_length = self.traj_length
                trajectories_reshaped = trajectories.reshape(batch_size, traj_length, 2)
                # Create output array
                denorm_trajectories = np.zeros_like(trajectories_reshaped)

                DENORM_METHOD = 'Affine'  # Choose denormalization method
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
                    epsilon = 1e-8  # Small number to avoid division by zero

                    for i in range(batch_size):
                        traj_norm = trajectories_reshaped[i]  # Shape (traj_length, 2)

                        if traj_norm.shape[0] < 2:  # 至少需要两个点来确定起终点
                            denorm_trajectories[i] = traj_norm  # 或者返回空/特定值？
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
                                print(
                                    f"Warning: Normalized start/end points are identical for batch item {i}, but target points differ. Cannot accurately recover scale. Using scale=1.0.")
                                s = 1.0  # Default scaling, likely inaccurate.

                        # --- Calculate the CORRECT offset based on the midpoint centering ---
                        # The transformation is: traj_target = traj_norm * s + center_point_target
                        # where center_point_target = (origin_target + dest_target) / 2.0
                        center_point_target = (origin_target + dest_target) / 2.0

                        # --- 1. Apply the initial inverse transformation ---
                        initial_denorm_traj = traj_norm * s + center_point_target

                        # --- 2. Calculate endpoint errors ---
                        if traj_length > 0:  # Ensure there are points to correct
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
                            correction = start_error_vector + (end_error_vector - start_error_vector) * (1 - w)[:,
                                                                                                        np.newaxis]

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

                else:
                    raise ValueError(f"Unknown denormalization method: {DENORM_METHOD}")
                # Reshape back to original format
                denormalized_list.append(denorm_trajectories.reshape(batch_size, traj_length * 2))

            return denormalized_list

        lat_mult = np.array([[o_lat_mult, d_lat_mult]])
        lon_mult = np.array([[o_lon_mult, d_lon_mult]])
        # denorm the reconstructed points
        reconstructed_points_realodfiner = np.expand_dims(reconstructed_points, axis=0).copy()
        reconstructed_points_realodfiner = denormalize_trajectories(self, [reconstructed_points_realodfiner], [example_condition[6:8]])[0]
        reconstructed_points_realodfiner = [x.reshape(self.traj_length, 2) for x in reconstructed_points_realodfiner]
        reconstructed_points_realodfiner = reconstructed_points_realodfiner[0]
        # denorm the original points
        raw_denorm_points_realodfiner = np.expand_dims(points, axis=0).copy()
        raw_denorm_points_realodfiner = denormalize_trajectories(self, [raw_denorm_points_realodfiner], [example_condition[6:8]])[0]
        raw_denorm_points_realodfiner = [x.reshape(self.traj_length, 2) for x in raw_denorm_points_realodfiner]
        raw_denorm_points_realodfiner = raw_denorm_points_realodfiner[0]

        lat_mult = np.array([[0.5, 0.5]])
        lon_mult = np.array([[0.5, 0.5]])
        # denorm the reconstructed points
        reconstructed_points = np.expand_dims(reconstructed_points, axis=0)
        reconstructed_points = denormalize_trajectories(self, [reconstructed_points], [example_condition[6:8]])[0]
        reconstructed_points = [x.reshape(self.traj_length, 2) for x in reconstructed_points]
        reconstructed_points = reconstructed_points[0]
        # denorm the original points
        raw_denorm_points = np.expand_dims(points, axis=0).copy()
        raw_denorm_points = denormalize_trajectories(self, [raw_denorm_points], [example_condition[6:8]])[0]
        raw_denorm_points = [x.reshape(self.traj_length, 2) for x in raw_denorm_points]
        raw_denorm_points = raw_denorm_points[0]


        if reconstructed_points is None:
            print(f"警告：方法{method}的重建失败")
            return

        # 计算误差
        mse1 = np.mean(np.sum((raw_points - reconstructed_points_realodfiner) ** 2, axis=1))
        mse2 = np.mean(np.sum((raw_points - reconstructed_points) ** 2, axis=1))

        # 创建可视化
        plt.figure(figsize=(10, 8))
        # set font for Chinese characters, not SimHei
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
        plt.plot(raw_points[:, 0], raw_points[:, 1], 'ro-', label='Original Trajectory', markersize=8)
        plt.plot(reconstructed_points_realodfiner[:, 0], reconstructed_points_realodfiner[:, 1], 'b^-',
                 label=f'{method_label} PARA+DENORM+Exact OD (MSE={mse1:.4e})', markersize=6)
        plt.plot(raw_denorm_points_realodfiner[:, 0], raw_denorm_points_realodfiner[:, 1], 'gx--',
                 label='DENORM+Exact OD', markersize=6)
        plt.plot(reconstructed_points[:, 0], reconstructed_points[:, 1], 'mv-',
                 label=f'{method_label} PARA+DENORM (MSE={mse2:.4e})', markersize=6)
        plt.plot(raw_denorm_points[:, 0], raw_denorm_points[:, 1], 'cd:', label='DENORM', markersize=6)
        plt.title(f'Trajectory Parameterization Method: {method}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend(loc='best')
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f"{vis_path}/trajectory_parameterization_{method}.png", dpi=300)
        plt.show()
        plt.close()

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
                    # data = super().__getitem__(idx) if hasattr(super(), '__getitem__') else self.data[idx].view(-1)
                    # if self.conditional:
                    #     return data, self.conditions[idx]
                    # return data

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
