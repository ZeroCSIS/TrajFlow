"""
This preprocessing module documents the pipeline originally used for the
private BW dataset. It is kept in the open-source release so users can
understand how the training data is transformed and what processed format is
expected by the later stages of the project.

Due to data policy restrictions, the BW dataset itself is not distributed in
this repository. Users can preprocess an authorized public dataset, such as
DiDi Taxi, into the same format and use it with the released code.
"""

import numpy as np
import torch
from data_utils import MiniTools
import pandas as pd
import random
import datetime
import jismesh.utils as ju
from torch.utils.data import TensorDataset
from sklearn.preprocessing import LabelEncoder
import pygeohash as pgh
transmode_switcher = {'WALK': 0, 'CAR': 1, 'BUS': 2, 'TRAIN': 3, 'BIKE': 4}
jismesh_switcher = {80000:1, 40000:40, 20000:20, 16000:16,
             10000:2, 8000:8, 5000:5, 4000:4, 2500:2.5, 2000:2,
             1000:3, 500:4, 250:5, 125:6}

def geohash_to_binary(geohash):
    # Define the base32 map
    base32_map = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                  'b': 10, 'c': 11, 'd': 12, 'e': 13, 'f': 14, 'g': 15, 'h': 16, 'j': 17, 'k': 18,
                  'm': 19, 'n': 20, 'p': 21, 'q': 22, 'r': 23, 's': 24, 't': 25, 'u': 26, 'v': 27,
                  'w': 28, 'x': 29, 'y': 30, 'z': 31}

    # Convert the geohash to base10
    base10 = 0
    for char in geohash:
        base10 = base10 * 32 + base32_map[char]

    # Convert the base10 to binary
    binary = bin(base10)[2:]

    return binary


# Place all the data preparation functions here
# gather(), time_to_decimal(), map_two_columns_to_shared_range(), getCondTraj(), pad_arrays_to_uniform_size(), resample_trajectory(), loadExistingCondition(), loadExistingData(), get_traj_data()
def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)

# Function to convert time string to decimal hours
def time_to_decimal(time_string):
    if type(time_string) is str:
        # Parse the time string to a datetime object
        time_obj = datetime.datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S')
    else:
        time_string = str(time_string)
        time_obj = datetime.datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S')
    # Extract hours and minutes
    hours = time_obj.hour
    minutes = time_obj.minute
    seconds = time_obj.second

    # Convert to decimal
    decimal_hours = hours + minutes / 60 + seconds / 3600

    return decimal_hours

# Function to map two columns of integers to a shared range and return two columns
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

def getCondTraj(traj_df):
    group = traj_df
    group = group[group['lat'] != 0]
    group = group[group['lon'] != 0].reset_index()
    # jismesh standard levels: 4 = 500 m, 3 = 1 km.
    MESH_DEGREE = jismesh_switcher[config.data.grid_size]
    group['meshocode']  = group.apply(lambda row:ju.to_meshcode(row['lat'],row['lon'],MESH_DEGREE),axis=1)

    # Preparing a list to store the results
    conditions = []
    traj_segments = []

    # Calculate departure time (start time of the first point)
    departure = group.iloc[0]['time']
    departure = time_to_decimal(departure)
    #devide the 24 hours into 5-minute-interval int
    departure = int(departure // 0.0833333)
    # Calculate total distance
    total_dis = sum(MiniTools.haversine_distance(lat1, lon1, lat2, lon2) for (lat1, lon1), (lat2, lon2) in
                    zip(group[['lat', 'lon']].values, group[['lat', 'lon']].values[1:]))*1000

    # Calculate total time (in seconds)
    start_time = datetime.datetime.strptime(str(group.iloc[0]['time']), '%Y-%m-%d %H:%M:%S')
    end_time = datetime.datetime.strptime(str(group.iloc[-1]['time']), '%Y-%m-%d %H:%M:%S')
    total_time = (end_time - start_time).total_seconds()

    # Total length of the segment (number of points)
    total_len = len(group)

    # Calculate average distance (total distance / total length)
    avg_dis = total_dis / total_len if total_len > 0 else 0
    # Calculate average speed (total distance / total time in hours)
    avg_speed = (total_dis / (total_time)) if total_time > 0 else 0

    # Starting and ending locations
    starting_location = group.iloc[0]['meshocode']
    ending_location = group.iloc[-1]['meshocode']

    # Get the transport mode
    transmode = group.iloc[0]['trans_mode2']
    transmode = transmode_switcher[transmode]

    # Append the result for this segment
    if total_len >= config.data.length_min and total_len <= config.data.length_max:
        conditions.append({
            'departure': departure,
            'total_dis': total_dis, #m
            'total_time': total_time, #s
            'total_len': total_len,
            'avg_dis': avg_dis, #m
            'avg_speed': avg_speed, #m/s
            'starting_location': starting_location,
            'ending_location': ending_location,
            'trans_mode': transmode
        })
    else:
        return [], []
    traj_segments.append(group[['lat', 'lon']].values)

    # Convert the results to a DataFrame
    conditions = pd.DataFrame(conditions).values

    return conditions,traj_segments

# Function like pad_arrays_to_uniform_size, but interpolation is used instead of padding
# Input is 2-D array with lat/lon
def resample_trajectory(data, max_length):
    data = data[0]
    # Calculate the number of points to interpolate
    num_points = max_length - data.shape[0]

    # Check if there are points to interpolate
    if num_points > 0:
        # interpolate from data.shape[0] to max_length for data
        new_data = np.zeros((max_length, data.shape[1]))
        for i in range(data.shape[1]):
            new_data[:, i] = np.interp(np.linspace(0, data.shape[0] - 1, max_length),
                                       np.arange(data.shape[0]), data[:, i])
    else:
        # longer than max_length, resample the trajectory to max_length
        # Resample, not cut
        new_data = np.zeros((max_length, data.shape[1]))
        for i in range(data.shape[1]):
            new_data[:, i] = np.interp(np.linspace(0, data.shape[0] - 1, max_length),
                                       np.arange(data.shape[0]), data[:, i])
    return [new_data]

# Function like pad_arrays_to_uniform_size, but interpolation is used instead of padding
# Input is a 3-D array with lat/lon.

def resample_existed_trajectory(data, max_length):
    length = data.shape[1]
    # Calculate the number of points to interpolate
    num_points = max_length - length

    # Check if there are points to interpolate
    if num_points > 0:
        # interpolate from length to max_length for data
        new_data = np.zeros((data.shape[0], max_length, data.shape[2]))
        for i in range(data.shape[0]):
            for j in range(data.shape[2]):
                new_data[i, :, j] = np.interp(np.linspace(0, length - 1, max_length),
                                              np.arange(length), data[i, :, j])
    else:
        # longer than max_length, resample the trajectory to max_length
        # Resample, not cut
        new_data = np.zeros((data.shape[0], max_length, data.shape[2]))
        for i in range(data.shape[0]):
            for j in range(data.shape[2]):
                new_data[i, :, j] = np.interp(np.linspace(0, length - 1, max_length),
                                              np.arange(length), data[i, :, j])
    return new_data

def loadExistingCondition(HEAD_PATH,TRAJ_ADJUST_PATH,HEAD_ADJUST_PATH):
    # Load the head data
    head = MiniTools.loadPKL(HEAD_PATH)
    with open(TRAJ_ADJUST_PATH, 'r') as f:
        traj_mean = []
        traj_std = []
        for i in range(2):
            traj_mean.append(float(f.readline().split(':')[-1]))
            traj_std.append(float(f.readline().split(':')[-1]))

    with open(HEAD_ADJUST_PATH, 'r') as f:
        cond_mean = []
        cond_std = []
        for i in range(5):
            cond_mean.append(float(f.readline().split(':')[-1]))
            cond_std.append(float(f.readline().split(':')[-1]))

    len_std = cond_std[2]
    len_mean = cond_mean[2]

    lengths = head[:, 3]
    lengths = lengths * len_std + len_mean
    lengths = lengths.astype(int)

    return head, traj_mean, traj_std, lengths, cond_mean, cond_std


def loadExistingData(FOLDER_PATH,resample_length = -1):
    HEAD_PATH = '%s/conditions.pkl'%(FOLDER_PATH)
    TRAJ_ADJUST_PATH = '%s/traj_mean_std.txt'%(FOLDER_PATH)
    HEAD_ADJUST_PATH = '%s/conditions_mean_std.txt'%(FOLDER_PATH)
    GT_DATA_PATH = '%s/traj_segments.pkl'%(FOLDER_PATH)
    GRID_MAPPING_PATH = '%s/mesh_mapping_dict.pkl'%(FOLDER_PATH)

    grid_mapping_dict = MiniTools.loadPKL(GRID_MAPPING_PATH)
    grid_dim = len(grid_mapping_dict)
    #exchange the key and value in the grid_mapping_dict
    grid_mapping_dict = {v: k for k, v in grid_mapping_dict.items()}

    # Load the gt data
    all_gt_data = MiniTools.loadPKL(GT_DATA_PATH)
    # Load the condition
    all_head, traj_mean, traj_std, lengths,cond_mean, cond_std = loadExistingCondition(HEAD_PATH,TRAJ_ADJUST_PATH,HEAD_ADJUST_PATH)

    # Resample the trajectory if needed
    if resample_length > 0:
        all_gt_data = resample_existed_trajectory(all_gt_data, resample_length)

    # Sample the data according to the config.data.user_num
    try:
        if config.data.user_num > 0:
            all_head = all_head[:config.data.user_num]
            all_gt_data = all_gt_data[:config.data.user_num]
    except:
        pass
    return all_head, traj_mean, traj_std, lengths, cond_mean, cond_std, all_gt_data, grid_mapping_dict

def get_traj_data(user_num=1000, sampling_segments_per_user=2000,files_save = '',random_seed=0,config=None):
    # Load a preprocessed dataset when available.
    if not config.data.load_existing:
        raise NotImplementedError(
            "The open-source release only supports preprocessed datasets. "
            "Set config.data.load_existing=True and point existing_data_folder to the prepared data directory."
        )

    FOLDER_PATH = config.data.existing_data_folder
    (input_condtions, traj_mean, traj_std, lengths, cond_mean, cond_std,
     input_traj_segments, grid_mapping_dict) = loadExistingData(
        FOLDER_PATH)
    input_traj_segments = resample_existed_trajectory(input_traj_segments, config.data.traj_length)
    if config.data.geohash == True:
        # Convert the condition[6,7] to jismesh
        o_list = [int(grid_mapping_dict[x]) for x in input_condtions[:, 6]]
        d_list = [int(grid_mapping_dict[x]) for x in input_condtions[:, 7]]
        # Convert the jismesh to lat/lon
        o_list = [ju.to_meshpoint(x,0.5,0.5) for x in o_list]
        d_list = [ju.to_meshpoint(x,0.5,0.5) for x in d_list]
        # Convert the lat/lon to geohash
        o_list = [geohash_to_binary(pgh.encode(x[0],x[1],precision=6)) for x in o_list]
        d_list = [geohash_to_binary(pgh.encode(x[0],x[1],precision=6)) for x in d_list]
        # Replace the original o,d with the geohash
        def string_to_int_vector(s):
            return [int(c) for c in s]
        o_geohash = np.array([string_to_int_vector(x) for x in o_list])
        d_geohash = np.array([string_to_int_vector(x) for x in d_list])
        input_condtions = np.concatenate([input_condtions,o_geohash,d_geohash],axis=1)
        grid_dim = 6 *5 # precision * 5
        config.model.grid_dim = grid_dim
    else:
        grid_dim = len(grid_mapping_dict)
        config.model.grid_dim = grid_dim

    return (input_condtions, input_traj_segments, grid_dim, cond_mean, cond_std,
            traj_mean, traj_std)

def prepare_data(input_config,exp_dir=''):
    # let config be the global variable in this python file
    global config
    config = input_config

    if config.data.load_existing != True:
        raise NotImplementedError(
            "The open-source release does not include private raw-data preprocessing. "
            "Use preprocessed data via config.data.existing_data_folder."
        )

    # Get the random seed
    random_seed = 0
    # Prepare the data
    head, traj, grid_dim, cond_mean, cond_std, traj_mean, traj_std = get_traj_data(
        files_save=exp_dir,
        random_seed=random_seed,
        config=config,
    )

    if config.data.norm_1by1 == True:
        # Normalize the traj one by one
        for m in range(traj.shape[0]):
            # get the geohash range (normalized by the lat/lon mean and std version)
            geohash_0 = head[m, -2 * grid_dim:-1 * grid_dim]
            geohash_d = head[m, -1 * grid_dim:]
            # get lat_min, lon_min, lat_max, lon_max from the traj
            lat_min, lon_min, lat_max, lon_max = (traj[m, :, 0].min(), traj[m, :, 1].min(),
                                                  traj[m, :, 0].max(), traj[m, :, 1].max())
            #Re-Normalize by the lat/lon min max
            if lat_max - lat_min == 0:
                traj[m, :, 0] = 0 + random.random() * 0.001
            else:
                traj[m, :, 0] = (traj[m, :, 0] - lat_min) / (lat_max - lat_min)
            #Normalize the lon
            if lon_max - lon_min == 0:
                traj[m, :, 1] = 0 + random.random() * 0.001
            else:
                traj[m, :, 1] = (traj[m, :, 1] - lon_min) / (lon_max - lon_min)
    else:
        pass
    # Swap the axes of the traj
    traj = np.swapaxes(traj, 1, 2)
    traj = torch.from_numpy(traj).float()
    head = torch.from_numpy(head).float()

    if config.model.classifier_type == 'classifier':
        # get the 7~8 of the head
        od_head = head[:, 6:8]
        unique_class_num = len(np.unique(od_head))
        # Initialize the LabelEncoder
        label_encoder = LabelEncoder()
        # Fit the encoder and transform the head to range [0, unique_class_num-1]
        head_encoded = label_encoder.fit_transform(od_head.flatten())
        # Save the mapping for later restoration
        head_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
        # Update the head with the encoded values
        od_head = head_encoded.reshape(od_head.shape)
        head[:, 6:8] = torch.from_numpy(od_head).long()
        # Get the number of unique classes of the head
        config.model.classifier_class_num = [unique_class_num for i in range(2)]    # Define a condition to decide whether to use fit_transform_all_data
        config.data.grid_size = unique_class_num

    dataset = TensorDataset(traj, head)
    return dataset, head, traj, grid_dim, cond_mean, cond_std, traj_mean, traj_std

if __name__ == '__main__':
    raise SystemExit(
        "data_utils/PrepareDataset.py CLI preprocessing is disabled in the open-source package. "
        "Use prepared public datasets under ./data/."
    )
