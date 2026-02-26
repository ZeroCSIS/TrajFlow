import sys
import os
import numpy as np
import torch
from data_utils import MiniTools
import pandas as pd
import random
import datetime
import jismesh.utils as ju
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from types import SimpleNamespace
import geopandas as gpd
from sklearn.preprocessing import LabelEncoder
import pygeohash as pgh
transmode_switcher = {'WALK': 0, 'CAR': 1, 'BUS': 2, 'TRAIN': 3, 'BIKE': 4}
# transmode_switcher = {'WALK': 0, 'CAR': 1, 'TRAIN': 3, 'BIKE': 4}
jismesh_switcher = {80000:1, 40000:40, 20000:20, 16000:16,
             10000:2, 8000:8, 5000:5, 4000:4, 2500:2.5, 2000:2,
             1000:3, 500:4, 250:5, 125:6}
# 1次(標準地域メッシュ 80km四方): 1
# 40倍(拡張統合地域メッシュ 40km四方): 40000
# 20倍(拡張統合地域メッシュ 20km四方): 20000
# 16倍(拡張統合地域メッシュ 16km四方): 16000
# 2次(標準地域メッシュ 10km四方): 2
# 8倍(拡張統合地域メッシュ 8km四方): 8000
# 5倍(統合地域メッシュ 5km四方): 5000
# 4倍(拡張統合地域メッシュ 4km四方): 4000
# 2.5倍(拡張統合地域メッシュ 2.5km四方): 2500
# 2倍(統合地域メッシュ 2km四方): 2000
# 3次(標準地域メッシュ 1km四方): 3
# 4次(分割地域メッシュ 500m四方): 4
# 5次(分割地域メッシュ 250m四方): 5
# 6次(分割地域メッシュ 125m四方): 6

def geohash_to_binary(geohash):
    # _PRECISION = {
    #     0: 20000000,
    #     1: 5003530,
    #     2: 625441,
    #     3: 123264,
    #     4: 19545,
    #     5: 3803,
    #     6: 610,
    #     7: 118,
    #     8: 19,
    #     9: 3.71,
    #     10: 0.6,
    # }
    # __base32 = '0123456789bcdefghjkmnpqrstuvwxyz'

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
    # if DATA_SOURCE == 'BW':
    group = traj_df
    # #Exclude error record
    group = group[group['lat'] != 0]
    group = group[group['lon'] != 0].reset_index()
    #remind the mesh degree from jismesh standard 4 = 500m, 3 = 1km
    MESH_DEGREE = jismesh_switcher[config.data.grid_size]
    group['meshocode']  = group.apply(lambda row:ju.to_meshcode(row['lat'],row['lon'],MESH_DEGREE),axis=1)
    #traj_df.to_csv('test_qwv.csv')

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
            #'segment_id': name,
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
# Input is 3-D array with lat/lon
import numpy as np

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
    #If loading existing data
    if config.data.load_existing:
        FOLDER_PATH = config.data.existing_data_folder
        (input_condtions, traj_mean, traj_std, lengths, cond_mean, cond_std,
         input_traj_segments, grid_mapping_dict) = loadExistingData(
            FOLDER_PATH)
        # traj_segment = pad_arrays_to_uniform_size(traj_segment, max_length=config.data.traj_length)
        input_traj_segments = resample_existed_trajectory(input_traj_segments, config.data.traj_length)
        # get the first and end of input_traj_segments
        # input_traj_segments = input_traj_segments[:,[0,-1],:]
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

        # with open('%s/conditions_mean_std.txt' % files_save, 'w') as f:
        #     for i in range(5):
        #         f.write('mean_%d: %f\n' % (i, cond_mean[i]))
        #         f.write('std_%d: %f\n' % (i, cond_std[i]))
        # with open('%s/traj_mean_std.txt' % files_save, 'w') as f:
        #     for i in range(2):
        #         f.write('traj_mean_%d: %f\n' % (i, traj_mean[i]))
        #         f.write('traj_std_%d: %f\n' % (i, traj_std[i]))
        return (input_condtions, input_traj_segments, grid_dim, cond_mean, cond_std,
                traj_mean, traj_std)
        # ONE_OD_TEST = False
        # if ONE_OD_TEST:
        #     # Only use data from one OD pair, OD is the -2 and -1 columns of conditions
        #     # Get the unique values of the OD pair and Select the top frequent OD pair
        #     unique_OD_pairs, counts = np.unique(input_condtions[:, -2:], axis=0, return_counts=True)
        #     selected_OD = unique_OD_pairs[np.argmax(counts)]
        #     # Filter the data based on the selected OD pair
        #     selected_indices = np.all(input_condtions[:, -2:] == selected_OD, axis=1)
        #     input_condtions = input_condtions[selected_indices]
        #
        #     # Filter the traj_segments based on the selected indices
        #     input_traj_segments = input_traj_segments[selected_indices]
        # return input_condtions, input_traj_segments
    else:
        if config.data.AOITYPE == True or config.data.AOIEMB == True:
            # Get the list of files
            TARGET_REGION = config.data.data_region
            print(TARGET_REGION)
            # 1. 查询表
            temp = MiniTools.loadPKL(config.data.user_region_list)
            # 使用列表推导式筛选出符合条件的userid
            traj_file_list = [userid for userid, info in temp.items() if info['prefecture'] in TARGET_REGION]
            # Load the key location file of the users
            aoikeylocation_dict = MiniTools.loadPKL('./lipeiran/TrajGenerationIntegrationLinux/'
                                                    'PreProcessing_202403Version/'
                                                    'UsersAOIKeyLocationManagementKashiwa.pkl')
            # Load the prefecture file of the users
            # keylocation_dict = keylocation_dict[traj_file_list]
            # aoikeylocation_dict = aoikeylocation_dict[[userid for userid, info in temp.items()]]
            # 处理每个文件
            input_condtion_list = []
            input_traj_segments_list = []
            for uid in tqdm(traj_file_list[0:user_num]):
                # Load the file
                file_path = '%s/%s.csv' % (config.data.RAWDATA_FOLDER, uid)
                traj_df = pd.read_csv(file_path)
                # clear lat/lon out of the region
                lat_min, lon_min = config.data.lat_min, config.data.lon_min
                lat_max, lon_max = config.data.lat_max, config.data.lon_max
                traj_df = traj_df[(traj_df['lat'] >= lat_min) & (traj_df['lat'] <= lat_max)]
                traj_df = traj_df[(traj_df['lon'] >= lon_min) & (traj_df['lon'] <= lon_max)]
                if len(traj_df) == 0:
                    continue
                # change the minimal time-interval to 1 minutes, and drop the row with duplicate time
                traj_df['time'] = pd.to_datetime(traj_df['time'])
                traj_df['time'] = traj_df['time'].dt.floor('1T')
                traj_df = traj_df.drop_duplicates(subset=['time'],keep='first')
                traj_df = traj_df.sort_values(by='time')
                traj_df = traj_df.reset_index(drop=True)
                # Filter what we want
                temp_aoi_dict = aoikeylocation_dict[uid]
                # label the traj_df by the key locaion and aoi key location based on meshcode (col:unknow4)
                # remember, the meshcode may not in the key of dict
                traj_df['kloc_type'] = traj_df['unknow4'].apply(
                    lambda x: temp_aoi_dict.get(x, [-1,-1])[0])
                traj_df['aoi_type'] = traj_df['unknow4'].apply(
                    lambda x: temp_aoi_dict.get(x, [-1,-1])[1])
                traj_df['aoi_emb'] = traj_df['unknow4'].apply(
                    lambda x: temp_aoi_dict.get(x, [-1,-1,[0.0 for temp_i in range(64)]])[2])
                # filter the traj_df with STAY->MOVE->STAY
                # (1) the current segment_id is within 1 day,
                # combining the time col (to get day) to make segment_id unique number
                traj_df['date'] = pd.to_datetime(traj_df['time']).dt.date
                traj_df['segment_id'] = [str(x) + '_' + str(y) for x, y in
                                            traj_df[['date', 'segment_id']].values]
                # make segment_id be unique number from 0 to length of unique segment_id
                traj_df['segment_id'] = pd.Categorical(traj_df['segment_id']).codes
                # (2) get the move_stay_df by the segment_id
                move_stay_df = traj_df.groupby('segment_id')[['segment_id','trans_mode1']].first()
                # (3) extract the MOVE segment_ids where previous is STAY and next is STAY
                segment_ids = []
                for i in range(1,len(move_stay_df)-1):
                    if (move_stay_df.iloc[i,1] == 'MOVE' and move_stay_df.iloc[i-1,1] == 'STAY'
                            and move_stay_df.iloc[i+1,1] == 'STAY'):
                        # get the segment_id
                        segment_id = move_stay_df.iloc[i,0]
                        segment_ids.append(segment_id)
                        # get the previous and next STAY's kloc_type and aoi_type
                        o_kloc_type = traj_df[traj_df['segment_id'] == move_stay_df.iloc[i-1,0]]['kloc_type'].values[0]
                        o_aoi_type = traj_df[traj_df['segment_id'] == move_stay_df.iloc[i-1,0]]['aoi_type'].values[0]
                        o_aoi_emb = traj_df[traj_df['segment_id'] == move_stay_df.iloc[i-1,0]]['aoi_emb'].values[0]
                        d_kloc_type = traj_df[traj_df['segment_id'] == move_stay_df.iloc[i+1,0]]['kloc_type'].values[0]
                        d_aoi_type = traj_df[traj_df['segment_id'] == move_stay_df.iloc[i+1,0]]['aoi_type'].values[0]
                        d_aoi_emb = traj_df[traj_df['segment_id'] == move_stay_df.iloc[i+1,0]]['aoi_emb'].values[0]
                        # get the traj_segment by the segment_id
                        traj_segment = traj_df[traj_df['segment_id'] == segment_id]
                        # Process the data by segment_id
                        # Calculate the required metrics
                        condition, traj_segment = getCondTraj(traj_segment)
                        if len(traj_segment) == 0 or o_aoi_type == -1 or d_aoi_type == -1:
                            continue
                        # onehot encoding aoi embedding and add to the condition
                        if config.data.AOIEMB == True:
                            condition = np.concatenate([condition, [np.array(o_aoi_emb)],[np.array(d_aoi_emb)]], axis=1)
                        elif config.data.AOITYPE == True:
                            condition = np.concatenate([condition, [[o_aoi_type, d_aoi_type]]], axis=1)
                        else:
                            condition = np.concatenate([condition, [[o_kloc_type,d_kloc_type]]], axis=1)
                        # Pad the arrays to the same length
                        # traj_segment = pad_arrays_to_uniform_size(traj_segment, max_length=config.data.traj_length)
                        traj_segment = resample_trajectory(traj_segment, config.data.traj_length)
                        # Append the results to the lists
                        input_condtion_list.append(condition)
                        input_traj_segments_list.append(traj_segment)
            # # just for test
            # input_condtion_list = [input_condtion_list[0],input_condtion_list[0]]
            # input_traj_segments_list = [input_traj_segments_list[0],input_traj_segments_list[0]]
            # Project into unique int, transfer the input_condtion_list into a numpy array
            input_condtions = np.concatenate(input_condtion_list)
            input_traj_segments = np.concatenate(input_traj_segments_list)
            # Map the last two columns (o,d) to a shared range
            input_condtions[:, 6:8], mapping_dict, max_unique_length = (
                map_two_columns_to_shared_range(input_condtions[:, 6:8]))
            # Normalize the 1~6 columns of conditions column by column, and keep the std, mean for the future use
            cond_std_list = []
            cond_mean_list = []
            for i in range(1, 6):
                cond_std_list.append(np.std(input_condtions[:, i]))
                cond_mean_list.append(np.mean(input_condtions[:, i]))
            for i in range(1, 6):
                if np.std(input_condtions[:, i]) != 0:
                    input_condtions[:, i] = ((input_condtions[:, i] - np.mean(input_condtions[:, i]))
                                             / np.std(input_condtions[:, i]))
                else:
                    input_condtions[:, i] = input_condtions[:, i] - np.mean(input_condtions[:, i])
            # Normalize the traj_segments, and keep the std, mean for the future use
            # a. Normalize by lat lon separately and record the mean and std separately
            traj_mean_list = []
            traj_std_list = []
            for j in range(2):
                traj_mean = np.mean(input_traj_segments[:, :, j])
                traj_std = np.std(input_traj_segments[:, :, j])
                traj_mean_list.append(traj_mean)
                traj_std_list.append(traj_std)
                input_traj_segments[:, :, j] = (input_traj_segments[:, :, j] - traj_mean) / traj_std
            with open('%s/traj_mean_std.txt' % files_save, 'w') as f:
                for i in range(2):
                    f.write('traj_mean_%d: %f\n' % (i, traj_mean_list[i]))
                    f.write('traj_std_%d: %f\n' % (i, traj_std_list[i]))
                # b. Normalize by all lat and lon
                # traj_mean = np.mean(input_traj_segments)
                # traj_std = np.std(input_traj_segments)
                # input_traj_segments = (input_traj_segments - traj_mean) / traj_std
                # with open('%s/traj_mean_std.txt' % files_save, 'w') as f:
                #     f.write('traj_mean: %f\n' % traj_mean)
                #     f.write('traj_std: %f\n' % traj_std)
                # Save the mapping dictionary and the conditions, and the traj_segments with std, mean
                MiniTools.savePKL(mapping_dict, '%s/mesh_mapping_dict.pkl' % files_save)
                MiniTools.savePKL(input_condtions, '%s/conditions.pkl' % files_save)
                MiniTools.savePKL(input_traj_segments, '%s/traj_segments.pkl' % files_save)
                with open('%s/conditions_mean_std.txt' % files_save, 'w') as f:
                    for i in range(5):
                        f.write('mean_%d: %f\n' % (i, cond_mean_list[i]))
                        f.write('std_%d: %f\n' % (i, cond_std_list[i]))
            return input_condtions, input_traj_segments, max_unique_length, cond_mean_list, cond_std_list, traj_mean_list, traj_std_list

        else:# config.data.AOITYPE == False
            # Get the list of files
            TARGET_REGION = config.data.data_region
            print(TARGET_REGION)
            # 1. 查询表
            temp = MiniTools.loadPKL(config.data.user_region_list)
            # 使用列表推导式筛选出符合条件的userid
            traj_file_list = [userid for userid, info in temp.items() if info['prefecture'] in TARGET_REGION]
            # 处理每个文件
            input_condtion_list = []
            input_traj_segments_list = []
            for uid in tqdm(traj_file_list[0:user_num]):
                # Load the file
                file_path = '%s/%s.csv'%(config.data.RAWDATA_FOLDER,uid)
                traj_df = pd.read_csv(file_path)
                traj_df = traj_df[traj_df['trans_mode1'] == 'MOVE']
                traj_df = traj_df[['time', 'lat', 'lon', 'segment_id','trans_mode2']]
                # change the minimal time-interval to 1 minutes, and drop the row with duplicate time
                traj_df['time'] = pd.to_datetime(traj_df['time'])
                traj_df['time'] = traj_df['time'].dt.floor('1T')
                traj_df = traj_df.drop_duplicates(subset=['time'],keep='first')
                traj_df = traj_df.sort_values(by='time')
                traj_df = traj_df.reset_index(drop=True)
                # clear lat/lon out of the region
                lat_min,lon_min = config.data.lat_min,config.data.lon_min
                lat_max,lon_max = config.data.lat_max,config.data.lon_max
                traj_df = traj_df[(traj_df['lat'] >= lat_min) & (traj_df['lat'] <= lat_max)]
                traj_df = traj_df[(traj_df['lon'] >= lon_min) & (traj_df['lon'] <= lon_max)]
                if len(traj_df) == 0:
                    continue
                #get the unique segment_id
                traj_df['date'] = traj_df['time'].dt.date
                traj_df['segment_id'] = [str(x) + '_' + str(y) for x, y in
                                                   traj_df[['date', 'segment_id']].values]
                #sample ramdom n 'segment_id''s data of the traj_df
                np.random.seed(random_seed)
                sampling_segment_ids = np.random.choice(traj_df['segment_id'].unique(),
                                                        sampling_segments_per_user)
                traj_df = traj_df[traj_df['segment_id'].isin(sampling_segment_ids)]
                traj_df = traj_df.dropna(subset=['lat', 'lon'])
                # Process the data by segment_id
                for segment_id, group in traj_df.groupby('segment_id'):
                    # Calculate the required metrics
                    condition, traj_segment = getCondTraj(group)
                    if len(traj_segment) == 0:
                        continue
                    # Pad the arrays to the same length
                    # traj_segment = pad_arrays_to_uniform_size(traj_segment, max_length=config.data.traj_length)
                    traj_segment = resample_trajectory(traj_segment, config.data.traj_length)
                    # Append the results to the lists
                    input_condtion_list.append(condition)
                    input_traj_segments_list.append(traj_segment)
            # # just for test
            # input_condtion_list = [input_condtion_list[0],input_condtion_list[0]]
            # input_traj_segments_list = [input_traj_segments_list[0],input_traj_segments_list[0]]
            #Project into unique int, transfer the input_condtion_list into a numpy array
            input_condtions = np.concatenate(input_condtion_list)
            input_traj_segments = np.concatenate(input_traj_segments_list)
            # Map the last two columns (o,d) to a shared range
            input_condtions[:, 6:8], mapping_dict, max_unique_length = (
                map_two_columns_to_shared_range(input_condtions[:, 6:8]))
            # Normalize the 1~6 columns of conditions column by column, and keep the std, mean for the future use
            cond_std_list = []
            cond_mean_list = []
            for i in range(1, 6):
                cond_std_list.append(np.std(input_condtions[:, i]))
                cond_mean_list.append(np.mean(input_condtions[:, i]))
            for i in range(1, 6):
                if np.std(input_condtions[:, i]) != 0:
                    input_condtions[:, i] = ((input_condtions[:, i] - np.mean(input_condtions[:, i]))
                                             / np.std(input_condtions[:, i]))
                else:
                    input_condtions[:, i] = input_condtions[:, i] - np.mean(input_condtions[:, i])
            #Normalize the traj_segments, and keep the std, mean for the future use
            # a. Normalize by lat lon separately and record the mean and std separately
            traj_mean_list = []
            traj_std_list = []
            for j in range(2):
                traj_mean = np.mean(input_traj_segments[:, :, j])
                traj_std = np.std(input_traj_segments[:, :, j])
                traj_mean_list.append(traj_mean)
                traj_std_list.append(traj_std)
                input_traj_segments[:, :, j] = (input_traj_segments[:, :, j] - traj_mean) / traj_std
            with open('%s/traj_mean_std.txt' % files_save, 'w') as f:
                for i in range(2):
                    f.write('traj_mean_%d: %f\n' % (i, traj_mean_list[i]))
                    f.write('traj_std_%d: %f\n' % (i, traj_std_list[i]))
                # b. Normalize by all lat and lon
                # traj_mean = np.mean(input_traj_segments)
                # traj_std = np.std(input_traj_segments)
                # input_traj_segments = (input_traj_segments - traj_mean) / traj_std
                # with open('%s/traj_mean_std.txt' % files_save, 'w') as f:
                #     f.write('traj_mean: %f\n' % traj_mean)
                #     f.write('traj_std: %f\n' % traj_std)
                # Save the mapping dictionary and the conditions, and the traj_segments with std, mean
                MiniTools.savePKL(mapping_dict, '%s/mesh_mapping_dict.pkl' % files_save)
                MiniTools.savePKL(input_condtions, '%s/conditions.pkl' % files_save)
                MiniTools.savePKL(input_traj_segments, '%s/traj_segments.pkl' % files_save)
                with open('%s/conditions_mean_std.txt' % files_save, 'w') as f:
                    for i in range(5):
                        f.write('mean_%d: %f\n' % (i, cond_mean_list[i]))
                        f.write('std_%d: %f\n' % (i, cond_std_list[i]))
            return input_condtions, input_traj_segments,max_unique_length,cond_mean_list, cond_std_list, traj_mean_list, traj_std_list

def prepare_data(input_config,exp_dir=''):
    # let config be the global variable in this python file
    global config
    config = input_config

    if config.data.load_existing == True:
        exp_dir = exp_dir
    else:
        if config.data.AOITYPE == True:
            AOINOTE = 'AOITYPE'
        elif config.data.AOIEMB == True:
            AOINOTE = 'AOIEMB'
        else:
            AOINOTE = ''
        data_note = '{}_{}_unum&tnum={}&{}_gridsize={}_len={}to{}{}'.format(config.data.dataset,
            config.data.data_region_note,config.data.user_num,
            config.data.traj_num_per_user,config.data.grid_size,
            config.data.length_min,config.data.length_max,AOINOTE)
        exp_dir = '../Datasets/%s'%data_note
        MiniTools.ifFolderExistThenCreate(exp_dir)

    # Get the random seed
    random_seed = 0
    # Prepare the data
    head, traj, grid_dim,cond_mean, cond_std, traj_mean, traj_std = get_traj_data(user_num=config.data.user_num,
                                         sampling_segments_per_user=config.data.traj_num_per_user,
                                         files_save=exp_dir,
                                         random_seed=random_seed,
                                         config=config)

    if config.data.norm_1by1 == True:
        # Normalize the traj one by one
        for m in range(traj.shape[0]):
            # get the geohash range (normalized by the lat/lon mean and std version)
            geohash_0 = head[m, -2 * grid_dim:-1 * grid_dim]
            geohash_d = head[m, -1 * grid_dim:]
            #lat_min, lon_min, lat_max, lon_max = MiniTools.getRangeByGeohash(geohash_0,geohash_d,traj_mean,traj_std)
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
    # dataloader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=8)
    return dataset, head, traj, grid_dim, cond_mean, cond_std, traj_mean, traj_std

if __name__ == '__main__':
    raise SystemExit(
        "data_utils/PrepareDataset.py CLI preprocessing is disabled in the open-source package. "
        "Use prepared public datasets under ./data/."
    )

# # Function to pad the arrays
# def pad_arrays_to_uniform_size(data, max_length):
#     padded_data = []
#     for array in data:
#         # Calculate the number of rows to pad
#         pad_rows = max_length - array.shape[0]
#
#         # Check if there are rows to pad
#         if pad_rows > 0:
#             # Create a zero array with the required padding size
#             padding = np.zeros((pad_rows, array.shape[1]))
#             # Append the padding to the array
#             new_array = np.vstack((array, padding))
#         else:
#             # If no padding is needed, use the array as is
#             new_array = array[:max_length]  # This also handles the case where array is larger than m
#
#         padded_data.append(new_array)
#     return padded_data
