# 1.Initial Lines
# !/usr/bin/env python
# -*- coding: utf-8 -*-


# 2.Note for this file.
'This file is for some utilization tools'
__author__ = 'Li Peiran'

# 3.Import the modules.
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import threading as td
import time
import pickle
import jismesh.utils as ju
import pandas as pd
from pandas import DataFrame
from scipy import sparse
# For NN
import torch
import torch.nn.functional as F  # 激励函数都在这
from torch.autograd import Variable
import torch.nn as nn
import scipy.stats
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import datetime
import pygeohash as pgh

# 4.Define the global variables. (if exists)

# 5.Define the class (if exsists)

# 6.Define the function (if exsists)


def getFilePath(root_path, file_list, dir_list=[], target_ext=[]):
    # 获取该目录下所有的文件名称和目录名称
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        # 获取目录或者文件的路径
        dir_file_path = os.path.join(root_path, dir_file)
        # 判断该路径为文件还是路径
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            # 递归获取所有文件和目录的路径
            getFilePath(dir_file_path, file_list, dir_list, target_ext)
        else:
            ext = os.path.splitext(dir_file_path)[1]  # 获取后缀名
            if ext == target_ext:
                file_list.append(dir_file_path)


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def normMinMaxAxis1(data):
    _range = np.max(data, axis=1) - np.min(data, axis=1)
    return (data - np.min(data, axis=1)[:, None]) / _range[:, None]

def normSum(data):
    _sum = np.sum(data)
    return np.nan_to_num(data / _sum) #trans the nan to zero


def normSumAxis1(data):
    _sum = np.sum(data, axis=1)
    return np.nan_to_num(data / _sum[:, None]) #trans the nan to zero

def normSum_Tensor(data):
    _sum = data.sum()
    return data / _sum #trans the nan to zero

def normSumAxis1_Tensor(data):
    _sum = data.sum(axis=1)
    return data / _sum[:, None] #trans the nan to zero



def numpyMSE(arr1, arr2):
    return np.square(np.subtract(arr1, arr2)).mean()


def numpyMAE(arr1, arr2):
    return np.abs(np.subtract(arr1, arr2)).mean()


def ifFolderExistThenCreate(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print('Create Folder: %s' % dir)
    return 1


def getGaussianPob(mu, std, value):
    return scipy.stats.norm(mu, std).pdf(value)


def getGaussianPobTensor(mean, std, value):
    return 1 / (std * np.sqrt(2 * np.pi)) * torch.exp(-(torch.pow((value - mean), 2) / 2 / std / std))


def savePKL(obj, name):
    """
    Save data as pickle file.
    """
    if name[-4:] == '.pkl':
        with open(name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def loadPKL(name):
    """
    Load data from a pickle file, with support for pickle protocol 5 and NumPy version compatibility.
    """
    import pickle
    import sys
    import numpy as np

    class NumpyCompatUnpickler(pickle.Unpickler):
        """Custom unpickler to handle NumPy version incompatibilities."""

        def find_class(self, module, name):
            # Handle NumPy 2.x -> 1.x: remap numpy._core to numpy.core
            if module.startswith('numpy._core'):
                module = module.replace('numpy._core', 'numpy.core')
            # Handle NumPy 1.x -> 2.x: remap numpy.core to numpy._core
            elif module.startswith('numpy.core') and hasattr(np, '_core'):
                module = module.replace('numpy.core', 'numpy._core')

            try:
                return super().find_class(module, name)
            except (AttributeError, ModuleNotFoundError):
                # Fallback: try original module path if remapping failed
                if 'numpy._core' in module:
                    fallback_module = module.replace('numpy._core', 'numpy.core')
                else:
                    fallback_module = module.replace('numpy.core', 'numpy._core')

                try:
                    return super().find_class(fallback_module, name)
                except Exception:
                    # If both fail, raise the original error
                    raise

    try:
        with open(name, 'rb') as f:
            return NumpyCompatUnpickler(f).load()
    except ValueError as e:
        if 'unsupported pickle protocol' in str(e):
            try:
                import pickle5
                with open(name, 'rb') as f:
                    class Pickle5CompatUnpickler(pickle5.Unpickler):
                        def find_class(self, module, name):
                            if module.startswith('numpy._core'):
                                module = module.replace('numpy._core', 'numpy.core')
                            elif module.startswith('numpy.core') and hasattr(np, '_core'):
                                module = module.replace('numpy.core', 'numpy._core')
                            return super().find_class(module, name)

                    return Pickle5CompatUnpickler(f).load()
            except ImportError:
                print("pickle5 module is not installed. Install it via pip to load this pickle file.")
                return None
            except Exception as e:
                print(f"An error occurred while trying to load the file with pickle5: {e}")
                return None
        else:
            print(f"An unexpected error occurred: {e}")
            return None
    except Exception as e:
        print(f"Failed to load pickle from {name}: {e}")
        print(f"Current NumPy version: {np.__version__}")
        print("Consider regenerating the dataset with the current NumPy version.")
        return None

import pandas as pd
import os
import chardet
def get_encoding(filename):
    """
    返回文件编码格式
    """
    with open(filename,'rb') as f:
        return chardet.detect(f.read())['encoding']

def lpVector2xyz(lp_list, lp_format):

    # 1. from lpVector to lp treeIndex
    from FastLabeling.NFM import lpVector2treeIndex
    nfm_list = lpVector2treeIndex(lp_list, lp_format)

    # 2. from treeIndex to xyz point
    from FastLabeling.NFM import nfm
    return nfm(nfm_list)

# Function to calculate distance between two latitude and longitude points
def haversine_distance(lat1, lon1, lat2, lon2):
    from math import radians, cos, sin, asin, sqrt

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def restartMongodb(db_path, log_path, port):
    import subprocess
    mongod_path = 'mongod'
    command = f"{mongod_path} --dbpath {db_path} --logpath {log_path} --port {port}"
    try:
        proc = subprocess.Popen(command.split())
        print("MongoDB has been started successfully.")
    except Exception as e:
        print(f"Failed to start MongoDB: {e}")

def binary_to_geohash(binary_array):
    """
    Convert a binary array to a geohash string.
    The binary array is assumed to be a numpy array where each element represents a binary digit.
    """
    # Convert binary array to integer
    geohash_int = int("".join(map(str, binary_array.astype(int))), 2)

    # Convert integer to base32 string
    base32_map = '0123456789bcdefghjkmnpqrstuvwxyz'
    geohash_str = ''
    while geohash_int > 0:
        geohash_str = base32_map[geohash_int % 32] + geohash_str
        geohash_int //= 32

    return geohash_str

def getRangeByGeohash(geohash_0,geohash_d,traj_mean,traj_std):
    # decode the geohash to lat/lon by 0,1 format
    geohash_0 = binary_to_geohash(geohash_0)
    geohash_d = binary_to_geohash(geohash_d)
    lat_o, lon_o = pgh.decode(geohash_0)
    lat_d, lon_d = pgh.decode(geohash_d)
    lat_min = min(lat_o, lat_d)
    lon_min = min(lon_o, lon_d)
    lat_max = max(lat_o, lat_d)
    lon_max = max(lon_o, lon_d)
    # Normalize the lat_min, lon_min, lat_max, lon_max
    lat_min = (lat_min - traj_mean[0]) / traj_std[0]
    lon_min = (lon_min - traj_mean[1]) / traj_std[1]
    lat_max = (lat_max - traj_mean[0]) / traj_std[0]
    lon_max = (lon_max - traj_mean[1]) / traj_std[1]
    return lat_min, lon_min, lat_max, lon_max

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
