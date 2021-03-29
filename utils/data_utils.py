'''
Python utils shared by many of my repos

Yingkai Sha
2020-01-23
'''

# general tools
import sys
from copy import copy
from glob import glob
from collections import Counter
from datetime import datetime, timedelta

# data tools
import h5py
import numpy as np
import pandas as pd

# stats tools
from scipy.spatial import cKDTree
from scipy.interpolate import interp2d
from scipy.interpolate import NearestNDInterpolator
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# geogrphical tools
from geopy.distance import great_circle

# utils
# def MAE_RMSE(data, obs):
#     '''
#     Calculate MAE and RMSE from 1d and 2d data.
#     does not accept NaNs
#     output: (single_val, single_val)
#     '''
#     MAE_out  = mean_absolute_error(data, obs)
#     RMSE_out = np.sqrt(mean_squared_error(data, obs))
#     return MAE_out, RMSE_out

def mae_cal(X1, X2):
    return np.nanmean(np.abs(X1-X2))

def MAE_spatial(grid1, grid2):
    '''
    Calculate MAE from 3d data with [MAE(t), x, y].
    NaNs are removed.
    output: grid
    '''
    grid_shape = grid1.shape
    MAE = np.zeros(grid_shape[1:])*np.nan
#     # nan flag
#     nan_flag = np.logical_or(np.isnan(grid1), np.isnan(grid2))
#     nan_flag = np.sum(nan_flag, 0)
    # loop over grid points
    for i in range(grid_shape[1]):
        for j in range(grid_shape[2]):
#             if nan_flag[i, j]==0:
            MAE[i, j] = mae_cal(grid1[:, i, j], grid2[:, i, j])
    return MAE

def MAE_temporal(grid1, grid2):
    '''
    Calculate MAE and RMSE from 3d data with [t, MAE(x, y)].
    NaNs are removed.
    output: timeseries
    '''
    grid_shape = grid1.shape
    MAE = np.zeros(grid_shape[0])
    # loop over time-axis
    for i in range(grid_shape[0]):
        MAE[i] = mae_cal(grid1[i, ...], grid2[i, ...])
    return MAE

# Move to qc
# def subset_valid(capa_secs, valid_range, test_range):
#     '''
#     Subseting indices by valid and test datetime
#     capa_secs: the datenum of timeseries, relative to 1970-01-01
#     see also: dt_to_sec
    
#     output: (flags, flags, flags)
#     '''
#     L = len(valid_range)
#     valid_ind = []; 
#     for i in range(L):
#         valid_secs = dt_to_sec(valid_range[i])
#         valid_bound = [np.searchsorted(capa_secs, valid_secs[0], 'left'), np.searchsorted(capa_secs, valid_secs[1], 'right')]
#         valid_ind += list(range(valid_bound[0], valid_bound[1]))
#     #
#     L = len(test_range)
#     test_ind = [];         
#     for i in range(L):    
#         test_secs = dt_to_sec(test_range[i])
#         test_bound = [np.searchsorted(capa_secs, test_secs[0], 'left'), np.searchsorted(capa_secs, test_secs[1], 'right')]
#         test_ind += list(range(test_bound[0], test_bound[1]))
#     #    
#     ind_all = list(range(len(capa_secs)))
#     train_ind = []
#     for ind_temp in ind_all:
#         if (ind_temp in valid_ind) or (ind_temp in test_ind):
#             continue;
#         else:
#             train_ind.append(ind_temp)
#     return train_ind, valid_ind, test_ind
  
# saving function

def save_hdf5(p_group, labels, out_dir, filename='example.hdf'):
    '''
    Save data into a signle hdf5
        - p_group: datasets combined in one tuple;
        - labels: list of strings;
        - out_dir: output path;
        - filename: example.hdf;
    **label has initial 'x' means ENCODED strings
    '''    
    name = out_dir+filename
    hdf = h5py.File(name, 'w')
    for i, label in enumerate(labels):
        if label[0] != 'x':
            hdf.create_dataset(label, data=p_group[i])
        else:
            string = p_group[i]
            hdf.create_dataset(label, (len(string), 1), 'S10', string)
    hdf.close()
    print('Save to {}'.format(name))

# def KDTree_wraper(dist_lon, dist_lat):
#     '''
#     A warper of scipy.spatial.cKDTree
#     Tree = KDTree_wraper(dist_lon, dist_lat)
#     '''
#     return cKDTree(list(zip(dist_lon.ravel(), dist_lat.ravel())))

def grid_search(xgrid, ygrid, stn_lon, stn_lat):
    '''
    kdtree-based nearest gridpoint search
    output: indices_lon, indices_lat
    '''
    gridTree = cKDTree(list(zip(xgrid.ravel(), ygrid.ravel()))) #KDTree_wraper(xgrid, ygrid)
    grid_shape = xgrid.shape
    dist, indexes = gridTree.query(list(zip(stn_lon, stn_lat)))
    return np.unravel_index(indexes, grid_shape)

def interp2d_wraper(nav_lon, nav_lat, grid_z, out_lon, out_lat, method='linear'):
    '''
    wrapper of interp2d, works for 2-d grid to grid interp.
    method = 'linear' or 'cubic'
    output: grid
    '''
    if np.sum(np.isnan(grid_z)) > 0:
        grid_z = fill_coast(grid_z, np.zeros(grid_z.shape).astype(bool))
    interp_obj = interp2d(nav_lon[0, :], nav_lat[:, 0], grid_z, kind=method)
    return interp_obj(out_lon[0, :], out_lat[:, 0])

def nearest_wraper(nav_lon, nav_lat, grid_z, out_lon, out_lat):
    '''
    wrapper of nearest neighbour
    '''
    f = NearestNDInterpolator((nav_lon.ravel(), nav_lat.ravel()), grid_z.ravel())
    out = f((out_lon.ravel(), out_lat.ravel()))
    return out.reshape(out_lon.shape)

def fillzero(arr):
    '''
    replace NaNs with zeros
    '''
    flag = np.isnan(arr)
    arr[flag] = 0.0
    return arr

def fillnan(arr):
    '''
    fill NaNs with nearest neighbour grid point val
    The first grid point (left and bottom) cannot be NaNs
    output: grid
    '''
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:, None], idx]
    return out

def fill_coast(arr, flag=False):
    '''
    Fill ocean grid points with the nearest land val
    sequence: left > top > right > bottom
    '''
    out = np.copy(arr) # copy
    # left fill
    out = np.fliplr(fillnan(np.fliplr(out)))
    # top fill
    out = np.rot90(fillnan(np.rot90(out, k=1)), k=3)
    # right fill
    out = fillnan(out)
    # bottom fill
    out = np.rot90(fillnan(np.rot90(out, k=3)), k=1)
    if type(flag) == bool:
        return out
    else:
        out[flag] = np.nan
        return out

def str_encode(string):
    '''
    encode string list to ascii (for saving to hdf5)
    '''
    out = [n.encode("ascii", "ignore") for n in string]
    return out

def str_decode(string):
    '''
    decode encoded numpy array to raw string list 
    (for extracting string from hdf5)
    '''
    string = string.tolist()
    out = [(n[0]).decode("ascii") for n in string]
    return out

def dt_to_sec(dt):
    '''
    python datetime to datenum relative to 1970-1-1
   output: list
    '''
    L = len(dt)
    base = datetime(1970, 1, 1)
    out = [0]*L
    for i, t in enumerate(dt):
        out[i] = int((t-base).total_seconds())
    return out

def sec_to_dt(secs):
    '''
    Converting datetime object to number of secs since 1970-01-01
    e.g. MATLAB datenum.m
    '''
    base = datetime(1970, 1, 1, 0, 0)
    L = len(secs)
    out = L*[0]
    for i, sec in enumerate(secs):
        out[i] = base+timedelta(seconds=int(sec))
    return out

# QC
def pick_by_flag(tuple_data, flag):
    '''
    Selecting tuple elements by a fix flag
    e.g.
        tuple_data = (X1, Y1); flag = [True, False]
        pick_by_flag --> (X1[flag], Y1[flag])
    '''
    L = len(tuple_data)
    out = {}
    for i in range(L):
        temp = tuple_data[i]
        temp = temp[flag]
        out[str(i)] = temp
    return tuple(out.values())

def latlon_to_dist(nav_lon, nav_lat, refp_lon='self', refp_lat='self'):
    '''
    Convert lat/lon arrays to distance [m] arrays
    'self' uses bottom/left row/columns as starting place
    provide your own as: list(zip(lat, lon))
    output: (grid, grid)
    '''
    if refp_lon=='self':
        refp_lon = list(zip(nav_lat[:, 0], nav_lon[:, 0]))
    if refp_lat=='self':
        refp_lat = list(zip(nav_lat[0, :], nav_lon[0, :]))
    grid_shape = nav_lon.shape
    dist_lon = np.empty(grid_shape)
    dist_lat = np.empty(grid_shape)
    # up-wind diff scheme
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            dist_lon[i, j] = great_circle(refp_lon[i], (nav_lat[i, j], nav_lon[i, j])).m
            dist_lat[i, j] = great_circle(refp_lat[j], (nav_lat[i, j], nav_lon[i, j])).m
    return dist_lon, dist_lat

# QC
def str_count(string, print_out=True):
    '''
    ?????
    '''
    count = Counter(string)
    keys = list(count.keys())
    freq = list(count.values())
    if print_out:
        L = len(freq)
        for i in range(L):
            print('Name: {}; Freq: {}'.format(keys[i], freq[i]))
    return keys, freq

def del_slash(strs):
    '''
    Delete the first element of each string in a list
    '''
    out = []
    for str_temp in strs:
        out.append(str_temp[1:])
    return out

def str_search(strs, keys):
    '''
    Return the index of each keys element from strs
    e.g.
        strs = ['a', 'b', 'c', 'd']; keys = ['a', 'c']
        str_serach(...) --> [0, 2]
    '''
    ind = []
    for key in keys:
        ind_temp = [i for i,s in enumerate(strs) if key in s]
        if len(ind_temp) == 1:
            ind.append(ind_temp[0])
        elif len(ind_temp) > 1:
            print('duplicate items (will pick the last one):')
            for ind_d in ind_temp:
                print('{} --> {}'.format(ind_d, strs[ind_d]))
            ind.append(ind_d)
        else:
            print('item {} not found.'.format(key))
            ind.append(9999)
    return ind

def ind_to_flag(inds, L):
    '''
    Converting inds to flags
    e.g.
        inds = [0, 2, 3]; L = 4 # must match
        ind_to_flag(...) --> [True, False, True, True]
    '''
    flag = np.zeros(L).astype(bool)
    for ind in inds:
        flag[ind] = True
    return flag.tolist()

def min_max(data):
    '''
    Min-max normalization, accepts NaNs
    '''
    return (data-np.nanmin(data))/(np.nanmax(data)-np.nanmin(data))

def inv_min_max(data, dmin, dmax):
    '''
    Inverse min-max normalization, accepts NaNs
    '''
    return data*(dmax-dmin)+dmin

def center_minmax(data):
    '''
    norm to [-1, 1]
    '''
    return 2*(data-np.nanmin(data))/(np.nanmax(data)-np.nanmin(data))-1

def inv_center_minmax(data, dmin, dmax):
    '''
    inv of center_minmax
    '''
    return ((data+1)*(dmax-dmin)+dmin)/2

def norm_std(data):
    '''
    Standardization, accepts NaNs
    '''
    return (data-np.nanmean(data))/np.nanstd(data)

def inv_norm_std(data, dmean, dstd):
    '''
    Inverse standardization, accepts NaNs
    '''
    return data*dstd+dmean

def log_trans(data):
    '''
    log transformation
    '''
    return np.log(data+1)

def inv_log_trans(data):
    '''
    inverse log transformation
    '''
    return np.exp(data)-1

def shuffle_ind(L):
    '''
    shuffle indices
    L: length of dimension
    '''
    ind = np.arange(L)
    np.random.shuffle(ind)
    return ind

def check_bounds(x, y, edge):
    '''
    Check if (x, y) within a rectangular box
    edge = [xmin, xmax, ymin, ymax]
    '''
    if x>=edge[0] and x<=edge[1] and y>= edge[2] and y<=edge[3]:
        return True
    else:
        return False
    
def check_bounds_2d(list_x, list_y, edge):
    L = len(list_x)
    out = []
    for i in range(L):
        out.append(check_bounds(list_x[i], list_y[i], edge))
    return out

def pad_val(grid_data, target_shape, pad_num=np.nan):
    '''
    zero pad to the desire shape
    '''
    grid_shape = grid_data.shape
    out = np.ones(target_shape)*pad_num
    indy_start = int((target_shape[0]-grid_shape[0])/2)
    indx_start = int((target_shape[1]-grid_shape[1])/2)
    out[indy_start:indy_start+grid_shape[0], indx_start:indx_start+grid_shape[1]] = grid_data
    return out

def season_ind_sep(date_list, key_format='_{}_'):
    '''
    group a date_list by seasons (DJF, MAM, JJA, SON)
    output: dictionary of indices (not True/False flags)
    *key_format usage example: 'my_{}'.format('djf')
    '''
    seasons = ['djf', 'mam', 'jja', 'son']
    
    IND = {}
    for sea in seasons:
        IND[key_format.format(sea)] = []
        
    for i, date in enumerate(date_list):
        if date.month in [12, 1, 2]:
            IND[key_format.format('djf')].append(i)
        elif date.month in [3, 4, 5]:
            IND[key_format.format('mam')].append(i)
        elif date.month in [6, 7, 8]:
            IND[key_format.format('jja')].append(i)
        elif date.month in [9, 10, 11]:
            IND[key_format.format('son')].append(i)
    return IND

# QC
def pad_sequence(data, target_len):
    '''
    pad sequence (by linear extrpolation)
    '''
    delta = data[1]-data[0]
    L = len(data)
    L_left = int((target_len-L)/2.0)
    #L_right = target_len - L_left
    start_val = data[0]-L_left*delta
    return np.arange(start_val, start_val+delta*(target_len+1), delta)

def dict_list_append(dict1, dict2, copy_flag=True):
    '''
    Appending 2 dictionaries, dictionary should only have lists
    (for appending lists of keras training loss)
    '''
    if copy_flag:
        out = copy(dict1)
    else:
        out = dict1
    keys = out.keys()
    for key in keys:
        out[key] += dict2[key]
    return out

def dt_match(dt_base, dt_sub):
    L = len(dt_base)
    flag = np.zeros(L).astype(bool)
    for i, dt_temp in enumerate(dt_sub):
        if dt_temp in dt_base:
            flag[np.searchsorted(dt_base, dt_temp)] = True
    return flag

def del2(M):
    '''
    Calculate the Laplacian of 2D fields
    output: grid (size shrinks)
    '''
    dx = 1
    dy = 1
    rows, cols = M.shape
    dx = dx * np.ones ((1, cols - 1))
    dy = dy * np.ones ((rows-1, 1))

    mr, mc = M.shape
    D = np.zeros ((mr, mc))

    if (mr >= 3):
        ## x direction
        ## left and right boundary
        D[:, 0] = (M[:, 0] - 2 * M[:, 1] + M[:, 2]) / (dx[:,0] * dx[:,1])
        D[:, mc-1] = (M[:, mc - 3] - 2 * M[:, mc - 2] + M[:, mc-1]) / (dx[:,mc - 3] * dx[:,mc - 2])
        ## interior points
        tmp1 = D[:, 1:mc - 1] 
        tmp2 = (M[:, 2:mc] - 2 * M[:, 1:mc - 1] + M[:, 0:mc - 2])
        tmp3 = np.kron (dx[:,0:mc -2] * dx[:,1:mc - 1], np.ones ((mr, 1)))
        D[:, 1:mc - 1] = tmp1 + tmp2 / tmp3

    if (mr >= 3):
        ## y direction
        ## top and bottom boundary
        D[0, :] = D[0,:]  + (M[0, :] - 2 * M[1, :] + M[2, :] ) / (dy[0,:] * dy[1,:])
        D[mr-1, :] = D[mr-1, :] + (M[mr-3,:] - 2 * M[mr-2, :] + M[mr-1, :]) / (dy[mr-3,:] * dx[:,mr-2])
        ## interior points
        tmp1 = D[1:mr-1, :] 
        tmp2 = (M[2:mr, :] - 2 * M[1:mr - 1, :] + M[0:mr-2, :])
        tmp3 = np.kron (dy[0:mr-2,:] * dy[1:mr-1,:], np.ones ((1, mc)))
        D[1:mr-1, :] = tmp1 + tmp2 / tmp3

    return D / 4

# dscale
def del_3d(data):
    '''
    Calculate the mean absolute Laplacian from [t, L(x, y)]
    output: timeseries
    '''
    L = data.shape[0]
    out = np.zeros(L)*np.nan
    for i in range(L):
        out[i] = np.nanmean(np.abs(del2(data[i, ...])))
    return out

