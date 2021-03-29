'''
The main script that formulates training and testing data
'''

import sys
import h5py
import subprocess
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime, timedelta
#
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/QC_OBS/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/QC_OBS/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')

import data_utils as du
import qc_utils as qu
from namelist import *

# ========== Functions ========== #
def merge_input(data_dir, out_dir, precip_thres=1.0):
    '''
    1. Combine input raw files as one
    2. Assign labels/learning targets
    3. Split positive and negative labels
    '''
    # filepath that has npy files of separated gridded and station inputs
    names = sorted(glob(data_dir))
    
    # allocation
    ## identifying array sizes by inspecting a single file
    data_temp = np.load(names[0])
    size_l = data_temp[()]['stn_input'].shape[1] # qc, raw, freq
    size_x, size_y, size_c = data_temp[()]['grid_input'].shape[1:]
    L = len(names)
    
    ## allocate bad obs
    grid_p = np.empty([L*500, size_x, size_y, size_c])
    stn_p  = np.empty([L*500, size_l])
    cate_p = np.empty([L*500])*np.nan
    
    ## allocate good obs
    grid_n = np.empty([L*500, size_x, size_y, size_c])
    stn_n  = np.empty([L*500, size_l])
    cate_n = np.empty([L*500])*np.nan
    
    # loop over npy files
    count_p = 0
    count_n = 0
    for i, name in enumerate(names):
        # import npy files
        print('processing: {}'.format(name))
        data_temp = np.load(name)
        stn_input = data_temp[()]['stn_input']
        grid_input = data_temp[()]['grid_input']
        
        # create flags (bad = positive = minority class)
        ## precip_thres = 1.0 mm/hr
        flag_bad = np.logical_and(np.abs(stn_input[:, 0]-stn_input[:, 1])>precip_thres, stn_input[:, 1]>0)
        flag_good = np.logical_and(np.abs(stn_input[:, 0]-stn_input[:, 1])<=precip_thres, stn_input[:, 1]>0)
        
        # fill pos and neg samples by loop and counts
        for j in range(len(stn_input[:, 0])):
            if flag_bad[j]:
                grid_p[count_p, ...] = grid_input[j, ...]
                stn_p [count_p, :]   = stn_input[j, :]
                cate_p[count_p] = flag_bad[j]
                count_p += 1
            else:
                grid_n[count_n, ...] = grid_input[j, ...]
                stn_n [count_n, :]   = stn_input[j, :]
                cate_n[count_n] = flag_bad[j]
                count_n += 1
                
    # return: tuple_pos, tuple_neg. Tuples are used for generating batches
    return (grid_p[:count_p, ...], stn_p[:count_p, ...], cate_p[:count_p, ...]), \
           (grid_n[:count_n, ...], stn_n[:count_n, ...], cate_n[:count_n, ...])

def split(file_dir, file_name, ens, prefix):
    for i, name in enumerate(file_name):
        print(name)
        data_temp = np.load(name)
        grid_input = data_temp[()]['grid_input']
        cate_out = data_temp[()]['cate_out']
        for j in range(ens):
            temp_name = prefix+str(j)+'_BATCH'+str(i)+'.npy'
            save_d = {'grid_input':grid_input[:, :, :, 2*j:2*(j+1)], 'cate_out':cate_out}
            np.save(file_dir+temp_name, save_d)
# ========================== #

import time
start_time = time.time()

del_old = False

stn_x = 'ALU' # <---- can modify to loop over
print('========== {} =========='.format(stn_x))

if del_od:
    # remove previous files
    print('Clean up previous files')
    subprocess.call('rm '+INPUT_stn_dir+'*npy', shell=True)
    subprocess.call('rm '+BATCH_stn_dir+'*npy', shell=True)

# ===== Stage 1 ===== #
# import station meta info
print('Loading station information')
with pd.HDFStore(BCH_meta_file, 'r') as hdf
    metadata = hdf['metadata']
    
# lon/lat/code
stn_lon = metadata['lon'][:].values
stn_lat = metadata['lat'][:].values
stn_code_raw = metadata['code'][:].values

# remove one station
stn_code = []
for i, temp_code in enumerate(stn_code_raw):
    if temp_code == stn_x:
        stn_lon[i] = np.nan
        stn_lat[i] = np.nan
    else:
        stn_code.append(temp_code)
stn_lon = stn_lon[~np.isnan(stn_lon)]
stn_lat = stn_lat[~np.isnan(stn_lat)]

# split BCH obs files
BCH_x = BACKUP_dir+'BCH_stn_test2.hdf'
BCH_hold = BACKUP_dir+'BCH_stn_test1.hdf'

with pd.HDFStore(BCH_hold, 'w') as hdf_temp:
    for j, code in enumerate(stn_code):
        try:
            # BCH_file = BACKUP_dir+'BCH_combine.hdf'
            with pd.HDFStore(BCH_file, 'r') as hdf_base:
                temp_pd = hdf_base[code]
        except:
            continue;
        hdf_temp[code] = temp_pd

with pd.HDFStore(BCH_x, 'w') as hdf_temp:
    try:
        with pd.HDFStore(BCH_file, 'r') as hdf_base:
            temp_pd = hdf_base[stn_x]
    except:
        print('stn {} not found'.format(stn_x))
        exit()
    if len(temp_pd)<5000:
        print('insufficient data for stn {}'.format(stn_x))
        exit()
    else:
        hdf_temp[code] = temp_pd
        
# import grided data
print('Loading gridded information')
capa_tuple, etopo_tuple, indx_tuple, indy_tuple = qu.load_capa_data(GRID_INPUT_file, stn_lon, stn_lat)

# available time for gridded data
# BACKUP_dir+'CaPA_compressed_datetime.npy'
data_temp = np.load(CaPA_datetime_file)
date_capa_dt = data_temp[()]['date']
date_capa = []
for date_temp in date_capa_dt:
    date_capa.append(np.datetime64(date_temp))
date_capa = np.array(date_capa)

# Process all
print('Processing train data')
perfix = INPUT_stn_dir+'INPUT_TRAIN'
_, _, _ = du.input_gen(capa_tuple, etopo_tuple, indx_tuple, indy_tuple, date_capa, stn_code, BCH_hold, perfix)
print('Processing testing data')
perfix = INPUT_stn_dir+'INPUT_TEST'
_, _, _ = du.input_gen(capa_tuple, etopo_tuple, indx_tuple, indy_tuple, date_capa, stn_code, BCH_x, perfix)

# ===== Stage 2 ===== #
print('Merging INPUTs')
labels = ['grid_input', 'stn_input', 'cate_out'] # 'x' for encoded string
# train
data_dir = INPUT_stn_dir+'INPUT_TRAIN*.npy'
p_group, n_group = merge_input(data_dir, BATCH_stn_dir, precip_thres = 1.0)
du.save_hdf5(p_group, labels, BATCH_stn_dir, 'TRAIN_p.hdf')
du.save_hdf5(n_group, labels, BATCH_stn_dir, 'TRAIN_n.hdf')
# test
data_dir = INPUT_stn_dir+'INPUT_TEST*.npy'
p_group, n_group = merge_input(data_dir, BATCH_stn_dir, precip_thres = 1.0)
du.save_hdf5(p_group, labels, BATCH_stn_dir, 'TEST_p.hdf')
du.save_hdf5(n_group, labels, BATCH_stn_dir, 'TEST_n.hdf')

# # ===== Stage 3 ===== #
print('Create batches')
train_name_p = BATCH_stn_dir+'TRAIN_p.hdf'
train_name_n = BATCH_stn_dir+'TRAIN_n.hdf'
test_name_p = BATCH_stn_dir+'TEST_p.hdf'
test_name_n = BATCH_stn_dir+'TEST_n.hdf'
# generating batches
labels = ['grid_input', 'stn_input', 'cate_out']
du.balanced_batches(train_name_p, train_name_n, labels, BATCH_stn_dir, pos_rate=0.5, batch_size=200, prefix='TRAIN')     
du.balanced_batches(test_name_p , test_name_n , labels, BATCH_stn_dir, pos_rate=0.5, batch_size=200, prefix='TEST')

print("--- %s seconds ---" % (time.time() - start_time))


# # ===== Stage 4 ===== #
# file_name = sorted(glob(BATCH_stn_dir+'TRAIN*npy'))
# split(BATCH_stn_dir, file_name, 6, 'ENS_TRAIN')
# file_name = sorted(glob(BATCH_stn_dir+'VALID*npy'))
# split(BATCH_stn_dir, file_name, 6, 'ENS_VALID')


