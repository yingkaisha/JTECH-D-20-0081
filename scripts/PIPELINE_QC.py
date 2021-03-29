import sys
import h5py
import subprocess
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime

sys.path.insert(0, '/glade/u/home/ksha/ML_repo/utils/')
import data_utils as du
import QC_utils as qu
from namelist_QC import *

# ==================== Import Raw Data ==================== #
if del_old:
    print('Re zeros: files under {} will be purged.'.format(INPUT_dir))
print('Import Raw Data')
# station metadata
hdf_io = pd.HDFStore(BCH_meta_file, 'r')
metadata = hdf_io['metadata']
hdf_io.close()
stn_lon = metadata['lon'][:].values
stn_lat = metadata['lat'][:].values
stn_code = metadata['code'][:].values
# capa datetime
temp_data = np.load(CaPA_datetime_file)
capa_dt = temp_data[()]['date']
capa_sec = np.array(du.dt_to_sec(capa_dt))
# load capa
capa_labels = ['capa1', 'capa2', 'capa3', 'capa4', 'capa5', 'capa6'][:ens]
etopo_labels = ['etopo1', 'etopo2', 'etopo3', 'etopo4', 'etopo5', 'etopo6'][:ens]
xgrid_labels = ['xgrid1', 'xgrid2', 'xgrid3', 'xgrid4', 'xgrid5', 'xgrid6'][:ens]
ygrid_labels = ['ygrid1', 'ygrid2', 'ygrid3', 'ygrid4', 'ygrid5', 'ygrid6'][:ens]
capa_tuple, indx_capa, indy_capa = qu.load_capa_data(GRID_INPUT_file, stn_lon, stn_lat, capa_labels, etopo_labels, xgrid_labels, ygrid_labels, ocean_flag=keep_ocean)
#
print('Generate Base Files by Station')
if del_old:
    # clean up
    cmd = 'rm '+INPUT_dir+freq+'*'
    print(cmd)
    subprocess.call(cmd, shell=True)
    cmd = 'rm '+BATCH_dir+freq+'*'
    print(cmd)
    subprocess.call(cmd, shell=True)
    cmd = 'rm '+HOLD_dir+freq+'*'
    print(cmd)
    subprocess.call(cmd, shell=True)
# gen by station
perfix = freq+'DBASE_'
out_dir = INPUT_dir
valid_range = [[datetime(2017, 2, 5)  , datetime(2017, 2, 20)],
               [datetime(2017, 4, 11) , datetime(2017, 4, 26)],
               [datetime(2017, 6, 4)  , datetime(2017, 6, 19)],
               [datetime(2017, 10, 25), datetime(2017, 11, 9)]]

test_range  = [[datetime(2017, 1, 5)  , datetime(2017, 1, 20)],
               [datetime(2017, 3, 21) , datetime(2017, 4, 5)] ,
               [datetime(2017, 8, 8)  , datetime(2017, 8, 23)],
               [datetime(2017, 11, 17), datetime(2017, 12, 2)]]
if freq == 'HIGH_':
    BCH_input = BCH_high_file
elif freq == 'LOW_':
    BCH_input = BCH_low_file
else:
    BCH_input = BCH_file
    
_, _ = qu.gen_by_stn_capa(capa_tuple, indx_capa, indy_capa, ens,
                                 half_edge, capa_sec, valid_range, test_range, 
                                 stn_code, BCH_input, perfix, out_dir)
# ========================================================= #
# ================ Assign Labels and Merge ================ #
print('Assign labels and Merge Files')
data_dir = INPUT_dir+freq+'DBASE*hdf'; filenames = glob(data_dir)

labels = ['stn_input', 'capa_input', 'time_info', 'cate_out', 'xstn_code']
p_group, n_group = qu.merge_input_capa(filenames, precip_thres=thres, option='TRAIN')
du.save_hdf5(p_group, labels, BATCH_dir, filename=freq+'TRAIN_p.hdf')
du.save_hdf5(n_group, labels, BATCH_dir, filename=freq+'TRAIN_n.hdf')

p_group, n_group = qu.merge_input_capa(filenames, precip_thres=thres, option='VALID')
du.save_hdf5(p_group, labels, BATCH_dir, filename=freq+'VALID_p.hdf')
du.save_hdf5(n_group, labels, BATCH_dir, filename=freq+'VALID_n.hdf')

p_group, n_group = qu.merge_input_capa(filenames, precip_thres=thres, option='TEST')
du.save_hdf5(p_group, labels, BATCH_dir, filename=freq+'TEST_p.hdf')
du.save_hdf5(n_group, labels, BATCH_dir, filename=freq+'TEST_n.hdf')
# ========================================================= #
# =================== Split into Batches ================== #
print('Split into Batches')
train_name_p = BATCH_dir+freq+'TRAIN_p.hdf'; train_name_n = BATCH_dir+freq+'TRAIN_n.hdf'
valid_name_p = BATCH_dir+freq+'VALID_p.hdf'; valid_name_n = BATCH_dir+freq+'VALID_n.hdf'
test_name_p = BATCH_dir+freq+'TEST_p.hdf'  ; test_name_n = BATCH_dir+freq+'TEST_n.hdf'
# # labels
labels_capa = ['capa_input', 'stn_input', 'time_info', 'cate_out', 'xstn_code']
# test, valid natural pack
qu.create_full_data(test_name_p , test_name_n , labels_capa, HOLD_dir, padx=[1, 1], pady=[1, 1], prefix=freq+'CAPA_TEST_natural.hdf', option='CAPA')
qu.create_full_data(valid_name_p , valid_name_n , labels_capa, HOLD_dir, padx=[1, 1], pady=[1, 1], prefix=freq+'CAPA_VALID_natural.hdf', option='CAPA')
# train, valid, test (balanced)
qu.balanced_batches(test_name_p, test_name_n, labels_capa, BATCH_dir, pos_rate=0.5, padx=padx, pady=pady, batch_size=200, prefix=freq+'CAPA_TEST', option='CAPA')
qu.balanced_batches(valid_name_p, valid_name_n, labels_capa, BATCH_dir, pos_rate=0.5, padx=padx, pady=pady, batch_size=200, prefix=freq+'CAPA_VALID', option='CAPA')
qu.balanced_batches(train_name_p, train_name_n, labels_capa, BATCH_dir, pos_rate=0.5, padx=padx, pady=pady, batch_size=200, prefix=freq+'CAPA_TRAIN', option='CAPA')
#
file_name = glob(BATCH_dir+freq+'CAPA_TRAIN*npy')
qu.split_batches(BATCH_dir, file_name, ens, freq+'ENS_CAPA_TRAIN')
file_name = glob(BATCH_dir+freq+'CAPA_VALID*npy')
qu.split_batches(BATCH_dir, file_name, ens, freq+'ENS_CAPA_VALID')
# ========================================================= #
# ==================== Regroup batches ==================== #
file_name = glob(BATCH_dir+freq+'CAPA_TRAIN*')
qu.packing_data(file_name, HOLD_dir, batch_size=200, labels=labels_capa, prefix=freq+'CAPA_TRAIN')
file_name = glob(BATCH_dir+freq+'CAPA_VALID*')
qu.packing_data(file_name, HOLD_dir, batch_size=200, labels=labels_capa, prefix=freq+'CAPA_VALID')
file_name = glob(BATCH_dir+freq+'CAPA_TEST*')
qu.packing_data(file_name, HOLD_dir, batch_size=200, labels=labels_capa, prefix=freq+'CAPA_TEST')
