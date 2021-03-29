import h5py
import random
import numpy as np
from glob import glob
from os.path import basename
#from scipy.stats import yeojohnson
#from utils import transform_cate, transform_cate_3d, mon_to_season

# path
data_dir = '/glade/scratch/ksha/DATA/QC_64_64/INPUT*.npy'
out_dir = '/glade/scratch/ksha/DATA/QC_64_64/BATCH_categorical/'
# thres for "dry" and "wet"
precip_thres = 0.1
scaling = 'none' # or 'yeo-johnson'
if scaling == 'min-max':
    print('---- min-max precip std -----')
    CaPA_interp_dir = '/glade/scratch/ksha/BACKUP/CaPA_interp_BC.hdf'
    hdf = h5py.File(CaPA_interp_dir, 'r')
    capa_grid = hdf['CaPA_interp'][...]
    hdf.close()
    capa_grid = capa_grid[90:-150, 190:-200]
    capa_max = np.nanmax(capa_grid)
elif scaling == 'yeo-johnson':
    print('----- yeo-johnson precip std -----')
    from scipy.stats import yeojohnson
    from utils import precip_std
    #
    std_dir = '/glade/scratch/ksha/BACKUP/precip_std_base_capa.npy'
    temp_data = np.load(std_dir)
    lmd = temp_data[()]['lmd']
    pmin = temp_data[()]['pmin']
    pmax = temp_data[()]['pmax']
    # precip_std(data, lmd, pmin, pmax)
else:
    print('----- not precip std provided -----')
# get filenames
names = sorted(glob(data_dir)) # <--- !!!
data_temp = np.load(names[0])
size_l = data_temp[()]['stn_input'].shape[0] # size_l --> qc, raw, capa, clim, lat, lon, time_diff, time
# minus two cz raw and qc-ed precip are not features here
size_x, size_y, size_c = data_temp[()]['grid_input'].shape[:-1]
L = len(names)
# allocation (pre-allocation does better than append)
# bad obs and raw rain
grid_p = np.empty([L*80, size_x, size_y, size_c])
stn_p  = np.empty([L*80, size_l])
cate_p = np.empty([L*80])*np.nan
# good obs abd raw rain
grid_n = np.empty([L*80, size_x, size_y, size_c])
stn_n  = np.empty([L*80, size_l])
cate_n = np.empty([L*80])*np.nan
# loop over files
count_p = 0
count_n = 0
# 
for i, name in enumerate(names):
    mon = int(name[-10:-8]) # get month from name, hard coded
    print('file num: {}. month: {}'.format(i, mon)) # viewing the progress
    # import npy files
    data_temp = np.load(name)
    stn_input = data_temp[()]['stn_input']
    grid_input = data_temp[()]['grid_input']
    grid_input[:, :, :, 0] = grid_input[:, :, :, 0]/6.0 # 6-hr precip to hourly
    # create categories
    flag_bad = np.abs(stn_input[0, :]-stn_input[1, :]) >= precip_thres
    flag_rain_qc = stn_input[0, :] >= precip_thres
    flag_rain_raw = stn_input[1, :] >= precip_thres
    # loop over samples
    for j in range(len(stn_input[0, :])):
        if (flag_rain_qc[j] and flag_rain_raw[j] and flag_bad[j]):
            grid_p[count_p, ...] = grid_input[:, :, :, j]
            stn_p[count_p, :] = stn_input[:, j]
            cate_p[count_p] = flag_bad[j]
            count_p += 1
        elif (flag_rain_qc[j] and flag_rain_raw[j] and ~flag_bad[j]):
            grid_n[count_n, ...] = grid_input[:, :, :, j]
            stn_n[count_n, :] = stn_input[:, j]
            cate_n[count_n] = flag_bad[j]
            count_n += 1
# train-validation-test splite
train_rate = 0.8
valid_rate = 0.1
count_p_train = int(count_p*train_rate)
count_p_valid = count_p_train+int(count_p*valid_rate)
count_n_train = int(count_n*train_rate)
count_n_valid = count_n_train+int(count_n*valid_rate)
# save data
count_p = [count_p_train, count_p_valid, count_p]
count_n = [count_n_train, count_n_valid, count_n]
# print out as a test of indexing
print(count_p); print(count_n)
# saving function
def save_data(grid, stn, cate, count, out_dir, key='pr'):
    print(key+' num: '+str(count[-1]))
    # train
    vali_name = out_dir+'TRAIN_'+key+'.hdf'
    hdf = h5py.File(vali_name, 'w')
    hdf.create_dataset('grid_input', data=grid[:count[0], ...])
    hdf.create_dataset('stn_input' , data= stn[:count[0], :])
    hdf.create_dataset('cate_out'  , data=cate[:count[0]])
    hdf.close()
    # pr_valid
    vali_name = out_dir+'VALID_'+key+'.hdf'
    hdf = h5py.File(vali_name, 'w')
    hdf.create_dataset('grid_input', data=grid[count[0]:count[1], ...])
    hdf.create_dataset('stn_input' , data= stn[count[0]:count[1], :])
    hdf.create_dataset('cate_out'  , data=cate[count[0]:count[1]])
    hdf.close()
    # pr test
    vali_name = out_dir+'TEST_'+key+'.hdf'
    hdf = h5py.File(vali_name, 'w')
    hdf.create_dataset('grid_input', data=grid[count[1]:count[2], ...])
    hdf.create_dataset('stn_input' , data= stn[count[1]:count[2], :])
    hdf.create_dataset('cate_out'  , data=cate[count[1]:count[2]])
    hdf.close()

save_data(grid_p, stn_p, cate_p, count_p, out_dir, key='p')
save_data(grid_n, stn_n, cate_n, count_n, out_dir, key='n')


