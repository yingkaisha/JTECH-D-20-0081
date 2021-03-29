'''
Subseting GPM satellite precipitation to BC domain
ETOPO1 included
(QC pre-processing but not used anymore)
'''

import sys
import h5py
import numpy as np
import pandas as pd
import netCDF4 as nc
from glob import glob
from scipy.interpolate import griddata
from datetime import datetime, timedelta

sys.path.insert(0, '/glade/u/home/ksha/ML_repo/utils/')
import data_utils as du
from namelist import *

GPM_dir = '/glade/scratch/ksha/BACKUP/GPM_Raw_Files/'

# datetime range (2016 jan - 2017 dec)
date_list = [datetime(2016, 1, 1, 0, 0)+timedelta(days=i) for i in range(0, 365+366)]
L = len(date_list) # <----- !!!!!!!
# subsetting info
nc_io = nc.Dataset(sorted(glob(GPM_dir+'*.V06B.HDF5.nc4*'))[0], 'r')
y = nc_io.variables['lat'][...]
x = nc_io.variables['lon'][...]
nc_io.close()
# subsetting inds
subsetx = [184, -201]; subsety_b = 81
lon, lat = np.meshgrid(x[subsetx[0]:subsetx[1]], y[subsety_b:])
# Allocation
GPM = np.empty((L*48,)+lon.shape)
RAIN = np.empty((L*48,)+lon.shape)
TIME = np.empty(L*48)
# GPM time = N seconds since 1970-01-01 00:00:00 UTC
base = datetime(1970, 1, 1, 0, 0, 0, 0)
count = 0
for i, dt in enumerate(date_list[:L]):
    file_names = sorted(glob(GPM_dir+dt.strftime('3B-HHR.MS.MRG.3IMERG.*%Y%m%d*.V06B.HDF5.nc4')))
    L_day = len(file_names)
    for j, name in enumerate(file_names):
        print(name)
        nc_io = nc.Dataset(name, 'r')
        # ----- variables ----- #
        GPM_temp = nc_io.variables['precipitationCal'][0, ...]
        rain_p = nc_io.variables['probabilityLiquidPrecipitation'][0, ...]
        time = nc_io.variables['time'][...]
        # --------------------- #
        nc_io.close()
        RAIN[count+j, ...] = np.ma.filled(rain_p, np.nan).T[subsety_b:, subsetx[0]:subsetx[1]]
        GPM[count+j, ...] = np.ma.filled(GPM_temp, np.nan).T[subsety_b:, subsetx[0]:subsetx[1]]
        TIME[count+j] = time[0]
        
    count += L_day
# subsetting from full alloc
GPM = GPM[:count, ...]
RAIN = RAIN[:count, ...]
TIME = TIME[:count]
# etopo
print('ETOPO interpolation')
ETOPO_path = BACKUP_dir+'etopo5.nc'
with nc.Dataset(ETOPO_path) as nc_obj:
    etopo_x = nc_obj.variables['topo_lon'][...]
    etopo_y = nc_obj.variables['topo_lat'][...]
    etopo_z = nc_obj.variables['topo'][...]
etopo_lon, etopo_lat = np.meshgrid(etopo_x, etopo_y)
input_points_etopo = (etopo_lon.flatten(), etopo_lat.flatten())
etopo_interp = griddata(input_points_etopo, etopo_z.flatten(), (lon, lat), method='linear')

# save 
hdf_io = h5py.File(BACKUP_dir+'GPM_compressed_BC.hdf', 'w')
hdf_io.create_dataset('lon'  , data=lon)
hdf_io.create_dataset('lat'  , data=lat)
hdf_io.create_dataset('GPM'  , data=GPM)
hdf_io.create_dataset('RAIN' , data=RAIN)
hdf_io.create_dataset('ETOPO', data=etopo_interp)
hdf_io.create_dataset('TIME' , data=TIME)
hdf_io.close()
print('Save to {}'.format(BACKUP_dir+'GPM_compressed_BC.hdf'))