'''
QC pre-processing
Interpolating CaPA data to different grid spcaings
using griddata, needs to spped-up

To do: replace griddata with faster interpolation method
'''


import h5py
import netCDF4 as nc
import numpy as np
from scipy.interpolate import griddata

# file path
data_dir = '/glade/scratch/ksha/BACKUP/'

# ========== CaPA data ========== #
with h5py.File(data_dir+'CaPA_compressed_BC.hdf', 'r') as hdf:
    CaPA_lon = hdf['lon'][...]
    CaPA_lat = hdf['lat'][...]
    CaPA = hdf['CaPA'][...]

# create lat/lon grids
L = len(CaPA); 
lonlim = [-150, -90]; latlim = [38, 72] # <----- !!! -145,-105,45,65
grid_shape = CaPA_lon.shape
print('Original shape: {}'.format(grid_shape))
input_points_capa = (CaPA_lon.flatten(), CaPA_lat.flatten())

# ========== ETOPO data ========== #
ETOPO_path = data_dir+'etopo5.nc'
with nc.Dataset(ETOPO_path) as nc_obj:
    etopo_x = nc_obj.variables['topo_lon'][...]
    etopo_y = nc_obj.variables['topo_lat'][...]
    etopo_z = nc_obj.variables['topo'][...]
etopo_lon, etopo_lat = np.meshgrid(etopo_x, etopo_y)
input_points_etopo = (etopo_lon.flatten(), etopo_lat.flatten())

# ========== loop over interp ========= #
# Produces: GRID_INPUT_FULL.hdf

factor = [0.2, 0.24, 0.3, 0.4, 0.6, 1.0, 1.75, 4] # 
count = 0

# interpolate and write as hdf
hdf_obj = h5py.File(data_dir+'GRID_INPUT_FULL.hdf', 'w')
# 
for i, f in enumerate(factor):  
    shape0 = (int(f*grid_shape[0]), int(f*grid_shape[1]))
    xgrid0, ygrid0 = np.meshgrid(np.linspace(lonlim[0], lonlim[1], shape0[1]), np.linspace(latlim[0], latlim[1], shape0[0]))
    print('===== factor: {} =====\n  shape: {}'.format(f, shape0))
    # allocation
    CaPA_interp0 = np.empty([L, shape0[0], shape0[1]])
    # loop
    for j in range(L):
        print('  CaPA frame: {}'.format(j))
        CaPA_interp0[j, ...] = griddata(input_points_capa, CaPA[j, ...].flatten(), (xgrid0, ygrid0), method='linear')
    print('  ETOPO5 interp')
    etopo_interp0 = griddata(input_points_etopo, etopo_z.flatten(), (xgrid0, ygrid0), method='linear')
    # varname and save
    lon_name = 'xgrid'+str(count)
    lat_name = 'ygrid'+str(count)
    capa_name = 'capa'+str(count)
    etopo_name = 'etopo'+str(count)
    count += 1
    # create dataset
    hdf_obj.create_dataset(lon_name, data=xgrid0)
    hdf_obj.create_dataset(lat_name, data=ygrid0)
    hdf_obj.create_dataset(capa_name, data=CaPA_interp0)
    hdf_obj.create_dataset(etopo_name, data=etopo_interp0)
#
hdf_obj.close()
