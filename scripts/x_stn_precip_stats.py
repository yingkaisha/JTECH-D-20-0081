'''
An old script for BCH station data pre-processing.
Not applied any more.
'''
import sys
import h5py
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime

sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/QC_OBS/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/QC_OBS/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
import utils

data_dir = '/glade/scratch/ksha/BACKUP/'
stn_obs_dir = data_dir+'BCH_QC_RAW.hdf'
stn_metadata_dir = data_dir+'BCH_metadata.hdf'

# import metadata
with pd.HDFStore(stn_metadata_dir, 'r') as hdf_temp:
    metadata = hdf_temp['metadata']
stn_code = metadata['code'].values.tolist()
#

# allocation
L = len(stn_code)
qc = np.empty(L*50000)
raw = np.empty(L*50000)
gap = np.empty(L*50000)
name = np.zeros(L*50000).astype(str)
# loop over stations
L = 0 # <-- reuse
for code in stn_code:
    try:
        with pd.HDFStore(stn_obs_dir, 'r') as hdf_temp:
            temp_pd = hdf_temp[code]
            print(code)
    except:
        continue; # jump to the next stn
    dL = len(temp_pd)
    qc[L:L+dL] = temp_pd['PREC_INST_QC'].values
    raw[L:L+dL] = temp_pd['PREC_INST_RAW'].values
    gap[L:L+dL] = temp_pd['FREQ'].values
    name[L:L+dL] = [code]*dL
    L += dL
# subtract
qc = qc[:L]; raw = raw[:L]; gap = gap[:L]; name = name[:L]
name_utf = utils.str_encode(name)
# save
hdf_io = h5py.File(data_dir+'stn_precip_series.hdf')
hdf_io.create_dataset('raw', data=raw)
hdf_io.create_dataset('qc', data=qc)
hdf_io.create_dataset('freq', data=gap)
hdf_io.create_dataset('name', (L,1), 'S10', name_utf)
hdf_io.close()




