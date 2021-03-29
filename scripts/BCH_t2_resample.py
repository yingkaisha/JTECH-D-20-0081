import sys
import h5py
import numpy as np
import numba as nb
import pandas as pd
from glob import glob
from datetime import datetime, timedelta

sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/QC_OBS/')
import data_utils as du
from namelist import *

#@nb.njit()
def bch_t2_resample(bch_tuple, sec_ref, gap=30*60):
    '''
    Coverting raw BCH precipitation data (imported into Pandas)
    into fixed temporal interval.
    
    Resampling is based on
    
    Input:
        bch_tuple: raw data, tuple of (obs_time, raw, qc, obs_period)
                   obs_time: time as seconds (relative to 1970-1-1).
                   raw: raw precip
                   qc: qc precip
                   obs_period: instrumental observation interval
                 
        sec_ref:   a list of seconds (relative to 1970-1-1)
                   corresponded to the "temp_pd".
                   this input is applied for resampling.
                   
        gap:       resampled interval, 
                   unit: second
                   default value: 30*60 (30 minutes)
    Output:
        Resampled raw precipitation
        Resampled QC'd precipitation
        *output matches "sec_ref". Missing vals are filled with nan 
    '''
    
    # importing pandas.Series
    ## Datetime
    stn_sec = bch_tuple[0]
    ## Raw value
    raw = bch_tuple[1]
    ## QC'd value
    qc = bch_tuple[2]
    ## Raw observing preiods
    freq = bch_tuple[3]
    
    L_raw = len(stn_sec)
    
    # expanding "sec_ref" to range of seconds
    base_sec = np.arange(np.min(sec_ref)+1, np.max(sec_ref)+1, 1)
    
    # for each sec within "base_sec", search its closest available obs
    ind_base = np.searchsorted(stn_sec, base_sec, 'left')
    ind_base[ind_base >= L_raw] = L_raw-1 # handling edged vals
    
    # Resample obs time and obs preiod to the "base_sec" 
    stn_on_base = stn_sec[ind_base]
    freq_on_base = freq[ind_base]
    # Resample precip to the "base_sec" 
    # Converting mm to mm/hour (regardless of resampling period)
    qc_on_base = qc[ind_base]/(60*60.0)
    raw_on_base = raw[ind_base]/(60*60.0)
    
    # Clean up resampled vals if:
    # 1. missings, i.e., the time interval of two neighbouring obs 
    #    is larger than the (instrumental) obs period. 
    # 2. Obs period is too large (two times large), 
    #    compared with the resampling period ("gap") 
    pick_flag = np.logical_and(np.logical_and(stn_on_base-base_sec<=freq_on_base, stn_on_base-base_sec>=0), freq_on_base<=gap*2)
    qc_on_base[~pick_flag] = np.nan
    raw_on_base[~pick_flag] = np.nan
    
    # truncate to the resampled length 
    L = len(qc_on_base)
    L_fix = (L//gap)*gap
    
    # average within each "gap"
    qc_hourly = np.sum(qc_on_base[:L_fix].reshape(-1, gap), axis=1)
    raw_hourly = np.sum(raw_on_base[:L_fix].reshape(-1, gap), axis=1)
    
    return raw_hourly, qc_hourly

# --------------------------------------------------- #
base = datetime(2016, 1, 1, 0, 0, 0)
#date_ref = [base + timedelta(days=x) for x in range(365+366)] # daily
date_ref = [base + timedelta(minutes=x) for x in range(0, (365+366)*24*60, 30)] # 30 min
sec_ref = np.asarray(du.dt_to_sec(date_ref))

# --------------------------------------------------- #
# get station code
with pd.HDFStore(BCH_meta_file, 'r') as hdf_temp:
    metadata = hdf_temp['metadata']

stn_code = metadata['code'].values.tolist()
key = []

# --------------------------------------------------- #
with pd.HDFStore(BACKUP_dir+'BCH_30min_T2_2020_10_28.hdf', 'w') as hdf_temp:
    for j, code in enumerate(stn_code):
        try:
            with pd.HDFStore(BACKUP_dir+'BCH_combine_T2.hdf', 'r') as hdf_temp2:
                temp_pd = hdf_temp2[code]
        except:
            continue;
        print(code)
        
        bch_tuple = ((temp_pd['datetime'].values.astype('O')/1e9).astype(int),
                     temp_pd['TEMP_INST_RAW'].values,
                     temp_pd['TEMP_INST_VAL'].values,
                     temp_pd['FREQ'].values
                    )
        raw_hourly, qc_hourly = bch_t2_resample(bch_tuple, sec_ref, gap=30*60) # 24*60*60 for daily
        
        # create a new dataframe 
        temp_pd2 = pd.DataFrame()
        
        # assigning datetime and vals
        temp_pd2['datetime'] = date_ref[1:]
        temp_pd2['TEMP_INST_RAW'] = qc_hourly
        temp_pd2['TEMP_INST_VAL'] = raw_hourly
        
        # drop nan
        temp_pd2 = temp_pd2.dropna()
        
        print('Number of resampled obs: {}'.format(len(temp_pd2)))
        hdf_temp[code] = temp_pd2
