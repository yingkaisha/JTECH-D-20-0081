import sys
import h5py
import numpy as np
import pandas as pd
from glob import glob
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from datetime import datetime, timedelta
#from collections import Counter
#
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
from data_utils import *

def crop_stn(grid0, series, padx=[1, 1], pady=[1, 1]):
    L = grid0.shape[0]
    size_x = grid0.shape[1]
    size_y = grid0.shape[2]
    indx1 = int(size_x/2)-padx[0]; indx2 = int(size_x/2)+padx[1]
    indy1 = int(size_y/2)-pady[0]; indy2 = int(size_y/2)+pady[1]
    for j in range(L):
        grid0[j, indx1:indx2, indy1:indy2] = series[j] # 1=raw
        #grid0[j, ...] -= series[j] # 1=raw
        capa_max0 = np.max(grid0[j, :, :])
        capa_min0 = np.min(grid0[j, :, :])
        if capa_max0 > 0:
            grid0[j, :, :] = (grid0[j, :, :]-capa_min0)/(capa_max0-capa_min0)
    return grid0

def load_gpm_data(GPM_dir, stn_lon, stn_lat):
    hdf_io = h5py.File(GPM_dir, 'r')
    GPM = hdf_io['GPM'][...]
    lon = hdf_io['lon'][...]
    lat = hdf_io['lat'][...]
    RAIN = hdf_io['RAIN'][...]/100.0
    ETOPO = hdf_io['ETOPO'][...]
    hdf_io.close()
    # pre-processing
    indx, indy = grid_search(lon, lat, stn_lon, stn_lat)
    ETOPO = grid_std(ETOPO)
    ETOPO[ETOPO<0] = 0
    return (GPM, RAIN, ETOPO), (indx, indx, indx), (indy, indy, indy)

def load_capa_data(CaPA_interp_dir, stn_lon, stn_lat, capa_labels, etopo_labels, xgrid_labels, ygrid_labels, ocean_flag=False):
    L = len(capa_labels)
    # dictionary
    capa_dict = {}; indx_dict = {}; indy_dict = {}
    # import files
    hdf = h5py.File(CaPA_interp_dir, 'r')
    for i in range(L):
        capa_dict[capa_labels[i]] = hdf[capa_labels[i]][...]/6.0 # from 6-hour to hourly
        xgrid_temp = hdf[xgrid_labels[i]][...]
        ygrid_temp = hdf[ygrid_labels[i]][...]
        indx_dict[xgrid_labels[i]], indy_dict[ygrid_labels[i]] = grid_search(xgrid_temp, ygrid_temp, stn_lon, stn_lat)
        etopo_temp = grid_std(hdf[etopo_labels[i]][...])
        if ocean_flag:
            etopo_temp[etopo_temp<0] = -0.05
        else:
            etopo_temp[etopo_temp<0] = 0
        capa_dict[etopo_labels[i]] = etopo_temp
    hdf.close()
    #
    capa_tuple  = tuple(capa_dict.values())
    indx_tuple  = tuple(indx_dict.values())
    indy_tuple  = tuple(indy_dict.values())
    return capa_tuple, indx_tuple, indy_tuple

def gen_by_stn_capa(capa_tuple, indx_capa, indy_capa, ens, half_edge, capa_sec, valid_range, test_range, stn_code, stn_obs_dir, perfix, out_dir):
    N_sample = []; N_code = []
    # size parameters
    half_edge_y = half_edge[1] #46 
    half_edge_x = half_edge[0] #23 # 46-by-92
    size_x, size_y, size_capa = half_edge_x*2, half_edge_y*2, len(capa_tuple)
    # adjusting time axis for gridded data
    hr6 = 6*60*60
    # gridded precip counts forward, now make it counting backward
    capa_sec += hr6
    for j, code in enumerate(stn_code):
        print(code)
        # ----- try import stn obs by code ----- #
        try:
            with pd.HDFStore(stn_obs_dir, 'r') as hdf_temp:
                temp_pd = hdf_temp[code]
        except:
            print('station {} missing'.format(code))
            continue; # jump to the next stn
        temp_qc   = temp_pd['PREC_INST_QC'].values
        temp_raw  = temp_pd['PREC_INST_RAW'].values
        temp_freq = temp_pd['FREQ'].values
        stn_sec   = temp_pd['datetime'].values.astype('O')/1e9
        # flag_out 0 precip
        flag_pick = temp_raw > 0
        temp_qc, temp_raw, temp_freq, stn_sec = pick_by_flag((temp_qc, temp_raw, temp_freq, stn_sec), flag_pick)
        # =========================================== #
        L_capa = len(capa_sec)
        # get corresponding capa index
        ind_stn = np.searchsorted(capa_sec, stn_sec)
        ## flag out out-of-bounds
        flag_pick = ind_stn < L_capa
        temp_qc, temp_raw, temp_freq, stn_sec, ind_stn = pick_by_flag((temp_qc, temp_raw, temp_freq, stn_sec, ind_stn), flag_pick)
        ## make sure capa is later than obs
        capa_match = capa_sec[ind_stn]
        flag_pad = (capa_match - stn_sec) < 0
        ind_stn[flag_pad] += 1
        # flag out out-of-bounds
        flag_pick = ind_stn < L_capa
        temp_qc, temp_raw, temp_freq, stn_sec, ind_stn = pick_by_flag((temp_qc, temp_raw, temp_freq, stn_sec, ind_stn), flag_pick)
        # make sure gap lower than 0.5hr, so that stn is within capa coverage 
        stn_on_capa = capa_sec[ind_stn]
        flag_pick = (stn_on_capa - stn_sec) <= hr6
        temp_qc, temp_raw, temp_freq, ind_stn, stn_sec = pick_by_flag((temp_qc, temp_raw, temp_freq, ind_stn, stn_sec), flag_pick)
        # =========================================== #
        print('Sample number: {}'.format(len(ind_stn)))
        N_sample.append(len(ind_stn))
        # allocation
        L_smaple = len(ind_stn)
        stn_input = np.empty([L_smaple, 3+ens]) # qc, raw, freq
        capa_input = np.empty([L_smaple, size_x, size_y, size_capa]) # channel last
        # assign stn/time info
        N_code.append(code)
        stn_info = str_encode([code]*L_smaple)
        time_info = np.hstack([capa_sec[ind_stn][:, None], stn_sec[:, None]]).astype(int)
        # valid/test subsetting
        time_info_capa = time_info[:, 0]
        train_ind, valid_ind, test_ind = subset_valid(time_info_capa, valid_range, test_range)
        # assign pointed data
        stn_input[:, 0] = temp_qc
        stn_input[:, 1] = temp_raw
        stn_input[:, 2] = temp_freq
        # assign capa data
        for c in range(int(size_capa/2)):
            c_capa = int(2*c)
            c_etopo = c_capa+1
            # stn locs
            x_s, x_e = indx_capa[c][j]-half_edge_x, indx_capa[c][j]+half_edge_x
            y_s, y_e = indy_capa[c][j]-half_edge_y, indy_capa[c][j]+half_edge_y
            capa_input[:, :, :, c_capa] = capa_tuple[c_capa][ind_stn, x_s:x_e, y_s:y_e]
            stn_input[..., 3+c] = capa_tuple[c_capa][ind_stn, indx_capa[c][j], indy_capa[c][j]]
            for n in range(L_smaple):
                capa_input[n, :, :, c_etopo] = capa_tuple[c_etopo][x_s:x_e, y_s:y_e]
        # save
        L = len(stn_input)
        if L <= 1:
            print('station {} missing'.format(code))
            continue;
        name = perfix+code+'.hdf'
        data_group = (capa_input, stn_input, time_info, train_ind, valid_ind, test_ind, stn_info)
        label_group = ['capa_input', 'stn_input', 'time_info', 'train_ind', 'valid_ind', 'test_ind', 'xstn_info'] # x for string
        save_hdf5(data_group, label_group, out_dir, filename=name)
    return N_sample, N_code

def gen_by_stn(grid_tuple, capa_tuple, indx_gpm, indy_gpm, indx_capa, indy_capa, half_edge, gpm_sec, capa_sec, valid_range, test_range, stn_code, stn_obs_dir, perfix, out_dir):
    N_sample = []; N_code = []
    # size parameters
    half_edge_y = half_edge[1] #46 
    half_edge_x = half_edge[0] #23 # 46-by-92
    size_x, size_y, size_gpm = half_edge_x*2, half_edge_y*2, len(grid_tuple)
    size_capa = len(capa_tuple)
    # adjusting time axis for gridded data
    hr6 = 6*60*60
    hr05 = 30*60
    # gridded precip counts forward, now make it counting backward
    gpm_sec += hr05 
    capa_sec += hr6
    # =========================================== #
    # get nearest capa time axis for gpm
    ind_gpm = np.searchsorted(capa_sec, gpm_sec)
    # clean up out-of-bounds
    flag_pick = ind_gpm < len(capa_sec)
    ind_gpm, gpm_sec = pick_by_flag((ind_gpm, gpm_sec), flag_pick)
    # make sure capa axis in the front (for counting backward)
    capa_match = capa_sec[ind_gpm]
    flag_pad = (capa_match - gpm_sec) <= 0
    ind_gpm[flag_pad] += 1
    # clean up out-of-bounds
    flag_pick = ind_gpm < len(capa_sec)
    ind_gpm, gpm_sec = pick_by_flag((ind_gpm, gpm_sec), flag_pick)
    # make sure gap lower than 6hr, so that gpm is within capa coverage 
    gpm_on_capa = capa_sec[ind_gpm]
    flag_pick = (gpm_on_capa - gpm_sec) <= hr6
    ind_gpm, gpm_sec = pick_by_flag((ind_gpm, gpm_sec), flag_pick)
    # =========================================== #
    # get nearest gpm time axis for stn obs
    L_gpm = len(gpm_sec)
    for j, code in enumerate(stn_code):
        print(code)
        # ----- try import stn obs by code ----- #
        try:
            with pd.HDFStore(stn_obs_dir, 'r') as hdf_temp:
                temp_pd = hdf_temp[code]
        except:
            print('station {} missing'.format(code))
            continue; # jump to the next stn
        temp_qc   = temp_pd['PREC_INST_QC'].values
        temp_raw  = temp_pd['PREC_INST_RAW'].values
        temp_freq = temp_pd['FREQ'].values
        stn_sec   = temp_pd['datetime'].values.astype('O')/1e9
        # flag_out 0 precip
        flag_pick = temp_raw > 0
        temp_qc, temp_raw, temp_freq, stn_sec = pick_by_flag((temp_qc, temp_raw, temp_freq, stn_sec), flag_pick)
        # =========================================== #
        # get corresponding gpm index
        ind_stn = np.searchsorted(gpm_sec, stn_sec)
        ## flag out out-of-bounds
        flag_pick = ind_stn < L_gpm
        temp_qc, temp_raw, temp_freq, stn_sec, ind_stn = pick_by_flag((temp_qc, temp_raw, temp_freq, stn_sec, ind_stn), flag_pick)
        ## make sure GPM is earlier than obs
        gpm_match = gpm_sec[ind_stn]
        flag_pad = (gpm_match - stn_sec) < 0
        ind_stn[flag_pad] += 1
        # flag out out-of-bounds
        flag_pick = ind_stn < L_gpm
        temp_qc, temp_raw, temp_freq, stn_sec, ind_stn = pick_by_flag((temp_qc, temp_raw, temp_freq, stn_sec, ind_stn), flag_pick)
        # make sure gap lower than 0.5hr, so that stn is within gpm coverage 
        stn_on_gpm = gpm_sec[ind_stn]
        flag_pick = (stn_on_gpm - stn_sec) <= hr05
        temp_qc, temp_raw, temp_freq, ind_stn, stn_sec = pick_by_flag((temp_qc, temp_raw, temp_freq, ind_stn, stn_sec), flag_pick)
        # =========================================== #
        print('Sample number: {}'.format(len(ind_stn)))
        N_sample.append(len(ind_stn))
        # allocation
        L_smaple = len(ind_stn)
        stn_input = np.empty([L_smaple, 3]) # qc, raw, freq
        grid_input = np.empty([L_smaple, size_x, size_y, size_gpm]) # channel last
        capa_input = np.empty([L_smaple, size_x, size_y, size_capa]) # channel last
        # assign stn/time info
        N_code.append(code)
        stn_info = str_encode([code]*L_smaple)
        time_info = np.hstack([capa_sec[ind_gpm][ind_stn][:, None], gpm_sec[ind_stn][:, None], stn_sec[:, None]]).astype(int)
        # valid/test subsetting
        time_info_capa = time_info[:, 0]
        train_ind, valid_ind, test_ind = subset_valid(time_info_capa, valid_range, test_range)
        # assign pointed data
        stn_input[:, 0] = temp_qc
        stn_input[:, 1] = temp_raw
        stn_input[:, 2] = temp_freq
        # assign gridded data
        for c in range(size_gpm):
            x_s, x_e = indx_gpm[c][j]-half_edge_x, indx_gpm[c][j]+half_edge_x
            y_s, y_e = indy_gpm[c][j]-half_edge_y, indy_gpm[c][j]+half_edge_y
            if len(grid_tuple[c].shape) == 3:
                grid_input[:, :, :, c] = grid_tuple[c][ind_stn, x_s:x_e, y_s:y_e]
            else:
                for n in range(L_smaple):
                    grid_input[n, :, :, c] = grid_tuple[c][x_s:x_e, y_s:y_e]
        # assign capa data
        for c in range(size_capa):
            c_ind = c//2
            x_s, x_e = indx_capa[c_ind][j]-half_edge_x, indx_capa[c_ind][j]+half_edge_x
            y_s, y_e = indy_capa[c_ind][j]-half_edge_y, indy_capa[c_ind][j]+half_edge_y
            if len(capa_tuple[c].shape) == 3:
                capa_input[:, :, :, c] = (capa_tuple[c][ind_gpm, x_s:x_e, y_s:y_e])[ind_stn, ...]
            else:
                for n in range(L_smaple):
                    capa_input[n, :, :, c] = capa_tuple[c][x_s:x_e, y_s:y_e]
        # save
        name = perfix+code+'.hdf'
        data_group = (grid_input, capa_input, stn_input, time_info, train_ind, valid_ind, test_ind, stn_info)
        label_group = ['gpm_input', 'capa_input', 'stn_input', 'time_info', 'train_ind', 'valid_ind', 'test_ind', 'xstn_info'] # x for string
        save_hdf5(data_group, label_group, out_dir, filename=name)
    return N_sample, N_code

def merge_input_capa(names, precip_thres=0.1, option='TRAIN'):
    # test one file + get file size
    hdf_io = h5py.File(names[0], 'r')
    stn_input = hdf_io['stn_input'][0, ...]
    capa_input = hdf_io['capa_input'][0, ...]
    hdf_io.close()
    stn_shape = stn_input.shape
    capa_shape = capa_input.shape
    
    # allocation
    L = len(names)
    if option == 'TRAIN' or option == 'ALL':
        factor = 5000
    else:
        factor = 750
    stn_p = np.empty((L*factor,)+stn_shape)
    capa_p = np.empty((L*factor,)+capa_shape)
    cate_p = np.empty([L*factor])*np.nan
    time_p = np.empty([L*factor, 2])
    code_p = []
    # 
    stn_n = np.empty((L*factor,)+stn_shape)
    capa_n = np.empty((L*factor,)+capa_shape)
    cate_n = np.empty([L*factor])*np.nan
    time_n = np.empty([L*factor, 2])
    code_n = []
    #
    count_p = 0; count_n = 0
    # assign values
    for i, name in enumerate(names):
        print('processing: {}'.format(name))
        hdf_io = h5py.File(name, 'r')
        stn_info = hdf_io['xstn_info'][...]
        time_info = hdf_io['time_info'][...]
        stn_input = hdf_io['stn_input'][...]
        capa_input = hdf_io['capa_input'][...]
        if option == 'ALL':
            pick_ind = np.arange(len(capa_input))
        elif option == 'TRAIN':
            pick_ind = hdf_io['train_ind'][...]
        elif option == 'VALID':
            pick_ind = hdf_io['valid_ind'][...]
        elif option == 'TEST':
            pick_ind = hdf_io['test_ind'][...]
        hdf_io.close()
        if len(pick_ind) < 1:
            print('{} has no picks on {}'.format(stn_info[0], option))
            continue;
        # subsetting
        stn_info = stn_info[pick_ind]
        time_info = time_info[pick_ind, ...]
        stn_input = stn_input[pick_ind, ...]
        capa_input = capa_input[pick_ind, ...]
        # create flags (bad = positive = minority class)
        flag_bad = np.logical_and(np.abs(stn_input[:, 0]-stn_input[:, 1])>precip_thres, stn_input[:, 1]>0)
        flag_good = np.logical_and(np.abs(stn_input[:, 0]-stn_input[:, 1])<=precip_thres, stn_input[:, 1]>0)
        # loop over samples
        for j in range(len(stn_input[:, 0])):
            if flag_bad[j]:
                stn_p[count_p, ...] = stn_input[j, ...]
                capa_p[count_p, ...] = capa_input[j, ...]
                time_p[count_p, ...] = time_info[j, ...]
                cate_p[count_p] = flag_bad[j]
                code_p.append(stn_info[j])
                count_p += 1
            elif flag_good[j]:
                stn_n[count_n, ...] = stn_input[j, ...]
                capa_n[count_n, ...] = capa_input[j, ...]
                time_n[count_n, ...] = time_info[j, ...]
                cate_n[count_n] = flag_bad[j]
                code_n.append(stn_info[j])
                count_n += 1
            else:
                print('!! zero raw !!')
    return (stn_p[:count_p, ...], capa_p[:count_p, ...], time_p[:count_p, ...], cate_p[:count_p], code_p[:count_p]), \
           (stn_n[:count_n, ...], capa_n[:count_n, ...], time_n[:count_n, ...], cate_n[:count_n], code_n[:count_n])

def merge_input(names, precip_thres=0.1, option='TRAIN'):
    # test one file + get file size
    hdf_io = h5py.File(names[0], 'r')
    stn_input = hdf_io['stn_input'][0, ...]
    gpm_input = hdf_io['gpm_input'][0, ...]
    capa_input = hdf_io['capa_input'][0, ...]
    hdf_io.close()
    stn_shape = stn_input.shape
    gpm_shape = gpm_input.shape
    capa_shape = capa_input.shape
    
    # allocation
    L = len(names)
    if option == 'TRAIN':
        factor = 5000
    else:
        factor = 750
    stn_p = np.empty((L*factor,)+stn_shape)
    gpm_p = np.empty((L*factor,)+gpm_shape)
    capa_p = np.empty((L*factor,)+capa_shape)
    cate_p = np.empty([L*factor])*np.nan
    time_p = np.empty([L*factor, 3])
    code_p = []
    # 
    stn_n = np.empty((L*factor,)+stn_shape)
    gpm_n = np.empty((L*factor,)+gpm_shape)
    capa_n = np.empty((L*factor,)+capa_shape)
    cate_n = np.empty([L*factor])*np.nan
    time_n = np.empty([L*factor, 3])
    code_n = []
    #
    count_p = 0; count_n = 0
    # assign values
    for i, name in enumerate(names):
        print('processing: {}'.format(name))
        hdf_io = h5py.File(name, 'r')
        stn_info = np.squeeze(hdf_io['xstn_info'][...])
        time_info = hdf_io['time_info'][...]
        stn_input = hdf_io['stn_input'][...]
        gpm_input = hdf_io['gpm_input'][...]
        capa_input = hdf_io['capa_input'][...]
        if option == 'TRAIN':
            pick_ind = hdf_io['train_ind'][...]
        elif option == 'VALID':
            pick_ind = hdf_io['valid_ind'][...]
        elif option == 'TEST':
            pick_ind = hdf_io['test_ind'][...]
        hdf_io.close()
        if len(pick_ind) < 1:
            print('{} has no picks on {}'.format(stn_info[0], option))
            continue;
        # subsetting
        stn_info = stn_info[pick_ind].tolist()
        time_info = time_info[pick_ind, ...]
        stn_input = stn_input[pick_ind, ...]
        gpm_input = gpm_input[pick_ind, ...]
        capa_input = capa_input[pick_ind, ...]
        # create flags (bad = positive = minority class)
        flag_bad = np.logical_and(np.abs(stn_input[:, 0]-stn_input[:, 1])>precip_thres, stn_input[:, 1]>0)
        flag_good = np.logical_and(np.abs(stn_input[:, 0]-stn_input[:, 1])<=precip_thres, stn_input[:, 1]>0)
        # loop over samples
        for j in range(len(stn_input[:, 0])):
            if flag_bad[j]:
                gpm_p[count_p, ...] = gpm_input[j, ...]
                stn_p[count_p, ...] = stn_input[j, ...]
                capa_p[count_p, ...] = capa_input[j, ...]
                time_p[count_p, ...] = time_info[j, ...]
                cate_p[count_p] = flag_bad[j]
                code_p.append(stn_info[j])
                count_p += 1
            elif flag_good[j]:
                gpm_n[count_n, ...] = gpm_input[j, ...]
                stn_n[count_n, ...] = stn_input[j, ...]
                capa_n[count_n, ...] = capa_input[j, ...]
                time_n[count_n, ...] = time_info[j, ...]
                cate_n[count_n] = flag_bad[j]
                code_n.append(stn_info[j])
                count_n += 1
            else:
                print('!! zero raw spotted !!')
    return (gpm_p[:count_p, ...], stn_p[:count_p, ...], capa_p[:count_p, ...], time_p[:count_p, ...], cate_p[:count_p], code_p[:count_p]), \
           (gpm_n[:count_n, ...], stn_n[:count_n, ...], capa_n[:count_n, ...], time_n[:count_n, ...], cate_n[:count_n], code_n[:count_n])


def aug_gen_capa(input_dir, out_dir, ens, label=1.0, rot=20, shift=0):
    hdf_io = h5py.File(input_dir, 'r')
    capa_p = hdf_io['capa_input'][...]
    stn_p  = hdf_io['stn_input'][...]
    cate_p = hdf_io['cate_out'][...] # ones
    hdf_io.close()
    #
    pick_flag = [True, False]*ens
    etopo_flag = [False, True]*ens
    L = len(cate_p)
    ind = np.arange(L); inds = []
    #
    grid_aug = np.empty(capa_p.shape)
    #
    aug_gen = keras.preprocessing.image.ImageDataGenerator(rotation_range=rot, width_shift_range=shift, height_shift_range=shift)
    
    aug_gen.fit(capa_p[..., pick_flag])
    aug_iter = aug_gen.flow(capa_p[..., pick_flag], ind, batch_size=1)
    # <---- 
    for i in range(L): 
        aug_temp, ind_temp = aug_iter.next()
        inds.append(int(ind_temp))
        grid_aug[i:i+1, ..., pick_flag] = aug_temp
        grid_aug[i, ..., etopo_flag] = capa_p[int(ind_temp), ..., etopo_flag]
    cate_aug = label*np.ones(L)
    print('save to {}'.format(out_dir))
    hdf_io = h5py.File(out_dir, 'w')
    hdf_io.create_dataset('capa_input', data=grid_aug)
    hdf_io.create_dataset('stn_input', data=stn_p[inds, :])
    hdf_io.create_dataset('cate_out', data=cate_aug)
    hdf_io.close()

def random_batches(h5name, labels, out_dir, batch_size=128, prefix='TRAIN', option='CAPA'):
    '''
    Split into batches
    '''
    hdf = h5py.File(h5name, 'r')
    p_dict = {}
    for i, label in enumerate(labels):
        p_dict[label] = hdf[label][...]
    hdf.close()
    # total available samples
    L = len(p_dict[labels[0]])
    print('Number of samples: {}'.format(L))
    # shuffle data
    indp = shuffle_ind(L)
    for i, label in enumerate(labels):
        p_dict[label] = p_dict[label][indp, ...]
    # number of batches
    N_batch = L//batch_size
    print('Total number of batches: {}'.format(N_batch))
    # loop over samples and split
    count = 0
    for n in range(N_batch):
        if count+batch_size < L:
            dict_batch = {}
            for i, label in enumerate(labels):
                dict_batch[label] = p_dict[label][count:(count+batch_size), ...] # pos
                # update indices
            count += batch_size
            # save
            temp_name = out_dir+prefix+str(n)+'.npy'
            np.save(temp_name, dict_batch)
            print(temp_name)
        else:
            break;

def balanced_batches(h5name_p, h5name_n, labels, out_dir, pos_rate=0.5, padx=[1, 1], pady=[1, 1], batch_size=128, prefix='TRAIN', option='CAPA'):
    '''
    Create balanced batches
    '''
    hdf = h5py.File(h5name_p, 'r')
    p_dict = {}
    for i, label in enumerate(labels):
        p_dict[label] = hdf[label][...]
    hdf.close()
    #
    hdf = h5py.File(h5name_n, 'r')
    n_dict = {}
    for i, label in enumerate(labels):
        n_dict[label] = hdf[label][...]
    hdf.close()
    # total available samples
    L_pos = len(p_dict[labels[0]])
    L_neg = len(n_dict[labels[0]])
    L_channel = p_dict[labels[0]].shape[-1] # channel last
    print('Number of positive samples: {}'.format(L_pos))
    print('Number of negative samples: {}'.format(L_neg))
    # shuffle data
    indp = shuffle_ind(L_pos)
    indn = shuffle_ind(L_neg)
    for i, label in enumerate(labels):
        p_dict[label] = p_dict[label][indp, ...]
        n_dict[label] = n_dict[label][indn, ...]
    # parameters that defines the batch components
    # number of pos and neg samples in each batch 
    L_batch_pos = int(pos_rate*batch_size)
    L_batch_neg = batch_size - L_batch_pos
    # number of batches
    N_batch = int(np.min(np.array([L_pos/L_batch_pos, L_neg/L_batch_neg])))
    print('Number of positive samples in each batch: {}'.format(L_batch_pos))
    print('Total number of batches: {}'.format(N_batch))
    # define indices
    count_pos1 = 0; count_pos2 = L_batch_pos
    count_neg1 = 0; count_neg2 = L_batch_neg
    # loop over samples and split
    for n in range(N_batch):
        #print(i)
        if count_pos2<L_pos and count_neg2<L_neg:
            dict_batch = {}
            for i, label in enumerate(labels):
                temp_shape = p_dict[label][...].shape
                if label[0] == 'x':
                    dict_batch[label] = np.zeros((batch_size, 1)).astype(str) # <--- set as str
                elif len(temp_shape)>1:
                    dict_batch[label] = np.empty((batch_size,)+temp_shape[1:])
                else:
                    dict_batch[label] = np.empty((batch_size,))
                dict_batch[label][:L_batch_pos, ...] = p_dict[label][count_pos1:count_pos2, ...] # pos
                dict_batch[label][L_batch_neg:, ...] = n_dict[label][count_neg1:count_neg2, ...] # neg
            # update indices
            count_pos1 += L_batch_pos; count_pos2 += L_batch_pos
            count_neg1 += L_batch_neg; count_neg2 += L_batch_neg
            # features
            # assuming grid, stn are the first two labels <----- !!!!!!!!
            if option == 'CAPA': 
                for j in range(L_channel):
                    if np.mod(j, 2) == 0: 
                        dict_batch[labels[0]][..., j]  = crop_stn(dict_batch[labels[0]][..., j], dict_batch[labels[1]][:, 1], padx, pady)
            elif option == 'GPM':
                dict_batch[labels[0]][..., 0]  = crop_stn(dict_batch[labels[0]][..., 0], dict_batch[labels[1]][:, 1], padx, pady)
            # save
            temp_name = out_dir+prefix+str(n)+'.npy'
            np.save(temp_name, dict_batch)
            print(temp_name)
        else:
            break;
            
def create_full_data(h5name_p, h5name_n, labels, out_dir, padx=[1, 1], pady=[1, 1], prefix='TEST', option='CAPA'):
    '''
    Create balanced "full" dataset (no train, valid split)
    '''
    hdf = h5py.File(h5name_p, 'r')
    p_dict = {}
    for i, label in enumerate(labels):
        p_dict[label] = hdf[label][...]
    hdf.close()
    #
    hdf = h5py.File(h5name_n, 'r')
    n_dict = {}
    for i, label in enumerate(labels):
        n_dict[label] = hdf[label][...]
    hdf.close()
    print('Data loaded')
    Lp = len(p_dict[labels[0]])
    Ln = len(n_dict[labels[0]])
    L_channel = p_dict[labels[0]].shape[-1] # channel last
    L = Lp+Ln
    dict_batch = {}
    for i, label in enumerate(labels):
        # allocation
        temp_shape = p_dict[label][...].shape
        if label[0] == 'x':
            dict_batch[label] = np.zeros(L).astype(str) # <--- set as str
        elif len(temp_shape)>1:
            dict_batch[label] = np.empty((L,)+temp_shape[1:])
        else:
            dict_batch[label] = np.empty((L,))
        # assign values
        if label[0] == 'x':
            dict_batch[label][:Lp, ...] = np.squeeze(p_dict[label]) # pos
            dict_batch[label][Lp:L, ...] = np.squeeze(n_dict[label]) # neg
            dict_batch[label] = str_encode(dict_batch[label])
        else:
            dict_batch[label][:Lp, ...] = p_dict[label] # pos
            dict_batch[label][Lp:L, ...] = n_dict[label] # neg
    # assuming grid, stn are the first two labels <----- !!!!!!!!
    if option == 'CAPA': 
        for j in range(L_channel):
            if np.mod(j, 2) == 0: 
                dict_batch[labels[0]][..., j]  = crop_stn(dict_batch[labels[0]][..., j], dict_batch[labels[1]][:, 1], padx, pady)
    elif option == 'GPM':
        dict_batch[labels[0]][..., 0]  = crop_stn(dict_batch[labels[0]][..., 0], dict_batch[labels[1]][:, 1], padx, pady)
    save_hdf5(tuple(dict_batch.values()), labels, out_dir, filename=prefix+'.hdf')

def split_batches(file_dir, file_name, ens, prefix):
    for i, name in enumerate(file_name):
        print(name)
        data_temp = np.load(name)
        grid_input = data_temp[()]['capa_input']
        cate_out = data_temp[()]['cate_out']
        for j in range(ens):
            temp_name = prefix+str(j)+'_BATCH'+str(i)+'.npy'
            save_d = {'capa_input':grid_input[:, :, :, 2*j:2*(j+1)], 'cate_out':cate_out}
            np.save(file_dir+temp_name, save_d)    

def split_batches_mlp(file_dir, file_name, ens, prefix):
    for i, name in enumerate(file_name):
        print(name)
        data_temp = np.load(name)
        grid_input = data_temp[()]['capa_input']
        stn_input = data_temp[()]['stn_input']
        cate_out = data_temp[()]['cate_out']
        for j in range(ens):
            temp_name = prefix+str(j)+'_BATCH'+str(i)+'.npy'
            save_d = {'capa_input':grid_input[:, :, :, 2*j:2*(j+1)], 'stn_input':stn_input[:, 1], 'cate_out':cate_out}
            np.save(file_dir+temp_name, save_d)  
            
def packing_data(names, out_dir, batch_size, labels, prefix='VALID'):
    L = len(names)
    L_out = L*batch_size
    print('Number of batches: {}'.format(L))
    temp_data = np.load(names[0])[()]
    dict_out = {}
    for i, label in enumerate(labels):
        temp_var = temp_data[label]
        if label[0] == 'x':
            dict_out[label] = np.zeros((L_out)).astype(str)
        elif len(temp_var.shape) > 1:
            dict_out[label] = np.empty((L_out,)+temp_var.shape[1:])
        else:
            dict_out[label] = np.empty(L_out)
    for i, name in enumerate(names):
        temp_data = np.load(name)
        for j, label in enumerate(labels):
            temp_var = temp_data[()][label][...]
            if label[0] == 'x':
                dict_out[label][i*batch_size:(i+1)*batch_size, ...] = np.squeeze(temp_data[()][label])
                
            elif len(temp_var.shape) > 1:
                dict_out[label][i*batch_size:(i+1)*batch_size, ...] = temp_data[()][label][...]
            else:
                dict_out[label][i*batch_size:(i+1)*batch_size] = temp_data[()][label][...]
    hdfname = out_dir+prefix+'_pack.hdf'    
    hdf_io = h5py.File(hdfname, 'w')
    for i, label in enumerate(labels):
        if label[0] == 'x':
            hdf_io.create_dataset(label, (L_out, 1), 'S10', str_encode(dict_out[label][...]))
        else:
            hdf_io.create_dataset(label, data=dict_out[label][...])
    hdf_io.close()
    print('Save to: {}'.format(hdfname))

    
def ensemble_predict(members, ensemble, grid_input):
    '''
    '''
    N_sample = grid_input.shape[0]
    N_model = len(members)
    out_member = np.empty([N_sample, N_model])
    print('Preparing member outputs')
    for i, model in enumerate(members):
        print('Member {}'.format(i))
        out_member[:, i] = np.squeeze(model.predict([grid_input[..., 2*i:2*(i+1)]]))
    print('Preparing ensemble outputs')
    return ensemble.predict([out_member])
        
def ensemble_train(members, train_tuple, valid_tuple, node_factor=2, out_dir='/glade/u/home/ksha/data/Keras/BACKUP/'):
    '''
    see 'ensemble_predict'
    '''
    grid_train, cate_train = train_tuple
    grid_valid, cate_valid = valid_tuple
    #
    N_train = grid_train.shape[0]
    N_valid = grid_valid.shape[0]
    N_model = len(members)
    out_train = np.empty([N_train, N_model])
    out_valid = np.empty([N_valid, N_model])
    #
    print('Preparing member outputs')
    for i, model in enumerate(members):
        print('\tmember {}'.format(i))
        out_train[:, i] = np.squeeze(model.predict([grid_train[..., 2*i:2*(i+1)]]))
        out_valid[:, i] = np.squeeze(model.predict([grid_valid[..., 2*i:2*(i+1)]]))
    # model structure
    print('Preparing ensemble sections')
    IN = keras.layers.Input(shape=(N_model,))
    X1 = keras.layers.Dense(node_factor*N_model)(IN)
    X1 = keras.layers.Activation('tanh')(X1)
    OUT = keras.layers.Dense(1, activation=keras.activations.sigmoid)(X1)
    model = keras.models.Model(inputs=IN, outputs=OUT)
    opt = keras.optimizers.Adam(lr=0.002, decay=0.0075)
    ## callbacks
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=opt, metrics=['accuracy'])
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0005),
                 keras.callbacks.ModelCheckpoint(filepath=out_dir+'QC_ENS.hdf', verbose=1, 
                                                 monitor='val_loss', save_best_only=True)]
    model.fit(out_train, y=cate_train, epochs=15, verbose=1, callbacks=callbacks, validation_data=(out_valid, cate_valid),
              shuffle=True, use_multiprocessing=True, max_queue_size=12, workers=6)
    print('Saved to {}'.format(out_dir+'QC_ENS.hdf'))
    return model


## ========== OLD functions ========== ##
# def split_and_save(grid, stn, ref, cate, code, count, out_dir, key='p'):
    
#     '''
#     Save merged data into single hdf5 files
#     Will split train, valid and test by "count"
#     '''
    
#     print(key+' num: '+str(count[-1]))
#     # train
#     vali_name = out_dir+'TRAIN_'+key+'.hdf'
#     hdf = h5py.File(vali_name, 'w')
#     hdf.create_dataset('grid_input', data=grid[:count[0], ...])
#     hdf.create_dataset('stn_input' , data= stn[:count[0], ...])
#     hdf.create_dataset('capa_ref'  , data= ref[:count[0], ...])
#     hdf.create_dataset('cate_out'  , data=cate[:count[0], ...])
#     hdf.create_dataset('stn_code'  , (len(code[:count[0]]), 1), 'S10', code[:count[0]])
#     hdf.close()
#     # pr_valid
#     vali_name = out_dir+'VALID_'+key+'.hdf'
#     hdf = h5py.File(vali_name, 'w')
#     hdf.create_dataset('grid_input', data=grid[count[0]:count[1], ...])
#     hdf.create_dataset('stn_input' , data= stn[count[0]:count[1], ...])
#     hdf.create_dataset('capa_ref'  , data= ref[count[0]:count[1], ...])
#     hdf.create_dataset('cate_out'  , data=cate[count[0]:count[1], ...])
#     hdf.create_dataset('stn_code'  , (len(code[count[0]:count[1]]), 1), 'S10', code[count[0]:count[1]])
#     hdf.close()
#     # pr test
#     vali_name = out_dir+'TEST_'+key+'.hdf'
#     hdf = h5py.File(vali_name, 'w')
#     hdf.create_dataset('grid_input', data=grid[count[1]:count[2], ...])
#     hdf.create_dataset('stn_input' , data= stn[count[1]:count[2], ...])
#     hdf.create_dataset('capa_ref'  , data= ref[count[1]:count[2], ...])
#     hdf.create_dataset('cate_out'  , data=cate[count[1]:count[2], ...])
#     hdf.create_dataset('stn_code'  , (len(code[count[1]:count[2]]), 1), 'S10', code[count[1]:count[2]])
#     hdf.close()

# def data_split(count_p, count_n, train_rate=0.8, valid_rate=0.1):
#     '''
#     Calculate indices of samples for spliting train, valid, test
#     '''
#     count_p_train = int(count_p*train_rate)
#     count_p_valid = count_p_train+int(count_p*valid_rate)
#     count_n_train = int(count_n*train_rate)
#     count_n_valid = count_n_train+int(count_n*valid_rate)
#     # save data
#     count_p_out = [count_p_train, count_p_valid, count_p]
#     count_n_out = [count_n_train, count_n_valid, count_n]
#     # print out as a test of indexing
#     print(count_p_out); print(count_n_out)
#     return count_p_out, count_n_out

# def balanced_full_data(h5name_p, h5name_n, out_dir, prefix='TRAIN'):
#     '''
#     Create balanced "full" dataset (no train, valid split)
#     '''
#     hdf = h5py.File(h5name_p, 'r')
#     grid_p = hdf['grid_input'][...]
#     stn_p  = hdf['stn_input'][...]
#     cate_p = hdf['cate_out'][...] # ones
#     ref_p  = hdf['capa_ref'][...]
#     code_p = du.str_decode(hdf['stn_code'][...])
#     hdf.close()
#     #
#     hdf = h5py.File(h5name_n, 'r')
#     grid_n = hdf['grid_input'][...]
#     stn_n  = hdf['stn_input'][...]
#     cate_n = hdf['cate_out'][...]
#     ref_n  = hdf['capa_ref'][...]
#     code_n = du.str_decode(hdf['stn_code'][...])
#     hdf.close()
#     print('Data loaded')
#     # flag to categories
#     cate_p = to_categorical(cate_p, 2)
#     cate_n = to_categorical(cate_n, 2)
#     # total available samples
#     L_pos = len(stn_p[:, 0])
#     L_neg = len(stn_n[:, 0])
#     # Allocation
#     # sizes
#     size_x, size_y, size_c = grid_p.shape[1:]
#     size_l = stn_p.shape[1]
#     L_full = int(np.min(np.array([L_pos, L_neg])))
#     # placeholders
#     grid_batch = np.empty([2*L_full, size_x, size_y, size_c])
#     stn_batch  = np.empty([2*L_full, size_l]) # qc, raw, freq
#     cate_batch = np.empty([2*L_full, 2]) # 2 for binary classification
#     ref_batch  = np.empty([2*L_full, 4, 4, int(size_c/2)])
#     code_batch = np.zeros([2*L_full]).astype('str')
#     # shuffle index
#     indp = du.shuffle_ind(L_pos)
#     indn = du.shuffle_ind(L_neg)
#     indf = du.shuffle_ind(2*L_full)
#     # assign values
#     # pos
#     grid_batch[:L_full, ...] = grid_p[indp, ...][:L_full, ...]
#     stn_batch [:L_full, ...] = stn_p [indp, ...][:L_full, ...]
#     cate_batch[:L_full, ...] = cate_p[indp, ...][:L_full, ...]
#     code_batch[:L_full] = np.array(code_p)[indp][:L_full]
#     # neg
#     grid_batch[L_full:, ...] = grid_n[indn, ...][:L_full, ...]
#     stn_batch [L_full:, ...] = stn_n [indn, ...][:L_full, ...]
#     cate_batch[L_full:, ...] = cate_n[indn, ...][:L_full, ...]
#     code_batch[L_full:] = np.array(code_n)[indn][:L_full]    
#     # shuffle entire
#     grid_batch = grid_batch[indf, ...]
#     stn_batch  = stn_batch [indf, ...]
#     cate_batch = cate_batch[indf, ...]
#     code_batch = code_batch[indf]
#     # blending features
#     print('generating features')
#     for j in range(size_c):
#         if np.mod(j, 2) == 0: 
#             grid_batch[..., j]  = du.crop_stn(grid_batch[..., j] , stn_batch[:, 1], [1, 1])
#     # save
#     name_temp = out_dir+prefix+'.hdf'
#     print('Saving to {}'.format(name_temp))
#     hdf_io = h5py.File(name_temp, 'w')
#     hdf_io.create_dataset('grid_input', data=grid_batch)
#     hdf_io.create_dataset('stn_input' , data=stn_batch)
#     hdf_io.create_dataset('capa_ref'  , data=ref_batch)
#     hdf_io.create_dataset('cate_out'  , data=cate_batch)
#     hdf_io.create_dataset('stn_code'  , (2*L_full, 1), 'S10', du.str_encode(code_batch.tolist()))
#     hdf_io.close()
#
# def dt_diff_check(dt_diff, thres):
#     '''
#     1. Checking the difference between two np.datetime64 values
#     2. Assuming positive diff, return True if diff lower than threshold
#     '''
#     temp_sec = dt_diff/np.timedelta64(1, 's')
#     if temp_sec >= 0 and temp_sec <= thres:
#         return True
#     else:
#         return False

# def gen_by_time(capa_tuple, etopo_tuple, indx_tuple, indy_tuple, half_edge, date_capa, stn_code, stn_obs_dir, perfix, obs_thres=21601):
#     '''
#     1. Subseting raw datasets
#     2. Matching CaPA datetime and obs datetime
#     3. Clean up all NaNs
#     4. Assuming 6 CaPA features and 6 etopo features
#     5. Return number of samples on all matched datetimes (for train, valid split)
#     '''
#     ens = len(capa_tuple)
#     N_date = []; N_sample = []
#     Lt = len(date_capa)
#     print('CaPA len: {}'.format(Lt))
#     # size parameters
#     half_edge_y = half_edge[1] #46 
#     half_edge_x = half_edge[0] #23 # 46-by-92
#     size_x, size_y, size_c = half_edge_x*2, half_edge_y*2, int(2*ens)
#     for i in range(len(date_capa)-1): # len(date_capa)-1
#         # datetim ref for pandas subset
#         date_ref2 = date_capa[i+1]
#         date_ref1 = date_capa[i]
#         print(date_ref2)
#         # allocation
#         stn_input = np.empty([700, 3]) # 3000 is a rough estimate
#         grid_input = np.empty([700, size_x, size_y, size_c]) # channel last
#         capa_ref = np.empty([700, 4, 4, int(size_c/2)])
#         metadata = []
#         count = 0 
#         # loop over stations
#         for j, code in enumerate(stn_code):
#             # ----- try import stn obs by code ----- #
#             try:
#                 with pd.HDFStore(stn_obs_dir, 'r') as hdf_temp:
#                     temp_pd = hdf_temp[code]
#             except:
#                 continue; # jump to the next stn
#             # ----- subset the full pandas by datetime ref ----- #
#             ind_flag = np.logical_and((temp_pd['datetime'] >= date_ref1).values, (temp_pd['datetime'] < date_ref2).values)
#             temp_subset = temp_pd.loc[ind_flag]
#             L = len(temp_subset)
#             if L < 1:
#                 continue; # jump to the next stn
#             # ----- examine the time difference between capa and stn obs ----- #
#             temp_date = temp_subset['datetime'].values
#             temp_qc   = temp_subset['PREC_INST_QC'].values
#             temp_raw  = temp_subset['PREC_INST_RAW'].values
#             temp_freq = temp_subset['FREQ'].values
#             # purge time diff longer than 1-day
#             flag = np.ones(L).astype(bool)
#             inds = []
#             for k in range(L):
#                 ind0 = np.searchsorted(date_capa, temp_date[k], 'left')
#                 if ind0 < Lt-1:
#                     ind1 = ind0+1
#                 else:
#                     ind1 = ind0
#                 #print('{}:{}:{}:{}'.format(i, ind0, ind1, i+1))
#                 dt_diff0 = date_capa[ind0] - temp_date[k] # stn time minus capa time must lower than zero
#                 dt_diff1 = date_capa[ind1] - temp_date[k]
#                 #print('{}-{}-{}'.format(date_capa[ind0], date_capa[ind1], temp_date[k]))
#                 if dt_diff_check(dt_diff0, thres=obs_thres):
#                     inds.append(ind0)
#                     flag[k] = True
#                 elif dt_diff_check(dt_diff1, thres=obs_thres):
#                     inds.append(ind1)
#                     flag[k] = True
#                 else:
#                     print('no good capa match for {}'.format(temp_date[k]))
#                     inds.append(999)
#                     flag[k] = False
#                     continue
#             # ----- flag out zero raws (they are not useful) ----- #
#             flag_nozero = temp_raw > 0 # 1e-5
#             flag_pick = np.logical_and(flag, flag_nozero)
#             if ~flag_pick.any():
#                 continue; # jump to the next stn
#             # ----- append to the current batch ----- #
#             ind_pick = []
#             for n in range(L):
#                 if flag_pick[n]:
#                     ind_pick.append(inds[n])
#             L = np.sum(flag_pick)
#             stn_input[count:count+L, 0] = temp_qc[flag_pick]
#             stn_input[count:count+L, 1] = temp_raw[flag_pick]
#             stn_input[count:count+L, 2] = temp_freq[flag_pick]
#             for k in range(ens):
#                 capa_ref[count:count+L, :, :, k] = capa_tuple[k][ind_pick, indx_tuple[k][j]-2:indx_tuple[k][j]+2, indy_tuple[k][j]-2:indy_tuple[k][j]+2]            
#                 grid_input[count:count+L, :, :, 2*k] = capa_tuple[k][ind_pick, indx_tuple[k][j]-half_edge_x:indx_tuple[k][j]+half_edge_x, indy_tuple[k][j]-half_edge_y:indy_tuple[k][j]+half_edge_y]   
#                 grid_input[count:count+L, :, :, 2*k+1] = etopo_tuple[k][indx_tuple[k][j]-half_edge_x:indx_tuple[k][j]+half_edge_x, indy_tuple[k][j]-half_edge_y:indy_tuple[k][j]+half_edge_y]
#             metadata.append([code]*L)
#             count += L
#         # ----- check length ----- #
#         stn_input = stn_input[:count, ...]
#         grid_input = grid_input[:count, ...]
#         # ----- filt out zero precip ----- #
#         L_out = len(stn_input)
#         if L_out>0:
#             print('num of elements: {}'.format(np.sum(L_out)))
#             N_sample.append(L_out)
#             N_date.append(date_ref2.astype('O'))
#             # flat list
#             stns = [item for sublist in metadata for item in sublist]
#             #print(stns)
#             npy_name = perfix+date_ref2.astype(datetime).strftime('%Y%m%d%H')+'.npy'
#             save_dict = {'grid_input':grid_input, 'stn_input':stn_input, 'capa_ref':capa_ref, 'metadata':stns, 'datetime':date_capa[ind_pick]}
#             np.save(npy_name, save_dict)
#         else:
#             print('all zeros')
#     return N_sample, N_date, ['grid_input', 'stn_input', 'capa_ref', 'metadata', 'datetime']
    