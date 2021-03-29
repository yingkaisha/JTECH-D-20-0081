import sys
import h5py
import numpy as np
import pandas as pd
from glob import glob
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.utils import resample

sys.path.insert(0, '/glade/u/home/ksha/ML_repo/utils/')
import data_utils as du
import keras_utils as ku
from namelist import *

R18 = 200 # number of boostrap cycles
save_dir = '/glade/work/ksha/data/Keras/QC_publish/'
eval_dir = '/glade/work/ksha/data/evaluation/'
data_dir2 = HOLD_dir+freq+'CAPA_TEST_pack.hdf'

# ===== import models ===== #
print('Loading models')
model_member = []
for i in range(ens):
    key = freq+'QC_noelev_CNN'+str(i)
    print('\t {}'.format(key))
    model_member.append(keras.models.load_model(save_dir+key+'.hdf'))
# ===== import data ===== #
# TEST
hdf_io = h5py.File(data_dir2, 'r')
capa_input = hdf_io['capa_input'][...]
cate_out = hdf_io['cate_out'][...]
hdf_io.close()
#
L2 = len(cate_out); inds = list(range(L2))
# boostraping
cate_p_test  = np.empty([L2, ens]); names = []
FP_boost = np.zeros([ens, R18, 10000])*np.nan
TP_boost = np.zeros([ens, R18, 10000])*np.nan
AUC_boost = np.empty([ens, R18])
cate_p = np.empty([L2, R18])
cate_test = np.empty([L2, ens, R18])
#
grid_boost = np.empty(capa_input.shape)
cate_boost = np.empty(cate_out.shape)
#
for r in range(R18):
    print('Boosting cycle {}'.format(r))
    # get boostraping indices
    inds_boost = resample(inds, replace=True, n_samples=L2)
    for j, ind in enumerate(inds_boost):
        grid_boost[j, ...] = capa_input[ind, ...]
        cate_boost[j] = cate_out[ind]
    # loop over members    
    for i in range(ens):
        # get model
        model_temp = model_member[i]
        cate_temp_test  = model_temp.predict(grid_boost[..., 2*i][..., None]) #
        # RUC
        FP_temp, TP_temp, _ = roc_curve(cate_boost, cate_temp_test)
        AUC_boost[i, r] = auc(FP_temp, TP_temp)
        TP_boost[i, r, :len(TP_temp)] = TP_temp
        FP_boost[i, r, :len(FP_temp)] = FP_temp
        # backup p values
        cate_test[:, i, r] = np.squeeze(cate_temp_test)
    cate_p[:, r] = np.squeeze(cate_boost)
    
# ===== Save data ===== #
#
name = eval_dir+'EVAL_QC_noelev_boost.hdf'
hdf_io = h5py.File(name, 'w')
hdf_io.create_dataset('cate_p', data=cate_p)
hdf_io.create_dataset('cate_boost', data=cate_test)
hdf_io.create_dataset('AUC_boost', data=AUC_boost)
hdf_io.create_dataset('TP_boost', data=TP_boost)
hdf_io.create_dataset('FP_boost', data=FP_boost)
hdf_io.close()
print('Save to {}'.format(name))

