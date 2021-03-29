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
from namelist import *

#ens = 5 # number of classifiers
freq = 'HIGH_' 
save_dir = '/glade/work/ksha/data/Keras/QC_publish/'
eval_dir = '/glade/work/ksha/data/evaluation/'
data_dir1 = HOLD_dir+freq+'CAPA_VALID_natural.hdf'
data_dir2 = HOLD_dir+freq+'CAPA_TEST_natural.hdf'
save_name = 'EVAL_QC_natural.npy'
# ===== import data ===== #
print('Importing data')
# VALID
hdf_io = h5py.File(data_dir1, 'r')
grid_valid = hdf_io['capa_input'][...]
cate_valid = hdf_io['cate_out'][...]
hdf_io.close()
# TEST
hdf_io = h5py.File(data_dir2, 'r')
grid_test = hdf_io['capa_input'][...]
cate_test = hdf_io['cate_out'][...]
hdf_io.close()

# get size of data
L1 = len(cate_valid)
L2 = len(cate_test)
# Train/valid/test set prediction
# ===== import models ===== #
print('Loading models')
model_member = []
for i in range(ens):
    key = freq+'QC_CNN'+str(i)
    print('\t {}'.format(key))
    model_member.append(keras.models.load_model(save_dir+key+'.hdf'))
#
print('Model evaluation started')
cate_p_test  = np.empty([L1, ens])
cate_p_valid  = np.empty([L2, ens])
names = []
REPORT = {}; FP = {}; TP = {}; AUC = {}
for i in range(ens):
    name = 'ENS'+str(i); names.append(name)
    print('\tProcessing {}'.format(name))
    model_temp = model_member[i]
    cate_temp_valid = np.squeeze(model_temp.predict(grid_valid[..., 2*i:2*(i+1)]))
    cate_temp_test = np.squeeze(model_temp.predict(grid_test[..., 2*i:2*(i+1)]))
    # backup probabilistic output
    cate_p_test[:, i] = cate_temp_test
    cate_p_valid[:, i] = cate_temp_valid
    # report
    REPORT[name] = classification_report(cate_test, cate_temp_test>=0.5)
    # RUC
    FP[name], TP[name], _ = roc_curve(cate_out, cate_temp_test)
    AUC[name] = auc(FP[name], TP[name])
# ens model
print('Processing ensembles')
model_ens = keras.models.load_model(save_dir+'QC_ENS.hdf')
cate_ens_valid  = np.squeeze(model_ens.predict(cate_p_valid))
cate_ens_test  = np.squeeze(model_ens.predict(cate_p_test))
# report
REPORT['ENS'] = classification_report(cate_out, cate_ens_test>=0.5)
FP['ENS'], TP['ENS'], _ = roc_curve(cate_out, cate_ens_test)
AUC['ENS'] = auc(FP['ENS'], TP['ENS'])
print(classification_report(cate_out, cate_ens_test>=0.5))
cate_p_valid = np.hstack([cate_p_valid, cate_ens_valid[:, None]])
cate_p_test = np.hstack([cate_p_test, cate_ens_test[:, None]])
    
save_d = {'AUC':AUC, 'TP':TP, 'FP':FP, 'REPORT':REPORT, 'cate_test':cate_p_test, 'cate_valid':cate_p_valid}
np.save(eval_dir+save_name, save_d)
print('Save to {}'.format(eval_dir+freq+save_name))

