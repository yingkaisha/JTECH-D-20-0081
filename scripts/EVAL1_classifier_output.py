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
data_dir0 = HOLD_dir+freq+'CAPA_TRAIN_pack.hdf'
data_dir1 = HOLD_dir+freq+'CAPA_VALID_pack.hdf'
data_dir2 = HOLD_dir+freq+'CAPA_TEST_pack.hdf'
save_name = 'EVAL_QC_members.npy'
# ===== import data ===== #
print('Importing data')
# TRAIN
hdf_io = h5py.File(data_dir0, 'r')
grid_input0 = hdf_io['capa_input'][...]
cate_out0   = hdf_io['cate_out'][...]
hdf_io.close()
print('\tTraining set')
# VALID
hdf_io = h5py.File(data_dir1, 'r')
grid_input1 = hdf_io['capa_input'][...]
cate_out1   = hdf_io['cate_out'][...]
hdf_io.close()
print('\tValidation set')
# TEST
hdf_io = h5py.File(data_dir2, 'r')
grid_input2 = hdf_io['capa_input'][...]
cate_out2   = hdf_io['cate_out'][...]
hdf_io.close()
print('\tTesting set')
# get size of data
L0 = len(cate_out0) # Train
L1 = len(cate_out1) # Valid
L2 = len(cate_out2) # Test
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
cate_p_train = np.empty([L0, ens])
cate_p_valid = np.empty([L1, ens])
cate_p_test  = np.empty([L2, ens])
names = []
REPORT = {}; FP = {}; TP = {}; AUC = {}
for i in range(ens):
    name = 'ENS'+str(i); names.append(name)
    print('\tProcessing {}'.format(name))
    model_temp = model_member[i]
    cate_temp_train = np.squeeze(model_temp.predict(grid_input0[..., 2*i:2*(i+1)])) # :2*(i+1)
    cate_temp_valid = np.squeeze(model_temp.predict(grid_input1[..., 2*i:2*(i+1)]))
    cate_temp_test  = np.squeeze(model_temp.predict(grid_input2[..., 2*i:2*(i+1)]))
    # backup probabilistic output
    cate_p_train[:, i] = cate_temp_train
    cate_p_valid[:, i] = cate_temp_valid
    cate_p_test [:, i] = cate_temp_test
    # report
    REPORT[name] = classification_report(cate_out2, cate_temp_test>=0.5)
    # RUC
    FP[name], TP[name], _ = roc_curve(cate_out2, cate_temp_test)
    AUC[name] = auc(FP[name], TP[name])

# ens model
print('Processing ensembles')
model_ens = keras.models.load_model(save_dir+'QC_ENS.hdf')
cate_ens_train = np.squeeze(model_ens.predict(cate_p_train))
cate_ens_valid = np.squeeze(model_ens.predict(cate_p_valid))
cate_ens_test  = np.squeeze(model_ens.predict(cate_p_test))
# report
REPORT['ENS'] = classification_report(cate_out2, cate_ens_test>=0.5)
FP['ENS'], TP['ENS'], _ = roc_curve(cate_out2, cate_ens_test)
AUC['ENS'] = auc(FP['ENS'], TP['ENS'])

print(classification_report(cate_out2, cate_ens_test>=0.5))

cate_p_train = np.hstack([cate_p_train, cate_ens_train[:, None]])
cate_p_valid = np.hstack([cate_p_valid, cate_ens_valid[:, None]])
cate_p_test = np.hstack([cate_p_test, cate_ens_test[:, None]])
    
save_d = {'AUC':AUC, 'TP':TP, 'FP':FP, 'REPORT':REPORT, 'cate_train':cate_p_train, 'cate_valid':cate_p_valid, 'cate_test':cate_p_test}
np.save(eval_dir+save_name, save_d)
print('Save to {}'.format(eval_dir+freq+save_name))

