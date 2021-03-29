import sys
import h5py
import numpy as np
#import tensorflow as tf
from tensorflow import keras
#import tensorflow.keras.backend as K
#
sys.path.insert(0, '/glade/u/home/ksha/ML_repo/utils/')
import keras_utils as ku
from namelist import *
# path
save_dir = '/glade/work/ksha/data/Keras/QC_publish/'
eval_dir = '/glade/work/ksha/data/evaluation/saliency_maps/'
# TRAIN data
print('Import training data')
hdf_io = h5py.File(HOLD_dir+freq+'CAPA_TRAIN_pack.hdf', 'r')
grid_train = hdf_io['capa_input'][...]
cate_train = hdf_io['cate_out'][...]
hdf_io.close()
# clean GPU memory to start my new run
import keras.backend as K
config = K.tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = K.tf.Session(config=config)
K.set_session(sess)
print('Import keras model')
#for n in range(4, ens, 1):
n = 3 # <------- 
# model
key = freq+'QC_CNN'+str(n)
print('\t {}'.format(key))
model = keras.models.load_model(save_dir+key+'.hdf')
# 
grid_train_temp = grid_train[..., 2*n:2*(n+1)] # samples
L_n = len(model.layers[-1].get_weights()[0])
L_e = 100
#
neuron_batch = 216; L_batch = L_n//neuron_batch # 216
for i in range(L_batch):
    print('\tneuron batch {}'.format(i))
    range_top = [i*neuron_batch, (i+1)*neuron_batch]
    ind_neuron, ind_example, gradient = ku.saliency_maps(model, grid_train_temp, range_top=range_top, range_ex=[0, L_e], batch_size=4096, layer_id=[-1, -2])
    #
    temp_name = eval_dir+key+'_saliency_top100_part{}.hdf'.format(i)
    hdf_io = h5py.File(temp_name, 'w')
    hdf_io.create_dataset('ind_neuron', data=ind_neuron)
    hdf_io.create_dataset('ind_example', data=ind_example)
    hdf_io.create_dataset('gradient', data=gradient)
    hdf_io.close()
    #
    print('Save to {}'.format(temp_name))

