import sys
import h5py
import numpy as np
import pandas as pd
from glob import glob
import tensorflow as tf
from tensorflow import keras
sys.path.insert(0, '/glade/u/home/ksha/ML_repo/utils/')
import data_utils as du
import keras_utils as ku
from namelist import *
#import importlib

save_dir = '/glade/work/ksha/data/Keras/QC_publish/'
# hyper params
x_size, y_size, c_size  = 64, 64, 2
l_size = 1
# training param
batch_size, m_size = 1, 200
epoch_num = 200
# ther param
labels = ['capa_input', 'cate_out']
flag=[True, True]
freq = 'HIGH_'
key = freq+'QC_'
# train members
model_member = []
for i in range(ens):
    i = int(i)
    name = 'CNN'+str(i)
    train_dir = BATCH_dir+freq+'ENS_CAPA_TRAIN{}*npy'.format(i)
    valid_dir = BATCH_dir+freq+'ENS_CAPA_VALID{}*npy'.format(i)
    steps = len(glob(train_dir))//batch_size
    # functional API
    IN1 = keras.layers.Input(shape=(x_size, y_size, c_size))
    H1 = ku.RES_BRANCH(IN1)
    FC1 = keras.layers.Flatten()(H1)
    OUT = keras.layers.Dense(1, activation=keras.activations.sigmoid)(FC1)
    # compile
    model = keras.models.Model(inputs=[IN1], outputs=OUT)
    opt = keras.optimizers.Adam(lr=0.0004, decay=0.0004)
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=opt, metrics=['accuracy'])
    # generator
    train_files = glob(train_dir)
    valid_files = glob(valid_dir)
    batch_size, m_size = 1, 200
    steps = len(train_files)//batch_size
    labels = ['capa_input', 'cate_out']
    gen_train = ku.grid_point_gen(train_files, batch_size, x_size, y_size, c_size, m_size, labels, flag)
    gen_valid = ku.grid_point_gen(valid_files, batch_size, x_size, y_size, c_size, m_size, labels, flag)
    # early stopping callbacks
    print('model save to {}'.format(save_dir+key+name+'.hdf'))
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3),
                 keras.callbacks.ModelCheckpoint(filepath=save_dir+key+name+'.hdf',
                                                 monitor='val_loss', save_best_only=True, verbose=1)]
    # train
    records = model.fit_generator(generator=gen_train, validation_data=gen_valid, callbacks=callbacks,
                                  steps_per_epoch=steps, epochs=epoch_num, verbose=1, shuffle=True,
                                  use_multiprocessing=True, max_queue_size=12, workers=6)
    history = records.history
    np.save(save_dir+key+name+'_records.hdf', records.history) # save history
    model_member.append(model)
    
# Train ens part
data_dir0 = HOLD_dir+freq+'CAPA_TRAIN_pack.hdf'
data_dir1 = HOLD_dir+freq+'CAPA_VALID_pack.hdf'
# TRAIN
hdf_io = h5py.File(data_dir0, 'r')
grid_input0 = hdf_io['capa_input'][...]
cate_out0   = hdf_io['cate_out'][...]
hdf_io.close()
# VALID
hdf_io = h5py.File(data_dir1, 'r')
grid_input1 = hdf_io['capa_input'][...]
cate_out1   = hdf_io['cate_out'][...]
hdf_io.close()
#
model = ku.ensemble_train(model_member, (grid_input0, cate_out0), (grid_input1, cate_out1), node_factor=2, out_dir=save_dir)
