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
from namelist_mlp import *
#import importlib

save_dir = '/glade/work/ksha/data/Keras/QC_publish/'
# hyper params
x_size, y_size = 8, 8
l_size = 64
# training param
batch_size, m_size = 1, 200
epoch_num = 200
# ther param
labels = ['capa_input', 'cate_out']
flag=[True, False]
freq = 'HIGH_'
key = freq+'QC_'
# train members
for i in range(ens):
    i = int(i)
    name = 'MLP'+str(i)
    train_dir = BATCH_dir+freq+'ENS_CAPA_TRAIN{}*npy'.format(i)
    valid_dir = BATCH_dir+freq+'ENS_CAPA_VALID{}*npy'.format(i)
    steps = len(glob(train_dir))//batch_size
    # functional API
    IN1 = keras.layers.Input(shape=(l_size,))
    X = keras.layers.Dense(128)(IN1)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Activation('tanh')(X)
    OUT = keras.layers.Dense(1, activation=keras.activations.sigmoid)(X)
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
    gen_train = ku.point_point_gen(train_files, batch_size, x_size, y_size, m_size, labels, flag)
    gen_valid = ku.point_point_gen(valid_files, batch_size, x_size, y_size, m_size, labels, flag)
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
    