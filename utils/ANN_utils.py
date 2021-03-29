import glob
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

def CONV_stack(X, channel, kernel, stack_num):
    for i in range(stack_num):
        X = keras.layers.Conv2D(channel, kernel, padding='same', use_bias=False, kernel_initializer='he_normal')(X)
        X = keras.layers.BatchNormalization(axis=3)(X)
        X = keras.layers.ReLU()(X)
    return X

def NIN(X, channel):
    X = keras.layers.Conv2D(channel, 1, use_bias=False, kernel_initializer='he_normal')(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.ReLU()(X)
    return X

def dense_res_block(X, channel, rate, layer):
    X = CONV_stack(X, channel, kernel=3, stack_num=1);
    for j in range(1, layer):
        X_temp = CONV_stack(X, rate, kernel=3, stack_num=1)
        X = keras.layers.concatenate([X, X_temp]);
    return X;

# ========= FCN-VGG Blocks ======== #
def FCN_left(X, channel, kernel, stack_num, pool_size=2):
    X_pool = keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size))(X)
    X_conv = CONV_stack(X_pool, channel, kernel, stack_num)
    # return pool for fusion
    return X_conv, X_pool

def FCN_right(X, channel_t, channel, kernel, stack_num, pool_size=2):
    X = keras.layers.Conv2DTranspose(channel_t, pool_size, strides=(pool_size, pool_size))(X)
    X = CONV_stack(X, channel, kernel, stack_num)
    return X
# ================================= #
# ========== UNet Blocks ========== #
def UNet_left(X, channel, kernel_size=3, pool_size=2):  
    X = keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size))(X)
    X = CONV_stack(X, channel, kernel_size, stack_num=1)
    return X

def UNet_right(X, X_left, channel, kernel_size=3, pool_size=2):
    X = keras.layers.UpSampling2D(size=(pool_size, pool_size))(X)
    X = CONV_stack(X, channel, kernel_size, stack_num=1)    
    H = keras.layers.concatenate([X_left, X], axis=3)
    H = CONV_stack(H, channel, kernel_size, stack_num=1)
    return H

def UNet_right_nin(X, X_left, channel, kernel_size=3, pool_size=2):
    X = keras.layers.UpSampling2D(size=(pool_size, pool_size))(X)
    X = CONV_stack(X, channel, kernel_size, stack_num=1)
    
    X = keras.layers.Conv2D(channel, 1, use_bias=False, kernel_initializer='he_normal')(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.ReLU()(X)
    
    H = keras.layers.concatenate([X_left, X], axis=3)
    H = CONV_stack(H, channel, kernel_size, stack_num=1)
    return H

def UNET(layer_N, c_size, in_stack_num=2):
    IN = keras.layers.Input((None, None, c_size))
    # left blocks
    X1 = CONV_stack(IN, layer_N[0], kernel=3, stack_num=in_stack_num)
    X2 = UNet_left(X1, layer_N[1])
    X3 = UNet_left(X2, layer_N[2])
    # bottom
    X4 = UNet_left(X3, layer_N[3])
    # right blocks
    X5 = UNet_right(X4, X3, layer_N[2])
    X6 = UNet_right(X5, X2, layer_N[1])
    X7 = UNet_right(X6, X1, layer_N[0])
    # output
    X7 = CONV_stack(X7, 2, kernel=3, stack_num=1)
    OUT = keras.layers.Conv2D(1, 1, activation=keras.activations.linear)(X7)
    model = keras.models.Model(inputs=[IN], outputs=OUT)
    return model

def UNET_AE(layer_N, c_size, in_stack_num=2, dropout=False, rate=0.1):
    IN = keras.layers.Input((None, None, c_size))
    # left blocks
    #X0 = keras.layers.GaussianNoise(0.01)(IN)
    X1 = CONV_stack(IN, layer_N[0], kernel=3, stack_num=in_stack_num)
    X2 = UNet_left(X1, layer_N[1])
    X3 = UNet_left(X2, layer_N[2])
    # bottom
    X4 = UNet_left(X3, layer_N[3])
    #
    if dropout:
        X4 = keras.layers.SpatialDropout2D(rate)(X4)
    # right blocks
    X5 = UNet_right(X4, X3, layer_N[2])
    if dropout:
        X5 = keras.layers.SpatialDropout2D(0.5*rate)(X5)
    X6 = UNet_right(X5, X2, layer_N[1])
    if dropout:
        X6 = keras.layers.SpatialDropout2D(0.5*rate)(X6)
    X7 = UNet_right(X6, X1, layer_N[0])
    # output
    X8 = CONV_stack(X7, 2, kernel=3, stack_num=1)
    X9 = CONV_stack(X7, 2, kernel=3, stack_num=1)
    OUT1 = keras.layers.Conv2D(1, 1, activation=keras.activations.linear, name='HR_temp')(X8)
    OUT2 = keras.layers.Conv2D(1, 1, activation=keras.activations.linear, name='HR_elev')(X9)
    model = keras.models.Model(inputs=[IN], outputs=[OUT1, OUT2])
    return model

def UNET_AE_PCT(layer_N, c_size, in_stack_num=2, dropout_rate=0):
    IN = keras.layers.Input((None, None, c_size))
    # left blocks
    #X0 = keras.layers.GaussianNoise(0.01)(IN)
    X1 = CONV_stack(IN, layer_N[0], kernel=3, stack_num=in_stack_num)
    X2 = UNet_left(X1, layer_N[1])
    X3 = UNet_left(X2, layer_N[2])
    # bottom
    X4 = UNet_left(X3, layer_N[3])
    #
    if dropout_rate > 0:
        X4 = keras.layers.SpatialDropout2D(dropout_rate)(X4)
    # right blocks
    X5 = UNet_right(X4, X3, layer_N[2])
    if dropout_rate > 0:
        X5 = keras.layers.SpatialDropout2D(0.5*dropout_rate)(X5)
    X6 = UNet_right(X5, X2, layer_N[1])
    if dropout_rate > 0:
        X6 = keras.layers.SpatialDropout2D(0.5*dropout_rate)(X6)
    X7 = UNet_right(X6, X1, layer_N[0])
    # output
    X8 = CONV_stack(X7, 2, kernel=3, stack_num=1)
    X9 = CONV_stack(X7, 2, kernel=3, stack_num=1)
    OUT1 = keras.layers.Conv2D(1, 1, activation=keras.activations.linear, name='HR_pct')(X8)
    OUT2 = keras.layers.Conv2D(1, 1, activation=keras.activations.linear, name='HR_clim')(X9)
    model = keras.models.Model(inputs=[IN], outputs=[OUT1, OUT2])
    return model

# ================================= #
# ======= UNet Plus Blocks ======== #
def down_block(X, channel, kernel=3):
    # down-sampling
    X_pool = keras.layers.MaxPooling2D(pool_size=(2, 2))(X)
    # conv+BN blocks
    X_conv = CONV_stack(X_pool, channel, kernel, stack_num=2)
    return X_conv
def up_block(X_conv, X_list, channel, kernel=3):
    # up-sampling + conv + concat
    X_unpool = keras.layers.UpSampling2D(size=(2, 2))(X_conv)
    X_unpool = keras.layers.concatenate([X_unpool]+X_list, axis=3)
    # conv+BN blocks 
    X_conv = CONV_stack(X_unpool, channel, kernel, stack_num=2)
    return X_conv
def input_conv(X, channel, kernel=3):
    X = CONV_stack(X, channel, kernel, stack_num=2)
    return X
# ================================= #
# ========== Dilated Conv ========= #
def dilated_conv(X, channel, dil_rate, kernel_size):
    X = keras.layers.Conv2D(channel, kernel_size, dilation_rate=dil_rate[0], padding='same', use_bias=False, kernel_initializer='he_normal')(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.PReLU(shared_axes=[1, 2])(X)
    X = keras.layers.Conv2D(channel, kernel_size, dilation_rate=dil_rate[1], padding='same', use_bias=False, kernel_initializer='he_normal')(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.PReLU(shared_axes=[1, 2])(X)
    X = keras.layers.Conv2D(channel, kernel_size, dilation_rate=dil_rate[2], padding='same', use_bias=False, kernel_initializer='he_normal')(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.PReLU(shared_axes=[1, 2])(X)
    return X
# ================================= #
# ========= ResNet Blocks ========= #
def identity_block(X, f, channels):
    '''
    Conv --- (indentity) --- Conv
    '''
    F1, F2, F3 = channels
    X_shortcut = X
    # main path
    # block 1
    X = keras.layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(1,1), padding ='valid')(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.Activation('relu')(X)
    # block 2
    X = keras.layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same')(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.PReLU(shared_axes=[1, 2])(X)
    X = keras.layers.SpatialDropout2D(0.1)(X)
    # block 3
    X = keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    # skip connection
    X = keras.layers.Add()([X, X_shortcut])
    X = keras.layers.Activation('relu')(X)
    return X

def RES_BRANCH(X1):

    X1 = keras.layers.Conv2D(16, kernel_size=(5, 5), strides=(2, 2), padding='valid', use_bias=False)(X1)
    X1 = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(X1)
    X1 = keras.layers.PReLU(shared_axes=[1, 2])(X1)
    X1 = keras.layers.SpatialDropout2D(0.1)(X1)
    
    X1 = keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(2, 2), padding='valid', use_bias=False)(X1)
    X1 = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(X1)
    X1 = keras.layers.PReLU(shared_axes=[1, 2])(X1)
    X1 = identity_block(X1, 3, [32, 32, 32])
    X1 = keras.layers.SpatialDropout2D(0.1)(X1)

    X1 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=1, padding='valid')(X1)
    
    X1 = keras.layers.Conv2D(48, kernel_size=(3, 3), strides=(1, 1), padding='valid', use_bias=False)(X1)
    X1 = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(X1)
    X1 = keras.layers.PReLU(shared_axes=[1, 2])(X1)
    X1 = identity_block(X1, 3, [48, 48, 48])
    X1 = keras.layers.SpatialDropout2D(0.1)(X1)
    
    X1 = keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid', use_bias=False)(X1)
    X1 = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(X1)
    X1 = keras.layers.PReLU(shared_axes=[1, 2])(X1)
    X1 = identity_block(X1, 3, [64, 64, 64])
    X1 = keras.layers.SpatialDropout2D(0.15)(X1)
    
    X1 = keras.layers.Conv2D(80, kernel_size=(3, 3), strides=(1, 1), padding='valid', use_bias=False)(X1)
    X1 = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(X1)
    X1 = keras.layers.PReLU(shared_axes=[1, 2])(X1)
    X1 = identity_block(X1, 3, [80, 80, 80])
    X1 = keras.layers.SpatialDropout2D(0.2)(X1)
    
    X1 = keras.layers.Conv2D(96, kernel_size=(3, 3), strides=(1, 1), padding='valid', use_bias=False)(X1)
    X1 = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(X1)
    X1 = keras.layers.PReLU(shared_axes=[1, 2])(X1)
    X1 = identity_block(X1, 3, [96, 96, 96])
    X1 = keras.layers.SpatialDropout2D(0.2)(X1)
    
    return X1
# ================================= #