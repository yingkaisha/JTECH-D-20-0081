import glob
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

class grid_point_gen(keras.utils.Sequence):
    def __init__(self, file_names, batch_size, x_size, y_size, c_size, m_size, labels, flag=[True, True]):
        '''
        Input
        ----------
        batch_size: how many files you want to access in one batch
        x_size, y_size: number of feature map pixels for 2d or 3d input
        c_size: number of feature map channels
        l_size: number of features for 1d input
        m_size: if one file have multiple samples, default is one-file-one-samle
        file_dir: the path that you have all files, must in reg. expr. (eg., *.jpg)
        
        Internal vars
        ---------------
        file_names: a list of filenames, returned by glob.glob()
        file_len: len(file_names)
        inds: indexes for the file sequence
        '''
        self.batch_size = batch_size # typically 4, 8, 16 etc.
        self.x_size = x_size
        self.y_size = y_size
        self.c_size = c_size
        self.m_size = m_size
        self.labels = labels
        self.flag = flag
        self.file_names = file_names
        self.file_len = len(self.file_names)
        self.inds = np.arange(self.file_len).astype(int)
    def __len__(self):
        return int(np.floor(self.file_len/self.batch_size))
    
    def __getitem__(self, index):
        random.shuffle(self.file_names)
        temp_file_names = self.file_names[index*self.batch_size:(index+1)*self.batch_size]
        return self.__readfile(temp_file_names, self.x_size, self.y_size, self.c_size, self.m_size, self.labels, self.flag)
    
    def __readfile(self, names, x_size, y_size, c_size, m_size, labels, flag):
        N = len(names)
        #print(names)
        # for gridded features
        GRID_IN = np.empty([N*m_size, x_size, y_size, c_size]) # gridded data
        Y_LABEL = np.empty([N*m_size]) # <--- binary classes
        for i, name in enumerate(names):
            data_temp = np.load(name, allow_pickle=True)          
            GRID_IN[i*m_size:(i+1)*m_size, ...] = data_temp[()][labels[0]][..., flag]
            Y_LABEL[i*m_size:(i+1)*m_size] = data_temp[()][labels[1]]
        return [GRID_IN], Y_LABEL
    
class grid_grid_gen(keras.utils.Sequence):
    def __init__(self, file_names, batch_size, c_size, m_size, labels, flag=[True, True]):
        '''
        Input
        ----------
        batch_size: how many files you want to access in one batch
        c_size: number of feature map channels
        l_size: number of features for 1d input
        m_size: if one file have multiple samples, default is one-file-one-samle
        file_dir: the path that you have all files, must in reg. expr. (eg., *.jpg)
        
        Internal vars
        ---------------
        file_names: a list of filenames, returned by glob.glob()
        file_len: len(file_names)
        inds: indexes for the file sequence
        '''
        self.batch_size = batch_size # typically 4, 8, 16 etc.
        self.c_size = c_size
        self.m_size = m_size
        self.labels = labels
        self.flag = flag
        self.file_names = file_names
        self.file_len = len(self.file_names)
        self.inds = np.arange(self.file_len).astype(int)
    def __len__(self):
        return int(np.floor(self.file_len/self.batch_size))
    
    def __getitem__(self, index):
        random.shuffle(self.file_names)
        temp_file_names = self.file_names[index*self.batch_size:(index+1)*self.batch_size]
        return self.__readfile(temp_file_names, self.c_size, self.m_size, self.labels, self.flag)
    
    def __readfile(self, names, c_size, m_size, labels, flag):
        N = len(names)
        l = 0.5
        data_temp = np.load(names[0], allow_pickle=True)
        GRID_IN = data_temp[()][labels[0]][...]
        Y_LABEL = data_temp[()][labels[1]][..., None]
        return [GRID_IN], [Y_LABEL, l*GRID_IN[..., 1][..., None]-(1-l)*GRID_IN[..., 2][..., None]]

class point_point_gen(keras.utils.Sequence):
    def __init__(self, file_names, batch_size, x_size, y_size, m_size, labels, flag=[True, False]):
        '''
        Input
        ----------
        batch_size: how many files you want to access in one batch
        x_size, y_size: number of feature map pixels for 2d or 3d input
        c_size: number of feature map channels
        l_size: number of features for 1d input
        m_size: if one file have multiple samples, default is one-file-one-samle
        file_dir: the path that you have all files, must in reg. expr. (eg., *.jpg)
        
        Internal vars
        ---------------
        file_names: a list of filenames, returned by glob.glob()
        file_len: len(file_names)
        inds: indexes for the file sequence
        '''
        self.batch_size = batch_size # typically 4, 8, 16 etc.
        self.x_size = x_size
        self.y_size = y_size
        self.m_size = m_size
        self.labels = labels
        self.flag = flag
        self.file_names = file_names
        self.file_len = len(self.file_names)
        self.inds = np.arange(self.file_len).astype(int)
    def __len__(self):
        return int(np.floor(self.file_len/self.batch_size))
    
    def __getitem__(self, index):
        random.shuffle(self.file_names)
        temp_file_names = self.file_names[index*self.batch_size:(index+1)*self.batch_size]
        return self.__readfile(temp_file_names, self.x_size, self.y_size, self.m_size, self.labels, self.flag)
    
    def __readfile(self, names, x_size, y_size, m_size, labels, flag):
        N = len(names)
        # for gridded features
        X_IN = np.empty([N*m_size, x_size*y_size]) # features
        Y_LABEL = np.empty([N*m_size]) # <--- binary classes
        for i, name in enumerate(names):
            data_temp = np.load(name, allow_pickle=True)
            X_IN[i*m_size:(i+1)*m_size, :] = data_temp[()][labels[0]][..., flag].reshape((N*m_size, x_size*y_size))
            Y_LABEL[i*m_size:(i+1)*m_size] = data_temp[()][labels[1]]
        return [X_IN], Y_LABEL

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

def identity_block2(X, f, channels):
    '''
    Conv --- (indentity) --- Conv
    '''
    F1, F2, F3 = channels
    X_shortcut = X
    # main path
    # block 1
    X = keras.layers.SeparableConv2D(filters=F1, kernel_size=(1, 1), strides=(1,1), padding ='valid')(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.Activation('relu')(X)
    # block 2
    X = keras.layers.SeparableConv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same')(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.PReLU(shared_axes=[1, 2])(X)
    X = keras.layers.SpatialDropout2D(0.1)(X)
    # block 3
    X = keras.layers.SeparableConv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(X)
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

def CONV_BRANCH(X1):
    X1 = keras.layers.Conv2D(16, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)(X1)
    X1 = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(X1)
    X1 = keras.layers.PReLU(shared_axes=[1, 2])(X1)
    X1 = keras.layers.SpatialDropout2D(0.1)(X1)
    
    X1 = keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False)(X1)
    X1 = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(X1)
    X1 = keras.layers.PReLU(shared_axes=[1, 2])(X1)
    X1 = keras.layers.SpatialDropout2D(0.1)(X1)

    X1 = keras.layers.MaxPooling2D(pool_size=(5, 5), strides=2, padding='valid')(X1)
    
    X1 = keras.layers.Conv2D(48, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X1)
    X1 = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(X1)
    X1 = keras.layers.PReLU(shared_axes=[1, 2])(X1)
    X1 = keras.layers.SpatialDropout2D(0.1)(X1)
    
    X1 = keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X1)
    X1 = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(X1)
    X1 = keras.layers.PReLU(shared_axes=[1, 2])(X1)
    X1 = keras.layers.SpatialDropout2D(0.15)(X1)
    
    X1 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=1, padding='valid')(X1)
    
    X1 = keras.layers.Conv2D(80, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X1)
    X1 = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(X1)
    X1 = keras.layers.PReLU(shared_axes=[1, 2])(X1)
    X1 = keras.layers.SpatialDropout2D(0.2)(X1)
    
    X1 = keras.layers.Conv2D(96, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(X1)
    X1 = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(X1)
    X1 = keras.layers.PReLU(shared_axes=[1, 2])(X1)
    X1 = keras.layers.SpatialDropout2D(0.2)(X1)   
    return X1

def saliency_maps(temp_model, grid_train_temp, range_top, range_ex, batch_size, layer_id=[-1, -2]):
    # allocation
    top_examples = np.empty((range_top[1]-range_top[0], range_ex[1]-range_ex[0]), dtype=int) # indices of (sorted neurons, sorted samples)
    # array that contains gradients, size: top neuron, top exp, size of gradients on the input end
    top_gradients = np.empty((range_top[1]-range_top[0], range_ex[1]-range_ex[0],)+grid_train_temp.shape[1:]) 
    batch_i = list(range(0, grid_train_temp.shape[0], batch_size)) + [grid_train_temp.shape[0]] # batch samples
    # get weight from the output end
    weights = temp_model.layers[layer_id[0]].get_weights()[0].ravel()
    top_neurons = weights.argsort()[::-1][range_top[0]:range_top[1]] # most activated neurals
    # loop over neurons
    print('Sorted order | neuron index | neuron weights')
    for n, neuron in enumerate(top_neurons):
        print(' {} |  {}  | {}'.format(n, neuron, weights[neuron])) # order, index of neuron, weights
        # define the activation of neurons as a backend function (for sorting the top examples)
        act_func = K.function([temp_model.input, K.learning_phase()], [temp_model.layers[layer_id[1]].output[:, neuron]])
        # loss = a monotonic function that takes neurons' final output 
        loss = (temp_model.layers[layer_id[1]].output[:, neuron]-4)**2
        # calculate gradients from loss (output end) to input end
        grads = K.gradients(loss, temp_model.input)[0]
        # standardizing gradients
        grads /= K.maximum(K.std(grads), K.epsilon())
        # define gradients calculation as a backend function
        grad_func = K.function([temp_model.input, K.learning_phase()], [grads])
        # allocation activation array
        act_values = np.zeros(grid_train_temp.shape[0])
        # loop over samples by batch
        for b in range(len(batch_i)-1):
            act_values[batch_i[b]:batch_i[b+1]] = act_func([grid_train_temp[batch_i[b]:batch_i[b+1]], 0])[0]
        # sort activation values and reteave examples index / gradients
        top_examples[n] = act_values.argsort()[::-1][range_ex[0]:range_ex[1]]
        top_gradients[n, ...] = -grad_func([grid_train_temp[top_examples[n]], 0])[0]  
    return top_neurons, top_examples, top_gradients

