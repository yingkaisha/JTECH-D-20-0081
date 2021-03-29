import sys
import h5py
import numpy as np
from glob import glob
import tensorflow as tf
from tensorflow import keras
sys.path.insert(0, '/glade/u/home/ksha/ML_repo/utils/')
import utils
# file path
save_dir = '/glade/u/home/ksha/data/Keras/BACKUP/'
data_dir = '/glade/scratch/ksha/DATA/QC_H_64_64/HOLD/TRAIN_new_pack.hdf' # TEST_pack.hdf5

# load training info
key = 'HydraRes' # <---- change keys to swap model
print('\tEvaluation: {}\n====================================='.format(key))
model = keras.models.load_model(save_dir+'QC_class_'+key+'.hdf')
# hdf access
hdf_io = h5py.File(data_dir, 'r')
grid_input = hdf_io['grid_input'][...]
#stn_input  = hdf_io['stn_input'][...]
cate_out   = hdf_io['cate_out'][...]
#ref        = hdf_io['ref'][...]
#stn_code   = np.array(utils.str_decode(hdf_io['stn_code'][...]))
hdf_io.close()
# no permutation case
cate_p = model.predict([grid_input[..., 0:2], grid_input[..., 2:4], grid_input[..., 4:6]])
cate_p = cate_p[:, 1]
# permutation function
def permute_predict(model, grid_input, R=1):
    '''
    model: keras model
    grid_input: (sample, x, y, channel)
    R: repeats
    '''
    grid_shuffle = np.copy(grid_input) # copy as backup (need big sys memory)
    # allocation
    L = grid_input.shape[0]
    N = grid_input.shape[-1]
    cate_permute = np.empty([L, N, R]) 
    grid_temp = np.empty(grid_input.shape)
    for r in range(R):
        print('---------- round: {} ----------'.format(r))
         # permutation
        ind = utils.shuffle_ind(L)
        grid_shuffle = grid_shuffle[ind, ...]
        # prediction
        for n in range(N):
            ind_flag = np.ones(N).astype(bool); ind_flag[n] = False # permute flag # [True]*n
            grid_temp[..., ind_flag] = grid_input[..., ind_flag] 
            grid_temp[..., np.logical_not(ind_flag)] = grid_shuffle[..., np.logical_not(ind_flag)]
            print('permuting feature: {}'.format(str(n)))
            cate_permute[:, n, r] = model.predict([grid_temp[..., 0:2], grid_temp[..., 2:4], grid_temp[..., 4:6]])[:, 1]
    return cate_permute

cate_permute = permute_predict(model, grid_input, R=50)
save_dir = '/glade/u/home/ksha/data/evaluation/permute_'+key+'.npy'
save_d = {'cate_p':cate_p, 'cate_permute':cate_permute}
np.save(save_dir, save_d)

