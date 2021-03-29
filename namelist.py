
# ========== namelist (CNN) ========== #

# Param.
ens = 5 # number of ensemble classifiers
thres = 0.5 # probability to binary flag thres
half_edge = [32, 32] # half-ed feature map size
padx=[1, 0]; pady=[1, 0] # padded station grids
use_GPM = False # GPM as input (not used)
use_AUG = False # Data augmentation (not used)
freq = 'HIGH_' # "HIGH" for 30 min; "LOW" for daily
del_old = True # Remove old results before generating new results
keep_ocean = False # preserve ocean grid point values

# Backup data (raw data) dir
BACKUP_dir = '/glade/scratch/ksha/BACKUP/'
BCH_file = BACKUP_dir+'BCH_combine.hdf'
BCH_high_file = BACKUP_dir+'BCH_half_hourly.hdf' # <------
BCH_low_file = BACKUP_dir+'BCH_half_hourly.hdf'
BCH_meta_file = BACKUP_dir+'BCH_metadata.hdf'
GRID_INPUT_file = BACKUP_dir+'GRID_INPUT_FULL.hdf'
CaPA_datetime_file = BACKUP_dir+'CaPA_compressed_datetime.npy'

# Processed data dir
eval_dir = '/glade/work/ksha/data/evaluation/'
fig_dir = '/glade/u/home/ksha/figures/'
QC_dir = '/glade/scratch/ksha/DATA/'
INPUT_dir = QC_dir+'QC_{}_{}/'.format(half_edge[0]*2, half_edge[1]*2) # <----- or 46_92
BATCH_dir = INPUT_dir + 'BATCH_categorical/'
HOLD_dir = INPUT_dir + 'HOLD/'

# Station withholding dir
INPUT_stn_dir = '/glade/scratch/ksha/DATA/QC_stn/'
BATCH_stn_dir = INPUT_stn_dir + 'BATCH_categorical/'
BACK_stn_dir  = INPUT_stn_dir + 'BACKUP/'
HOLD_stn_dir  = INPUT_stn_dir + 'HOLD/'

# figure dir
fig_dir = '/glade/u/home/ksha/figures/'