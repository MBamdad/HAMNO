import torch
import numpy as np
from pathlib import Path
import random
import torch as _torch

# ——— Core ———
SEED = 42

DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# ——— Problem selector ———
# One of: 'AC3D', 'CH3D', 'SH3D'
PROBLEM = 'AC3D'

# ——— Model ———
#"UNO3d, UNet3d, UFNO3d, 'DeepONet3D_Robust', or 'FNO4d', 'FFNO4d', 'HAMNO3d'
MODEL = 'HAMNO3d'


## DeepONet (DeepONet3D_Robust) hyperparameters

DEEPONET_WIDTH = 64
DEEPONET_BRANCH_DEPTH = 4
DEEPONET_TRUNK_HIDDEN = 64
DEEPONET_TRUNK_DEPTH = 4
DEEPONET_HEAD_WIDTH = 64
DEEPONET_COORDS_IN_BRANCH = True # True Test 3

PROBLEM_SPECS = {
    'AC3D': dict(
        #GRID_RESOLUTION=32,
        GRID_RESOLUTION=32,
        L_DOMAIN=2.0,
        DT=1e-4,
        TOTAL_TIME_STEPS=100,
        EPSILON_PARAM=0.1,
        MAT_DATA_PATH="/scratch/noqu8762/PENCO4NonPeriodicBCs/NeumanBCs_DCM/Data/AC3D_DCM32_600_grf3d.mat",
        #MAT_DATA_PATH="/scratch/noqu8762/PENCO4NonPeriodicBCs/data/AC3D_DCM32_250_grf3d.mat",
        CH_LINEAR_COEF=3.0,
    ),
    'CH3D': dict(
        GRID_RESOLUTION=32,
        L_DOMAIN=2.0,
        DT= 1e-3,
        TOTAL_TIME_STEPS=100,
        EPSILON_PARAM=0.05,
        MAT_DATA_PATH='/scratch/noqu8762/PENCO4NonPeriodicBCs/NeumanBCs_DCM/Data/CH3D_DCM_32_600_grf3d.mat',
    ),
    'SH3D': dict(
        GRID_RESOLUTION=32,   # Nx = Ny = Nz
        L_DOMAIN=15.0,
        DT = 5e-2,              # 0.05
        TOTAL_TIME_STEPS=100,
        EPSILON_PARAM=0.15,   # appears as (1 - eps) in the SH linear term
        MAT_DATA_PATH = "/scratch/noqu8762/PENCO4NonPeriodicBCs/data/SH3D_NonPeriodic_DCM_32_600_grf3d.mat", # non periodic BC
    ),
}

def seed_everything(seed: int = SEED):
    print(f"[Seed] Using seed={seed}")
    random.seed(seed)
    np.random.seed(seed)
    _torch.manual_seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(seed)

    # Deterministic CuDNN for reproducibility
    _torch.backends.cudnn.deterministic = True
    _torch.backends.cudnn.benchmark = False

# Load active problem spec
_spec = PROBLEM_SPECS[PROBLEM]

# For CH3D only: u_t = ∇²(u^3 - alpha * u) - ε^2 ∇^4 u
CH_LINEAR_COEF = PROBLEM_SPECS.get('CH3D', {}).get('CH_LINEAR_COEF', 3.0)

# ——— Geometry & Time (per PROBLEM) ———
GRID_RESOLUTION = _spec['GRID_RESOLUTION']
L_DOMAIN = _spec['L_DOMAIN']
#DX = L_DOMAIN / GRID_RESOLUTION
#DX = L_DOMAIN / (GRID_RESOLUTION - 1)
DX = L_DOMAIN / GRID_RESOLUTION

DT = _spec['DT']
TOTAL_TIME_STEPS = _spec['TOTAL_TIME_STEPS']
TIME_END = DT * TOTAL_TIME_STEPS
SAVED_STEPS = TOTAL_TIME_STEPS + 1

# ——— Physics ———
EPSILON_PARAM = _spec['EPSILON_PARAM']
EPS2 = EPSILON_PARAM ** 2
PHYS_MAX_SCALE = 2.0
# ——— Data ———
MAT_DATA_PATH = _spec['MAT_DATA_PATH']


SCALE_STEPS_WITH_NTRAIN = False # True # False # True   # set True to mimic "beam behavior"
use_lbfgs = False # True
N_TRAIN_REF = 50                 # reference N_TRAIN that matches your current STEPS_PER_EPOCH


if PROBLEM == 'SH3D':
    STEPS_PER_EPOCH = 25 # 30 # 20 # 5  # SH3D
elif PROBLEM == 'CH3D':
    STEPS_PER_EPOCH = 25 # 10 # 50 # 30 # 20 # 40 # CH ok,  # 25 # 40 # 30 # 80  # CH3D
elif PROBLEM == 'AC3D':
    STEPS_PER_EPOCH = 10  # AC3D

else:
    print('Enter the right PROBLEM !')



PURE_PHYSICS_USE_ALL = True    # <- when PDE_WEIGHT==1.0, ignore N_TRAIN for train


if MODEL in ('FFNO4d', 'MHNO_FFNO'):
    MODES = 12 # 10 # 12  # Or whatever value was used during training
    if PROBLEM == "SH3D":
        WIDTH = 24
    else:
        WIDTH = 12 # 12  #  12 # This is the most likely one to change
elif MODEL in ('HAMNO3d'):
    MODES = 8 # 6 bad# 8 good # 16 # 12  # --> AC # Or whatever value was used during training
    WIDTH = 12 # 16 # 16 # 12  # --> AC # This is the most likely one to change
else:
    MODES = 8  #   # Or whatever value was used during training
    WIDTH = 12 # 16 # 10  # 10 # 12 # This is the most likely one to change

if MODEL in ('FFNO4d', 'MHNO_FNO'):
    N_LAYERS =  4 # 2
else:
    N_LAYERS = 2

if PROBLEM == 'MBE3D':
    WIDTH_Q = 11
    WIDTH_H = 11
else:
    WIDTH_Q = 12 # 12 # 16 # 10
    WIDTH_H = 10 # 10

if PROBLEM == 'AC3D':
    Expansion = 2
elif PROBLEM == 'CH3D':
    Expansion = 2 # 6 is bad for HMNO
elif PROBLEM == 'SH3D':
    if MODEL == "FFNO4d":
        Expansion = 6  #
    else:
        Expansion = 2 # HAMNO
else:
    Expansion =  2

# ——— Training ———
if  PROBLEM == 'CH3D': # old
    EPOCHS = 100 # 30 # 150 # 150 # 70 # 200 # 100 # 100 # 70
elif PROBLEM == 'SH3D':
    EPOCHS = 170
else:
    EPOCHS =  50

BATCH_SIZE = 8
if PROBLEM == 'MBE3D' or PROBLEM == 'CH3D':
    LEARNING_RATE =  5e-4 # 1e-4 #  5e-4 # 5e-4
elif MODEL == 'HAMNO3d' and PROBLEM == 'AC3D':
    LEARNING_RATE = 1e-2 #  1e-2 for manuscript
else:
    LEARNING_RATE = 1e-3

T_IN_CHANNELS = 5 # number of past frames the model sees
T_OUT = 1



if PROBLEM == 'CH3D':
    WEIGHT_DECAY =   5e-6 # 5e-4 # 5e-6 # 1e-4 # 5e-6 # 5e-6
elif MODEL == 'HAMNO3d' and PROBLEM == 'AC3D':
    WEIGHT_DECAY = 1e-3 # 5e-3 #   1e-3 --> for manuscript
elif MODEL == 'HAMNO3d' and PROBLEM == 'SH3D':
    WEIGHT_DECAY = 5e-6 #
else:
    WEIGHT_DECAY =   5e-6 # 5e-6 # 1e-5 #

PDE_WEIGHT = 1.0
N_TRAIN = 50 # 50 #  100 # 200

N_TEST_FIXED = 50 #50 AC, 100 # 100 # 100             # <- constant, test set size is fixed now
TEST_MODE = 'manual'   # or 'manual'
TEST_PICK = 1       # 0 spherical   # only used if TEST_MODE == 'manual'
N_TEST = 50 # max(1, N_TRAIN // 4)


# ——— Eval/plots ———
TRAIN_TMAX = 50 # 100 # Training capped at {favorite frame: 50}, rollout tested through 100.
EVAL_TIME_FRAMES = [0, 50, 100]
#EVAL_TIME_FRAMES = [25, 50, 75, 100]
#EVAL_TIME_FRAMES = [20, 40, 60, 80, 100]