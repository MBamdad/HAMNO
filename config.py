import torch
import numpy as np
from pathlib import Path
import random
import torch as _torch

# ——— Core ———
SEED = 42

DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# ——— Problem selector ———
# One of: 'AC3D', 'CH3D', 'SH3D', 'MBE3D', 'PFC3D'
PROBLEM = 'AC3D'   # <- set here when you want Swift–Hohenberg
# ——— Model ———
#"DeepONet3D_TNOPhysics", UNO3d, UNet3d, UFNO3d, 'DeepONet3D_Robust', , 'TNO3d' or 'FNO4d', 'FFNO4d', 'MHNO_FFNO'
# 'FNO_PMNO'--> fails for CH;
MODEL = 'HAMNO3d'
BC_TYPE = "neumann"   # "periodic" or "neumann" PhysicsResidualTNO3d_AC, 'TNO3d_PureRobust', 'MambaNO3d', 'HMNO3dPlus','HAMNO3d'
Training_Type = "PurePhysics" # "PurePhysics" , "Teacher_Based" ---> PENCO


## DeepONet hyperparameters

DEEPONET_WIDTH = 64
DEEPONET_BRANCH_DEPTH = 4
DEEPONET_TRUNK_HIDDEN = 64
DEEPONET_TRUNK_DEPTH = 4
DEEPONET_HEAD_WIDTH = 64
DEEPONET_COORDS_IN_BRANCH = True # True Test 3

PROBLEM_SPECS = {
    'AC3D': dict(
        #GRID_RESOLUTION=32,
        GRID_RESOLUTION=128,
        L_DOMAIN=2.0,
        DT=1e-4,
        TOTAL_TIME_STEPS=100,
        EPSILON_PARAM=0.1,
        #MAT_DATA_PATH="/scratch/noqu8762/PENCO4NonPeriodicBCs/NeumanBCs_DCM/Data/AC3D_DCM32_600_grf3d.mat",
        MAT_DATA_PATH="/scratch/noqu8762/PENCO4NonPeriodicBCs/data/AC3D_DCM128_250_grf3d.mat",
        CH_LINEAR_COEF=3.0,
    ),
    'CH3D': dict(
        GRID_RESOLUTION=32,
        L_DOMAIN=2.0,
        DT= 1e-3, #5e-4, #5e-4, # 5e-3,
        TOTAL_TIME_STEPS=100,
        EPSILON_PARAM=0.05,
        #MAT_DATA_PATH = "/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/data/CH3D_500_Nt_101_Nx_32.mat", # correct, dt= 0.005
        MAT_DATA_PATH='/scratch/noqu8762/PENCO4NonPeriodicBCs/NeumanBCs_DCM/Data/CH3D_DCM_32_600_grf3d.mat', # dt = 0.001
    ),
    'SH3D': dict(
        GRID_RESOLUTION=32,   # Nx = Ny = Nz
        L_DOMAIN=15.0,
        DT = 5e-2,              # 0.05
        TOTAL_TIME_STEPS=100,
        EPSILON_PARAM=0.15,   # appears as (1 - eps) in the SH linear term
        #MAT_DATA_PATH="/scratch/noqu8762/PENCOO/data/SH3D_grf3d_ff_250_Nt_101_Nx_32.mat", # periodic BC
        #MAT_DATA_PATH = "/scratch/noqu8762/PENCO4NonPeriodicBCs/data/SH3D_NonPeriodic_DCM_32_250_grf3d.mat", # non periodic BC
        MAT_DATA_PATH = "/scratch/noqu8762/PENCO4NonPeriodicBCs/data/SH3D_NonPeriodic_DCM_32_600_grf3d.mat", # non periodic BC
    ),

    'MBE3D': dict(
        GRID_RESOLUTION=32,
        L_DOMAIN=2*np.pi,
        DT=5e-3,
        TOTAL_TIME_STEPS=100,
        EPSILON_PARAM=0.1,
        MAT_DATA_PATH = '/scratch/noqu8762/PENCOO/data/MBE3D_Augmented_250_Nt_101_Nx_32.mat', # dt=0.005
        #MAT_DATA_PATH = '/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/data/MBE3D_Augmented_250_Nt_101_Nx_32.mat', # dt=0.005
    ),
    'PFC3D': dict(
        GRID_RESOLUTION=32,
        L_DOMAIN=10*np.pi,
        DT=1e-2, # 1e-2, old and valid
        TOTAL_TIME_STEPS=100,
        EPSILON_PARAM=0.5,
        MAT_DATA_PATH="/scratch/noqu8762/PENCOO/data/PFC3D_Augmented_250_Nt_101_Nx_32.mat", # dt=0.01
        #MAT_DATA_PATH ="/data/PFC3D_Augmented_250_Nt_101_Nx_32_dt05.mat", # dt=0.05
        #MAT_DATA_PATH="/scratch/noqu8762/phase_field_equations_4d/AC3D_Hybrid/data/PFC3D_Augmented_250_Nt_101_Nx_32.mat", # dt=0.01
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

if PROBLEM == 'MBE3D':
    STEPS_PER_EPOCH = 30  # MBE  # pick once; same training budget regardless of N_TRAIN
elif PROBLEM == 'PFC3D':
    STEPS_PER_EPOCH = 10  # PFC
elif PROBLEM == 'SH3D':
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
#T_OUT = 1 # 96 # 1 # 96 # 96 + 5 - 1 = 100 # determines how much future the model predicts

if MODEL in ("DeepONet_PMNO", "TNO3d", "FNO_PMNO", "DeepONet3D_TNOPhysics"):
    T_OUT = 1
else:
    T_OUT = 1


if PROBLEM == 'CH3D':
    WEIGHT_DECAY =   5e-6 # 5e-4 # 5e-6 # 1e-4 # 5e-6 # 5e-6
elif MODEL == 'HAMNO3d' and PROBLEM == 'AC3D':
    WEIGHT_DECAY = 1e-3 # 5e-3 #   1e-3 --> for manuscript
elif MODEL == 'HAMNO3d' and PROBLEM == 'SH3D':
    WEIGHT_DECAY = 5e-6 #
else:
    WEIGHT_DECAY =   5e-6 # 5e-6 # 1e-5 #

#AC:  LR=0.01, WD=0.001, WIDTH_Q=12, WIDTH_H=10, MODES=12, WIDTH=12, N_LAYERS=2, EXP=2, EPOCHS=50, BATCH_SIZE=8 --> mean: 2.15%
#CH:  LR=0.0005, WD=5e-06,: mean=4.2341e-02
PDE_WEIGHT = 0.25

N_TRAIN = 100 # 50 # 200 # 200 # 50 # 200 #


W_AC_IC = 1e-3
W_AC_BC = 1e-4

ZETA1 = 1   # 1 = use collocation architecture term, 0 = disable it --> collocation on/off
ZETA2 = 1   # 1 = use energy architecture gate, 0 = disable it

# --- AC + TNO residual-head physics ---
USE_AC_TNO_RESIDUAL_HEAD = True
AC_CORR_SCALE = 0.25          # try 0.20, 0.25, 0.35
AC_W_PDE = 10.0
AC_W_DEFECT = 1.0
AC_W_ENERGY = 1.0
AC_W_SMOOTH = 0.0


N_TEST_FIXED = 50 #50 AC, 100 # 100 # 100             # <- constant, test set size is fixed now
TEST_MODE = 'manual'   # or 'manual'
TEST_PICK = 1       # 0 spherical   # only used if TEST_MODE == 'manual'
N_TEST = 50 # max(1, N_TRAIN // 4)

# ——— Debug print scaling ———
DEBUG_MU_SCALE = 0.25

# ——— Eval/plots ———
TRAIN_TMAX = 100 # 100 # Training capped at {favorite frame: 50}, rollout tested through 100.
#EVAL_TIME_FRAMES = [0, 50, 100]
#EVAL_TIME_FRAMES = [25, 50, 75, 100]
EVAL_TIME_FRAMES = [20, 40, 60, 80, 100]
###

# -------------------------
# Physics-consistent warm start
# -------------------------
USE_NUMERICAL_WARMSTART = True
WARMSTART_EPOCHS = 0
WARMSTART_LR = 5e-4

############

# -------------------------
# AC collocation family sweep
AC_COLLOCATION_FAMILY = "random"
AC_COLLOCATION_NPTS = 3
AC_RANDOM_TAU_CLIP = 0.08
CH_COLLOCATION_NPTS = 3

###

# -------------------------
# Rollout-aware training
# -------------------------
TRAIN_ROLLOUT_STEPS = 4      # try 4 first, then 5
ROLLOUT_LOSS_END_WEIGHT = 3.0