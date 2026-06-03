import os  # <- minimal addition
import torch, numpy as np, random
import config
from networks import FNO4d, TNO3d, FFNO4d, MHNO_FFNO, DeepONet3D_Robust
from Trainer import build_loaders, train_fno_hybrid, evaluate_stats_and_plot #, train_ac_beamstyle, train_ch_beamstyle,train_sh_beamstyle

#from Trainer_out_rollout import build_loaders, train_fno_hybrid, evaluate_stats_and_plot #, train_ac_beamstyle, train_ch_beamstyle,train_sh_beamstyle

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from timeit import default_timer as timer
import gc
import math

def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
def set_full_determinism(seed=42):
    import os
    import random
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # strict deterministic mode
    torch.use_deterministic_algorithms(True)
def main():
    config.seed_everything(config.SEED)   # <--- important, before loaders/model
    set_full_determinism(config.SEED)
    device = config.DEVICE
    print("Using device:", device)
    print('PDE_WEIGHT is: ', config.PDE_WEIGHT)
    print('MODEL name is: ', config.MODEL)
    print('Problem is: ', config.PROBLEM)
    print('N_Train is: ', config.N_TRAIN)
    print('T_out is: ', config.T_OUT)
    print('STEPS_PER_EPOCH is: ', config.STEPS_PER_EPOCH)
    print(
        f"LR={config.LEARNING_RATE}, WD={config.WEIGHT_DECAY}, WIDTH_Q={config.WIDTH_Q}, WIDTH_H={config.WIDTH_H}, MODES={config.MODES}, WIDTH={config.WIDTH}, N_LAYERS={config.N_LAYERS}, EXP={config.Expansion}, EPOCHS={config.EPOCHS}, BATCH_SIZE={config.BATCH_SIZE}, TRAIN_TMAX={config.TRAIN_TMAX}")
    #print('DATA_LOSS_SCALE is: ', config.DATA_LOSS_SCALE)


    # (tiny clarity prints; no behavioral change)
    print(f"Grid: N={config.GRID_RESOLUTION}, L={config.L_DOMAIN:g}, dx={config.DX:g}")
    print(f"Time: dt={config.DT:g}, Nt={config.TOTAL_TIME_STEPS}, T_end={config.DT*config.TOTAL_TIME_STEPS:g}")
    print(f"ε (epsilon): {config.EPSILON_PARAM:g}")
    print(f"Data path: {config.MAT_DATA_PATH}")
    if isinstance(config.MAT_DATA_PATH, str) and (config.MAT_DATA_PATH.startswith("/") or config.MAT_DATA_PATH.startswith(".")):
        if not os.path.exists(config.MAT_DATA_PATH):
            print(f"[WARN] MAT_DATA_PATH not found: {config.MAT_DATA_PATH}")

    # Model
    if config.MODEL == 'FNO4d':
        model = FNO4d(
            modes1=config.MODES, modes2=config.MODES, modes3=config.MODES, modes4_internal=None,
            width=config.WIDTH, width_q=config.WIDTH_Q, T_in_channels=config.T_IN_CHANNELS,
            n_layers=config.N_LAYERS
        ).to(device)

    elif config.MODEL == 'FFNO4d':
        model = FFNO4d(
            modes=config.MODES,
            width=config.WIDTH,
            width_q=config.WIDTH_Q,
            T_in_channels=config.T_IN_CHANNELS,
            n_layers=config.N_LAYERS,
            expansion=config.Expansion,  # start with 2 or 4
            use_pairwise=True,
            use_local_conv=  True
        ).to(device)

    elif config.MODEL == 'TNO3d_PureRobust':
        from networks import TNO3d_PureRobust
        model = TNO3d_PureRobust(
            modes=config.MODES,
            width=max(config.WIDTH, 16),
            width_q=max(config.WIDTH_Q, 16),
            T_in=config.T_IN_CHANNELS,
            n_layers=3,
            heads=4,
            dim_head=16,
            mlp_ratio=2,
            dropout=0.0,
            use_coords_in_lift=True,
        ).to(device)

    elif config.MODEL == 'HAMNO3d':
        from networks import HAMNO3d
        model = HAMNO3d(
            modes=config.MODES,
            width=config.WIDTH,
            width_q=config.WIDTH_Q,
            T_in=config.T_IN_CHANNELS,
            n_layers=config.N_LAYERS,
            expansion=config.Expansion,
        ).to(device)

    elif config.MODEL == 'UNO3d':
        from networks import UNO3d
        model = UNO3d(
            modes=config.MODES,
            width=config.WIDTH,
            width_q=config.WIDTH_Q,
            T_in=config.T_IN_CHANNELS,
            n_layers=config.N_LAYERS,
            expansion=config.Expansion,
        ).to(device)

    elif config.MODEL == 'UNet3d':
        from networks import UNet3d
        model = UNet3d(
            T_in=config.T_IN_CHANNELS,
            width=config.WIDTH,
            out_channels=1,
        ).to(device)

    elif config.MODEL == 'UFNO3d':
        from networks import UFNO3d
        model = UFNO3d(
            modes=config.MODES,
            width=config.WIDTH,
            width_q=config.WIDTH_Q,
            T_in=config.T_IN_CHANNELS,
            n_layers=config.N_LAYERS,
            expansion=config.Expansion,
        ).to(device)


    elif config.MODEL == 'HMNO3dPlus':
        from networks import HMNO3dPlus
        model = HMNO3dPlus(
            modes=config.MODES,
            width=config.WIDTH,
            width_q=config.WIDTH_Q,
            T_in=config.T_IN_CHANNELS,
            n_layers=config.N_LAYERS,
            expansion=config.Expansion,
        ).to(device)


    elif config.MODEL == 'DeepONet_PMNO':
        from networks import DeepONet3D_Robust, PMNO_DeepONet

        deeponet = DeepONet3D_Robust(
            T_in_channels=config.T_IN_CHANNELS,
            width=getattr(config, "DEEPONET_WIDTH", 64),
            branch_depth=getattr(config, "DEEPONET_BRANCH_DEPTH", 4),
            trunk_hidden=getattr(config, "DEEPONET_TRUNK_HIDDEN", 64),
            trunk_depth=getattr(config, "DEEPONET_TRUNK_DEPTH", 4),
            head_width=getattr(config, "DEEPONET_HEAD_WIDTH", 64),
            use_coords_in_input=getattr(config, "DEEPONET_COORDS_IN_BRANCH", True),
        )

        model = PMNO_DeepONet(
            deeponet=deeponet,
            k=config.T_IN_CHANNELS,
            T_out=config.T_OUT,
            pm_width=16,
        ).to(device)

    elif config.MODEL == 'DeepONet3D_TNOPhysics':
        from networks import DeepONet3D_Robust, DeepONet3D_TNOPhysics
        model = DeepONet3D_TNOPhysics(
            T_in_channels=config.T_IN_CHANNELS,
            T_out=config.T_OUT,
            width=getattr(config, "DEEPONET_WIDTH", 64),
            branch_depth=getattr(config, "DEEPONET_BRANCH_DEPTH", 4),
            trunk_hidden=getattr(config, "DEEPONET_TRUNK_HIDDEN", 64),
            trunk_depth=getattr(config, "DEEPONET_TRUNK_DEPTH", 4),
            head_width=getattr(config, "DEEPONET_HEAD_WIDTH", 64),
            use_coords_in_input=getattr(config, "DEEPONET_COORDS_IN_BRANCH", True),
            q_width=getattr(config, "WIDTH_Q", 64),
            h_width=getattr(config, "WIDTH_H", 32),
            corr_width=32,
            n_layers_q=2,
            n_layers_h=2,
        ).to(device)


    elif config.MODEL == 'FNO_PMNO':
        from networks import FNO3D_PMNO, PMNO_AC

        fno = FNO3D_PMNO(
            modes=config.MODES,
            width=config.WIDTH,
            T_in=config.T_IN_CHANNELS,
            width_q=config.WIDTH_Q,
            n_layers=config.N_LAYERS,
        )

        model = PMNO_AC(
            fno=fno,
            k=config.T_IN_CHANNELS
        ).to(device)


    # In your main script or config

    elif config.MODEL == "PhysicsGuidedTNO3d":
        from networks import PhysicsGuidedTNO3d
        model = PhysicsGuidedTNO3d(
            modes1=config.MODES,
            modes2=config.MODES,
            modes3=config.MODES,
            width=config.WIDTH,
            width_q=config.WIDTH_Q,
            width_h=config.WIDTH_H,
            T_in=config.T_IN_CHANNELS,
            T_out=config.T_OUT,
            n_layers=config.N_LAYERS,
        ).to(config.DEVICE)


    elif config.MODEL == 'MHNO_FFNO':
        model = MHNO_FFNO(
            modes=config.MODES,
            width=config.WIDTH,
            width_q=config.WIDTH_Q,
            width_h=config.WIDTH_H,
            T_in=config.T_IN_CHANNELS,
            T_out=config.T_OUT,
            n_layers=config.N_LAYERS,
            expansion=config.Expansion,
            use_pairwise=True,
            use_local_conv=  True # False #
        ).to(device)

    elif config.MODEL == 'DeepONet3D_Robust':
        from networks import DeepONet3D_Robust
        model = DeepONet3D_Robust(
            T_in_channels=config.T_IN_CHANNELS,
            width=getattr(config, "DEEPONET_WIDTH", 64),
            branch_depth=getattr(config, "DEEPONET_BRANCH_DEPTH", 4),
            trunk_hidden=getattr(config, "DEEPONET_TRUNK_HIDDEN", 64),
            trunk_depth=getattr(config, "DEEPONET_TRUNK_DEPTH", 4),
            head_width=getattr(config, "DEEPONET_HEAD_WIDTH", 64),
            use_coords_in_input=getattr(config, "DEEPONET_COORDS_IN_BRANCH", True),
        ).to(device)


    else:
        model = TNO3d(
            modes1=config.MODES,  # spectral modes in x
            modes2=config.MODES,  # spectral modes in y
            modes3=config.MODES,  # spectral modes in z
            width=config.WIDTH,   # channel width in trunk
            width_q=config.WIDTH_Q,  # width in the projection MLP (q)
            width_h=config.WIDTH_H,  # temporal memory width
            T_in=config.T_IN_CHANNELS,
            T_out= config.T_OUT,      # how many future time frames the network outputs in one forward pass
            n_layers=config.N_LAYERS
        ).to(config.DEVICE)

    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Data
    train_loader, test_loader, test_ids, normalizers = build_loaders()
    base_steps = getattr(config, "STEPS_PER_EPOCH", 20)

    N_ref = max(1, int(getattr(config, "N_TRAIN_REF", 50)))
    N_cur = max(1, int(getattr(config, "N_TRAIN_ACTUAL",
                               getattr(config, "N_TRAIN", N_ref))))

    if config.PDE_WEIGHT == 1.0:
        # -------- PURE PHYSICS --------
        # Do NOT depend on N_TRAIN
        STEPS_PER_EPOCH_EFF = base_steps

    elif config.PDE_WEIGHT == 0.0:
        # -------- PURE DATA --------
        # Fully scale with dataset size
        scale = N_cur / N_ref
        STEPS_PER_EPOCH_EFF = math.ceil(len(train_loader.dataset) / config.BATCH_SIZE)

    else:
        # -------- HYBRID --------
        # Partial scaling between physics and data
        scale = N_cur / N_ref
        hybrid_scale = 1.0 + (1.0 - config.PDE_WEIGHT) * (scale - 1.0)
        STEPS_PER_EPOCH_EFF = max(1, int(round(base_steps * hybrid_scale)))

    setattr(config, "STEPS_PER_EPOCH_EFF", STEPS_PER_EPOCH_EFF)

    setattr(config, "STEPS_PER_EPOCH_EFF", STEPS_PER_EPOCH_EFF)
    TOTAL_STEPS = config.EPOCHS * STEPS_PER_EPOCH_EFF
    print(f"[Budget] steps/epoch={STEPS_PER_EPOCH_EFF}, total updates={TOTAL_STEPS}, "
          f"approx windows={TOTAL_STEPS * config.BATCH_SIZE}")

    # Optim
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    #scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    scheduler = CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS)  # step this every batch

    # Training function

    start_train = timer()
    train_fno_hybrid(model, train_loader, test_loader, optimizer, scheduler, config.DEVICE, pde_weight=config.PDE_WEIGHT)

    end_train = timer()
    total_sec = end_train - start_train
    print(f"\n[Timing] Total training time = {total_sec:.2f} seconds ({total_sec / 60:.2f} minutes)\n")


    # Evaluate: stats + 3×len(times) plot
    # free training memory before evaluation
    del optimizer
    del scheduler
    del train_loader
    del test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Evaluate: stats + 3×len(times) plot
    evaluate_stats_and_plot(model, config.MAT_DATA_PATH, test_ids, times=config.EVAL_TIME_FRAMES)

if __name__ == "__main__":
    main()
