import h5py, numpy as np, torch, random
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
import config
from itertools import cycle
from torch.cuda.amp import autocast
from pathlib import Path

import math

from functions import weak_form_AC_gauss
from functions import ac_physical_gauss_single_step
from functions import ac_weak_fe_loss_tetra_neumann
from functions import ch_weak_fe_loss_tetra_neumann
from functions import physics_collocation_tau_L2_SH_neumann,semi_implicit_step_sh_neumann, energy_penalty_sh_neumann, low_k_mse_neumann
from functions import  mixed_form_CH_physical_gauss_single_step
from functions import mixed_form_SH_physical_gauss_single_step, sh_weak_fe_loss_tetra_neumann


import matplotlib
matplotlib.use('TkAgg')

with h5py.File(config.MAT_DATA_PATH, "r") as f:
    print(config.PROBLEM, f["phi"].shape)

# ---------------------
# Dataset: load chosen trajectories into RAM ONCE
# ---------------------
class AC3DTrajectoryDataset(Dataset):
    """
    Holds full trajectories for selected sample_ids in RAM.
    __getitem__ returns the trajectory tensor (Nt, S, S, S).
    """
    def __init__(self, mat_path, sample_ids, dtype=np.float32):
        super().__init__()
        self.sample_ids = np.array(sample_ids)
        with h5py.File(mat_path, "r") as f:
            dset = f["phi"]  # (Nz,Ny,Nx,Nt,Ns)
            Nz, Ny, Nx, Nt, Ns = dset.shape
            self.Nz, self.Ny, self.Nx, self.Nt = Nz, Ny, Nx, Nt
            self.data = []
            for sid in self.sample_ids:
                raw = np.array(dset[:, :, :, :, sid], dtype=dtype)   # (Nz,Ny,Nx,Nt)
                traj = np.transpose(raw, (3,2,1,0))                  # (Nt,Nx,Ny,Nz)
                self.data.append(traj)
            self.data = np.stack(self.data, axis=0)                  # (Ns_sel,Nt,Nx,Ny,Nz)

        X = self.data
        self._mean = float(X.mean()); self._std = float(X.std() + 1e-8)
        class _Norm:
            def __init__(self, m, s): self.m, self.s = m, s
            def encode(self, t): return (t - self.m)/self.s
            def decode(self, t): return t*self.s + self.m
        self.normalizer_x = _Norm(self._mean, self._std)
        self.normalizer_y = _Norm(self._mean, self._std)

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return torch.from_numpy(self.data[idx])  # (Nt,S,S,S)

# ---------------------
# Collate: convert trajectories -> (x,y) windows
# ---------------------
'''
def make_windowed_collate(T_in=4, t_min=None, t_max=None, normalized=False, normalizers=None):
    y_norm = normalizers[1] if (normalized and normalizers is not None) else None
    def _collate(batch):
        Nt = batch[0].shape[0]
        t0 = T_in - 1 if t_min is None else max(T_in-1, t_min)
        t1 = Nt - 2   if t_max is None else min(Nt-2,   t_max)

        xs, ys = [], []
        for traj in batch:                 # traj: (Nt,S,S,S)
            t = random.randint(t0, t1)
            x = traj[t-(T_in-1):t+1]       # (T_in,S,S,S)
            y = traj[t+1]                  # (S,S,S)
            x = x.permute(1,2,3,0).contiguous()   # (S,S,S,T_in)
            y = y.unsqueeze(-1).contiguous()      # (S,S,S,1)
            xs.append(x); ys.append(y)
        x = torch.stack(xs, dim=0)         # (B,S,S,S,T_in)
        y = torch.stack(ys, dim=0)         # (B,S,S,S,1)
        if y_norm is not None:
            x = (x - y_norm.m)/y_norm.s
            y = (y - y_norm.m)/y_norm.s
        return x, y
    return _collate
'''
def make_windowed_collate(T_in=4, T_out=1, t_min=None, t_max=None,
                          normalized=False, normalizers=None):
    y_norm = normalizers[1] if (normalized and normalizers is not None) else None

    def _collate(batch):
        # traj: (Nt, S, S, S)
        Nt = batch[0].shape[0]

        # we need t such that:
        #  - we have T_in frames ending at t         -> t >= T_in - 1
        #  - we can take T_out frames (t+1 ... t+T_out) within trajectory.
        # traj indices: 0 ... Nt-1
        # last valid t satisfies: t + T_out <= Nt - 1  -> t <= Nt - 1 - T_out
        t0 = T_in - 1 if t_min is None else max(T_in - 1, t_min)
        t1_raw = Nt - 1 - T_out
        if t_max is None:
            t1 = t1_raw
        else:
            t1 = min(t1_raw, t_max)

        xs, ys = [], []
        for traj in batch:                 # traj: (Nt,S,S,S)
            t = random.randint(t0, t1)

            # input window: [t-(T_in-1), ..., t]
            x = traj[t-(T_in-1):t+1]           # (T_in,S,S,S)

            # output window: [t+1, ..., t+T_out]
            y = traj[t+1:t+1+T_out]           # (T_out,S,S,S)

            # reorder to channel-last
            x = x.permute(1, 2, 3, 0).contiguous()  # (S,S,S,T_in)
            y = y.permute(1, 2, 3, 0).contiguous()  # (S,S,S,T_out)

            xs.append(x); ys.append(y)

        x = torch.stack(xs, dim=0)  # (B,S,S,S,T_in)
        y = torch.stack(ys, dim=0)  # (B,S,S,S,T_out)

        if y_norm is not None:
            x = (x - y_norm.m)/y_norm.s
            y = (y - y_norm.m)/y_norm.s

        return x, y

    return _collate

# ---------------------
# Loaders (your preferred split API)
# ---------------------

def build_loaders():
    rng = np.random.default_rng(config.SEED)

    # --- read Ns from the file (phi: (Nz,Ny,Nx,Nt,Ns)) ---
    with h5py.File(config.MAT_DATA_PATH, "r") as f:
        Ns = int(f["phi"].shape[-1])

    # deterministically shuffle IDs once
    all_ids = np.arange(Ns)
    rng.shuffle(all_ids)

    # ----- FIX 1: make test set independent of N_TRAIN -----
    n_test = min(int(getattr(config, "N_TEST_FIXED", 100)), Ns - 1)  # keep at least 1 train
    test_ids = all_ids[:n_test]
    train_pool = all_ids[n_test:]

    # ----- FIX 2: in pure-physics mode, ignore N_TRAIN and use full pool -----
    use_all = bool(getattr(config, "PURE_PHYSICS_USE_ALL", True)) and (config.PDE_WEIGHT == 1.0)
    if use_all:
        chosen_train_ids = train_pool
    else:
        n_train_req = int(getattr(config, "N_TRAIN", len(train_pool)))
        n_train = max(1, min(n_train_req, len(train_pool)))
        chosen_train_ids = train_pool[:n_train]

    # build RAM datasets over the chosen IDs
    train_base = AC3DTrajectoryDataset(config.MAT_DATA_PATH, chosen_train_ids)
    test_base  = AC3DTrajectoryDataset(config.MAT_DATA_PATH, test_ids)

    normalizers = [train_base.normalizer_x, train_base.normalizer_y]

    #collate = make_windowed_collate(
    #    T_in=config.T_IN_CHANNELS, t_min=0, t_max=config.TOTAL_TIME_STEPS-1,
    #    normalized=False, normalizers=normalizers
    #)

    collate = make_windowed_collate(
        T_in=config.T_IN_CHANNELS,
        T_out=getattr(config, "T_OUT", 1),  # 👈 depend on T_OUT
        t_min=0, t_max=config.TOTAL_TIME_STEPS - 1,
        normalized=False, normalizers=normalizers
    )

    train_loader = DataLoader(train_base, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True,
                              collate_fn=collate)
    test_loader = DataLoader(test_base, batch_size=config.BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=True, persistent_workers=True,
                             collate_fn=collate)

    # deterministic list of test IDs (now fixed!)
    test_indices = list(test_ids)
    setattr(config, "N_TRAIN_ACTUAL", len(chosen_train_ids))
    print(f"[Split] train_ids={len(chosen_train_ids)}, test_ids={len(test_ids)}")

    return train_loader, test_loader, test_indices, normalizers
####
# ---------------------
# Utilities
# ---------------------
def relative_l2(a, b, eps=1e-12):
    diff = (a - b).flatten(start_dim=1)
    denom = b.flatten(start_dim=1)
    num = torch.sqrt(torch.sum(diff**2, dim=1) + eps)
    den = torch.sqrt(torch.sum(denom**2, dim=1) + eps)
    return (num / den)  # (B,)

def weighted_tout_mse(y_pred, y_true, end_weight=2.5):
    """
    Weighted multi-step MSE.
    Later predicted steps receive larger weight.
    """
    assert y_pred.shape == y_true.shape
    T = y_pred.shape[-1]

    if T == 1:
        return F.mse_loss(y_pred, y_true)

    w = torch.linspace(
        1.0, end_weight, T,
        device=y_pred.device,
        dtype=y_pred.dtype
    )  # (T,)

    err = (y_pred - y_true) ** 2              # (B,S,S,S,T)
    err_t = err.mean(dim=(0, 1, 2, 3))        # (T,)
    return (w * err_t).sum() / w.sum()

def rollout_block_from_one_step_model(model, x, T_out):
    """
    Build a multi-step prediction block from a one-step model.

    model(x): (B,S,S,S,T_in) -> (B,S,S,S,1)

    Returns:
        y_block: (B,S,S,S,T_out)
    """
    preds = []
    x_cur = x

    for _ in range(T_out):
        y_next = model(x_cur)                  # (B,S,S,S,1)
        preds.append(y_next)
        x_cur = torch.cat([x_cur[..., 1:], y_next], dim=-1)

    return torch.cat(preds, dim=-1)           # (B,S,S,S,T_out)



# Save model helper funtion

def make_model_save_path():
    """
    Build a clean checkpoint path using selected config values.

    Example:
    Trained_Models/AC3D_HAMNO3d_pde0p25_fine64_fw0p5.pt
    """
    save_dir = Path("Trained_Models")
    save_dir.mkdir(parents=True, exist_ok=True)

    def clean_float(x):
        return str(x).replace(".", "p").replace("-", "m")

    problem = config.PROBLEM
    model = config.MODEL
    pde_weight = clean_float(config.PDE_WEIGHT)
    fine_res = getattr(config, "PHYSICS_FINE_RESOLUTION", config.GRID_RESOLUTION)
    fine_weight = clean_float(getattr(config, "PHYSICS_FINE_WEIGHT", 0.0))

    filename = (
        f"{problem}_{model}"
        f"_pde{pde_weight}"
        f"_fine{fine_res}"
        f"_fw{fine_weight}_cosine.pt"
    )

    return save_dir / filename

# ---------------------
# Training: hybrid
# ---------------------

def train_fno_hybrid(model, train_loader, test_loader, optimizer, scheduler, device, pde_weight=None):
    pde_weight = config.PDE_WEIGHT if pde_weight is None else pde_weight

    # physics term weights (baseline-compatible)
    USE_AC = (config.PROBLEM == 'AC3D')  # NEW
    USE_CH = (config.PROBLEM == 'CH3D')
    USE_SH = (config.PROBLEM == 'SH3D')  # <-- NEW
    USE_PFC = (config.PROBLEM == 'PFC3D')  # NEW
    USE_MBE = (config.PROBLEM == 'MBE3D')  # <-- add this
    CLIP_NORM = 1.0

    print("Epoch |   Time   | DataLoss | PhysLoss | TotalLoss | Test relL2 | energy_loss | scheme_loss | Loss_strong | Loss_weak | loss_u| loss_q |weak_u| weak_p | LR")

    # FIXED number of updates per epoch (independent of N_TRAIN)
    steps_per_epoch = getattr(config, "STEPS_PER_EPOCH_EFF", getattr(config, "STEPS_PER_EPOCH", 20))

    train_iter = cycle(train_loader)  # infinite stream of batches

    best_test_rel = float("inf")
    save_path = make_model_save_path()

    for ep in range(config.EPOCHS):
        model.train()
        t1 = default_timer()
        data_loss_acc = phys_loss_acc = total_loss_acc = l_mid_norm_ch_cc = l_lowk_cc = Loss_strong = Loss_weak = loss_uu = loss_qq = weak_uu = weak_pp  = 0.0
        energy_loss_acc = scheme_loss_acc = 0.0
        n_batches = 0
        for _ in range(steps_per_epoch):  # ← fixed number of updates each epoch
            x, y = next(train_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            u_in_last = x[..., -1:]  # (B,S,S,S,1)
            optimizer.zero_grad(set_to_none=True)
            # forward
            y_pred = model(x)
            y_hat = y_pred
            loss_data = F.mse_loss(y_pred, y)


            if USE_SH:
                y_hat = y_pred
                # ---------- first predicted step ----------
                y_first = y_hat[..., 0:1]
                # SH is 4th order, so use mixed/split form like CH
                loss_strong_raw, loss_u, loss_q  =  mixed_form_SH_physical_gauss_single_step(u_in_last,y_first,config.DT,config.DX,config.EPSILON_PARAM,tau=0.5, normalize=True,robust=False)
                loss_strong= 0.005 * loss_strong_raw

                loss_weak_raw, weak_u, weak_p = sh_weak_fe_loss_tetra_neumann(u_in_last, y_first,config.DT,config.DX,config.EPSILON_PARAM,tau=0.5, normalize=True, robust=False,p_weight=1.0, return_parts=True)
                loss_weak =  0.005 *  loss_weak_raw
                # ---------- total SH physics ----------
                loss_phys = (loss_strong + loss_weak)

                loss_energy = torch.zeros_like(loss_phys)

            elif USE_AC:
                # ---------- first predicted step ----------
                y_first = y_pred[..., 0:1]
                loss_weak = 0.02 * ac_weak_fe_loss_tetra_neumann(u_in_last,y_first,config.DT,config.DX,config.EPS2,normalize=True, robust=False)
                loss_strong =  0.02 * ac_physical_gauss_single_step( u_in_last,y_first,config.DT, config.DX, config.EPS2)
                # ---------- total AC physics ----------
                loss_phys = (loss_strong + loss_weak)
            # correct
            elif USE_CH:
                y_hat = y_pred
                # ---------- first predicted step ----------
                y_first = y_hat[..., 0:1]
                loss_strong = mixed_form_CH_physical_gauss_single_step( u_in_last, y_first, config.DT, config.DX, config.EPS2, tau=0.5, normalize=True, robust=False, mass_weight=0.0)
                loss_weak =  2.0 * ch_weak_fe_loss_tetra_neumann( u_in_last, y_first,config.DT,config.DX, config.EPS2,tau=0.5, normalize=True, robust=False, mass_weight=0.0)
                # ---------- total CH physics ----------
                loss_phys = (0.02 * loss_strong+  0.02 * loss_weak)

            else:
                raise RuntimeError(f"Unknown/unsupported PROBLEM: {config.PROBLEM}")

            # ---- total loss (unchanged structure) ----
            loss_total = loss_data * (1 - pde_weight) + pde_weight * loss_phys

            # backward
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()


            # accumulators Loss_strong, Loss_weak
            data_loss_acc  += loss_data.item()
            phys_loss_acc  += loss_phys.item()
            Loss_strong += loss_strong.item() # l_mid_norm.item() #loss_strong # loss_strong.item() # l_mid_norm.item()
            Loss_weak += loss_weak.item()
            loss_uu += 0.0 #loss_u.item()
            loss_qq += 0.0 #loss_q.item()
            weak_uu += 0.0 #weak_u.item()
            weak_pp += 0.0 #weak_p.item()

            l_lowk_cc = 0.0 # l_lowk.item()
            energy_loss_acc += 0.0 # loss_energy.item()
            scheme_loss_acc += 0.0 # CH_residual_integral # CH_residual_integral.item()
            total_loss_acc += loss_total.item()
            n_batches      += 1
            scheduler.step()

        # eval
        model.eval()
        with torch.no_grad():
            rels = []
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                #y_pred = model(x)
                if config.MODEL == 'FNO_PMNO':
                    y_pred = model(x, steps=config.T_OUT)
                else:
                    y_pred = model(x)

                #if config.PROBLEM == 'CH3D':
                #    y_pred = mass_project_pred(y_pred, x[..., -1:])

                rels.append(relative_l2(y_pred, y))
            test_rel = torch.cat(rels, dim=0).mean().item()

            # ---------------------
            # Save best checkpoint
            # ---------------------
            if test_rel < best_test_rel:
                best_test_rel = test_rel

                torch.save({
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_test_rel": best_test_rel,

                    # key experiment identifiers
                    "problem": config.PROBLEM,
                    "model": config.MODEL,
                    "pde_weight": config.PDE_WEIGHT,
                    "physics_fine_resolution": getattr(config, "PHYSICS_FINE_RESOLUTION", None),
                    "physics_fine_weight": getattr(config, "PHYSICS_FINE_WEIGHT", None),

                    # useful extra metadata
                    "grid_resolution": config.GRID_RESOLUTION,
                    "t_in_channels": config.T_IN_CHANNELS,
                    "t_out": getattr(config, "T_OUT", 1),
                    "epochs": config.EPOCHS,
                    "batch_size": config.BATCH_SIZE,
                    "learning_rate": config.LEARNING_RATE,
                    "weight_decay": config.WEIGHT_DECAY,
                }, save_path)

                #print(f"[Save] Best model saved: {save_path} | test_rel={best_test_rel:.4e}")

        #scheduler.step()
        t2 = default_timer()
        lr = optimizer.param_groups[0]['lr']
        print(f"{ep:5d} | {t2-t1:7.3f} | "
              f"{data_loss_acc/n_batches:8.3e} | {phys_loss_acc/n_batches:8.3e} | {total_loss_acc/n_batches:9.3e} | "
              f"{test_rel:10.3e} |  {energy_loss_acc:10.3e} |  {scheme_loss_acc:10.3e} | {Loss_strong:10.3e} | {Loss_weak:10.3e} |{loss_uu/n_batches:10.3e} |{loss_qq/n_batches:10.3e} | {weak_uu/n_batches:10.3e} |{weak_pp/n_batches:10.3e} |{lr: .2e}")

# ---------------------
# Evaluation: rollout
# ---------------------
def rollout_autoregressive(model, traj_np, T_in, Nt=100):
    """
    traj_np: (Nt+1, S,S,S) ground truth trajectory for one sample
    returns pred: same shape, with pred[0:T_in]=gt[0:T_in],
    and the rest filled autoregressively in blocks of T_OUT.
    """
    device = next(model.parameters()).device
    Nt_plus1, Sx, Sy, Sz = traj_np.shape
    assert Nt_plus1 >= Nt + 1

    T_out = int(getattr(config, "T_OUT", 1))

    traj_torch = torch.from_numpy(traj_np).to(device)  # (Nt+1,Sx,Sy,Sz)
    pred = traj_torch.clone()
    model.eval()

    t = T_in - 1
    while t < Nt:
        # input window ending at time t
        x_win = pred[t - (T_in - 1):t + 1]  # (T_in,Sx,Sy,Sz)
        x = x_win.permute(1, 2, 3, 0).unsqueeze(0)  # (1,Sx,Sy,Sz,T_in)

        with torch.no_grad():
            #y_block = model(x)  # (1,Sx,Sy,Sz,T_out_expected)
            if config.MODEL == 'FNO_PMNO':
                y_block = model(x, steps=config.T_OUT)
            else:
                y_block = model(x)

            #if config.PROBLEM == 'CH3D':
            #    y_block = mass_project_pred(y_block, x[..., -1:])
            #x_model = append_mass_channel_CH(x) if (config.PROBLEM == 'CH3D') else x
            #y_block = model(x_model)  # (1,Sx,Sy,Sz,T_out_expected)

        # Make sure we handle either T_OUT=1 or >1
        assert y_block.dim() == 5
        this_T_out = y_block.shape[-1]   # usually = config.T_OUT

        # write predicted steps back into pred
        for k in range(this_T_out):
            t_next = t + 1 + k
            if t_next > Nt:
                break
            y_step = y_block[..., k]      # (1,Sx,Sy,Sz)
            y_step = y_step.squeeze(0)    # (Sx,Sy,Sz)
            pred[t_next] = y_step

        t += this_T_out  # jump forward by the number of predicted steps

    pred_np = pred.detach().cpu().numpy()
    return pred_np


def relative_l2_scalar(a, b, eps=1e-12):
    num = np.linalg.norm(a.ravel() - b.ravel())
    den = np.linalg.norm(b.ravel()) + eps
    return num / den

def evaluate_stats_and_plot(model, mat_path, test_ids, times):
    import matplotlib
    matplotlib.use('TkAgg')
    import h5py, numpy as np
    import matplotlib.pyplot as plt

    def sym_vlims(A, sym_frac=0.995):
        m = np.mean(A)
        a = np.quantile(np.abs(A - m), sym_frac)
        return m - a, m + a


    with h5py.File(mat_path, "r") as f:
        dset = f["phi"]  # (Nz,Ny,Nx,Nt,Ns)
        Nz, Ny, Nx, Nt, Ns = dset.shape
        assert Nt == config.SAVED_STEPS

        rel_errors = {t: [] for t in times}

        # pick first test id for plotting
        #pid = int(test_ids[0])
        mode = getattr(config, "TEST_MODE", "random")
        if mode == "manual":
            pick = int(getattr(config, "TEST_PICK", 0))
            pid = int(test_ids[pick % len(test_ids)])  # pick a specific ID from the already-random test_ids
        else:
            pid = int(test_ids[0])
        print(f"[Eval] Visualization sample id (from test_ids): {pid}")

        gt_raw = np.array(dset[:, :, :, :, pid], dtype=np.float32)  # (Nz,Ny,Nx,Nt)
        gt = np.transpose(gt_raw, (3,2,1,0))                        # (Nt,Nx,Ny,Nz)
        # ---- Inference timing: one full rollout for a new PDE instance ----
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # make sure GPU is idle before timing
        t_inf_start = default_timer()

        pred = rollout_autoregressive(model, gt, config.T_IN_CHANNELS,
                                      Nt=config.TOTAL_TIME_STEPS)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # ensure all kernels are finished
        t_inf_end = default_timer()

        inf_time = t_inf_end - t_inf_start
        print(f"[Timing] Inference time for one rollout (Nt={config.TOTAL_TIME_STEPS}) "
              f"= {inf_time:.4f} s ({inf_time / 60:.4f} min)")


        # stats across test ids
        for sid in test_ids:
            gt_raw = np.array(dset[:, :, :, :, sid], dtype=np.float32)
            gt_s = np.transpose(gt_raw, (3,2,1,0))
            pred_s = rollout_autoregressive(model, gt_s, config.T_IN_CHANNELS,
                                            Nt=config.TOTAL_TIME_STEPS)
            for t in times:
                num = np.linalg.norm(pred_s[t].ravel() - gt_s[t].ravel())
                den = np.linalg.norm(gt_s[t].ravel()) + 1e-12
                rel_errors[t].append(num/den)

        # print stats
        all_vals = []
        print("\nRelative L2 error stats:")
        for t in times:
            arr = np.array(rel_errors[t])
            print(f"t={t:3d}: mean={arr.mean():.4e}  min={arr.min():.4e}  max={arr.max():.4e}  std={arr.std():.4e}")
            all_vals.extend(arr.tolist())
        all_vals = np.array(all_vals)
        print(f"OVERALL frames {times}: mean={all_vals.mean():.4e}  min={all_vals.min():.4e}  "
              f"max={all_vals.max():.4e}  std={all_vals.std():.4e}")

        # Mean relative L2 error over ALL time frames 0...TOTAL_TIME_STEPS
        all_time_errors = []

        for sid in test_ids:
            gt_raw = np.array(dset[:, :, :, :, sid], dtype=np.float32)
            gt_s = np.transpose(gt_raw, (3, 2, 1, 0))

            pred_s = rollout_autoregressive(
                model,
                gt_s,
                config.T_IN_CHANNELS,
                Nt=config.TOTAL_TIME_STEPS
            )

            for t in range(config.TOTAL_TIME_STEPS + 1):
                num = np.linalg.norm(pred_s[t].ravel() - gt_s[t].ravel())
                den = np.linalg.norm(gt_s[t].ravel()) + 1e-12
                all_time_errors.append(num / den)

        all_time_errors = np.array(all_time_errors)

        print(
            f"ALL TIME FRAMES [0, {config.TOTAL_TIME_STEPS}]: "
            f"mean={all_time_errors.mean():.4e}  "
            f"min={all_time_errors.min():.4e}  "
            f"max={all_time_errors.max():.4e}  "
            f"std={all_time_errors.std():.4e}"
        )

        # 3×len(times) subplot (central z-slice)
        S = gt.shape[1]; zc = S // 2
        fig, axes = plt.subplots(3, len(times), figsize=(4*len(times), 9))
        for j, t in enumerate(times):
            exact = gt[t, :, :, zc]
            predt = pred[t, :, :, zc]
            rel   = np.abs(predt - exact) / (np.abs(exact) + 1e-8)

            v0, V0 = sym_vlims(exact)
            v1, V1 = sym_vlims(predt)
            im0 = axes[0, j].imshow(exact, origin='lower', cmap='RdBu_r', vmin=v0, vmax=V0)
            im1 = axes[1, j].imshow(predt, origin='lower', cmap='RdBu_r', vmin=v1, vmax=V1)

            #im0 = axes[0, j].imshow(exact, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)
            axes[0, j].set_title(f"Exact t={t}");  fig.colorbar(im0, ax=axes[0, j], shrink=0.8)
            #im1 = axes[1, j].imshow(predt, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)
            axes[1, j].set_title(f"Pred t={t}");   fig.colorbar(im1, ax=axes[1, j], shrink=0.8)
            im2 = axes[2, j].imshow(rel, origin='lower', cmap='viridis')
            axes[2, j].set_title(f"Rel. L2 (px) t={t}"); fig.colorbar(im2, ax=axes[2, j], shrink=0.8)

            for r in range(3):
                axes[r, j].set_xlabel('x'); axes[r, j].set_ylabel('y')
        plt.tight_layout(); plt.show()
