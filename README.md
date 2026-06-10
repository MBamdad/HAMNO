# HAMNO

**HAMNO** is a **Hierarchical Adaptive Multi-scale Neural Operator** for learning long-time solution operators of nonlinear three-dimensional phase-field dynamical systems. The model is designed for non-periodic PDEs with homogeneous Neumann boundary conditions and is evaluated on the Allen--Cahn, Cahn--Hilliard, and Swift--Hohenberg equations.

<p align="center">
  <img src="HAMNO_Architecture/HAMNO_Architecture.png" width="850">
</p>

<p align="center">
  <b>Schematic overview of HAMNO and its physics-informed extension PI-HAMNO.</b>
</p>

---

## Overview

HAMNO learns a one-step solution operator from a short temporal history of three-dimensional phase fields:

```math
\mathcal{G}_{\theta}:
\left(
u^{n-T_{\mathrm{in}}+1},
\ldots,
u^{n-1},
u^{n}
\right)
\longmapsto
u_{\theta}^{n+1}.
```

Here, \(u^n\) denotes the phase-field solution at time step \(n\), and \(T_{\mathrm{in}}\) is the number of previous frames used as input. During evaluation, the learned one-step map is applied autoregressively to reconstruct the full time evolution.

The framework is developed for nonlinear, multi-scale, time-dependent PDEs where accurate long-horizon prediction requires both local feature resolution and global operator interaction.

---

## Main idea of HAMNO

HAMNO combines three key components:

1. **Local convolutional operators**  
   Capture nearby spatial interactions, sharp interfaces, and fine-scale structures.

2. **Global spectral operators**  
   Model long-range dependencies and global solution behavior through Fourier-based operator learning.

3. **Hierarchical encoder--decoder structure**  
   Learns solution features across multiple spatial resolutions using downsampling, upsampling, and skip connections.

The central innovation is an **adaptive local--global gating mechanism**. Instead of combining local and global features with fixed weights, HAMNO learns data-dependent fusion weights at each spatial location:

```math
h_{\mathrm{fused}}(\mathbf{x})
=
\alpha(\mathbf{x})\,h_{\mathrm{local}}(\mathbf{x})
+
\beta(\mathbf{x})\,h_{\mathrm{global}}(\mathbf{x}).
```

This allows the model to decide how much local or global information is needed depending on the evolving solution field and spatial scale.

---

## HAMNO architecture

The input history is written as

```math
\mathbf{U}_{\mathrm{in}}^{n}
=
\left[
u^{n-T_{\mathrm{in}}+1},
\ldots,
u^{n-1},
u^{n}
\right].
```

HAMNO first lifts the input history and spatial coordinates into a latent feature representation:

```math
v_0(\mathbf{x})
=
P
\left(
\mathbf{U}_{\mathrm{in}}^{n}(\mathbf{x}),
\mathbf{x}
\right),
```

where \(P\) is a coordinate-aware lifting map and \(\mathbf{x}=(x,y,z)\) denotes the spatial coordinate.

Each HAMNO block uses two complementary operator branches. The local branch captures nearby spatial patterns:

```math
h_{\mathrm{local}}
=
\mathcal{K}_{\mathrm{local}}(h),
```

while the global branch captures long-range spectral interactions:

```math
h_{\mathrm{global}}
=
\mathcal{K}_{\mathrm{global}}(h).
```

The two branches are then combined using adaptive local--global fusion:

```math
h_{\mathrm{fused}}
=
\alpha\,h_{\mathrm{local}}
+
\beta\,h_{\mathrm{global}}.
```

The fused representation is passed through a channel-mixing operator and added back through a residual update:

```math
v_{\ell}^{\prime}
=
v_{\ell}
+
\mathcal{M}
\left(
h_{\mathrm{fused}}
\right).
```

A second residual update applies nonlinear feature mixing:

```math
v_{\ell+1}
=
v_{\ell}^{\prime}
+
\Psi_{\ell}
\left(
\mathrm{Norm}
\left(
v_{\ell}^{\prime}
\right)
\right).
```

Here, \(\mathcal{M}\) is a channel-mixing operator and \(\Psi_{\ell}\) is a nonlinear pointwise feature transformation.

The encoder progressively reduces the spatial resolution to capture coarse global structures, while the decoder reconstructs fine-scale details using upsampling and skip connections. The final prediction is obtained by

```math
u_{\theta}^{n+1}(\mathbf{x})
=
Q
\left(
v_{\mathrm{final}}(\mathbf{x})
\right),
```

where \(Q\) is the final projection head.

---

## Difference from related neural operators

HAMNO differs from common neural-operator baselines as follows:

| Model | Main feature | Limitation addressed by HAMNO |
|---|---|---|
| **FNO** | Global Fourier operator | Limited recovery of localized and high-frequency structures |
| **F-FNO** | Factorized spectral operators | Efficient spectral mixing, but limited adaptive multi-scale fusion |
| **DeepONet** | Branch--trunk operator representation | Separates input and coordinate encoding but lacks hierarchical local--global fusion |
| **U-Net** | Local encoder--decoder | Strong local representation but no explicit global spectral operator |
| **U-NO** | U-shaped neural operator | Multi-resolution structure, mainly spectral-driven |
| **U-FNO** | Fourier layers with U-Net components | Uses fixed additive local--global fusion |
| **HAMNO** | Adaptive local--global hierarchical neural operator | Learns when and where to use local, global, and multi-scale information |

In summary, HAMNO combines the strengths of Fourier neural operators, convolutional models, and U-shaped architectures, while replacing fixed feature fusion with adaptive local--global operator coupling.

---

## Physics-informed extension: PI-HAMNO

**PI-HAMNO** extends HAMNO by introducing a **multi-objective physics regularization strategy**. Instead of relying only on data fitting, the model is trained with both data loss and complementary physics-based constraints:

```math
\mathcal{L}_{\mathrm{total}}
=
(1-\lambda)\mathcal{L}_{\mathrm{data}}
+
\lambda\mathcal{L}_{\mathrm{phys}},
```

where <img src="https://latex.codecogs.com/svg.image?\lambda\in[0,1]" alt="\lambda\in[0,1]" height="16"> controls the balance between data-driven learning and physics-informed regularization.

The physics loss combines two complementary PDE residuals:

```math
\mathcal{L}_{\mathrm{phys}}
=
\mathcal{L}_{\mathrm{strong}}
+
\mathcal{L}_{\mathrm{weak}}.
```

This strong--weak coupling provides two different views of the same governing law: the strong form enforces local differential consistency, while the weak form enforces variational consistency over finite elements.

### Strong-form residual

In the **strong-form** part, the PDE residual is evaluated on the structured nodal solution field. Spatial derivatives are computed using finite-difference differential operators, and the one-step residual is written as

```math
R_{\mathrm{FD}}
=
\frac{
u_{\theta}^{n+1}
-
u^{n}
}{
\Delta t
}
-
\mathcal{N}
\left(
u_{\theta}^{n+1}
\right),
```

where <img src="https://latex.codecogs.com/svg.image?\mathcal{N}(\cdot)" alt="\mathcal{N}(\cdot)" height="18"> denotes the spatial PDE operator.

The strong-form loss minimizes the domain-integrated squared residual. In the implementation, each cubic grid cell is decomposed into tetrahedra, and the residual integral is evaluated using tetrahedral quadrature:

```math
\mathcal{L}_{\mathrm{strong}}
\approx
\int_{\Omega}
R_{\mathrm{FD}}^2
\,d\mathbf{x}.
```

This term directly penalizes local PDE imbalance and is sensitive to sharp gradients, interfacial motion, and high-frequency residual errors.

### Weak-form residual

In the **weak-form** part, the PDE is enforced in a variational sense. The residual is multiplied by finite-element test functions and integrated over the domain:

```math
\int_{\Omega}
\left(
u_t
-
\mathcal{N}(u)
\right)
v
\,d\Omega
=
0.
```

The domain is decomposed into tetrahedral elements, and the weak residual is assembled using P1 tetrahedral finite-element basis functions. For each tetrahedron <img src="https://latex.codecogs.com/svg.image?K" alt="K" height="16"> and each local test function, an element-wise residual <img src="https://latex.codecogs.com/svg.image?r_i^K" alt="r_i^K" height="18"> is computed. The weak-form loss minimizes the mean squared residual over all tetrahedra and local basis functions:

```math
\mathcal{L}_{\mathrm{weak}}
=
\frac{1}{N_e}
\sum_{K}
\sum_{i}
\left(
r_i^K
\right)^2.
```

This variational formulation improves global physical consistency, numerical conditioning, and compatibility with homogeneous Neumann boundary conditions.

This dual-residual formulation establishes a complementary multi-objective physics constraint, where the strong form enforces local differential consistency and the weak form promotes element-wise variational consistency within a unified training objective.



---

## Supported benchmark problems

The current implementation supports three 3D non-periodic phase-field equations:

- **AC3D**: Allen--Cahn equation
- **CH3D**: Cahn--Hilliard equation
- **SH3D**: Swift--Hohenberg equation

The active problem is selected in `config.py`:

```python
PROBLEM = 'AC3D'   # Options: 'AC3D', 'CH3D', 'SH3D'
MODEL = 'HAMNO3d'
```

---

## Repository structure

```text
HAMNO/
├── config.py              # Main configuration file
├── main.py                # Training and evaluation entry point
├── Trainer.py             # Data loading, training, rollout, and plotting utilities
├── functions.py           # Physics losses and numerical operators
├── networks.py            # Neural-operator architectures
├── requirements.txt       # Required Python packages
├── HAMNO_Architecture/    # Architecture figures
├── Results/               # Result figures and outputs
└── Trained_Models/        # Saved model checkpoints
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/MBamdad/HAMNO.git
cd HAMNO
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

If you are using a GPU, make sure that your PyTorch installation matches your CUDA version.

---

## How to run

First, edit `config.py` and select the desired problem and model:

```python
PROBLEM = 'AC3D'
MODEL = 'HAMNO3d'
```

Then run:

```bash
python main.py
```

The code will:

1. Load the selected 3D phase-field dataset.
2. Build the selected neural-operator model.
3. Train the model using the configured loss setting.
4. Evaluate long-horizon autoregressive rollout.
5. Save results and trained checkpoints.

---

## Data path

The dataset path is set in `config.py`:

```python
MAT_DATA_PATH = "path/to/your/data.mat"
```

Each `.mat` file should contain the phase-field trajectories used for training and testing.

---

## Key features

- Hierarchical adaptive multi-scale neural operator
- Local convolutional and global spectral operator coupling
- Data-dependent local--global gating
- Encoder--decoder structure with skip connections
- Long-horizon autoregressive rollout
- Strong-form and weak-form physics-informed training
- Support for non-periodic 3D phase-field dynamics
- Applications to AC, CH, and SH equations

---

## Key achievements

HAMNO and PI-HAMNO are designed to improve:

- Long-time rollout accuracy
- Stability of autoregressive prediction
- Representation of local interfaces and global dynamics
- Physical consistency under non-periodic boundary conditions
- Data efficiency in limited-training-data regimes
- Robustness for multi-scale nonlinear phase-field systems

---

## Citation

If you use this repository, please cite the related HAMNO manuscript:

```bibtex
@article{bamdad2026hamno,
  title={HAMNO: A Hierarchical Adaptive Multi-scale Neural Operator with Physics-Informed Learning for Dynamical Systems},
  author={Bamdad, Mostafa and Eshaghi, Mohammad Sadegh and Rabczuk, Timon},
  journal={Transactions on Mathematical Sciences and Computational Engineering},
  year={2026}
}
```

---

## Contact

For questions, please contact:

**Mostafa Bamdad**  
Bauhaus-Universität Weimar  
mostafa.bamdad@uni-weimar.de
