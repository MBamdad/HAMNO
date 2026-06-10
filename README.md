# HAMNO

**HAMNO** is a Hierarchical Adaptive Multi-scale Neural Operator for learning long-time solution operators of nonlinear three-dimensional phase-field dynamical systems. The model is designed for non-periodic PDEs with homogeneous Neumann boundary conditions and is evaluated on the Allen--Cahn, Cahn--Hilliard, and Swift--Hohenberg equations.

<p align="center">
  <img src="HAMNO_Architecture/HAMNO_Architecture.png" width="850">
</p>

<p align="center">
  <b>Schematic overview of HAMNO and its physics-informed extension PI-HAMNO.</b>
</p>

---

## Overview

HAMNO learns the one-step solution operator

\[
\mathcal{G}_{\theta}:
\left(u^{n-T_{\mathrm{in}}+1}, \ldots, u^{n}\right)
\mapsto u_{\theta}^{n+1},
\]

where a short temporal history of 3D phase fields is used to predict the next state. During evaluation, the learned one-step map is applied autoregressively to reconstruct the full trajectory.

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

The central innovation is an **adaptive local--global gating mechanism**. Instead of adding local and global features with fixed weights, HAMNO learns data-dependent weights at each spatial location:

\[
h_{\mathrm{fused}}(\mathbf{x})
=
\alpha(\mathbf{x}) h_{\mathrm{local}}(\mathbf{x})
+
\beta(\mathbf{x}) h_{\mathrm{global}}(\mathbf{x}).
\]

This allows the model to decide how much local or global information is needed depending on the evolving solution field.

---

## Difference from related neural operators

HAMNO differs from common neural-operator baselines in the following way:

| Model | Main feature | Limitation addressed by HAMNO |
|---|---|---|
| FNO | Global Fourier operator | Limited recovery of localized and high-frequency structures |
| F-FNO | Factorized spectral operators | More efficient spectral mixing, but still limited adaptive multi-scale fusion |
| DeepONet | Branch--trunk operator representation | Separates input and coordinate encoding but lacks hierarchical local--global fusion |
| U-Net | Local encoder--decoder | Strong local representation but no explicit global spectral operator |
| U-NO | U-shaped neural operator | Multi-resolution structure but mainly spectral-driven |
| U-FNO | Fourier layers with U-Net components | Uses fixed additive local--global fusion |
| **HAMNO** | Adaptive local--global hierarchical operator | Learns when and where to use local, global, and multi-scale information |

In short, HAMNO combines the strengths of Fourier neural operators, convolutional models, and U-shaped architectures, while replacing fixed fusion by adaptive local--global operator coupling.

---

## Physics-informed extension: PI-HAMNO

PI-HAMNO extends HAMNO by adding physics-based regularization during training. The total loss is

\[
\mathcal{L}_{\mathrm{total}}
=
(1-\lambda)\mathcal{L}_{\mathrm{data}}
+
\lambda \mathcal{L}_{\mathrm{phys}},
\]

where \(\lambda\) controls the balance between data fitting and physics enforcement.

The physics loss combines two complementary PDE constraints:

\[
\mathcal{L}_{\mathrm{phys}}
=
\mathcal{L}_{\mathrm{strong}}
+
\mathcal{L}_{\mathrm{weak}}.
\]

### Strong-form residual

The strong-form loss directly penalizes the PDE residual in physical coordinates. Spatial derivatives are evaluated using finite-difference operators, and the squared residual is integrated over the 3D domain.

This term improves local PDE consistency and is sensitive to sharp gradients, interfaces, and high-frequency errors.

### Weak-form residual

The weak-form loss enforces the PDE in a variational sense. The residual is multiplied by finite-element test functions, integrated over tetrahedral elements, and evaluated using centroid-based quadrature.

This term improves global physical consistency, numerical conditioning, and compatibility with homogeneous Neumann boundary conditions.

Together, the strong and weak residuals provide complementary physical constraints: the strong form controls local differential errors, while the weak form improves global variational consistency.

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
