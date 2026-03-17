# Vlasov Latent Dynamics with PDE-FIND

Discovery of latent dynamical models for variables learned from Vlasov-Poisson plasma simulations using autoencoder compression, including PDE-FIND, DeepMoD sparse regression, and dense linear transport fits.

## Overview

This project applies machine learning to discover low-dimensional latent dynamics governing the Vlasov-Poisson system, specifically the two-stream instability. The workflow consists of three main stages:

1. **Compression**: Train a convolutional autoencoder on velocity distribution functions `f(x, v, t)` to learn a low-dimensional latent representation `z_k(x, t)` for each spatial point.

2. **Discovery**: Fit latent dynamics with PDE-FIND, with DeepMoD sparse regression, or with a dense linear transport model.

3. **Analysis**: Examine the discovered equations to understand the essential physics captured in the latent space.

## Project Structure

```
PDE_find/
├── Conv_velocity_AE.py          # Convolutional autoencoder training
├── latent_dynamics.py           # Full PDE-FIND discovery with flexible library
├── latent_dynamics_compact.py   # Compact discovery with fixed coupled library
├── latent_dynamics_linear.py    # Dense fit of Z_t = c + A Z + B Z_x
├── latent_dynamics_deepymod.py  # DeepMoD-based sparse discovery
├── latent_dynamics_pdenet.py    # PDE-Net-style discovery with learned derivative kernels
├── simulate_latent_pde_rk45.py  # Integrate learned latent dynamics and decode
├── plot_latent_3d.py            # 3D visualization of latent modes
├── reconstruct_e_field.py       # Electric field reconstruction from distributions
├── generate_all_e_fields.py     # Batch electric field generation
├── tutorials/
│   └── PDE_FIND.py              # Reference PDE-FIND implementation
├── vlasov_twostream_param_grid/ # Parameter grid of simulation data
└── results/                     # Output directory for latent dynamics
```

## Data

The project uses numerical solutions of the Vlasov-Poisson system for the two-stream instability:

- **Source**: `vlasov_twostream_param_grid/` contains 443 simulation cases
- **Each case** (`T_{T}_k_{k}/`):
  - `distribution_full.npz`: Distribution function `f(x, v, t)`
  - `electric_field_full.npz`: Electric field `E(x, t)`
- **Grid**: Varies temperature `T` and wavenumber `k` parameters

### Data Format

```python
# distribution_full.npz
{
    'f': (nt, nx, nv) array,  # Distribution function
    't': (nt,) array,          # Time points
    'x': (nx,) array,          # Spatial grid
    'v': (nv,) array,          # Velocity grid
    ...
}

# electric_field_full.npz
{
    'E': (nt, nx) array,       # Electric field
    't': (nt,) array,          # Time points
    'x': (nx,) array,          # Spatial grid
}
```

## Workflow

### 1. Train Autoencoder

**Script**: `Conv_velocity_AE.py`

Train a 1D convolutional autoencoder that compresses the velocity dimension:

```bash
# Basic usage
python Conv_velocity_AE.py --case-dir vlasov_twostream_param_grid/T_1.00_k_1.00

# With custom latent dimension
python Conv_velocity_AE.py --case-dir vlasov_twostream_param_grid/T_1.00_k_1.00 --latent-dim 8

# Full options
python Conv_velocity_AE.py \
    --case-dir vlasov_twostream_param_grid/T_1.00_k_1.00 \
    --latent-dim 8 \
    --hidden-dim 64 \
    --conv-channels 8,16,32 \
    --kernel-size 5 \
    --epochs 50 \
    --batch-size 1024 \
    --learning-rate 1e-3
```

**Output**: `velocity_autoencoder_results.npz` containing:
- `latent`: `(nt, nx, nz)` array of latent modes
- `t`, `x`: time and space grids
- `case_name`, `case_dir`, `nt`, `nx`, `nz`: metadata

### 2. Discover Latent Dynamics

Four modes of operation are available:

#### A. Full Library Discovery (`latent_dynamics.py`)

Flexible library construction with polynomial terms, spatial derivatives, and optional reciprocal features:

```bash
# Independent mode: discover separate PDE for each latent mode
python latent_dynamics.py --latent-file results/velocity_autoencoder_results.npz --system independent

# Coupled mode: discover coupled PDE system
python latent_dynamics.py --latent-file results/velocity_autoencoder_results.npz --system coupled

# Without electric field (latent-only dynamics)
python latent_dynamics.py --latent-file results/velocity_autoencoder_results.npz --no-electric-field

# With reciprocal features
python latent_dynamics.py --latent-file results/velocity_autoencoder_results.npz --include-reciprocal

# Custom PDE-FIND parameters
python latent_dynamics.py \
    --latent-file results/velocity_autoencoder_results.npz \
    --D 2 \                    # Max spatial derivative order
    --P 2 \                    # Max polynomial power
    --time-diff FD \           # Time differentiation method
    --space-diff FD \          # Space differentiation method
    --lam 1e-2 \               # Ridge regression parameter
    --d-tol 0.5                # Tolerance increment for STRidge
```

**Output**:
- `<latent-file-stem>_pde_find.npz`: Discovered PDE coefficients and metrics
- `<latent-file-stem>_pde_find.txt`: Human-readable report

#### B. Dense Linear Transport Fit (`latent_dynamics_linear.py`)

Fit the coupled model
`Z_t = c + A Z + B Z_x`
with a single ridge regression over all latent modes:

```bash
# Coupled linear transport without constant forcing
python latent_dynamics_linear.py --latent-file results/velocity_autoencoder_results.npz

# Add a constant term and adjust the ridge strength
python latent_dynamics_linear.py \
    --latent-file results/velocity_autoencoder_results.npz \
    --include-constant \
    --ridge-alpha 1e-5 \
    --space-diff Fourier \
    --time-diff FD
```

**Output**:
- `<latent-file-stem>_linear_transport.npz`: Dense linear coefficients and metrics
- `<latent-file-stem>_linear_transport.txt`: Human-readable report

#### C. Compact Library Discovery (`latent_dynamics_compact.py`)

Fixed library with physically-motivated terms for coupled latent-E-field dynamics:

```bash
# Discover with compact library [z_i, z_{i,x}, E, z_i*E, z_i*z_{i,x}]
python latent_dynamics_compact.py --latent-file results/velocity_autoencoder_results.npz

# Analyze specific modes only
python latent_dynamics_compact.py --latent-file results/velocity_autoencoder_results.npz --modes 0 1 2

# Custom PDE-FIND parameters
python latent_dynamics_compact.py \
    --latent-file results/velocity_autoencoder_results.npz \
    --lam 1e-2 \
    --d-tol 0.5 \
    --time-diff FD \
    --space-diff FD
```

**Output**:
- `<latent-file-stem>_compact_pde_find.npz`: Discovered PDE coefficients
- `<latent-file-stem>_compact_pde_find.txt`: Human-readable report

#### D. PDE-Net-Style Discovery (`latent_dynamics_pdenet.py`)

Inspired by `PDE-Net-2.0`, this script learns spatial derivative filters with 1D circular convolutions and uses a SymNet right-hand side. For interpretability it also fits a post-hoc sparse polynomial proxy to the learned SymNet dynamics.

```bash
# Coupled latent PDE-Net fit on all modes
python latent_dynamics_pdenet.py \
    --latent-file results/velocity_autoencoder_results.npz

# Restrict to a few modes and use a smaller library
python latent_dynamics_pdenet.py \
    --latent-file results/velocity_autoencoder_results.npz \
    --modes 0 1 2 \
    --diff-order 1 \
    --poly-order 2 \
    --kernel-size 5
```

**Output**:
- `<latent-file-stem>_pdenet.npz`: Learned kernels, SymNet parameters, symbolic proxy coefficients, and latent rollout prediction
- `<latent-file-stem>_pdenet.txt`: Human-readable report with a post-hoc symbolic proxy of the learned SymNet

### 3. Decode and Evaluate Learned Dynamics

Use `simulate_latent_pde_rk45.py` to decode discovered latent trajectories back to `f(x, v, t)` and compare them to the Vlasov truth. The same script supports PDE-FIND, DeepMoD, linear transport, and PDE-Net-style outputs.

```bash
# Evaluate a PDE-Net latent model and decode it with the trained autoencoder
python simulate_latent_pde_rk45.py \
    --pde-file results/velocity_autoencoder_results_pdenet.npz \
    --device cpu
```

### 4. Visualize Results

**Script**: `plot_latent_3d.py`

```bash
# Plot all latent modes
python plot_latent_3d.py --latent-file results/velocity_autoencoder_results.npz

# Plot specific modes
python plot_latent_3d.py --latent-file results/velocity_autoencoder_results.npz --modes 0 1 2

# Include electric field in visualization
python plot_latent_3d.py --latent-file results/velocity_autoencoder_results.npz --electric-file path/to/electric_field_full.npz

# Custom visualization parameters
python plot_latent_3d.py \
    --latent-file results/velocity_autoencoder_results.npz \
    --t-stride 10 \
    --x-stride 2 \
    --cmap viridis \
    --elev 30 \
    --azim -130
```

## Key Features

### Autoencoder Architecture

- **Encoder**: 1D convolutions along velocity dimension with channel expansion
- **Bottleneck**: Linear compression to latent dimension `N_z`
- **Decoder**: Transposed convolutions to reconstruct velocity distribution
- **Loss**: Mean squared error on velocity distribution reconstruction

### PDE-FIND Integration

The project uses a modified version of the PDE-FIND algorithm (from `tutorials/PDE_FIND.py`) with:

- **Differentiation methods**: Finite difference, polynomial, Tikhonov, Fourier, central difference
- **Library construction**: Polynomial terms up to order `P`, spatial derivatives up to order `D`
- **Sparse regression**: STRidge (sequential thresholded ridge regression)
- **Complex support**: Handles complex-valued latent variables

### Library Options

1. **Independent mode**: Each latent mode evolves independently
   - Library: `[z_k, z_k^x, z_k^xx, ..., E, E^x, ...]`
   - Equation: `∂z_k/∂t = L[z_k, E]`

2. **Coupled mode**: Latent modes interact with each other
   - Library: `[z_i, z_i^x, z_i*z_j, ..., E, E^x, ...]` for all modes
   - Equation: `∂z_k/∂t = L[z_0, ..., z_{N_z-1}, E]`

3. **Compact mode**: Physically-motivated fixed library
   - Library: `[z_i, z_i^x, E, z_i*E, z_i*z_i^x]` for all modes
   - Equation: `∂z_k/∂t = L[z, z_x, E, z*E, z*z_x]`

4. **Linear transport mode**: Dense coupled regression
   - Library: `[1, z_0, ..., z_{N_z-1}, z_{0,x}, ..., z_{N_z-1,x}]`
   - Equation: `Z_t = c + A Z + B Z_x`
   - Solver: ridge regression on the full matrix system

5. **PDE-Net-style mode**: Learned derivative kernels plus sparse symbolic library
   - Library: polynomial combinations of `[z_i, z_{i,x}, z_{i,xx}, ...]`
   - Equation: `Z_t = L_theta[Z, Z_x, Z_xx, ...]`
   - Solver: Adam on derivative filters and coefficients, followed by sparse ridge refit

## Output Format

### PDE-FIND Results `.npz`

```python
{
    # Source data
    'latent_file': str,           # Path to latent file
    'electric_file': str,         # Path to electric field file
    'used_electric_field': bool,  # Whether E was included in library

    # Grid
    't': (nt,) array,             # Time grid
    'x': (nx,) array,             # Space grid
    'mode_indices': (nz,) array,  # Mode indices analyzed

    # Discovered equations
    'coefficients_real': (nz, n_terms) array,  # Real parts of coefficients
    'coefficients_imag': (nz, n_terms) array,  # Imaginary parts of coefficients
    'equations': (nz,) array,     # String representations of PDEs
    'library_description': (n_terms,) array,   # RHS term descriptions

    # Metrics
    'residual_l2': (nz,) array,   # L2 norm of residuals
    'relative_residual_l2': (nz,) array,  # Relative residuals
    'nonzero_terms': (nz,) array, # Number of nonzero coefficients

    # Parameters
    'D': int,                     # Max spatial derivative order
    'P': int,                     # Max polynomial power
    'dt': float,                  # Time step
    'dx': float,                  # Space step
    'time_diff': str,             # Time differentiation method
    'space_diff': str,            # Space differentiation method
    'lam': float,                 # Ridge parameter
    'd_tol': float,               # Tolerance increment
    'l0_penalty': float,          # L0 penalty (optional)
    'maxit': int,                 # Max iterations
    'str_iters': int,             # STRidge iterations
    'normalize': int,             # Normalization option
    'split': float,               # Train/validation split
}
```

## Example Results

Typical discovered latent dynamics show:

- **Latent modes** capturing coherent structures in phase space
- **Coupling to electric field** through linear and nonlinear terms
- **Spatial derivatives** representing wave propagation and dispersion
- **Sparse representations** with 3-7 nonzero terms per equation

## Dependencies

- Python 3.9+
- NumPy
- PyTorch (for autoencoder training)
- Matplotlib (for visualization)

## Notes

- The autoencoder is trained on velocity distributions at each `(x, t)` point independently
- Latent variables can be complex-valued if the distribution function is complex
- PDE-FIND requires uniformly spaced time and space grids
- Electric field data can be loaded from file or reconstructed from distributions
- The compact library mode requires electric field data

## References

- PDE-FIND: Rudy, S. H., et al. "Data-driven discovery of partial differential equations." (2017)
- Vlasov-Poisson: Standard model for collisionless plasma dynamics
- Two-stream instability: Classic plasma instability with rich nonlinear dynamics
