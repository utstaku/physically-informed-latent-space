#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import operator
import random
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from Conv_velocity_AE import (
    ConvVelocityAutoencoder,
    build_dataloader,
    evaluate_loss,
    extract_state_dict_numpy,
    flatten_snapshots,
    load_distribution,
    normalize_from_train,
    parse_conv_channels,
    reconstruction_metrics,
    resolve_case_dir,
    reshape_sample_grid,
    split_indices,
)
from latent_dynamics import (
    check_uniform_spacing,
    discover_coupled_mode_pde,
    discover_mode_pde,
    patch_pde_find_compatibility,
    resolve_electric_data,
    resolve_modes,
)

try:
    import torch
    from torch import nn
    import tutorials.PDE_FIND as pde_find
except ImportError as exc:
    raise SystemExit(
        "PyTorch and the local PDE-FIND tutorial module are required. "
        "Run this script in the 'lasdi' environment, for example:\n"
        "  conda run -n lasdi python joint_ae_latent_pdefind.py --case-dir <case>"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the convolutional velocity autoencoder while simultaneously enforcing "
            "a sparse latent PDE constraint using STRidge, following latent_dynamics.py."
        )
    )
    parser.add_argument(
        "--grid-dir",
        type=Path,
        default=Path("vlasov_twostream_param_grid"),
        help="Directory that contains all parameter-grid cases.",
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=None,
        help="Specific case directory. If omitted, the first case in sorted order is used.",
    )
    parser.add_argument("--latent-dim", type=int, default=8, help="Latent size Nz for each x cell.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden width of the AE bottleneck MLP.")
    parser.add_argument(
        "--conv-channels",
        type=str,
        default="8,16,32",
        help="Comma-separated Conv1d channel widths used along the v direction.",
    )
    parser.add_argument("--kernel-size", type=int, default=5, help="Odd kernel size used in all conv layers.")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Mini-batch size over flattened (t, x) samples for the reconstruction update.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="plateau",
        choices=("none", "plateau"),
        help="Learning-rate scheduler. 'plateau' reduces lr when validation loss stalls.",
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=5,
        help="Epochs to wait before reducing lr when using plateau scheduler.",
    )
    parser.add_argument("--lr-factor", type=float, default=0.5, help="Multiplicative lr decay factor.")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Lower bound for the learning rate.")
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.9,
        help="Fraction of flattened samples used for reconstruction training/normalization.",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="independent",
        choices=("independent", "coupled"),
        help="Whether to fit one PDE per latent mode or a coupled latent PDE system.",
    )
    parser.add_argument(
        "--modes",
        type=int,
        nargs="*",
        default=None,
        help="Latent mode indices constrained by the PDE loss. Default is all modes.",
    )
    parser.add_argument(
        "--electric-file",
        type=Path,
        default=None,
        help="Optional electric_field_full.npz. If omitted, the script tries to infer it from the case.",
    )
    parser.add_argument(
        "--no-electric-field",
        action="store_true",
        help="Disable electric-field loading entirely and use a latent-only PDE library.",
    )
    parser.add_argument(
        "--include-reciprocal",
        action="store_true",
        help="Augment the PDE library with reciprocal features such as 1/z_k and 1/E.",
    )
    parser.add_argument(
        "--reciprocal-eps",
        type=float,
        default=1e-6,
        help="Small positive regularization used in reciprocal features to avoid division by zero.",
    )
    parser.add_argument("--D", type=int, default=1, help="Maximum spatial derivative order in the PDE library.")
    parser.add_argument("--P", type=int, default=1, help="Maximum polynomial power in the PDE library.")
    parser.add_argument(
        "--time-diff",
        type=str,
        default="FD",
        choices=("FD",),
        help="Joint training uses differentiable finite differences in time.",
    )
    parser.add_argument(
        "--space-diff",
        type=str,
        default="FD",
        choices=("FD",),
        help="Joint training uses differentiable finite differences in space.",
    )
    parser.add_argument("--lam", type=float, default=1e-2, help="Ridge parameter for STRidge.")
    parser.add_argument("--d-tol", type=float, default=0.5, help="Tolerance increment for STRidge.")
    parser.add_argument(
        "--l0-penalty",
        type=float,
        default=None,
        help="Optional penalty on the number of nonzero terms used by TrainSTRidge.",
    )
    parser.add_argument("--maxit", type=int, default=25, help="Max tolerance-search iterations for STRidge.")
    parser.add_argument("--str-iters", type=int, default=10, help="Inner STRidge iterations.")
    parser.add_argument("--normalize", type=int, default=2, help="Normalization passed to TrainSTRidge.")
    parser.add_argument("--split", type=float, default=0.8, help="Train/validation split passed to TrainSTRidge.")
    parser.add_argument("--width-x", type=int, default=None, help="Unused for FD, kept for latent_dynamics.py parity.")
    parser.add_argument("--width-t", type=int, default=None, help="Unused for FD, kept for latent_dynamics.py parity.")
    parser.add_argument("--deg-x", type=int, default=5, help="Unused for FD, kept for latent_dynamics.py parity.")
    parser.add_argument("--deg-t", type=int, default=None, help="Unused for FD, kept for latent_dynamics.py parity.")
    parser.add_argument("--sigma", type=float, default=2.0, help="Unused for FD, kept for latent_dynamics.py parity.")
    parser.add_argument("--sg-window-x", type=int, default=7, help="Unused for FD, kept for latent_dynamics.py parity.")
    parser.add_argument("--sg-window-t", type=int, default=7, help="Unused for FD, kept for latent_dynamics.py parity.")
    parser.add_argument("--sg-poly-x", type=int, default=3, help="Unused for FD, kept for latent_dynamics.py parity.")
    parser.add_argument("--sg-poly-t", type=int, default=3, help="Unused for FD, kept for latent_dynamics.py parity.")
    parser.add_argument(
        "--print-best-tol",
        action="store_true",
        help="Print PDE-FIND's best tolerance during STRidge search.",
    )
    parser.add_argument(
        "--pde-weight",
        type=float,
        default=1e-2,
        help="Weight applied to the differentiable latent PDE residual.",
    )
    parser.add_argument(
        "--full-recon-weight",
        type=float,
        default=1.0,
        help="Reconstruction weight used during the full-grid PDE update step.",
    )
    parser.add_argument(
        "--pde-warmup-epochs",
        type=int,
        default=10,
        help="Number of initial epochs trained with reconstruction loss only.",
    )
    parser.add_argument(
        "--pde-fit-every",
        type=int,
        default=1,
        help="How often to refit STRidge coefficients from the current latent trajectory.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for initialization and data split.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Torch device selection.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npz path. Defaults to <case-dir>/conv_velocity_autoencoder_joint_pdefind.npz.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Human-readable text summary. Defaults to <output-stem>.txt.",
    )
    parser.add_argument(
        "--save-reconstruction",
        action="store_true",
        help="Store the final decoded reconstruction f(t, x, v) in the output file.",
    )
    return parser.parse_args()


def cuda_is_available() -> bool:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="CUDA initialization: Unexpected error from cudaGetDeviceCount.*",
            category=UserWarning,
        )
        return torch.cuda.is_available()


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not cuda_is_available():
            raise RuntimeError("CUDA was requested but is not available in the current lasdi environment.")
        return torch.device("cuda")
    if cuda_is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_is_available():
        torch.cuda.manual_seed_all(seed)


def infer_output_path(case_dir: Path) -> Path:
    return case_dir / "conv_velocity_autoencoder_joint_pdefind.npz"


def infer_report_path(output_path: Path) -> Path:
    return output_path.with_suffix(".txt")


def save_report(report_path: Path, lines: Iterable[str]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def encode_grid(model: ConvVelocityAutoencoder, samples_grid: torch.Tensor) -> torch.Tensor:
    nt, nx, nv = samples_grid.shape
    flattened = samples_grid.reshape(nt * nx, nv).unsqueeze(1)
    latent = model.encode(flattened)
    return latent.reshape(nt, nx, model.latent_dim)


def decode_grid(model: ConvVelocityAutoencoder, latent_grid: torch.Tensor) -> torch.Tensor:
    nt, nx, nz = latent_grid.shape
    decoded = model.decode(latent_grid.reshape(nt * nx, nz)).squeeze(1)
    return decoded.reshape(nt, nx, model.input_dim)


def finite_difference_last(u: torch.Tensor, dx: float, order: int) -> torch.Tensor:
    n = u.shape[-1]
    if order <= 0:
        raise ValueError(f"Derivative order must be positive, got {order}")
    if n < 3:
        raise ValueError(f"At least 3 points are required for finite differences, got n={n}")

    ux = torch.zeros_like(u)
    if order == 1:
        ux[..., 1:-1] = (u[..., 2:] - u[..., :-2]) / (2.0 * dx)
        ux[..., 0] = (-1.5 * u[..., 0] + 2.0 * u[..., 1] - 0.5 * u[..., 2]) / dx
        ux[..., -1] = (1.5 * u[..., -1] - 2.0 * u[..., -2] + 0.5 * u[..., -3]) / dx
        return ux

    if order == 2:
        if n < 4:
            raise ValueError(f"At least 4 points are required for second derivatives, got n={n}")
        ux[..., 1:-1] = (u[..., 2:] - 2.0 * u[..., 1:-1] + u[..., :-2]) / (dx**2)
        ux[..., 0] = (2.0 * u[..., 0] - 5.0 * u[..., 1] + 4.0 * u[..., 2] - u[..., 3]) / (dx**2)
        ux[..., -1] = (2.0 * u[..., -1] - 5.0 * u[..., -2] + 4.0 * u[..., -3] - u[..., -4]) / (dx**2)
        return ux

    if order == 3:
        if n < 6:
            raise ValueError(f"At least 6 points are required for third derivatives, got n={n}")
        ux[..., 2:-2] = (
            0.5 * u[..., 4:]
            - u[..., 3:-1]
            + u[..., 1:-3]
            - 0.5 * u[..., :-4]
        ) / (dx**3)
        ux[..., 0] = (
            -2.5 * u[..., 0]
            + 9.0 * u[..., 1]
            - 12.0 * u[..., 2]
            + 7.0 * u[..., 3]
            - 1.5 * u[..., 4]
        ) / (dx**3)
        ux[..., 1] = (
            -2.5 * u[..., 1]
            + 9.0 * u[..., 2]
            - 12.0 * u[..., 3]
            + 7.0 * u[..., 4]
            - 1.5 * u[..., 5]
        ) / (dx**3)
        ux[..., -1] = (
            2.5 * u[..., -1]
            - 9.0 * u[..., -2]
            + 12.0 * u[..., -3]
            - 7.0 * u[..., -4]
            + 1.5 * u[..., -5]
        ) / (dx**3)
        ux[..., -2] = (
            2.5 * u[..., -2]
            - 9.0 * u[..., -3]
            + 12.0 * u[..., -4]
            - 7.0 * u[..., -5]
            + 1.5 * u[..., -6]
        ) / (dx**3)
        return ux

    return finite_difference_last(finite_difference_last(u, dx, 3), dx, order - 3)


def finite_difference(tensor: torch.Tensor, dx: float, order: int, dim: int) -> torch.Tensor:
    moved = tensor.movedim(dim, -1)
    diffed = finite_difference_last(moved, dx=dx, order=order)
    return diffed.movedim(-1, dim)


def stabilized_reciprocal_torch(values: torch.Tensor, eps: float) -> torch.Tensor:
    if eps <= 0.0:
        raise ValueError(f"reciprocal-eps must be positive, got {eps}")
    sign = torch.where(values >= 0.0, torch.ones_like(values), -torch.ones_like(values))
    safe = torch.where(values.abs() >= eps, values, sign * eps)
    return 1.0 / safe


def flatten_fortran_order(field_x_t: torch.Tensor) -> torch.Tensor:
    return field_x_t.transpose(0, 1).reshape(-1, 1)


def build_power_tuples(num_variables: int, max_degree: int) -> List[Tuple[int, ...]]:
    powers: List[Tuple[int, ...]] = []
    for degree in range(1, max_degree + 1):
        size = num_variables + degree - 1
        for indices in itertools.combinations(range(size), num_variables - 1):
            starts = [0] + [index + 1 for index in indices]
            stops = list(indices) + [size]
            powers.append(tuple(map(operator.sub, stops, starts)))
    return powers


def build_theta_torch(
    library_fields: Sequence[Tuple[str, torch.Tensor]],
    dx: float,
    derivative_order: int,
    polynomial_order: int,
    include_reciprocal: bool,
    reciprocal_eps: float,
) -> Tuple[torch.Tensor, List[str]]:
    if not library_fields:
        raise ValueError("At least one library field is required.")

    sample_field = library_fields[0][1]
    num_rows = sample_field.shape[0] * sample_field.shape[1]
    dtype = sample_field.dtype
    device = sample_field.device

    data_columns: List[torch.Tensor] = []
    data_descriptions: List[str] = []
    derivative_columns: List[torch.Tensor] = [torch.ones((num_rows, 1), device=device, dtype=dtype)]
    derivative_descriptions = [""]

    for field_name, field_x_t in library_fields:
        data_columns.append(flatten_fortran_order(field_x_t))
        data_descriptions.append(field_name)
        if include_reciprocal:
            reciprocal = stabilized_reciprocal_torch(field_x_t, reciprocal_eps)
            data_columns.append(flatten_fortran_order(reciprocal))
            data_descriptions.append(f"inv_{field_name}")

        for order in range(1, derivative_order + 1):
            derivative = finite_difference(field_x_t, dx=dx, order=order, dim=0)
            derivative_columns.append(flatten_fortran_order(derivative))
            derivative_descriptions.append(f"{field_name}_{{{'x' * order}}}")

    data_matrix = torch.cat(data_columns, dim=1)
    derivatives_matrix = torch.cat(derivative_columns, dim=1)
    powers = build_power_tuples(data_matrix.shape[1], polynomial_order)

    theta_columns: List[torch.Tensor] = [torch.ones((num_rows, 1), device=device, dtype=dtype)]
    descriptions = [""]

    for deriv_index in range(1, derivatives_matrix.shape[1]):
        theta_columns.append(derivatives_matrix[:, deriv_index : deriv_index + 1])
        descriptions.append(derivative_descriptions[deriv_index])

    for deriv_index in range(derivatives_matrix.shape[1]):
        deriv_column = derivatives_matrix[:, deriv_index : deriv_index + 1]
        for power in powers:
            new_column = deriv_column
            function_description = ""
            for data_index, exponent in enumerate(power):
                if exponent == 0:
                    continue
                new_column = new_column * (data_matrix[:, data_index : data_index + 1] ** exponent)
                if exponent == 1:
                    function_description += data_descriptions[data_index]
                else:
                    function_description += f"{data_descriptions[data_index]}^{exponent}"
            theta_columns.append(new_column)
            descriptions.append(function_description + derivative_descriptions[deriv_index])

    return torch.cat(theta_columns, dim=1), descriptions


def build_target_and_library_fields(
    latent_grid: torch.Tensor,
    target_mode: int,
    mode_indices: Sequence[int],
    electric_field: torch.Tensor | None,
    system: str,
) -> Tuple[torch.Tensor, List[Tuple[str, torch.Tensor]]]:
    target_field_x_t = latent_grid[:, :, target_mode].transpose(0, 1)
    if system == "coupled":
        library_fields = [(f"z{mode}", latent_grid[:, :, mode].transpose(0, 1)) for mode in mode_indices]
    else:
        library_fields = [("z", target_field_x_t)]

    if electric_field is not None:
        library_fields.append(("E", electric_field.transpose(0, 1)))
    return target_field_x_t, library_fields


def fit_sparse_pdes(
    latent: np.ndarray,
    modes: Sequence[int],
    electric_field: np.ndarray | None,
    dt: float,
    dx: float,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, List[str], List[str], Dict[str, float]]:
    coefficients: List[np.ndarray] = []
    equations: List[str] = []
    rhs_description_ref: List[str] | None = None
    residuals: List[float] = []
    relative_residuals: List[float] = []
    nonzero_terms: List[float] = []

    for mode in modes:
        if args.system == "coupled":
            weights, rhs_description, metrics, equation = discover_coupled_mode_pde(
                mode_index=mode,
                latent=latent,
                modes=modes,
                electric_field=electric_field,
                dt=dt,
                dx=dx,
                pde_find=pde_find,
                args=args,
            )
        else:
            weights, rhs_description, metrics, equation = discover_mode_pde(
                mode_index=mode,
                latent=latent,
                electric_field=electric_field,
                dt=dt,
                dx=dx,
                pde_find=pde_find,
                args=args,
            )

        weights_real = np.real_if_close(weights, tol=1_000)
        if np.iscomplexobj(weights_real):
            if np.max(np.abs(weights_real.imag)) > 1e-8:
                raise ValueError("Complex STRidge coefficients are not supported in the joint training loss.")
            weights_real = weights_real.real
        weights_real = np.asarray(weights_real, dtype=np.float64).reshape(-1)

        if rhs_description_ref is None:
            rhs_description_ref = list(rhs_description)
        elif list(rhs_description) != rhs_description_ref:
            raise RuntimeError("PDE library descriptions changed across modes, which should not happen.")

        coefficients.append(weights_real)
        equations.append(equation)
        residuals.append(float(metrics["residual_l2"]))
        relative_residuals.append(float(metrics["relative_residual_l2"]))
        nonzero_terms.append(float(metrics["nonzero_terms"]))

    if rhs_description_ref is None:
        raise RuntimeError("No PDE coefficients were fitted.")

    summary = {
        "mean_residual_l2": float(np.mean(residuals)),
        "mean_relative_residual_l2": float(np.mean(relative_residuals)),
        "mean_nonzero_terms": float(np.mean(nonzero_terms)),
    }
    return np.stack(coefficients, axis=0), rhs_description_ref, equations, summary


def compute_joint_pde_loss(
    latent_grid: torch.Tensor,
    mode_indices: Sequence[int],
    electric_field: torch.Tensor | None,
    coefficients: np.ndarray,
    rhs_description: Sequence[str],
    dt: float,
    dx: float,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    losses = []
    relative_losses = []
    nonzero_terms = []

    for row_index, mode in enumerate(mode_indices):
        target_field_x_t, library_fields = build_target_and_library_fields(
            latent_grid=latent_grid,
            target_mode=mode,
            mode_indices=mode_indices,
            electric_field=electric_field,
            system=args.system,
        )
        ut = flatten_fortran_order(finite_difference(target_field_x_t, dx=dt, order=1, dim=1))
        theta, theta_description = build_theta_torch(
            library_fields=library_fields,
            dx=dx,
            derivative_order=args.D,
            polynomial_order=args.P,
            include_reciprocal=args.include_reciprocal,
            reciprocal_eps=args.reciprocal_eps,
        )
        if list(theta_description) != list(rhs_description):
            raise RuntimeError("Torch PDE library term ordering does not match latent_dynamics.py.")

        weight_tensor = torch.as_tensor(coefficients[row_index], device=theta.device, dtype=theta.dtype).reshape(-1, 1)
        residual = ut - theta @ weight_tensor
        denom = ut.pow(2).mean().clamp_min(1e-12)
        normalized_loss = residual.pow(2).mean() / denom
        losses.append(normalized_loss)
        relative_losses.append(float(normalized_loss.detach().cpu().item()))
        nonzero_terms.append(float(np.count_nonzero(coefficients[row_index])))

    total_loss = torch.stack(losses).mean()
    return total_loss, {
        "relative_pde_loss": float(np.mean(relative_losses)),
        "mean_nonzero_terms": float(np.mean(nonzero_terms)),
    }


def train_joint_model(
    train_samples: np.ndarray,
    val_samples: np.ndarray,
    samples_grid: np.ndarray,
    latent_dim: int,
    hidden_dim: int,
    conv_channels: Sequence[int],
    kernel_size: int,
    dt: float,
    dx: float,
    electric_field: np.ndarray | None,
    mode_indices: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[
    ConvVelocityAutoencoder,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[int],
]:
    model = ConvVelocityAutoencoder(
        input_dim=train_samples.shape[1],
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        conv_channels=conv_channels,
        kernel_size=kernel_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )

    criterion = nn.MSELoss(reduction="mean")
    train_loader = build_dataloader(train_samples, batch_size=args.batch_size, shuffle=True)
    eval_train_loader = build_dataloader(train_samples, batch_size=args.batch_size, shuffle=False)
    val_loader = build_dataloader(val_samples, batch_size=args.batch_size, shuffle=False)

    samples_grid_tensor = torch.from_numpy(samples_grid).to(device=device, dtype=torch.float32)
    electric_field_tensor = (
        torch.from_numpy(electric_field.astype(np.float32)).to(device=device, dtype=torch.float32)
        if electric_field is not None
        else None
    )

    train_losses: List[float] = []
    val_losses: List[float] = []
    lr_history: List[float] = []
    pde_loss_history: List[float] = []
    pde_residual_history: List[float] = []
    pde_nonzero_history: List[float] = []
    anchor_recon_history: List[float] = []
    refit_epochs: List[int] = []

    current_coefficients: np.ndarray | None = None
    current_rhs_description: List[str] | None = None
    current_fit_summary: Dict[str, float] | None = None

    for epoch in range(args.epochs):
        model.train()
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            recon_loss = criterion(recon, batch)
            recon_loss.backward()
            optimizer.step()

        epoch_pde_loss = 0.0
        epoch_anchor_loss = 0.0
        epoch_relative_residual = float("nan")
        epoch_nonzero_terms = float("nan")

        if args.pde_weight > 0.0 and (epoch + 1) > args.pde_warmup_epochs:
            need_refit = current_coefficients is None or ((epoch + 1 - args.pde_warmup_epochs - 1) % args.pde_fit_every == 0)
            if need_refit:
                with torch.no_grad():
                    latent_grid_np = encode_grid(model, samples_grid_tensor).detach().cpu().numpy().astype(np.float64)
                current_coefficients, current_rhs_description, _equations, current_fit_summary = fit_sparse_pdes(
                    latent=latent_grid_np,
                    modes=mode_indices,
                    electric_field=electric_field.astype(np.float64) if electric_field is not None else None,
                    dt=dt,
                    dx=dx,
                    args=args,
                )
                refit_epochs.append(epoch + 1)

            optimizer.zero_grad(set_to_none=True)
            latent_grid = encode_grid(model, samples_grid_tensor)
            reconstruction_grid = decode_grid(model, latent_grid)
            pde_loss, pde_stats = compute_joint_pde_loss(
                latent_grid=latent_grid,
                mode_indices=mode_indices,
                electric_field=electric_field_tensor,
                coefficients=current_coefficients,
                rhs_description=current_rhs_description,
                dt=dt,
                dx=dx,
                args=args,
            )
            anchor_recon_loss = criterion(reconstruction_grid, samples_grid_tensor)
            total_joint_loss = args.pde_weight * pde_loss + args.full_recon_weight * anchor_recon_loss
            total_joint_loss.backward()
            optimizer.step()

            epoch_pde_loss = float(pde_loss.detach().cpu().item())
            epoch_anchor_loss = float(anchor_recon_loss.detach().cpu().item())
            epoch_relative_residual = (
                float(current_fit_summary["mean_relative_residual_l2"]) if current_fit_summary is not None else pde_stats["relative_pde_loss"]
            )
            epoch_nonzero_terms = (
                float(current_fit_summary["mean_nonzero_terms"]) if current_fit_summary is not None else pde_stats["mean_nonzero_terms"]
            )

        train_loss = evaluate_loss(model, eval_train_loader, device)
        val_loss = evaluate_loss(model, val_loader, device)
        if scheduler is not None:
            scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        lr_history.append(current_lr)
        pde_loss_history.append(epoch_pde_loss)
        pde_residual_history.append(epoch_relative_residual)
        pde_nonzero_history.append(epoch_nonzero_terms)
        anchor_recon_history.append(epoch_anchor_loss)

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs:03d} "
            f"train_mse={train_loss:.6e} val_mse={val_loss:.6e} "
            f"pde_loss={epoch_pde_loss:.6e} pde_rel={epoch_relative_residual:.6e} "
            f"anchor_recon={epoch_anchor_loss:.6e} lr={current_lr:.6e}"
        )

    return (
        model,
        np.asarray(train_losses, dtype=np.float32),
        np.asarray(val_losses, dtype=np.float32),
        np.asarray(lr_history, dtype=np.float32),
        np.asarray(pde_loss_history, dtype=np.float32),
        np.asarray(pde_residual_history, dtype=np.float32),
        np.asarray(pde_nonzero_history, dtype=np.float32),
        np.asarray(anchor_recon_history, dtype=np.float32),
        refit_epochs,
    )


def save_results(
    output_path: Path,
    case_dir: Path,
    t: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    latent: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    train_losses: np.ndarray,
    val_losses: np.ndarray,
    lr_history: np.ndarray,
    pde_loss_history: np.ndarray,
    pde_residual_history: np.ndarray,
    pde_nonzero_history: np.ndarray,
    anchor_recon_history: np.ndarray,
    refit_epochs: Sequence[int],
    coefficients: np.ndarray,
    rhs_description: Sequence[str],
    equations: Sequence[str],
    model: ConvVelocityAutoencoder,
    device: torch.device,
    args: argparse.Namespace,
    metrics: Dict[str, np.ndarray],
    electric_source: str | None,
    reconstruction: np.ndarray | None,
) -> None:
    payload: Dict[str, np.ndarray] = {
        "case_name": np.asarray(case_dir.name),
        "case_dir": np.asarray(str(case_dir)),
        "torch_device": np.asarray(str(device)),
        "model_type": np.asarray("conv_velocity_autoencoder_joint_pdefind"),
        "t": t.astype(np.float32),
        "x": x.astype(np.float32),
        "v": v.astype(np.float32),
        "nt": np.asarray(latent.shape[0], dtype=np.int32),
        "nx": np.asarray(latent.shape[1], dtype=np.int32),
        "nv": np.asarray(len(v), dtype=np.int32),
        "nz": np.asarray(latent.shape[2], dtype=np.int32),
        "latent": latent.astype(np.float32),
        "feature_mean": mean.astype(np.float32),
        "feature_std": std.astype(np.float32),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "learning_rate_history": lr_history,
        "pde_loss_history": pde_loss_history,
        "pde_relative_residual_history": pde_residual_history,
        "pde_nonzero_terms_history": pde_nonzero_history,
        "anchor_reconstruction_history": anchor_recon_history,
        "pde_refit_epochs": np.asarray(list(refit_epochs), dtype=np.int32),
        "coefficients": coefficients.astype(np.float64),
        "rhs_description": np.asarray(list(rhs_description)),
        "equations": np.asarray(list(equations)),
        "mode_indices": np.asarray(list(resolve_modes(args.modes, latent.shape[2])), dtype=np.int32),
        "conv_channels": np.asarray(model.conv_channels, dtype=np.int32),
        "kernel_size": np.asarray(model.kernel_size, dtype=np.int32),
        "hidden_dim": np.asarray(model.hidden_dim, dtype=np.int32),
        "system": np.asarray(args.system),
        "D": np.asarray(args.D, dtype=np.int32),
        "P": np.asarray(args.P, dtype=np.int32),
        "lam": np.asarray(args.lam, dtype=np.float64),
        "d_tol": np.asarray(args.d_tol, dtype=np.float64),
        "l0_penalty": np.asarray(np.nan if args.l0_penalty is None else args.l0_penalty, dtype=np.float64),
        "normalize": np.asarray(args.normalize, dtype=np.int32),
        "split": np.asarray(args.split, dtype=np.float64),
        "include_reciprocal": np.asarray(args.include_reciprocal),
        "reciprocal_eps": np.asarray(args.reciprocal_eps, dtype=np.float64),
        "pde_weight": np.asarray(args.pde_weight, dtype=np.float32),
        "full_recon_weight": np.asarray(args.full_recon_weight, dtype=np.float32),
        "pde_warmup_epochs": np.asarray(args.pde_warmup_epochs, dtype=np.int32),
        "pde_fit_every": np.asarray(args.pde_fit_every, dtype=np.int32),
        "electric_source": np.asarray("" if electric_source is None else electric_source),
    }
    payload.update(extract_state_dict_numpy(model))
    payload.update(metrics)
    if reconstruction is not None:
        payload["reconstruction"] = reconstruction.astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)


def main() -> None:
    args = parse_args()
    if args.pde_fit_every <= 0:
        raise ValueError("--pde-fit-every must be positive.")
    if args.pde_warmup_epochs < 0:
        raise ValueError("--pde-warmup-epochs must be non-negative.")

    patch_pde_find_compatibility(pde_find)
    conv_channels = parse_conv_channels(args.conv_channels)
    set_seed(args.seed)
    device = resolve_device(args.device)

    case_dir = resolve_case_dir(args.grid_dir, args.case_dir)
    f_t_x_v, t, x, v = load_distribution(case_dir)
    dt = check_uniform_spacing(t.astype(np.float64), "t")
    dx = check_uniform_spacing(x.astype(np.float64), "x")
    nt, nx, nv = f_t_x_v.shape

    output_path = args.output.resolve() if args.output is not None else infer_output_path(case_dir.resolve())
    report_path = args.report.resolve() if args.report is not None else infer_report_path(output_path)

    samples = flatten_snapshots(f_t_x_v)
    rng = np.random.default_rng(args.seed)
    train_idx, val_idx = split_indices(len(samples), args.train_fraction, rng)
    samples_norm, mean, std = normalize_from_train(samples, train_idx)
    samples_grid = reshape_sample_grid(samples_norm, nt=nt, nx=nx)
    train_samples = samples_norm[train_idx]
    val_samples = samples_norm[val_idx]

    latent_meta = {
        "case_name": np.asarray(case_dir.name),
        "case_dir": np.asarray(str(case_dir.resolve())),
        "nt": np.asarray(nt, dtype=np.int32),
        "nx": np.asarray(nx, dtype=np.int32),
        "nz": np.asarray(args.latent_dim, dtype=np.int32),
    }
    electric_field, electric_source = resolve_electric_data(
        latent_file=output_path,
        meta=latent_meta,
        expected_t=t.astype(np.float64),
        expected_x=x.astype(np.float64),
        requested_electric_file=args.electric_file,
        disable_electric_field=args.no_electric_field,
    )

    mode_indices = resolve_modes(args.modes, args.latent_dim)

    print(f"Using case: {case_dir}")
    print(f"Loaded f with shape (Nt, Nx, Nv)=({nt}, {nx}, {nv})")
    print(f"Using torch device: {device}")
    print(f"Latent dim: {args.latent_dim}, constrained modes: {mode_indices}")
    print(f"PDE system: {args.system}")
    print(f"Electric field source: {'none' if electric_source is None else electric_source}")
    print(f"dx={dx:.6e}, dt={dt:.6e}")

    (
        model,
        train_losses,
        val_losses,
        lr_history,
        pde_loss_history,
        pde_residual_history,
        pde_nonzero_history,
        anchor_recon_history,
        refit_epochs,
    ) = train_joint_model(
        train_samples=train_samples,
        val_samples=val_samples,
        samples_grid=samples_grid,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        conv_channels=conv_channels,
        kernel_size=args.kernel_size,
        dt=dt,
        dx=dx,
        electric_field=electric_field,
        mode_indices=mode_indices,
        args=args,
        device=device,
    )

    with torch.no_grad():
        samples_grid_tensor = torch.from_numpy(samples_grid).to(device=device, dtype=torch.float32)
        latent_torch = encode_grid(model, samples_grid_tensor)
        reconstruction_torch = decode_grid(model, latent_torch)

    latent = latent_torch.cpu().numpy().astype(np.float32)
    reconstruction = reconstruction_torch.cpu().numpy().astype(np.float32)
    reconstruction = (reconstruction * std + mean).reshape(nt, nx, nv).astype(np.float32)

    coefficients, rhs_description, equations, pde_fit_summary = fit_sparse_pdes(
        latent=latent.astype(np.float64),
        modes=mode_indices,
        electric_field=electric_field.astype(np.float64) if electric_field is not None else None,
        dt=dt,
        dx=dx,
        args=args,
    )

    metrics: Dict[str, np.ndarray] = {
        "final_train_loss": np.asarray(train_losses[-1], dtype=np.float32),
        "final_val_loss": np.asarray(val_losses[-1], dtype=np.float32),
        "final_pde_loss": np.asarray(pde_loss_history[-1], dtype=np.float32),
        "final_pde_relative_residual": np.asarray(pde_residual_history[-1], dtype=np.float32),
        "final_pde_nonzero_terms": np.asarray(pde_nonzero_history[-1], dtype=np.float32),
        "final_anchor_reconstruction_loss": np.asarray(anchor_recon_history[-1], dtype=np.float32),
        "mean_pde_residual_l2": np.asarray(pde_fit_summary["mean_residual_l2"], dtype=np.float64),
        "mean_pde_relative_residual_l2": np.asarray(pde_fit_summary["mean_relative_residual_l2"], dtype=np.float64),
        "mean_pde_nonzero_terms": np.asarray(pde_fit_summary["mean_nonzero_terms"], dtype=np.float64),
        **reconstruction_metrics(reconstruction, f_t_x_v),
    }

    save_results(
        output_path=output_path,
        case_dir=case_dir.resolve(),
        t=t,
        x=x,
        v=v,
        latent=latent,
        mean=mean,
        std=std,
        train_losses=train_losses,
        val_losses=val_losses,
        lr_history=lr_history,
        pde_loss_history=pde_loss_history,
        pde_residual_history=pde_residual_history,
        pde_nonzero_history=pde_nonzero_history,
        anchor_recon_history=anchor_recon_history,
        refit_epochs=refit_epochs,
        coefficients=coefficients,
        rhs_description=rhs_description,
        equations=equations,
        model=model,
        device=device,
        args=args,
        metrics=metrics,
        electric_source=electric_source,
        reconstruction=reconstruction if args.save_reconstruction else None,
    )

    report_lines = [
        f"case_dir: {case_dir.resolve()}",
        f"shape: (Nt={nt}, Nx={nx}, Nv={nv})",
        f"device: {device}",
        f"latent_dim: {args.latent_dim}",
        f"constrained_modes: {mode_indices}",
        f"system: {args.system}",
        f"electric_source: {'none' if electric_source is None else electric_source}",
        f"dt: {dt:.8e}",
        f"dx: {dx:.8e}",
        f"epochs: {args.epochs}",
        f"batch_size: {args.batch_size}",
        f"learning_rate: {args.learning_rate:.8e}",
        f"pde_weight: {args.pde_weight:.8e}",
        f"full_recon_weight: {args.full_recon_weight:.8e}",
        f"pde_warmup_epochs: {args.pde_warmup_epochs}",
        f"pde_fit_every: {args.pde_fit_every}",
        f"final_train_loss: {float(train_losses[-1]):.8e}",
        f"final_val_loss: {float(val_losses[-1]):.8e}",
        f"final_pde_loss: {float(pde_loss_history[-1]):.8e}",
        f"mean_pde_relative_residual_l2: {float(metrics['mean_pde_relative_residual_l2']):.8e}",
        f"reconstruction_relative_l2_percent: {float(metrics['reconstruction_relative_l2_percent']):.4f}",
        f"output: {output_path}",
    ]
    report_lines.append("")
    report_lines.append("discovered_pdes:")
    report_lines.extend(equations)
    save_report(report_path, report_lines)

    print(f"Saved joint results to: {output_path}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
