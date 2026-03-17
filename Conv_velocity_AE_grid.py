#!/usr/bin/env python3

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required. Run this script in the 'lasdi' environment, for example:\n"
        "  conda run -n lasdi python Conv_velocity_AE_grid.py"
    ) from exc

from Conv_velocity_AE import (
    ConvVelocityAutoencoder,
    apply_normalization,
    compute_density_loss,
    compute_regularization_loss,
    default_encoder_output_path,
    encode_all,
    evaluate_electric_field_loss_on_grid,
    evaluate_loss,
    extract_state_dict_numpy,
    flatten_snapshots,
    load_distribution,
    normalize_from_train,
    parse_conv_channels,
    reconstruct_all,
    reconstruction_metrics,
    reference_profile_scale,
    relative_reference_target,
    resolve_device,
    set_seed,
    trapezoid_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a convolutional velocity autoencoder on a 5x5 subsampling of "
            "vlasov_twostream_param_grid and validate on one random unused case."
        )
    )
    parser.add_argument(
        "--grid-dir",
        type=Path,
        default=Path("vlasov_twostream_param_grid"),
        help="Directory that contains all parameter-grid cases.",
    )
    parser.add_argument(
        "--num-t-samples",
        type=int,
        default=5,
        help="Number of evenly spaced temperature values to sample from the grid.",
    )
    parser.add_argument(
        "--num-k-samples",
        type=int,
        default=5,
        help="Number of evenly spaced wavenumber values to sample from the grid.",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=8,
        help="Latent size Nz for each x cell.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden width of the linear bottleneck layers around the convolutional encoder.",
    )
    parser.add_argument(
        "--conv-channels",
        type=str,
        default="8,16,32",
        help="Comma-separated Conv1d channel widths used along the v direction.",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=5,
        help="Odd kernel size used in all convolutional encoder/decoder layers.",
    )
    parser.add_argument(
        "--padding-mode",
        type=str,
        default="zeros",
        choices=("zeros", "replicate", "reflect"),
        help="Padding mode used in encoder Conv1d layers along the v direction.",
    )
    parser.add_argument(
        "--f0-epsilon",
        type=float,
        default=1e-3,
        help="Positive epsilon used in the relative target (f - f0) / (f0 + epsilon).",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Mini-batch size over flattened (case, t, x) samples.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate.",
    )
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
    parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.5,
        help="Multiplicative lr decay factor for plateau scheduler.",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
        help="Lower bound for the learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for case selection and initialization.",
    )
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
        default=Path("results/vlasov_twostream_5x5_conv_velocity_autoencoder_results.npz"),
        help="Output .npz path for the trained model and all encoded cases.",
    )
    parser.add_argument(
        "--encoder-output",
        type=Path,
        default=None,
        help="Output .pt path for an encoder-only checkpoint. Defaults to <output-stem>_encoder.pt.",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=None,
        help="Optional .npz path where the latent trajectory of the validation case is saved.",
    )
    parser.add_argument(
        "--save-reconstruction",
        action="store_true",
        help="Save reconstructed f(t, x, v) for the training and validation cases in the output file.",
    )
    parser.add_argument(
        "--density-weight",
        type=float,
        default=0.0,
        help="Weight for an auxiliary density reconstruction loss on n(x, t) = integral f dv.",
    )
    parser.add_argument(
        "--electric-weight",
        type=float,
        default=0.0,
        help="Weight for an auxiliary electric-field reconstruction loss on E(x, t) recovered from Poisson's equation.",
    )
    parser.add_argument(
        "--vlasov-residual-weight",
        type=float,
        default=0.0,
        help="Weight for a Vlasov residual loss on reconstructed f using f_t + v f_x + E f_v.",
    )
    parser.add_argument(
        "--smoothness-weight",
        type=float,
        default=0.0,
        help="Weight for latent smoothness regularization based on second differences in t and x.",
    )
    parser.add_argument(
        "--smoothness-time-weight",
        type=float,
        default=1.0,
        help="Relative weight for latent smoothness along time.",
    )
    parser.add_argument(
        "--smoothness-space-weight",
        type=float,
        default=1.0,
        help="Relative weight for latent smoothness along space.",
    )
    parser.add_argument(
        "--dynamics-weight",
        type=float,
        default=0.0,
        help="Weight for a latent dynamics regularizer that encourages z_t to be explainable by z and z_x.",
    )
    parser.add_argument(
        "--dynamics-ridge",
        type=float,
        default=1e-4,
        help="Ridge parameter used in the local latent dynamics fit.",
    )
    parser.add_argument(
        "--reg-time-window",
        type=int,
        default=17,
        help="Temporal window size used when sampling latent regularization patches.",
    )
    parser.add_argument(
        "--reg-x-window",
        type=int,
        default=33,
        help="Spatial window size used when sampling latent regularization patches.",
    )
    parser.add_argument(
        "--reg-patches-per-epoch",
        type=int,
        default=1,
        help="Number of randomly chosen case patches used for latent regularization in each epoch.",
    )
    return parser.parse_args()


def parse_case_parameters(case_dir: Path) -> Tuple[float, float]:
    parts = case_dir.name.split("_")
    if len(parts) != 4 or parts[0] != "T" or parts[2] != "k":
        raise ValueError(f"Unexpected case directory name: {case_dir.name}")
    return float(parts[1]), float(parts[3])


def list_case_dirs(grid_dir: Path) -> List[Path]:
    case_dirs = [path for path in sorted(grid_dir.iterdir()) if path.is_dir() and path.name.startswith("T_")]
    if not case_dirs:
        raise FileNotFoundError(f"No case directories found in {grid_dir}")
    return case_dirs


def select_evenly_spaced(values: Sequence[float], count: int) -> List[float]:
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if count > len(values):
        raise ValueError(f"Requested {count} values from a set of size {len(values)}")
    if count == 1:
        return [float(values[len(values) // 2])]

    raw_indices = np.linspace(0, len(values) - 1, count)
    indices = [int(round(index)) for index in raw_indices]
    selected = [float(values[index]) for index in indices]
    if len(set(selected)) != count:
        raise ValueError(f"Could not choose {count} unique evenly spaced values from {values}")
    return selected


def resolve_training_case_dirs(grid_dir: Path, num_t_samples: int, num_k_samples: int) -> Tuple[List[Path], np.ndarray, np.ndarray]:
    case_dirs = list_case_dirs(grid_dir)
    case_map = {parse_case_parameters(case_dir): case_dir for case_dir in case_dirs}
    t_values = sorted({t for t, _k in case_map})
    k_values = sorted({k for _t, k in case_map})

    selected_t = select_evenly_spaced(t_values, num_t_samples)
    selected_k = select_evenly_spaced(k_values, num_k_samples)

    training_case_dirs = []
    for t_value, k_value in product(selected_t, selected_k):
        key = (float(t_value), float(k_value))
        if key not in case_map:
            raise FileNotFoundError(
                f"Selected case does not exist in {grid_dir}: T={t_value:.2f}, k={k_value:.2f}"
            )
        training_case_dirs.append(case_map[key])

    return training_case_dirs, np.asarray(selected_t, dtype=np.float32), np.asarray(selected_k, dtype=np.float32)


def choose_validation_case(case_dirs: Sequence[Path], training_case_dirs: Sequence[Path], seed: int) -> Path:
    training_names = {case_dir.name for case_dir in training_case_dirs}
    candidates = [case_dir for case_dir in case_dirs if case_dir.name not in training_names]
    if not candidates:
        raise ValueError("No unused cases are available for validation.")
    rng = np.random.default_rng(seed)
    return candidates[int(rng.integers(0, len(candidates)))]


def validate_matching_grid(
    case_dir: Path,
    t_reference: np.ndarray,
    x_reference: np.ndarray,
    v_reference: np.ndarray,
    f_t_x_v: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
) -> None:
    if f_t_x_v.shape != (len(t_reference), len(x_reference), len(v_reference)):
        raise ValueError(
            f"Case {case_dir} has shape {f_t_x_v.shape}, expected "
            f"({len(t_reference)}, {len(x_reference)}, {len(v_reference)})"
        )
    if not np.allclose(t, t_reference, rtol=1e-6, atol=1e-8):
        raise ValueError(f"Time grid mismatch for case {case_dir}")
    if not np.allclose(x, x_reference, rtol=1e-6, atol=1e-8):
        raise ValueError(f"Space grid mismatch for case {case_dir}")
    if not np.allclose(v, v_reference, rtol=1e-6, atol=1e-8):
        raise ValueError(f"Velocity grid mismatch for case {case_dir}")


def estimate_reference_profile_multi(train_cases: np.ndarray) -> np.ndarray:
    if train_cases.ndim != 4:
        raise ValueError(f"Expected train_cases with shape (Nc, Nt, Nx, Nv), got {train_cases.shape}")
    return np.asarray(train_cases[:, 0, :, :].mean(axis=(0, 1)), dtype=np.float32)


def build_normalized_case_arrays(
    train_cases: np.ndarray,
    validation_case: np.ndarray,
    f0_v: np.ndarray,
    f0_epsilon: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    transformed_train = relative_reference_target(train_cases, f0_v, f0_epsilon)
    transformed_val = relative_reference_target(validation_case, f0_v, f0_epsilon)

    num_cases, nt, nx, nv = transformed_train.shape
    flattened_train = transformed_train.reshape(num_cases * nt * nx, nv)
    normalized_train, transformed_mean, transformed_std = normalize_from_train(
        flattened_train,
        np.arange(flattened_train.shape[0], dtype=np.int64),
    )
    normalized_val = apply_normalization(
        flatten_snapshots(transformed_val),
        transformed_mean,
        transformed_std,
    )
    train_grids = normalized_train.reshape(num_cases, nt, nx, nv)
    val_grid = normalized_val.reshape(validation_case.shape)
    return train_grids, val_grid, transformed_mean, transformed_std


def compute_regularization_loss_multi(
    model: ConvVelocityAutoencoder,
    sample_grids: np.ndarray,
    dt: float,
    dx: float,
    smoothness_weight: float,
    smoothness_time_weight: float,
    smoothness_space_weight: float,
    dynamics_weight: float,
    dynamics_ridge: float,
    reg_time_window: int,
    reg_x_window: int,
    reg_patches_per_epoch: int,
    electric_weight: float,
    vlasov_residual_weight: float,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    density_weights: torch.Tensor,
    v_coords: torch.Tensor,
    device: torch.device,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    total_loss = torch.zeros((), device=device)
    smoothness_total = torch.zeros((), device=device)
    dynamics_total = torch.zeros((), device=device)
    electric_total = torch.zeros((), device=device)
    residual_total = torch.zeros((), device=device)

    if sample_grids.shape[0] == 0 or reg_patches_per_epoch <= 0:
        return total_loss, smoothness_total, dynamics_total, electric_total, residual_total

    for _ in range(reg_patches_per_epoch):
        grid_index = int(rng.integers(0, sample_grids.shape[0]))
        grid = np.ascontiguousarray(sample_grids[grid_index])
        reg_loss, smoothness_loss, dynamics_loss, electric_loss, residual_loss = compute_regularization_loss(
            model=model,
            sample_grid=grid,
            dt=dt,
            dx=dx,
            smoothness_weight=smoothness_weight,
            smoothness_time_weight=smoothness_time_weight,
            smoothness_space_weight=smoothness_space_weight,
            dynamics_weight=dynamics_weight,
            dynamics_ridge=dynamics_ridge,
            reg_time_window=reg_time_window,
            reg_x_window=reg_x_window,
            reg_patches_per_epoch=1,
            electric_weight=electric_weight,
            vlasov_residual_weight=vlasov_residual_weight,
            feature_mean=feature_mean,
            feature_std=feature_std,
            density_weights=density_weights,
            v_coords=v_coords,
            device=device,
            rng=rng,
        )
        total_loss = total_loss + reg_loss
        smoothness_total = smoothness_total + smoothness_loss
        dynamics_total = dynamics_total + dynamics_loss
        electric_total = electric_total + electric_loss
        residual_total = residual_total + residual_loss

    scale = 1.0 / float(reg_patches_per_epoch)
    return (
        total_loss * scale,
        smoothness_total * scale,
        dynamics_total * scale,
        electric_total * scale,
        residual_total * scale,
    )


def train_autoencoder_multi(
    train_samples: np.ndarray,
    validation_samples: np.ndarray,
    train_case_grids: np.ndarray,
    validation_grid: np.ndarray,
    latent_dim: int,
    hidden_dim: int,
    conv_channels: Sequence[int],
    kernel_size: int,
    padding_mode: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    lr_scheduler_name: str,
    lr_patience: int,
    lr_factor: float,
    min_lr: float,
    dt: float,
    dx: float,
    smoothness_weight: float,
    smoothness_time_weight: float,
    smoothness_space_weight: float,
    dynamics_weight: float,
    dynamics_ridge: float,
    reg_time_window: int,
    reg_x_window: int,
    reg_patches_per_epoch: int,
    density_weight: float,
    electric_weight: float,
    vlasov_residual_weight: float,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    density_weights: np.ndarray,
    v_coords: np.ndarray,
    seed: int,
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
    np.ndarray,
]:
    model = ConvVelocityAutoencoder(
        input_dim=train_samples.shape[1],
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        conv_channels=conv_channels,
        kernel_size=kernel_size,
        padding_mode=padding_mode,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = None
    if lr_scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_factor,
            patience=lr_patience,
            min_lr=min_lr,
        )
    criterion = nn.MSELoss(reduction="mean")
    feature_mean_tensor = torch.as_tensor(feature_mean.reshape(1, 1, -1), dtype=torch.float32, device=device)
    feature_std_tensor = torch.as_tensor(feature_std.reshape(1, 1, -1), dtype=torch.float32, device=device)
    density_weights_tensor = torch.as_tensor(density_weights.reshape(1, 1, -1), dtype=torch.float32, device=device)
    v_coords_tensor = torch.as_tensor(v_coords, dtype=torch.float32, device=device)

    from Conv_velocity_AE import build_dataloader

    train_loader = build_dataloader(train_samples, batch_size=batch_size, shuffle=True)
    eval_train_loader = build_dataloader(train_samples, batch_size=batch_size, shuffle=False)
    val_loader = build_dataloader(validation_samples, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []
    lr_history = []
    smoothness_history = []
    dynamics_history = []
    electric_history = []
    vlasov_residual_history = []
    validation_electric_history = []
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        model.train()
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = criterion(recon, batch)
            if density_weight > 0.0:
                loss = loss + density_weight * compute_density_loss(
                    recon=recon,
                    batch=batch,
                    feature_mean=feature_mean_tensor,
                    feature_std=feature_std_tensor,
                    density_weights=density_weights_tensor,
                )
            loss.backward()
            optimizer.step()

        epoch_smoothness = 0.0
        epoch_dynamics = 0.0
        epoch_electric = 0.0
        epoch_vlasov_residual = 0.0
        if smoothness_weight > 0.0 or dynamics_weight > 0.0 or electric_weight > 0.0 or vlasov_residual_weight > 0.0:
            optimizer.zero_grad(set_to_none=True)
            reg_loss, smoothness_loss, dynamics_loss, electric_loss, residual_loss = compute_regularization_loss_multi(
                model=model,
                sample_grids=train_case_grids,
                dt=dt,
                dx=dx,
                smoothness_weight=smoothness_weight,
                smoothness_time_weight=smoothness_time_weight,
                smoothness_space_weight=smoothness_space_weight,
                dynamics_weight=dynamics_weight,
                dynamics_ridge=dynamics_ridge,
                reg_time_window=reg_time_window,
                reg_x_window=reg_x_window,
                reg_patches_per_epoch=reg_patches_per_epoch,
                electric_weight=electric_weight,
                vlasov_residual_weight=vlasov_residual_weight,
                feature_mean=feature_mean_tensor,
                feature_std=feature_std_tensor,
                density_weights=density_weights_tensor,
                v_coords=v_coords_tensor,
                device=device,
                rng=rng,
            )
            if reg_loss.requires_grad:
                reg_loss.backward()
                optimizer.step()
                epoch_smoothness = float(smoothness_loss.detach().cpu().item())
                epoch_dynamics = float(dynamics_loss.detach().cpu().item())
                epoch_electric = float(electric_loss.detach().cpu().item())
                epoch_vlasov_residual = float(residual_loss.detach().cpu().item())

        train_loss = evaluate_loss(
            model,
            eval_train_loader,
            device,
            density_weight,
            feature_mean_tensor,
            feature_std_tensor,
            density_weights_tensor,
        )
        val_loss = evaluate_loss(
            model,
            val_loader,
            device,
            density_weight,
            feature_mean_tensor,
            feature_std_tensor,
            density_weights_tensor,
        )
        validation_electric = 0.0
        if electric_weight > 0.0:
            validation_electric = evaluate_electric_field_loss_on_grid(
                model=model,
                sample_grid=validation_grid,
                feature_mean=feature_mean_tensor,
                feature_std=feature_std_tensor,
                density_weights=density_weights_tensor,
                dx=dx,
                batch_size=batch_size,
                device=device,
            )
            val_loss = val_loss + electric_weight * validation_electric
        if scheduler is not None:
            scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        lr_history.append(current_lr)
        smoothness_history.append(epoch_smoothness)
        dynamics_history.append(epoch_dynamics)
        electric_history.append(epoch_electric)
        vlasov_residual_history.append(epoch_vlasov_residual)
        validation_electric_history.append(validation_electric)
        print(
            f"Epoch {epoch + 1:03d}/{epochs:03d} "
            f"train_obj={train_loss:.6e} val_obj={val_loss:.6e} "
            f"smooth={epoch_smoothness:.6e} dyn={epoch_dynamics:.6e} "
            f"elec={epoch_electric:.6e} vlasov={epoch_vlasov_residual:.6e} "
            f"val_elec={validation_electric:.6e} lr={current_lr:.6e}"
        )

    return (
        model,
        np.asarray(train_losses, dtype=np.float32),
        np.asarray(val_losses, dtype=np.float32),
        np.asarray(lr_history, dtype=np.float32),
        np.asarray(smoothness_history, dtype=np.float32),
        np.asarray(dynamics_history, dtype=np.float32),
        np.asarray(electric_history, dtype=np.float32),
        np.asarray(vlasov_residual_history, dtype=np.float32),
        np.asarray(validation_electric_history, dtype=np.float32),
    )


def save_encoder_checkpoint_multi(
    encoder_output_path: Path,
    model: ConvVelocityAutoencoder,
    mean: np.ndarray,
    std: np.ndarray,
    f0_v: np.ndarray,
    f0_scale_v: np.ndarray,
    f0_epsilon: float,
    v: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    grid_dir: Path,
    training_case_dirs: Sequence[Path],
    validation_case_dir: Path,
    selected_t: np.ndarray,
    selected_k: np.ndarray,
) -> None:
    checkpoint = {
        "model_type": "conv_velocity_encoder_grid",
        "grid_dir": str(grid_dir),
        "training_case_dirs": [str(case_dir) for case_dir in training_case_dirs],
        "training_case_names": [case_dir.name for case_dir in training_case_dirs],
        "validation_case_dir": str(validation_case_dir),
        "validation_case_name": validation_case_dir.name,
        "selected_t_values": torch.from_numpy(selected_t.astype(np.float32)),
        "selected_k_values": torch.from_numpy(selected_k.astype(np.float32)),
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "latent_dim": model.latent_dim,
        "conv_channels": tuple(model.conv_channels),
        "kernel_size": model.kernel_size,
        "padding_mode": model.padding_mode,
        "feature_lengths": tuple(model.feature_lengths),
        "encoded_channels": model.encoded_channels,
        "encoded_length": model.encoded_length,
        "feature_mean": torch.from_numpy(mean.astype(np.float32)),
        "feature_std": torch.from_numpy(std.astype(np.float32)),
        "f0_v": torch.from_numpy(f0_v.astype(np.float32)),
        "f0_scale_v": torch.from_numpy(f0_scale_v.astype(np.float32)),
        "f0_epsilon": np.float32(f0_epsilon),
        "training_target": "relative_f_minus_f0",
        "v": torch.from_numpy(v.astype(np.float32)),
        "x": torch.from_numpy(x.astype(np.float32)),
        "t": torch.from_numpy(t.astype(np.float32)),
        "encoder_features_state_dict": model.encoder_features.state_dict(),
        "encoder_projection_state_dict": model.encoder_projection.state_dict(),
    }
    encoder_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, encoder_output_path)


def save_validation_latent(
    output_path: Path,
    validation_case_dir: Path,
    training_case_dirs: Sequence[Path],
    t: np.ndarray,
    x: np.ndarray,
    latent: np.ndarray,
) -> None:
    payload: Dict[str, np.ndarray] = {
        "case_name": np.asarray(validation_case_dir.name),
        "case_dir": np.asarray(str(validation_case_dir)),
        "training_case_names": np.asarray([case_dir.name for case_dir in training_case_dirs]),
        "training_case_dirs": np.asarray([str(case_dir) for case_dir in training_case_dirs]),
        "t": np.asarray(t, dtype=np.float32),
        "x": np.asarray(x, dtype=np.float32),
        "nt": np.asarray(latent.shape[0], dtype=np.int32),
        "nx": np.asarray(latent.shape[1], dtype=np.int32),
        "nz": np.asarray(latent.shape[2], dtype=np.int32),
        "latent": np.asarray(latent, dtype=np.float32),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)


def save_results(
    output_path: Path,
    grid_dir: Path,
    training_case_dirs: Sequence[Path],
    validation_case_dir: Path,
    selected_t: np.ndarray,
    selected_k: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    train_latent: np.ndarray,
    validation_latent: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    f0_v: np.ndarray,
    f0_scale_v: np.ndarray,
    f0_epsilon: float,
    train_losses: np.ndarray,
    val_losses: np.ndarray,
    lr_history: np.ndarray,
    smoothness_history: np.ndarray,
    dynamics_history: np.ndarray,
    electric_history: np.ndarray,
    vlasov_residual_history: np.ndarray,
    validation_electric_history: np.ndarray,
    model: ConvVelocityAutoencoder,
    device: torch.device,
    args: argparse.Namespace,
    train_metrics: Dict[str, np.ndarray],
    validation_metrics: Dict[str, np.ndarray],
    train_reconstruction: np.ndarray | None,
    validation_reconstruction: np.ndarray | None,
) -> None:
    payload: Dict[str, np.ndarray] = {
        "grid_dir": np.asarray(str(grid_dir)),
        "torch_device": np.asarray(str(device)),
        "model_type": np.asarray("conv_velocity_autoencoder_grid"),
        "training_case_names": np.asarray([case_dir.name for case_dir in training_case_dirs]),
        "training_case_dirs": np.asarray([str(case_dir) for case_dir in training_case_dirs]),
        "validation_case_name": np.asarray(validation_case_dir.name),
        "validation_case_dir": np.asarray(str(validation_case_dir)),
        "selected_t_values": np.asarray(selected_t, dtype=np.float32),
        "selected_k_values": np.asarray(selected_k, dtype=np.float32),
        "num_training_cases": np.asarray(len(training_case_dirs), dtype=np.int32),
        "t": t,
        "x": x,
        "v": v,
        "nt": np.asarray(train_latent.shape[1], dtype=np.int32),
        "nx": np.asarray(train_latent.shape[2], dtype=np.int32),
        "nv": np.asarray(v.shape[0], dtype=np.int32),
        "nz": np.asarray(train_latent.shape[-1], dtype=np.int32),
        "train_latent": np.asarray(train_latent, dtype=np.float32),
        "validation_latent": np.asarray(validation_latent, dtype=np.float32),
        "feature_mean": mean.astype(np.float32),
        "feature_std": std.astype(np.float32),
        "f0_v": f0_v.astype(np.float32),
        "f0_scale_v": f0_scale_v.astype(np.float32),
        "f0_epsilon": np.asarray(f0_epsilon, dtype=np.float32),
        "training_target": np.asarray("relative_f_minus_f0"),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "learning_rate_history": lr_history,
        "smoothness_history": smoothness_history,
        "dynamics_history": dynamics_history,
        "electric_history": electric_history,
        "vlasov_residual_history": vlasov_residual_history,
        "validation_electric_history": validation_electric_history,
        "conv_channels": np.asarray(model.conv_channels, dtype=np.int32),
        "kernel_size": np.asarray(model.kernel_size, dtype=np.int32),
        "padding_mode": np.asarray(model.padding_mode),
        "hidden_dim": np.asarray(model.hidden_dim, dtype=np.int32),
        "feature_lengths": np.asarray(model.feature_lengths, dtype=np.int32),
        "smoothness_weight": np.asarray(args.smoothness_weight, dtype=np.float32),
        "smoothness_time_weight": np.asarray(args.smoothness_time_weight, dtype=np.float32),
        "smoothness_space_weight": np.asarray(args.smoothness_space_weight, dtype=np.float32),
        "dynamics_weight": np.asarray(args.dynamics_weight, dtype=np.float32),
        "dynamics_ridge": np.asarray(args.dynamics_ridge, dtype=np.float32),
        "density_weight": np.asarray(args.density_weight, dtype=np.float32),
        "electric_weight": np.asarray(args.electric_weight, dtype=np.float32),
        "vlasov_residual_weight": np.asarray(args.vlasov_residual_weight, dtype=np.float32),
        "reg_time_window": np.asarray(args.reg_time_window, dtype=np.int32),
        "reg_x_window": np.asarray(args.reg_x_window, dtype=np.int32),
        "reg_patches_per_epoch": np.asarray(args.reg_patches_per_epoch, dtype=np.int32),
    }
    payload.update(extract_state_dict_numpy(model))
    for key, value in train_metrics.items():
        payload[f"train_{key}"] = np.asarray(value)
    for key, value in validation_metrics.items():
        payload[f"validation_{key}"] = np.asarray(value)
    if train_reconstruction is not None:
        payload["train_reconstruction"] = np.asarray(train_reconstruction, dtype=np.float32)
    if validation_reconstruction is not None:
        payload["validation_reconstruction"] = np.asarray(validation_reconstruction, dtype=np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)


def main() -> None:
    args = parse_args()
    conv_channels = parse_conv_channels(args.conv_channels)
    set_seed(args.seed)
    device = resolve_device(args.device)

    grid_dir = args.grid_dir.resolve()
    all_case_dirs = list_case_dirs(grid_dir)
    training_case_dirs, selected_t, selected_k = resolve_training_case_dirs(
        grid_dir=grid_dir,
        num_t_samples=args.num_t_samples,
        num_k_samples=args.num_k_samples,
    )
    validation_case_dir = choose_validation_case(all_case_dirs, training_case_dirs, args.seed)

    train_cases = []
    t_ref = None
    x_ref = None
    v_ref = None
    for case_dir in training_case_dirs:
        f_t_x_v, t, x, v = load_distribution(case_dir)
        if t_ref is None:
            t_ref = t
            x_ref = x
            v_ref = v
        else:
            validate_matching_grid(case_dir, t_ref, x_ref, v_ref, f_t_x_v, t, x, v)
        train_cases.append(f_t_x_v)

    assert t_ref is not None and x_ref is not None and v_ref is not None
    train_cases_array = np.asarray(train_cases, dtype=np.float32)
    validation_case, validation_t, validation_x, validation_v = load_distribution(validation_case_dir)
    validate_matching_grid(
        validation_case_dir,
        t_ref,
        x_ref,
        v_ref,
        validation_case,
        validation_t,
        validation_x,
        validation_v,
    )

    num_cases, nt, nx, nv = train_cases_array.shape
    print(f"Using grid directory: {grid_dir}")
    print(f"Selected T values: {selected_t.tolist()}")
    print(f"Selected k values: {selected_k.tolist()}")
    print(f"Training cases ({num_cases}): {[case_dir.name for case_dir in training_case_dirs]}")
    print(f"Validation case: {validation_case_dir.name}")
    print(f"Loaded training tensor with shape (Nc, Nt, Nx, Nv)=({num_cases}, {nt}, {nx}, {nv})")
    print(f"Using torch device: {device}")
    print(f"Conv channels: {conv_channels}")
    print(f"Kernel size: {args.kernel_size}")
    print(f"Padding mode: {args.padding_mode}")
    print(f"f0 epsilon: {args.f0_epsilon}")

    f0_v = estimate_reference_profile_multi(train_cases_array)
    f0_scale_v = reference_profile_scale(f0_v, args.f0_epsilon)
    train_case_grids, validation_grid, transformed_mean, transformed_std = build_normalized_case_arrays(
        train_cases=train_cases_array,
        validation_case=validation_case,
        f0_v=f0_v,
        f0_epsilon=args.f0_epsilon,
    )
    mean = transformed_mean * f0_scale_v.reshape(1, -1) + f0_v.reshape(1, -1)
    std = transformed_std * f0_scale_v.reshape(1, -1)
    train_samples = train_case_grids.reshape(num_cases * nt * nx, nv)
    validation_samples = validation_grid.reshape(validation_case.shape[0] * validation_case.shape[1], nv)
    density_weights = trapezoid_weights(v_ref)
    dt = float(np.mean(np.diff(t_ref)))
    dx = float(np.mean(np.diff(x_ref)))

    (
        model,
        train_losses,
        val_losses,
        lr_history,
        smoothness_history,
        dynamics_history,
        electric_history,
        vlasov_residual_history,
        validation_electric_history,
    ) = train_autoencoder_multi(
        train_samples=train_samples,
        validation_samples=validation_samples,
        train_case_grids=train_case_grids,
        validation_grid=validation_grid,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        conv_channels=conv_channels,
        kernel_size=args.kernel_size,
        padding_mode=args.padding_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_name=args.lr_scheduler,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        min_lr=args.min_lr,
        dt=dt,
        dx=dx,
        smoothness_weight=args.smoothness_weight,
        smoothness_time_weight=args.smoothness_time_weight,
        smoothness_space_weight=args.smoothness_space_weight,
        dynamics_weight=args.dynamics_weight,
        dynamics_ridge=args.dynamics_ridge,
        reg_time_window=args.reg_time_window,
        reg_x_window=args.reg_x_window,
        reg_patches_per_epoch=args.reg_patches_per_epoch,
        density_weight=args.density_weight,
        electric_weight=args.electric_weight,
        vlasov_residual_weight=args.vlasov_residual_weight,
        feature_mean=mean,
        feature_std=std,
        density_weights=density_weights,
        v_coords=v_ref,
        seed=args.seed,
        device=device,
    )

    train_latent = []
    train_reconstruction = [] if args.save_reconstruction else None
    train_metric_values: Dict[str, List[np.ndarray]] = {}
    for case_dir, case_grid, case_truth in zip(training_case_dirs, train_case_grids, train_cases_array):
        case_samples = case_grid.reshape(nt * nx, nv)
        latent = encode_all(
            model=model,
            samples=case_samples,
            nt=nt,
            nx=nx,
            latent_dim=args.latent_dim,
            batch_size=args.batch_size,
            device=device,
        )
        reconstruction = reconstruct_all(
            model=model,
            samples=case_samples,
            mean=mean,
            std=std,
            original_shape=case_truth.shape,
            batch_size=args.batch_size,
            device=device,
        )
        metrics = reconstruction_metrics(reconstruction, case_truth, v_ref)
        train_latent.append(latent)
        for key, value in metrics.items():
            train_metric_values.setdefault(key, []).append(np.asarray(value))
        if train_reconstruction is not None:
            train_reconstruction.append(reconstruction)
        print(
            f"{case_dir.name}: relL2={float(metrics['reconstruction_relative_l2']):.6e} "
            f"density_relL2={float(metrics['density_reconstruction_relative_l2']):.6e}"
        )

    validation_latent = encode_all(
        model=model,
        samples=validation_samples,
        nt=validation_case.shape[0],
        nx=validation_case.shape[1],
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        device=device,
    )
    validation_reconstruction_array = reconstruct_all(
        model=model,
        samples=validation_samples,
        mean=mean,
        std=std,
        original_shape=validation_case.shape,
        batch_size=args.batch_size,
        device=device,
    )
    validation_metrics = reconstruction_metrics(validation_reconstruction_array, validation_case, v_ref)
    print(f"Validation latent shape: {validation_latent.shape}")
    print(f"Validation relL2: {float(validation_metrics['reconstruction_relative_l2']):.6e}")
    print(
        "Validation density relL2: "
        f"{float(validation_metrics['density_reconstruction_relative_l2']):.6e}"
    )

    train_latent_array = np.asarray(train_latent, dtype=np.float32)
    train_metrics = {key: np.asarray(values) for key, values in train_metric_values.items()}
    train_metrics["reconstruction_relative_l2_mean"] = np.asarray(
        np.mean(train_metrics["reconstruction_relative_l2"], dtype=np.float64),
        dtype=np.float64,
    )
    train_metrics["density_reconstruction_relative_l2_mean"] = np.asarray(
        np.mean(train_metrics["density_reconstruction_relative_l2"], dtype=np.float64),
        dtype=np.float64,
    )

    output_path = args.output.resolve()
    encoder_output_path = (
        args.encoder_output.resolve()
        if args.encoder_output is not None
        else default_encoder_output_path(output_path)
    )
    validation_output_path = (
        args.validation_output.resolve()
        if args.validation_output is not None
        else output_path.with_name(f"{output_path.stem}_validation.npz")
    )
    train_reconstruction_array = (
        np.asarray(train_reconstruction, dtype=np.float32) if train_reconstruction is not None else None
    )
    validation_reconstruction = validation_reconstruction_array if args.save_reconstruction else None

    save_results(
        output_path=output_path,
        grid_dir=grid_dir,
        training_case_dirs=training_case_dirs,
        validation_case_dir=validation_case_dir,
        selected_t=selected_t,
        selected_k=selected_k,
        t=np.asarray(t_ref, dtype=np.float32),
        x=np.asarray(x_ref, dtype=np.float32),
        v=np.asarray(v_ref, dtype=np.float32),
        train_latent=train_latent_array,
        validation_latent=np.asarray(validation_latent, dtype=np.float32),
        mean=mean,
        std=std,
        f0_v=f0_v,
        f0_scale_v=f0_scale_v,
        f0_epsilon=args.f0_epsilon,
        train_losses=train_losses,
        val_losses=val_losses,
        lr_history=lr_history,
        smoothness_history=smoothness_history,
        dynamics_history=dynamics_history,
        electric_history=electric_history,
        vlasov_residual_history=vlasov_residual_history,
        validation_electric_history=validation_electric_history,
        model=model,
        device=device,
        args=args,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        train_reconstruction=train_reconstruction_array,
        validation_reconstruction=validation_reconstruction,
    )
    save_validation_latent(
        output_path=validation_output_path,
        validation_case_dir=validation_case_dir,
        training_case_dirs=training_case_dirs,
        t=np.asarray(validation_t, dtype=np.float32),
        x=np.asarray(validation_x, dtype=np.float32),
        latent=np.asarray(validation_latent, dtype=np.float32),
    )
    save_encoder_checkpoint_multi(
        encoder_output_path=encoder_output_path,
        model=model,
        mean=mean,
        std=std,
        f0_v=f0_v,
        f0_scale_v=f0_scale_v,
        f0_epsilon=args.f0_epsilon,
        v=np.asarray(v_ref, dtype=np.float32),
        x=np.asarray(x_ref, dtype=np.float32),
        t=np.asarray(t_ref, dtype=np.float32),
        grid_dir=grid_dir,
        training_case_dirs=training_case_dirs,
        validation_case_dir=validation_case_dir,
        selected_t=selected_t,
        selected_k=selected_k,
    )
    print(f"Results written to: {output_path}")
    print(f"Validation latent written to: {validation_output_path}")
    print(f"Encoder checkpoint written to: {encoder_output_path}")


if __name__ == "__main__":
    main()
