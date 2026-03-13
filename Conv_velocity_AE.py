#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
import warnings
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required. Run this script in the 'lasdi' environment, for example:\n"
        "  conda run -n lasdi python Conv_velocity_AE.py"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a PyTorch convolutional velocity-only autoencoder on one case from "
            "vlasov_twostream_param_grid/distribution_full.npz."
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
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Mini-batch size over flattened (t, x) samples.",
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
        "--train-fraction",
        type=float,
        default=0.9,
        help="Fraction of flattened samples used for training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for initialization and train/validation split.",
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
        default=None,
        help="Output .npz path. Defaults to <case-dir>/conv_velocity_autoencoder_results.npz.",
    )
    parser.add_argument(
        "--encoder-output",
        type=Path,
        default=None,
        help="Output .pt path for an encoder-only checkpoint. Defaults to <output-stem>_encoder.pt.",
    )
    parser.add_argument(
        "--save-reconstruction",
        action="store_true",
        help="Save reconstructed f(t, x, v) in the output file.",
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
        help="Number of spatiotemporal patches used for latent regularization in each epoch.",
    )
    return parser.parse_args()


def parse_conv_channels(spec: str) -> Tuple[int, ...]:
    try:
        channels = tuple(int(token.strip()) for token in spec.split(",") if token.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid --conv-channels specification: {spec!r}") from exc

    if not channels:
        raise ValueError("--conv-channels must contain at least one positive integer.")
    if any(channel <= 0 for channel in channels):
        raise ValueError(f"All conv channels must be positive, got {channels}")
    return channels


def resolve_case_dir(grid_dir: Path, case_dir: Path | None) -> Path:
    if case_dir is not None:
        if not case_dir.is_dir():
            raise FileNotFoundError(f"Case directory not found: {case_dir}")
        return case_dir

    candidates = sorted(path for path in grid_dir.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No case directories found in {grid_dir}")
    return candidates[0]


def load_distribution(case_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dist_path = case_dir / "distribution_full.npz"
    if not dist_path.exists():
        raise FileNotFoundError(f"distribution_full.npz not found: {dist_path}")

    with np.load(dist_path) as data:
        t = np.asarray(data["t"], dtype=np.float32)
        x = np.asarray(data["x"], dtype=np.float32)
        v = np.asarray(data["v"], dtype=np.float32)
        f = np.asarray(data["f"], dtype=np.float32)

    expected_nt = len(t)
    expected_nx = len(x)
    expected_nv = len(v)

    if f.shape == (expected_nt, expected_nx, expected_nv):
        f_t_x_v = f
    elif f.shape == (expected_nx, expected_nv, expected_nt):
        f_t_x_v = np.transpose(f, (2, 0, 1))
    else:
        raise ValueError(
            "Unsupported f shape. Expected either (Nt, Nx, Nv) or (Nx, Nv, Nt), "
            f"got {f.shape} with Nt={expected_nt}, Nx={expected_nx}, Nv={expected_nv}."
        )

    return f_t_x_v, t, x, v


def flatten_snapshots(f_t_x_v: np.ndarray) -> np.ndarray:
    nt, nx, nv = f_t_x_v.shape
    return f_t_x_v.reshape(nt * nx, nv)


def split_indices(
    num_samples: int,
    train_fraction: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")

    indices = np.arange(num_samples)
    rng.shuffle(indices)
    split = int(num_samples * train_fraction)
    train_idx = indices[:split]
    val_idx = indices[split:]
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("train_fraction produced an empty train or validation split.")
    return train_idx, val_idx


def normalize_from_train(
    samples: np.ndarray,
    train_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = samples[train_idx].mean(axis=0, keepdims=True)
    std = samples[train_idx].std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    normalized = (samples - mean) / std
    return normalized.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_is_available():
        torch.cuda.manual_seed_all(seed)


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


class ConvVelocityAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        conv_channels: Sequence[int],
        kernel_size: int,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.conv_channels = tuple(int(channel) for channel in conv_channels)
        self.kernel_size = kernel_size

        padding = kernel_size // 2
        stride = 2
        activation = nn.Tanh

        encoder_layers = []
        in_channels = 1
        for out_channels in self.conv_channels:
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    activation(),
                )
            )
            in_channels = out_channels
        self.encoder_features = nn.Sequential(*encoder_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim, dtype=torch.float32)
            feature_lengths = [input_dim]
            encoded = dummy
            for layer in self.encoder_features:
                encoded = layer(encoded)
                feature_lengths.append(int(encoded.shape[-1]))

        self.feature_lengths = tuple(feature_lengths)
        self.encoded_channels = int(encoded.shape[1])
        self.encoded_length = int(encoded.shape[2])
        flattened_dim = self.encoded_channels * self.encoded_length

        self.encoder_projection = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(flattened_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder_projection = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, flattened_dim),
            activation(),
        )

        decoder_layers = []
        decoder_in_channels = list(reversed(self.conv_channels))
        decoder_out_channels = list(reversed(self.conv_channels[:-1])) + [1]
        current_lengths = list(reversed(self.feature_lengths[1:]))
        target_lengths = list(reversed(self.feature_lengths[:-1]))

        for idx, (in_ch, out_ch, current_len, target_len) in enumerate(
            zip(decoder_in_channels, decoder_out_channels, current_lengths, target_lengths)
        ):
            output_padding = target_len - (2 * current_len - 1)
            if output_padding not in (0, 1):
                raise ValueError(
                    "Could not infer a valid ConvTranspose1d output padding for "
                    f"length transition {current_len} -> {target_len}."
                )

            layers = [
                nn.ConvTranspose1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                )
            ]
            if idx < len(decoder_in_channels) - 1:
                layers.append(activation())
            decoder_layers.append(nn.Sequential(*layers))

        self.decoder_features = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder_features(x)
        return self.encoder_projection(features)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        features = self.decoder_projection(z)
        features = features.view(z.shape[0], self.encoded_channels, self.encoded_length)
        return self.decoder_features(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def build_dataloader(samples: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor = torch.from_numpy(samples).unsqueeze(1)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def reshape_sample_grid(samples: np.ndarray, nt: int, nx: int) -> np.ndarray:
    return samples.reshape(nt, nx, samples.shape[-1])


def sample_regularization_patch(
    sample_grid: np.ndarray,
    time_window: int,
    x_window: int,
    rng: np.random.Generator,
) -> np.ndarray:
    nt, nx, _nv = sample_grid.shape
    patch_nt = min(max(time_window, 1), nt)
    patch_nx = min(max(x_window, 1), nx)
    start_t = int(rng.integers(0, nt - patch_nt + 1))
    start_x = int(rng.integers(0, nx - patch_nx + 1))
    return np.ascontiguousarray(sample_grid[start_t : start_t + patch_nt, start_x : start_x + patch_nx, :])


def encode_patch(model: ConvVelocityAutoencoder, patch: torch.Tensor) -> torch.Tensor:
    nt_patch, nx_patch, nv = patch.shape
    flattened = patch.reshape(nt_patch * nx_patch, nv).unsqueeze(1)
    latent = model.encode(flattened)
    return latent.reshape(nt_patch, nx_patch, model.latent_dim)


def second_difference(tensor: torch.Tensor, dim: int) -> torch.Tensor | None:
    if tensor.shape[dim] < 3:
        return None
    return (
        tensor.narrow(dim, 2, tensor.shape[dim] - 2)
        - 2.0 * tensor.narrow(dim, 1, tensor.shape[dim] - 2)
        + tensor.narrow(dim, 0, tensor.shape[dim] - 2)
    )


def compute_smoothness_loss(
    latent_patch: torch.Tensor,
    dt: float,
    dx: float,
    time_weight: float,
    space_weight: float,
) -> torch.Tensor:
    loss = latent_patch.new_zeros(())

    if time_weight > 0.0:
        z_tt = second_difference(latent_patch, dim=0)
        if z_tt is not None:
            loss = loss + time_weight * ((z_tt / (dt * dt)) ** 2).mean()

    if space_weight > 0.0:
        z_xx = second_difference(latent_patch, dim=1)
        if z_xx is not None:
            loss = loss + space_weight * ((z_xx / (dx * dx)) ** 2).mean()

    return loss


def compute_dynamics_loss(
    latent_patch: torch.Tensor,
    dt: float,
    dx: float,
    ridge: float,
) -> torch.Tensor:
    nt_patch, nx_patch, nz = latent_patch.shape
    if nt_patch < 2 or nx_patch < 3:
        return latent_patch.new_zeros(())

    z_now = latent_patch[:-1, 1:-1, :]
    z_t = (latent_patch[1:, 1:-1, :] - latent_patch[:-1, 1:-1, :]) / dt
    z_x = (latent_patch[:-1, 2:, :] - latent_patch[:-1, :-2, :]) / (2.0 * dx)

    features = torch.cat([torch.ones_like(z_now[..., :1]), z_now, z_x], dim=-1).reshape(-1, 1 + 2 * nz)
    targets = z_t.reshape(-1, nz)
    if features.shape[0] < features.shape[1]:
        return latent_patch.new_zeros(())

    fit_features = features.detach()
    fit_targets = targets.detach()

    if ridge > 0.0:
        xtx = fit_features.T @ fit_features
        xty = fit_features.T @ fit_targets
        eye = torch.eye(xtx.shape[0], device=xtx.device, dtype=xtx.dtype)
        coefficients = torch.linalg.solve(xtx + ridge * eye, xty)
    else:
        coefficients = torch.linalg.lstsq(fit_features, fit_targets).solution

    residual = targets - features @ coefficients
    return (residual**2).mean()


def compute_regularization_loss(
    model: ConvVelocityAutoencoder,
    sample_grid: np.ndarray,
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
    device: torch.device,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_loss = torch.zeros((), device=device)
    smoothness_loss_total = torch.zeros((), device=device)
    dynamics_loss_total = torch.zeros((), device=device)

    if reg_patches_per_epoch <= 0:
        return total_loss, smoothness_loss_total, dynamics_loss_total
    if smoothness_weight <= 0.0 and dynamics_weight <= 0.0:
        return total_loss, smoothness_loss_total, dynamics_loss_total

    for _ in range(reg_patches_per_epoch):
        patch_np = sample_regularization_patch(
            sample_grid=sample_grid,
            time_window=reg_time_window,
            x_window=reg_x_window,
            rng=rng,
        )
        patch = torch.from_numpy(patch_np).to(device=device, dtype=torch.float32)
        latent_patch = encode_patch(model, patch)

        smoothness_loss = compute_smoothness_loss(
            latent_patch=latent_patch,
            dt=dt,
            dx=dx,
            time_weight=smoothness_time_weight,
            space_weight=smoothness_space_weight,
        )
        dynamics_loss = compute_dynamics_loss(
            latent_patch=latent_patch,
            dt=dt,
            dx=dx,
            ridge=dynamics_ridge,
        )

        total_loss = total_loss + smoothness_weight * smoothness_loss + dynamics_weight * dynamics_loss
        smoothness_loss_total = smoothness_loss_total + smoothness_loss
        dynamics_loss_total = dynamics_loss_total + dynamics_loss

    scale = 1.0 / float(reg_patches_per_epoch)
    return total_loss * scale, smoothness_loss_total * scale, dynamics_loss_total * scale


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    criterion = nn.MSELoss(reduction="mean")
    total_loss = 0.0
    total_count = 0
    for (batch,) in loader:
        batch = batch.to(device)
        recon = model(batch)
        loss = criterion(recon, batch)
        batch_size = batch.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
    return total_loss / max(total_count, 1)


def train_autoencoder(
    train_samples: np.ndarray,
    val_samples: np.ndarray,
    samples_grid: np.ndarray,
    latent_dim: int,
    hidden_dim: int,
    conv_channels: Sequence[int],
    kernel_size: int,
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
    seed: int,
    device: torch.device,
) -> Tuple[ConvVelocityAutoencoder, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = ConvVelocityAutoencoder(
        input_dim=train_samples.shape[1],
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        conv_channels=conv_channels,
        kernel_size=kernel_size,
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

    train_loader = build_dataloader(train_samples, batch_size=batch_size, shuffle=True)
    eval_train_loader = build_dataloader(train_samples, batch_size=batch_size, shuffle=False)
    val_loader = build_dataloader(val_samples, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []
    lr_history = []
    smoothness_history = []
    dynamics_history = []
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        model.train()
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()

        epoch_smoothness = 0.0
        epoch_dynamics = 0.0
        if smoothness_weight > 0.0 or dynamics_weight > 0.0:
            optimizer.zero_grad(set_to_none=True)
            reg_loss, smoothness_loss, dynamics_loss = compute_regularization_loss(
                model=model,
                sample_grid=samples_grid,
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
                device=device,
                rng=rng,
            )
            if reg_loss.requires_grad:
                reg_loss.backward()
                optimizer.step()
                epoch_smoothness = float(smoothness_loss.detach().cpu().item())
                epoch_dynamics = float(dynamics_loss.detach().cpu().item())

        train_loss = evaluate_loss(model, eval_train_loader, device)
        val_loss = evaluate_loss(model, val_loader, device)
        if scheduler is not None:
            scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        lr_history.append(current_lr)
        smoothness_history.append(epoch_smoothness)
        dynamics_history.append(epoch_dynamics)
        print(
            f"Epoch {epoch + 1:03d}/{epochs:03d} "
            f"train_mse={train_loss:.6e} val_mse={val_loss:.6e} "
            f"smooth={epoch_smoothness:.6e} dyn={epoch_dynamics:.6e} lr={current_lr:.6e}"
        )

    return (
        model,
        np.asarray(train_losses, dtype=np.float32),
        np.asarray(val_losses, dtype=np.float32),
        np.asarray(lr_history, dtype=np.float32),
        np.asarray(smoothness_history, dtype=np.float32),
        np.asarray(dynamics_history, dtype=np.float32),
    )


@torch.no_grad()
def encode_all(
    model: ConvVelocityAutoencoder,
    samples: np.ndarray,
    nt: int,
    nx: int,
    latent_dim: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    loader = build_dataloader(samples, batch_size=batch_size, shuffle=False)
    encoded_batches = []
    for (batch,) in loader:
        batch = batch.to(device)
        encoded_batches.append(model.encode(batch).cpu().numpy().astype(np.float32))
    latent = np.concatenate(encoded_batches, axis=0)
    return latent.reshape(nt, nx, latent_dim)


@torch.no_grad()
def reconstruct_all(
    model: ConvVelocityAutoencoder,
    samples: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    original_shape: Tuple[int, int, int],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    loader = build_dataloader(samples, batch_size=batch_size, shuffle=False)
    recon_batches = []
    for (batch,) in loader:
        batch = batch.to(device)
        recon_batches.append(model(batch).squeeze(1).cpu().numpy().astype(np.float32))
    recon = np.concatenate(recon_batches, axis=0)
    recon = recon * std + mean
    return recon.reshape(original_shape).astype(np.float32)


def reconstruction_metrics(reconstruction: np.ndarray, target: np.ndarray) -> Dict[str, np.ndarray]:
    diff = reconstruction - target
    mse = np.mean(diff**2, dtype=np.float64)
    rmse = np.float64(np.sqrt(mse))
    denom = np.linalg.norm(target.reshape(-1), ord=2)
    rel_l2 = np.linalg.norm(diff.reshape(-1), ord=2) / denom if denom > 0.0 else 0.0
    rel_l2_percent = rel_l2 * 100.0
    return {
        "reconstruction_mse": np.asarray(mse, dtype=np.float64),
        "reconstruction_rmse": np.asarray(rmse, dtype=np.float64),
        "reconstruction_relative_l2": np.asarray(rel_l2, dtype=np.float64),
        "reconstruction_relative_l2_percent": np.asarray(rel_l2_percent, dtype=np.float64),
    }


def extract_state_dict_numpy(model: nn.Module) -> Dict[str, np.ndarray]:
    state = {}
    for name, tensor in model.state_dict().items():
        key = name.replace(".", "__")
        state[f"param_{key}"] = tensor.detach().cpu().numpy().astype(np.float32)
    return state


def default_output_path(case_dir: Path) -> Path:
    return case_dir / "conv_velocity_autoencoder_results.npz"


def default_encoder_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_encoder.pt")


def save_encoder_checkpoint(
    encoder_output_path: Path,
    model: ConvVelocityAutoencoder,
    mean: np.ndarray,
    std: np.ndarray,
    v: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    case_dir: Path,
) -> None:
    checkpoint = {
        "model_type": "conv_velocity_encoder",
        "case_name": case_dir.name,
        "case_dir": str(case_dir),
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "latent_dim": model.latent_dim,
        "conv_channels": tuple(model.conv_channels),
        "kernel_size": model.kernel_size,
        "feature_lengths": tuple(model.feature_lengths),
        "encoded_channels": model.encoded_channels,
        "encoded_length": model.encoded_length,
        "feature_mean": torch.from_numpy(mean.astype(np.float32)),
        "feature_std": torch.from_numpy(std.astype(np.float32)),
        "v": torch.from_numpy(v.astype(np.float32)),
        "x": torch.from_numpy(x.astype(np.float32)),
        "t": torch.from_numpy(t.astype(np.float32)),
        "encoder_features_state_dict": model.encoder_features.state_dict(),
        "encoder_projection_state_dict": model.encoder_projection.state_dict(),
    }
    encoder_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, encoder_output_path)


def save_results(
    output_path: Path,
    case_dir: Path,
    t: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    original_shape: Tuple[int, int, int],
    latent: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    train_losses: np.ndarray,
    val_losses: np.ndarray,
    lr_history: np.ndarray,
    smoothness_history: np.ndarray,
    dynamics_history: np.ndarray,
    model: ConvVelocityAutoencoder,
    device: torch.device,
    args: argparse.Namespace,
    metrics: Dict[str, np.ndarray],
    reconstruction: np.ndarray | None,
) -> None:
    payload: Dict[str, np.ndarray] = {
        "case_name": np.asarray(case_dir.name),
        "case_dir": np.asarray(str(case_dir)),
        "torch_device": np.asarray(str(device)),
        "model_type": np.asarray("conv_velocity_autoencoder"),
        "t": t,
        "x": x,
        "v": v,
        "nt": np.asarray(original_shape[0], dtype=np.int32),
        "nx": np.asarray(original_shape[1], dtype=np.int32),
        "nv": np.asarray(original_shape[2], dtype=np.int32),
        "nz": np.asarray(latent.shape[-1], dtype=np.int32),
        "latent": latent,
        "feature_mean": mean.astype(np.float32),
        "feature_std": std.astype(np.float32),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "learning_rate_history": lr_history,
        "smoothness_history": smoothness_history,
        "dynamics_history": dynamics_history,
        "conv_channels": np.asarray(model.conv_channels, dtype=np.int32),
        "kernel_size": np.asarray(model.kernel_size, dtype=np.int32),
        "hidden_dim": np.asarray(model.hidden_dim, dtype=np.int32),
        "feature_lengths": np.asarray(model.feature_lengths, dtype=np.int32),
        "smoothness_weight": np.asarray(args.smoothness_weight, dtype=np.float32),
        "smoothness_time_weight": np.asarray(args.smoothness_time_weight, dtype=np.float32),
        "smoothness_space_weight": np.asarray(args.smoothness_space_weight, dtype=np.float32),
        "dynamics_weight": np.asarray(args.dynamics_weight, dtype=np.float32),
        "dynamics_ridge": np.asarray(args.dynamics_ridge, dtype=np.float32),
        "reg_time_window": np.asarray(args.reg_time_window, dtype=np.int32),
        "reg_x_window": np.asarray(args.reg_x_window, dtype=np.int32),
        "reg_patches_per_epoch": np.asarray(args.reg_patches_per_epoch, dtype=np.int32),
    }
    payload.update(extract_state_dict_numpy(model))
    payload.update(metrics)

    if reconstruction is not None:
        payload["reconstruction"] = reconstruction

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)


def main() -> None:
    args = parse_args()
    conv_channels = parse_conv_channels(args.conv_channels)
    set_seed(args.seed)
    device = resolve_device(args.device)

    case_dir = resolve_case_dir(args.grid_dir, args.case_dir)
    f_t_x_v, t, x, v = load_distribution(case_dir)
    nt, nx, nv = f_t_x_v.shape

    print(f"Using case: {case_dir}")
    print(f"Loaded f with shape (Nt, Nx, Nv)=({nt}, {nx}, {nv})")
    print(f"Using torch device: {device}")
    print(f"Conv channels: {conv_channels}")
    print(f"Kernel size: {args.kernel_size}")

    samples = flatten_snapshots(f_t_x_v)
    rng = np.random.default_rng(args.seed)
    train_idx, val_idx = split_indices(len(samples), args.train_fraction, rng)
    samples_norm, mean, std = normalize_from_train(samples, train_idx)
    samples_grid = reshape_sample_grid(samples_norm, nt=nt, nx=nx)
    dt = float(np.mean(np.diff(t)))
    dx = float(np.mean(np.diff(x)))

    train_samples = samples_norm[train_idx]
    val_samples = samples_norm[val_idx]

    model, train_losses, val_losses, lr_history, smoothness_history, dynamics_history = train_autoencoder(
        train_samples=train_samples,
        val_samples=val_samples,
        samples_grid=samples_grid,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        conv_channels=conv_channels,
        kernel_size=args.kernel_size,
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
        seed=args.seed,
        device=device,
    )

    latent = encode_all(
        model=model,
        samples=samples_norm,
        nt=nt,
        nx=nx,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        device=device,
    )
    print(f"Latent shape per snapshot: ({nx}, {args.latent_dim})")
    print(f"Saved latent array shape: {latent.shape}")

    full_reconstruction = reconstruct_all(
        model=model,
        samples=samples_norm,
        mean=mean,
        std=std,
        original_shape=f_t_x_v.shape,
        batch_size=args.batch_size,
        device=device,
    )
    metrics = reconstruction_metrics(full_reconstruction, f_t_x_v)
    print(f"Reconstruction MSE   : {float(metrics['reconstruction_mse']):.6e}")
    print(f"Reconstruction RMSE  : {float(metrics['reconstruction_rmse']):.6e}")
    print(f"Reconstruction relL2 : {float(metrics['reconstruction_relative_l2']):.6e}")
    print(f"Reconstruction error : {float(metrics['reconstruction_relative_l2_percent']):.3f}%")

    reconstruction = full_reconstruction if args.save_reconstruction else None

    output_path = args.output if args.output is not None else default_output_path(case_dir)
    encoder_output_path = (
        args.encoder_output if args.encoder_output is not None else default_encoder_output_path(output_path)
    )
    save_results(
        output_path=output_path,
        case_dir=case_dir,
        t=t,
        x=x,
        v=v,
        original_shape=f_t_x_v.shape,
        latent=latent,
        mean=mean,
        std=std,
        train_losses=train_losses,
        val_losses=val_losses,
        lr_history=lr_history,
        smoothness_history=smoothness_history,
        dynamics_history=dynamics_history,
        model=model,
        device=device,
        args=args,
        metrics=metrics,
        reconstruction=reconstruction,
    )
    save_encoder_checkpoint(
        encoder_output_path=encoder_output_path,
        model=model,
        mean=mean,
        std=std,
        v=v,
        x=x,
        t=t,
        case_dir=case_dir,
    )
    print(f"Results written to: {output_path}")
    print(f"Encoder checkpoint written to: {encoder_output_path}")


if __name__ == "__main__":
    main()
