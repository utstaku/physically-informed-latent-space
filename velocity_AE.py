#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required. Run this script in the 'lasdi' environment, for example:\n"
        "  conda run -n lasdi python velocity_AE.py"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a PyTorch velocity-only autoencoder on one case from "
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
        help="Hidden width of the fully connected autoencoder.",
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
        help="Output .npz path. Defaults to <case-dir>/velocity_autoencoder_results.npz.",
    )
    parser.add_argument(
        "--save-reconstruction",
        action="store_true",
        help="Save reconstructed f(t, x, v) in the output file.",
    )
    return parser.parse_args()


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


class VelocityAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


def build_dataloader(samples: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor = torch.from_numpy(samples)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


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
    latent_dim: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    lr_scheduler_name: str,
    lr_patience: int,
    lr_factor: float,
    min_lr: float,
    device: torch.device,
) -> Tuple[VelocityAutoencoder, np.ndarray, np.ndarray, np.ndarray]:
    model = VelocityAutoencoder(
        input_dim=train_samples.shape[1],
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
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

    for epoch in range(epochs):
        model.train()
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()

        train_loss = evaluate_loss(model, eval_train_loader, device)
        val_loss = evaluate_loss(model, val_loader, device)
        if scheduler is not None:
            scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        lr_history.append(current_lr)
        print(
            f"Epoch {epoch + 1:03d}/{epochs:03d} "
            f"train_mse={train_loss:.6e} val_mse={val_loss:.6e} lr={current_lr:.6e}"
        )

    return (
        model,
        np.asarray(train_losses, dtype=np.float32),
        np.asarray(val_losses, dtype=np.float32),
        np.asarray(lr_history, dtype=np.float32),
    )


@torch.no_grad()
def encode_all(
    model: VelocityAutoencoder,
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
    model: VelocityAutoencoder,
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
        recon_batches.append(model(batch).cpu().numpy().astype(np.float32))
    recon = np.concatenate(recon_batches, axis=0)
    recon = recon * std + mean
    return recon.reshape(original_shape).astype(np.float32)


def reconstruction_metrics(reconstruction: np.ndarray, target: np.ndarray) -> Dict[str, np.ndarray]:
    diff = reconstruction - target
    mse = np.mean(diff**2, dtype=np.float64)
    rmse = np.float64(np.sqrt(mse))
    denom = np.linalg.norm(target.reshape(-1), ord=2)
    rel_l2 = np.linalg.norm(diff.reshape(-1), ord=2) / denom if denom > 0.0 else 0.0
    return {
        "reconstruction_mse": np.asarray(mse, dtype=np.float64),
        "reconstruction_rmse": np.asarray(rmse, dtype=np.float64),
        "reconstruction_relative_l2": np.asarray(rel_l2, dtype=np.float64),
    }


def extract_state_dict_numpy(model: nn.Module) -> Dict[str, np.ndarray]:
    state = {}
    for name, tensor in model.state_dict().items():
        key = name.replace(".", "__")
        state[f"param_{key}"] = tensor.detach().cpu().numpy().astype(np.float32)
    return state


def default_output_path(case_dir: Path) -> Path:
    return case_dir / "velocity_autoencoder_results.npz"


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
    model: VelocityAutoencoder,
    device: torch.device,
    metrics: Dict[str, np.ndarray],
    reconstruction: np.ndarray | None,
) -> None:
    payload: Dict[str, np.ndarray] = {
        "case_name": np.asarray(case_dir.name),
        "case_dir": np.asarray(str(case_dir)),
        "torch_device": np.asarray(str(device)),
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
    }
    payload.update(extract_state_dict_numpy(model))
    payload.update(metrics)

    if reconstruction is not None:
        payload["reconstruction"] = reconstruction

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    case_dir = resolve_case_dir(args.grid_dir, args.case_dir)
    f_t_x_v, t, x, v = load_distribution(case_dir)
    nt, nx, nv = f_t_x_v.shape

    print(f"Using case: {case_dir}")
    print(f"Loaded f with shape (Nt, Nx, Nv)=({nt}, {nx}, {nv})")
    print(f"Using torch device: {device}")

    samples = flatten_snapshots(f_t_x_v)
    rng = np.random.default_rng(args.seed)
    train_idx, val_idx = split_indices(len(samples), args.train_fraction, rng)
    samples_norm, mean, std = normalize_from_train(samples, train_idx)

    train_samples = samples_norm[train_idx]
    val_samples = samples_norm[val_idx]

    model, train_losses, val_losses, lr_history = train_autoencoder(
        train_samples=train_samples,
        val_samples=val_samples,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_name=args.lr_scheduler,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        min_lr=args.min_lr,
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

    reconstruction = full_reconstruction if args.save_reconstruction else None

    output_path = args.output if args.output is not None else default_output_path(case_dir)
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
        model=model,
        device=device,
        metrics=metrics,
        reconstruction=reconstruction,
    )
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()
