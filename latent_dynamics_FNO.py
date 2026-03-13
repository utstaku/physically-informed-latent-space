#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from latent_dynamics import check_uniform_spacing, load_latent_data, resolve_modes

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from neuralop.models import FNO
except ImportError as exc:
    raise SystemExit(
        "PyTorch and neuraloperator are required. Run this script in the 'lasdi' environment, for example:\n"
        "  conda run -n lasdi python latent_dynamics_FNO.py --latent-file <path>"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a 1D Fourier Neural Operator on latent trajectories so that "
            "z(x, t) -> z(x, t + dt)."
        )
    )
    parser.add_argument(
        "--latent-file",
        type=Path,
        required=True,
        help="Path to velocity_autoencoder_results.npz or compatible latent file.",
    )
    parser.add_argument(
        "--modes",
        type=int,
        nargs="*",
        default=None,
        help="Latent mode indices to use as FNO channels. Default is all modes.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of one-step transitions used for training.",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default="sequential",
        choices=("sequential", "random"),
        help="Whether to split transitions sequentially in time or randomly.",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-6, help="Adam weight decay.")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="plateau",
        choices=("none", "plateau"),
        help="Learning-rate scheduler.",
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=20,
        help="Epochs before reducing lr for the plateau scheduler.",
    )
    parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.5,
        help="Factor for the plateau scheduler.",
    )
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate.")
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=64,
        help="Hidden width of the FNO.",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=4,
        help="Number of Fourier layers.",
    )
    parser.add_argument(
        "--n-modes",
        type=int,
        default=16,
        help="Number of Fourier modes kept in the 1D spectral convolution.",
    )
    parser.add_argument(
        "--lifting-channel-ratio",
        type=float,
        default=2.0,
        help="FNO lifting block width ratio.",
    )
    parser.add_argument(
        "--projection-channel-ratio",
        type=float,
        default=2.0,
        help="FNO projection block width ratio.",
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="instance_norm",
        choices=("none", "instance_norm", "group_norm", "ada_in"),
        help="Normalization layer used inside the FNO.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout used in the FNO channel MLP.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for initialization and data split.",
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
        help="Output .npz path. Defaults to <latent-file-stem>_fno_dynamics.npz.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Output .pt checkpoint path. Defaults to <output-stem>.pt.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Human-readable text summary. Defaults to <output-stem>.txt.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Store one-step and rollout predictions in the output .npz file.",
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


def infer_output_path(latent_file: Path) -> Path:
    return latent_file.with_name(f"{latent_file.stem}_fno_dynamics.npz")


def infer_checkpoint_path(output_path: Path) -> Path:
    return output_path.with_suffix(".pt")


def infer_report_path(output_path: Path) -> Path:
    return output_path.with_suffix(".txt")


def resolve_spectral_modes(requested_modes: int, nx: int) -> int:
    return min(requested_modes, max(nx // 2, 1))


def build_one_step_pairs(latent: np.ndarray, modes: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    selected = latent[:, :, modes].astype(np.float32)
    inputs = np.transpose(selected[:-1], (0, 2, 1))
    targets = np.transpose(selected[1:], (0, 2, 1))
    return inputs, targets


def split_transition_indices(
    num_samples: int,
    train_fraction: float,
    split_mode: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if num_samples < 2:
        raise ValueError("At least two one-step transition samples are required.")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"train_fraction must lie in (0, 1), got {train_fraction}")

    if split_mode == "sequential":
        split = int(num_samples * train_fraction)
        split = min(max(split, 1), num_samples - 1)
        train_idx = np.arange(split, dtype=np.int64)
        val_idx = np.arange(split, num_samples, dtype=np.int64)
        return train_idx, val_idx

    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples, dtype=np.int64)
    rng.shuffle(indices)
    split = int(num_samples * train_fraction)
    split = min(max(split, 1), num_samples - 1)
    train_idx = np.sort(indices[:split])
    val_idx = np.sort(indices[split:])
    return train_idx, val_idx


def normalize_from_train(
    inputs: np.ndarray,
    targets: np.ndarray,
    train_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stacked = np.concatenate([inputs[train_idx], targets[train_idx]], axis=0)
    mean = stacked.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = stacked.std(axis=(0, 2), keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    normalized_inputs = ((inputs - mean) / std).astype(np.float32)
    normalized_targets = ((targets - mean) / std).astype(np.float32)
    return normalized_inputs, normalized_targets, mean, std


def build_loader(
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def build_model(
    num_channels: int,
    nx: int,
    args: argparse.Namespace,
) -> FNO:
    spectral_modes = resolve_spectral_modes(args.n_modes, nx)
    if spectral_modes < 1:
        raise ValueError(f"Invalid number of spectral modes derived from Nx={nx}")

    norm = None if args.norm == "none" else args.norm
    return FNO(
        n_modes=(spectral_modes,),
        in_channels=num_channels,
        out_channels=num_channels,
        hidden_channels=args.hidden_channels,
        n_layers=args.n_layers,
        lifting_channel_ratio=args.lifting_channel_ratio,
        projection_channel_ratio=args.projection_channel_ratio,
        positional_embedding="grid",
        norm=norm,
        channel_mlp_dropout=args.dropout,
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    criterion = nn.MSELoss(reduction="mean")
    total_loss = 0.0
    total_count = 0
    for batch_inputs, batch_targets in loader:
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        batch_preds = model(batch_inputs)
        loss = criterion(batch_preds, batch_targets)
        batch_size = batch_inputs.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
    return total_loss / max(total_count, 1)


def train_model(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    val_inputs: np.ndarray,
    val_targets: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[nn.Module, np.ndarray, np.ndarray, np.ndarray]:
    model = build_model(num_channels=train_inputs.shape[1], nx=train_inputs.shape[-1], args=args).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )

    criterion = nn.MSELoss(reduction="mean")
    train_loader = build_loader(train_inputs, train_targets, batch_size=args.batch_size, shuffle=True)
    eval_train_loader = build_loader(train_inputs, train_targets, batch_size=args.batch_size, shuffle=False)
    val_loader = build_loader(val_inputs, val_targets, batch_size=args.batch_size, shuffle=False)

    train_losses: List[float] = []
    val_losses: List[float] = []
    lr_history: List[float] = []

    for epoch in range(args.epochs):
        model.train()
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            batch_preds = model(batch_inputs)
            loss = criterion(batch_preds, batch_targets)
            loss.backward()
            optimizer.step()

        train_loss = evaluate_model(model, eval_train_loader, device)
        val_loss = evaluate_model(model, val_loader, device)
        if scheduler is not None:
            scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        lr_history.append(current_lr)
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs:03d} "
            f"train_mse={train_loss:.6e} val_mse={val_loss:.6e} lr={current_lr:.6e}"
        )

    return (
        model,
        np.asarray(train_losses, dtype=np.float32),
        np.asarray(val_losses, dtype=np.float32),
        np.asarray(lr_history, dtype=np.float32),
    )


@torch.no_grad()
def predict_batches(
    model: nn.Module,
    inputs: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(torch.from_numpy(inputs), batch_size=batch_size, shuffle=False, drop_last=False)
    outputs: List[np.ndarray] = []
    for batch_inputs in loader:
        batch_inputs = batch_inputs.to(device)
        outputs.append(model(batch_inputs).cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


@torch.no_grad()
def rollout_sequence(
    model: nn.Module,
    initial_state: np.ndarray,
    num_steps: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    del batch_size
    model.eval()
    state = torch.from_numpy(initial_state[None, ...]).to(device=device, dtype=torch.float32)
    outputs: List[np.ndarray] = []
    for _ in range(num_steps):
        state = model(state)
        outputs.append(state.squeeze(0).cpu().numpy().astype(np.float32))
    return np.stack(outputs, axis=0) if outputs else np.empty((0,) + initial_state.shape, dtype=np.float32)


def unnormalize(array: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (array * std + mean).astype(np.float32)


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, np.ndarray]:
    diff = pred - target
    mse = np.mean(diff**2, dtype=np.float64)
    rmse = np.float64(np.sqrt(mse))
    denom = np.linalg.norm(target.reshape(-1), ord=2)
    rel_l2 = np.linalg.norm(diff.reshape(-1), ord=2) / denom if denom > 0.0 else 0.0
    return {
        "mse": np.asarray(mse, dtype=np.float64),
        "rmse": np.asarray(rmse, dtype=np.float64),
        "relative_l2": np.asarray(rel_l2, dtype=np.float64),
        "relative_l2_percent": np.asarray(rel_l2 * 100.0, dtype=np.float64),
    }


def compute_rollout_payload(
    model: nn.Module,
    latent_channels_x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int,
    device: torch.device,
    split_mode: str,
) -> Tuple[np.ndarray | None, np.ndarray | None, Dict[str, np.ndarray]]:
    if len(val_idx) == 0:
        return None, None, {}

    if split_mode == "sequential":
        start = int(train_idx[-1] + 1)
        target = latent_channels_x[start + 1 :]
        if len(target) == 0:
            return None, None, {}
        initial_state = ((latent_channels_x[start : start + 1] - mean) / std)[0]
        pred_norm = rollout_sequence(model, initial_state, num_steps=len(target), batch_size=batch_size, device=device)
        pred = unnormalize(pred_norm, mean, std)
        return pred, target.astype(np.float32), {f"rollout_{k}": v for k, v in compute_metrics(pred, target).items()}

    prev_states = latent_channels_x[val_idx]
    targets = latent_channels_x[val_idx + 1]
    prev_states_norm = ((prev_states - mean) / std).astype(np.float32)
    pred_norm = predict_batches(model, prev_states_norm, batch_size=batch_size, device=device)
    pred = unnormalize(pred_norm, mean, std)
    return pred, targets.astype(np.float32), {f"rollout_{k}": v for k, v in compute_metrics(pred, targets).items()}


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    modes: Sequence[int],
    args: argparse.Namespace,
) -> None:
    checkpoint = {
        "model_type": "latent_fno_dynamics",
        "state_dict": model.state_dict(),
        "feature_mean": torch.from_numpy(mean.astype(np.float32)),
        "feature_std": torch.from_numpy(std.astype(np.float32)),
        "t": torch.from_numpy(t.astype(np.float32)),
        "x": torch.from_numpy(x.astype(np.float32)),
        "modes": torch.as_tensor(list(modes), dtype=torch.int64),
        "config": vars(args),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)


def save_report(report_path: Path, lines: Iterable[str]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_results(
    output_path: Path,
    latent_file: Path,
    t: np.ndarray,
    x: np.ndarray,
    dt: float,
    dx: float,
    modes: Sequence[int],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    spectral_modes: int,
    train_losses: np.ndarray,
    val_losses: np.ndarray,
    lr_history: np.ndarray,
    metrics: Dict[str, np.ndarray],
    args: argparse.Namespace,
    one_step_pred: np.ndarray | None,
    one_step_target: np.ndarray | None,
    rollout_pred: np.ndarray | None,
    rollout_target: np.ndarray | None,
) -> None:
    payload: Dict[str, np.ndarray] = {
        "latent_file": np.asarray(str(latent_file)),
        "model_type": np.asarray("latent_fno_dynamics"),
        "t": t.astype(np.float32),
        "x": x.astype(np.float32),
        "modes": np.asarray(list(modes), dtype=np.int32),
        "dt": np.asarray(dt, dtype=np.float32),
        "dx": np.asarray(dx, dtype=np.float32),
        "train_transition_indices": train_idx.astype(np.int32),
        "val_transition_indices": val_idx.astype(np.int32),
        "feature_mean": mean.astype(np.float32),
        "feature_std": std.astype(np.float32),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "learning_rate_history": lr_history,
        "epochs": np.asarray(args.epochs, dtype=np.int32),
        "batch_size": np.asarray(args.batch_size, dtype=np.int32),
        "learning_rate": np.asarray(args.learning_rate, dtype=np.float32),
        "weight_decay": np.asarray(args.weight_decay, dtype=np.float32),
        "hidden_channels": np.asarray(args.hidden_channels, dtype=np.int32),
        "n_layers": np.asarray(args.n_layers, dtype=np.int32),
        "n_modes_fourier": np.asarray(spectral_modes, dtype=np.int32),
        "train_fraction": np.asarray(args.train_fraction, dtype=np.float32),
        "split_mode": np.asarray(args.split_mode),
        "seed": np.asarray(args.seed, dtype=np.int32),
    }
    payload.update(metrics)

    if args.save_predictions:
        if one_step_pred is not None and one_step_target is not None:
            payload["one_step_prediction"] = one_step_pred.astype(np.float32)
            payload["one_step_target"] = one_step_target.astype(np.float32)
        if rollout_pred is not None and rollout_target is not None:
            payload["rollout_prediction"] = rollout_pred.astype(np.float32)
            payload["rollout_target"] = rollout_target.astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    latent_file = args.latent_file.resolve()
    latent, t, x, meta = load_latent_data(latent_file)
    del meta

    device = resolve_device(args.device)
    dt = check_uniform_spacing(t, "t")
    dx = check_uniform_spacing(x, "x")
    nt, nx, nz = latent.shape
    if nt < 3:
        raise ValueError(f"At least Nt=3 is required for one-step dynamics training, got Nt={nt}")

    modes = resolve_modes(args.modes, nz)
    spectral_modes = resolve_spectral_modes(args.n_modes, nx)
    inputs, targets = build_one_step_pairs(latent, modes)
    train_idx, val_idx = split_transition_indices(
        num_samples=inputs.shape[0],
        train_fraction=args.train_fraction,
        split_mode=args.split_mode,
        seed=args.seed,
    )
    inputs_norm, targets_norm, mean, std = normalize_from_train(inputs, targets, train_idx)

    train_inputs = inputs_norm[train_idx]
    train_targets = targets_norm[train_idx]
    val_inputs = inputs_norm[val_idx]
    val_targets = targets_norm[val_idx]

    output_path = args.output.resolve() if args.output is not None else infer_output_path(latent_file)
    checkpoint_path = (
        args.checkpoint.resolve() if args.checkpoint is not None else infer_checkpoint_path(output_path)
    )
    report_path = args.report.resolve() if args.report is not None else infer_report_path(output_path)

    print(f"Loaded latent data from: {latent_file}")
    print(f"Latent shape: Nt={nt}, Nx={nx}, Nz={nz}")
    print(f"Using latent modes: {modes}")
    print(f"Device: {device}")
    print(f"dx={dx:.6e}, dt={dt:.6e}")
    print(f"Train transitions: {len(train_idx)}, validation transitions: {len(val_idx)}")

    model, train_losses, val_losses, lr_history = train_model(
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        args=args,
        device=device,
    )

    one_step_pred_norm = predict_batches(model, val_inputs, batch_size=args.batch_size, device=device)
    one_step_pred = unnormalize(one_step_pred_norm, mean, std)
    one_step_target = targets[val_idx].astype(np.float32)
    one_step_metrics = {f"one_step_{k}": v for k, v in compute_metrics(one_step_pred, one_step_target).items()}

    latent_channels_x = np.transpose(latent[:, :, modes].astype(np.float32), (0, 2, 1))
    rollout_pred, rollout_target, rollout_metrics = compute_rollout_payload(
        model=model,
        latent_channels_x=latent_channels_x,
        mean=mean,
        std=std,
        train_idx=train_idx,
        val_idx=val_idx,
        batch_size=args.batch_size,
        device=device,
        split_mode=args.split_mode,
    )

    metrics: Dict[str, np.ndarray] = {
        "final_train_loss": np.asarray(train_losses[-1], dtype=np.float32),
        "final_val_loss": np.asarray(val_losses[-1], dtype=np.float32),
        **one_step_metrics,
        **rollout_metrics,
    }

    save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        mean=mean,
        std=std,
        t=t,
        x=x,
        modes=modes,
        args=args,
    )
    save_results(
        output_path=output_path,
        latent_file=latent_file,
        t=t,
        x=x,
        dt=dt,
        dx=dx,
        modes=modes,
        train_idx=train_idx,
        val_idx=val_idx,
        mean=mean,
        std=std,
        spectral_modes=spectral_modes,
        train_losses=train_losses,
        val_losses=val_losses,
        lr_history=lr_history,
        metrics=metrics,
        args=args,
        one_step_pred=one_step_pred,
        one_step_target=one_step_target,
        rollout_pred=rollout_pred,
        rollout_target=rollout_target,
    )

    report_lines = [
        f"latent_file: {latent_file}",
        f"latent_shape: (Nt={nt}, Nx={nx}, Nz={nz})",
        f"used_modes: {modes}",
        f"device: {device}",
        f"dt: {dt:.8e}",
        f"dx: {dx:.8e}",
        f"train_fraction: {args.train_fraction:.4f}",
        f"split_mode: {args.split_mode}",
        f"train_transitions: {len(train_idx)}",
        f"val_transitions: {len(val_idx)}",
        f"hidden_channels: {args.hidden_channels}",
        f"n_layers: {args.n_layers}",
        f"n_modes_fourier: {spectral_modes}",
        f"epochs: {args.epochs}",
        f"batch_size: {args.batch_size}",
        f"learning_rate: {args.learning_rate:.8e}",
        f"weight_decay: {args.weight_decay:.8e}",
        f"final_train_loss: {float(train_losses[-1]):.8e}",
        f"final_val_loss: {float(val_losses[-1]):.8e}",
        f"one_step_mse: {float(one_step_metrics['one_step_mse']):.8e}",
        f"one_step_rmse: {float(one_step_metrics['one_step_rmse']):.8e}",
        f"one_step_relative_l2: {float(one_step_metrics['one_step_relative_l2']):.8e}",
        f"one_step_relative_l2_percent: {float(one_step_metrics['one_step_relative_l2_percent']):.4f}",
    ]
    if rollout_metrics:
        report_lines.extend(
            [
                f"rollout_mse: {float(rollout_metrics['rollout_mse']):.8e}",
                f"rollout_rmse: {float(rollout_metrics['rollout_rmse']):.8e}",
                f"rollout_relative_l2: {float(rollout_metrics['rollout_relative_l2']):.8e}",
                f"rollout_relative_l2_percent: {float(rollout_metrics['rollout_relative_l2_percent']):.4f}",
            ]
        )
    report_lines.extend(
        [
            f"checkpoint: {checkpoint_path}",
            f"output: {output_path}",
        ]
    )
    save_report(report_path, report_lines)

    print(f"Saved checkpoint to: {checkpoint_path}")
    print(f"Saved metrics to: {output_path}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
