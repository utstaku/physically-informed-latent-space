#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "PyTorch and neuraloperator are required. Run this script in the 'lasdi' environment, for example:\n"
        "  conda run -n lasdi python latent_dynamics_FNO_grid.py --latent-file <path>"
    ) from exc

from Conv_velocity_AE import load_distribution
from latent_dynamics import resolve_modes
from latent_dynamics_FNO import (
    compute_metrics,
    infer_checkpoint_path,
    infer_report_path,
    predict_batches,
    resolve_device,
    resolve_rollout_weights,
    rollout_sequence,
    save_checkpoint,
    set_seed,
    train_model,
    unnormalize,
)
from simulate_latent_FNO import (
    build_full_latent_prediction,
    decode_latent_trajectory,
    encode_distribution_snapshot,
    load_autoencoder_from_latent_file,
    relative_l2_per_time,
    resolve_rollout_window,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a latent FNO on Conv_velocity_AE_grid.py output and evaluate decoded "
            "rollouts on the full Vlasov two-stream parameter grid."
        )
    )
    parser.add_argument(
        "--latent-file",
        type=Path,
        required=True,
        help="Path to Conv_velocity_AE_grid.py output (.npz).",
    )
    parser.add_argument(
        "--grid-dir",
        type=Path,
        default=Path("vlasov_twostream_param_grid"),
        help="Directory that contains the full Vlasov parameter grid.",
    )
    parser.add_argument(
        "--modes",
        type=int,
        nargs="*",
        default=None,
        help="Latent mode indices to model with the FNO. Default is all latent modes.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=5,
        help="Number of recursive rollout steps K used in the training loss.",
    )
    parser.add_argument(
        "--rollout-loss-weights",
        type=str,
        default=None,
        help="Optional comma-separated weights for steps 2..K in the rollout loss.",
    )
    parser.add_argument(
        "--lambda-one-step",
        type=float,
        default=1.0,
        help="Weight applied to the one-step loss term.",
    )
    parser.add_argument(
        "--lambda-rollout",
        type=float,
        default=0.1,
        help="Weight applied to rollout loss terms l_2,...,l_K.",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="FNO mini-batch size.")
    parser.add_argument(
        "--decode-batch-size",
        type=int,
        default=2048,
        help="Batch size used while encoding initial conditions and decoding latent trajectories.",
    )
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
        "--start-index",
        type=int,
        default=0,
        help="Time index used as the rollout initial condition for full-grid evaluation.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Number of rollout steps evaluated on each full-grid case. Defaults to the full remaining horizon.",
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=None,
        help="Optional final time for evaluation. Overrides --num-steps when provided.",
    )
    parser.add_argument(
        "--fill-unmodeled",
        type=str,
        default="truth",
        choices=("truth", "initial", "zero"),
        help=(
            "How to fill latent channels not modeled by the FNO before decoding. "
            "Use 'truth' only when you intentionally fit a subset of latent modes."
        ),
    )
    parser.add_argument(
        "--clip-min",
        type=float,
        default=None,
        help="Optional lower clip applied to decoded f before error evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for initialization.",
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
        help="Output .npz path for FNO training metrics. Defaults to <latent-file-stem>_fno_grid_dynamics.npz.",
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
        "--eval-output",
        type=Path,
        default=None,
        help="Output .npz path for full-grid rollout metrics. Defaults to <output-stem>_full_grid_eval.npz.",
    )
    parser.add_argument(
        "--error-map-output",
        type=Path,
        default=None,
        help="Output image path for the full-grid error map. Defaults to <output-stem>_error_map.png.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="FNO Error Map on Vlasov Full Grid (convae, max relative error %)",
        help="Title used for the saved error-map figure.",
    )
    parser.add_argument(
        "--annotate-decimals",
        type=int,
        default=1,
        help="Number of decimals shown in each error-map cell.",
    )
    return parser.parse_args()


def infer_output_path(latent_file: Path) -> Path:
    return latent_file.with_name(f"{latent_file.stem}_fno_grid_dynamics.npz")


def infer_eval_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_full_grid_eval.npz")


def infer_error_map_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_error_map.png")


def scalar_str(value: np.ndarray) -> str:
    array = np.asarray(value)
    return str(array.item()) if array.shape == () else str(array.tolist())


def parse_case_parameters(case_dir: Path) -> Tuple[float, float]:
    parts = case_dir.name.split("_")
    if len(parts) != 4 or parts[0] != "T" or parts[2] != "k":
        raise ValueError(f"Unexpected case directory name: {case_dir.name}")
    return float(parts[1]), float(parts[3])


def grid_value_key(value: float) -> str:
    return f"{float(value):.8f}"


def list_case_dirs(grid_dir: Path) -> List[Path]:
    case_dirs = [path for path in sorted(grid_dir.iterdir()) if path.is_dir() and path.name.startswith("T_")]
    if not case_dirs:
        raise FileNotFoundError(f"No case directories found in {grid_dir}")
    return case_dirs


def load_grid_latent_data(
    latent_file: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    with np.load(latent_file, allow_pickle=False) as data:
        required = ("train_latent", "validation_latent", "t", "x")
        for key in required:
            if key not in data:
                raise KeyError(f"{latent_file} does not contain required key '{key}'.")
        train_latent = np.asarray(data["train_latent"], dtype=np.float32)
        validation_latent = np.asarray(data["validation_latent"], dtype=np.float32)
        t = np.asarray(data["t"], dtype=np.float32)
        x = np.asarray(data["x"], dtype=np.float32)
        meta: Dict[str, np.ndarray] = {}
        for key in ("training_case_names", "validation_case_name", "selected_t_values", "selected_k_values"):
            if key in data:
                meta[key] = np.asarray(data[key])
    if train_latent.ndim != 4:
        raise ValueError(f"Expected train_latent with shape (Nc, Nt, Nx, Nz), got {train_latent.shape}")
    if validation_latent.ndim != 3:
        raise ValueError(f"Expected validation_latent with shape (Nt, Nx, Nz), got {validation_latent.shape}")
    return train_latent, validation_latent, t, x, meta


def build_case_channel_sequences(latent_cases: np.ndarray, modes: Sequence[int]) -> np.ndarray:
    selected = np.asarray(latent_cases[..., modes], dtype=np.float32)
    return np.transpose(selected, (0, 1, 3, 2))


def build_single_channel_sequence(latent_case: np.ndarray, modes: Sequence[int]) -> np.ndarray:
    selected = np.asarray(latent_case[:, :, modes], dtype=np.float32)
    return np.transpose(selected, (0, 2, 1))


def normalize_train_sequences(
    train_sequences: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_sequences.mean(axis=(0, 1, 3), keepdims=True).astype(np.float32)
    std = train_sequences.std(axis=(0, 1, 3), keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    normalized = ((train_sequences - mean) / std).astype(np.float32)
    return normalized, mean[:, 0], std[:, 0]


def apply_sequence_normalization(
    sequence: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return ((sequence - mean) / std).astype(np.float32)


def build_rollout_windows(
    sequences: np.ndarray,
    rollout_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if sequences.ndim != 4:
        raise ValueError(f"Expected sequences with shape (Nc, Nt, C, Nx), got {sequences.shape}")
    num_cases, nt, channels, nx = sequences.shape
    if nt < rollout_steps + 1:
        raise ValueError(
            f"At least Nt={rollout_steps + 1} is required for rollout_steps={rollout_steps}, got Nt={nt}"
        )
    num_windows = nt - rollout_steps
    inputs = sequences[:, :num_windows]
    targets = np.stack(
        [sequences[:, step : step + num_windows] for step in range(1, rollout_steps + 1)],
        axis=2,
    )
    return (
        inputs.reshape(num_cases * num_windows, channels, nx).astype(np.float32),
        targets.reshape(num_cases * num_windows, rollout_steps, channels, nx).astype(np.float32),
    )


def encode_distribution_trajectory(
    model: torch.nn.Module,
    distribution_t_x_v: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    field = np.asarray(distribution_t_x_v, dtype=np.float32)
    nt, nx, nv = field.shape
    normalized = ((field - feature_mean.reshape(1, 1, nv)) / feature_std.reshape(1, 1, nv)).astype(np.float32)
    flattened = normalized.reshape(nt * nx, nv)
    outputs: List[np.ndarray] = []
    for start in range(0, flattened.shape[0], batch_size):
        batch = torch.from_numpy(flattened[start : start + batch_size]).unsqueeze(1).to(device=device, dtype=torch.float32)
        encoded = model.encode(batch).cpu().numpy().astype(np.float32)
        outputs.append(encoded)
    return np.concatenate(outputs, axis=0).reshape(nt, nx, -1).astype(np.float32)


def compute_validation_rollout_metrics(
    model: torch.nn.Module,
    validation_sequence: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    if validation_sequence.shape[0] < 2:
        return {}

    prev_states = validation_sequence[:-1]
    targets = validation_sequence[1:]
    prev_states_norm = apply_sequence_normalization(prev_states, mean[0], std[0])
    one_step_pred_norm = predict_batches(model, prev_states_norm, batch_size=batch_size, device=device)
    one_step_pred = unnormalize(one_step_pred_norm, mean[0], std[0])
    one_step_metrics = {f"one_step_{key}": value for key, value in compute_metrics(one_step_pred, targets).items()}

    initial_state_norm = apply_sequence_normalization(validation_sequence[0], mean[0], std[0])
    rollout_future_norm = rollout_sequence(
        model=model,
        initial_state=initial_state_norm,
        num_steps=validation_sequence.shape[0] - 1,
        batch_size=batch_size,
        device=device,
    )
    rollout_future = unnormalize(rollout_future_norm, mean[0], std[0])
    rollout_pred = np.concatenate([validation_sequence[0:1], rollout_future], axis=0)
    rollout_metrics = {f"rollout_{key}": value for key, value in compute_metrics(rollout_pred, validation_sequence).items()}
    return {**one_step_metrics, **rollout_metrics}


def case_relative_error_curve(
    f_true: np.ndarray,
    f_pred: np.ndarray,
) -> np.ndarray:
    return relative_l2_per_time(f_pred, f_true).astype(np.float32)


def evaluate_case_rollout(
    case_dir: Path,
    autoencoder: torch.nn.Module,
    fno_model: torch.nn.Module,
    modeled_modes: Sequence[int],
    latent_dim: int,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    fno_mean: np.ndarray,
    fno_std: np.ndarray,
    start_index: int,
    num_steps: int,
    batch_size: int,
    fill_unmodeled: str,
    clip_min: float | None,
    device: torch.device,
    t_reference: np.ndarray,
    x_reference: np.ndarray,
) -> np.ndarray:
    f_t_x_v, t_case, x_case, _v_case = load_distribution(case_dir)
    if not np.allclose(t_case, t_reference, rtol=1e-6, atol=1e-8):
        raise ValueError(f"Time grid mismatch for case {case_dir}")
    if not np.allclose(x_case, x_reference, rtol=1e-6, atol=1e-8):
        raise ValueError(f"Space grid mismatch for case {case_dir}")

    truth_window = np.asarray(f_t_x_v[start_index : start_index + num_steps + 1], dtype=np.float32)
    initial_latent_full = encode_distribution_snapshot(
        model=autoencoder,
        snapshot_x_v=f_t_x_v[start_index],
        feature_mean=feature_mean,
        feature_std=feature_std,
        batch_size=batch_size,
        device=device,
    )
    initial_modeled = initial_latent_full[:, modeled_modes].T.astype(np.float32)
    initial_modeled_norm = apply_sequence_normalization(initial_modeled, fno_mean[0], fno_std[0])
    latent_future_norm = rollout_sequence(
        model=fno_model,
        initial_state=initial_modeled_norm,
        num_steps=num_steps,
        batch_size=batch_size,
        device=device,
    )
    latent_future = unnormalize(latent_future_norm, fno_mean[0], fno_std[0])
    latent_future = np.transpose(latent_future, (0, 2, 1))
    latent_modeled = np.concatenate([initial_latent_full[None, :, modeled_modes], latent_future], axis=0)

    if len(modeled_modes) == latent_dim and fill_unmodeled == "truth":
        latent_pred_full = latent_modeled
    else:
        if fill_unmodeled == "truth":
            latent_truth_full = encode_distribution_trajectory(
                model=autoencoder,
                distribution_t_x_v=truth_window,
                feature_mean=feature_mean,
                feature_std=feature_std,
                batch_size=batch_size,
                device=device,
            )
        elif fill_unmodeled == "initial":
            latent_truth_full = np.repeat(initial_latent_full[None, ...], num_steps + 1, axis=0)
        else:
            latent_truth_full = np.zeros((num_steps + 1, truth_window.shape[1], latent_dim), dtype=np.float32)

        latent_pred_full = build_full_latent_prediction(
            latent_true=np.asarray(latent_truth_full, dtype=np.float64),
            latent_pred_modeled=np.asarray(latent_modeled, dtype=np.float64),
            modeled_modes=modeled_modes,
            fill_unmodeled=fill_unmodeled,
        ).astype(np.float32)

    f_pred = decode_latent_trajectory(
        model=autoencoder,
        latent_t_x_z=latent_pred_full,
        feature_mean=feature_mean,
        feature_std=feature_std,
        batch_size=batch_size,
        device=device,
    )
    if clip_min is not None:
        f_pred = np.maximum(f_pred, clip_min)
    error_curve = case_relative_error_curve(truth_window, f_pred)
    return error_curve


def build_error_grid(
    case_names: Sequence[str],
    case_max_errors: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    parsed = [(name, *parse_case_parameters(Path(name))) for name in case_names]
    t_values = np.asarray(sorted({t for _name, t, _k in parsed}), dtype=np.float32)
    k_values = np.asarray(sorted({k for _name, _t, k in parsed}), dtype=np.float32)
    grid = np.full((len(k_values), len(t_values)), np.nan, dtype=np.float32)
    t_index = {grid_value_key(value): idx for idx, value in enumerate(t_values.tolist())}
    k_index = {grid_value_key(value): idx for idx, value in enumerate(k_values.tolist())}
    for (name, t_value, k_value), error in zip(parsed, case_max_errors):
        del name
        grid[k_index[grid_value_key(k_value)], t_index[grid_value_key(t_value)]] = float(error)
    return grid, t_values, k_values


def plot_error_map(
    output_path: Path,
    error_grid: np.ndarray,
    t_values: np.ndarray,
    k_values: np.ndarray,
    training_case_names: Sequence[str],
    title: str,
    annotate_decimals: int,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    image = ax.imshow(
        error_grid,
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        extent=[float(t_values[0]), float(t_values[-1]), float(k_values[0]), float(k_values[-1])],
    )
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Wavenumber k")
    ax.set_title(title)

    if len(t_values) > 1:
        dt = float(t_values[1] - t_values[0])
    else:
        dt = 1.0
    if len(k_values) > 1:
        dk = float(k_values[1] - k_values[0])
    else:
        dk = 1.0

    for j, k_value in enumerate(k_values):
        for i, t_value in enumerate(t_values):
            value = error_grid[j, i]
            if np.isfinite(value):
                ax.text(
                    float(t_value),
                    float(k_value),
                    f"{value:.{annotate_decimals}f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black",
                )

    for case_name in training_case_names:
        t_value, k_value = parse_case_parameters(Path(case_name))
        rect = Rectangle(
            (t_value - 0.5 * dt, k_value - 0.5 * dk),
            dt,
            dk,
            fill=False,
            edgecolor="black",
            linewidth=1.5,
        )
        ax.add_patch(rect)

    colorbar = fig.colorbar(image, ax=ax, shrink=0.96)
    colorbar.set_label("Maximum relative error (%)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_results(
    output_path: Path,
    latent_file: Path,
    t: np.ndarray,
    x: np.ndarray,
    modes: Sequence[int],
    mean: np.ndarray,
    std: np.ndarray,
    train_losses: np.ndarray,
    val_losses: np.ndarray,
    lr_history: np.ndarray,
    metrics: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> None:
    payload: Dict[str, np.ndarray] = {
        "latent_file": np.asarray(str(latent_file)),
        "model_type": np.asarray("latent_fno_grid_dynamics"),
        "t": np.asarray(t, dtype=np.float32),
        "x": np.asarray(x, dtype=np.float32),
        "modes": np.asarray(list(modes), dtype=np.int32),
        "feature_mean": np.asarray(mean, dtype=np.float32),
        "feature_std": np.asarray(std, dtype=np.float32),
        "train_losses": np.asarray(train_losses, dtype=np.float32),
        "val_losses": np.asarray(val_losses, dtype=np.float32),
        "learning_rate_history": np.asarray(lr_history, dtype=np.float32),
        "epochs": np.asarray(args.epochs, dtype=np.int32),
        "batch_size": np.asarray(args.batch_size, dtype=np.int32),
        "learning_rate": np.asarray(args.learning_rate, dtype=np.float32),
        "weight_decay": np.asarray(args.weight_decay, dtype=np.float32),
        "hidden_channels": np.asarray(args.hidden_channels, dtype=np.int32),
        "n_layers": np.asarray(args.n_layers, dtype=np.int32),
        "n_modes_fourier": np.asarray(args.n_modes, dtype=np.int32),
        "rollout_steps": np.asarray(args.rollout_steps, dtype=np.int32),
        "rollout_loss_weights": resolve_rollout_weights(args.rollout_steps, args.rollout_loss_weights),
        "lambda_one_step": np.asarray(args.lambda_one_step, dtype=np.float32),
        "lambda_rollout": np.asarray(args.lambda_rollout, dtype=np.float32),
        "seed": np.asarray(args.seed, dtype=np.int32),
    }
    payload.update({key: np.asarray(value) for key, value in metrics.items()})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)


def save_eval_results(
    output_path: Path,
    case_names: Sequence[str],
    case_error_curves: np.ndarray,
    case_max_errors: np.ndarray,
    error_grid: np.ndarray,
    t_values: np.ndarray,
    k_values: np.ndarray,
    training_case_names: Sequence[str],
    validation_case_name: str,
    t_eval: np.ndarray,
) -> None:
    payload: Dict[str, np.ndarray] = {
        "case_names": np.asarray(list(case_names)),
        "case_error_curves": np.asarray(case_error_curves, dtype=np.float32),
        "case_max_relative_error_percent": np.asarray(case_max_errors, dtype=np.float32),
        "error_grid_percent": np.asarray(error_grid, dtype=np.float32),
        "temperature_values": np.asarray(t_values, dtype=np.float32),
        "wavenumber_values": np.asarray(k_values, dtype=np.float32),
        "training_case_names": np.asarray(list(training_case_names)),
        "validation_case_name": np.asarray(validation_case_name),
        "evaluation_time": np.asarray(t_eval, dtype=np.float32),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)


def save_report(report_path: Path, lines: Iterable[str]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.latent_file = args.latent_file.resolve()
    args.grid_dir = args.grid_dir.resolve()
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_latent, validation_latent, t, x, meta = load_grid_latent_data(args.latent_file)
    training_case_names = [str(item) for item in np.asarray(meta.get("training_case_names", []))]
    validation_case_name = scalar_str(meta.get("validation_case_name", np.asarray("unknown")))

    num_train_cases, nt, nx, nz = train_latent.shape
    if nt < args.rollout_steps + 1:
        raise ValueError(
            f"At least Nt={args.rollout_steps + 1} is required for rollout_steps={args.rollout_steps}, got Nt={nt}"
        )

    modeled_modes = resolve_modes(args.modes, nz)
    rollout_weights = resolve_rollout_weights(args.rollout_steps, args.rollout_loss_weights)

    train_sequences = build_case_channel_sequences(train_latent, modeled_modes)
    validation_sequence = build_single_channel_sequence(validation_latent, modeled_modes)
    train_sequences_norm, mean, std = normalize_train_sequences(train_sequences)
    validation_sequence_norm = apply_sequence_normalization(validation_sequence, mean[0], std[0])

    train_inputs, train_targets = build_rollout_windows(train_sequences_norm, args.rollout_steps)
    val_inputs, val_targets = build_rollout_windows(validation_sequence_norm[None, ...], args.rollout_steps)

    output_path = args.output.resolve() if args.output is not None else infer_output_path(args.latent_file)
    checkpoint_path = args.checkpoint.resolve() if args.checkpoint is not None else infer_checkpoint_path(output_path)
    report_path = args.report.resolve() if args.report is not None else infer_report_path(output_path)
    eval_output_path = args.eval_output.resolve() if args.eval_output is not None else infer_eval_output_path(output_path)
    error_map_path = (
        args.error_map_output.resolve() if args.error_map_output is not None else infer_error_map_path(output_path)
    )

    print(f"Loaded grid latent data from: {args.latent_file}")
    print(f"Training latent shape: {train_latent.shape}")
    print(f"Validation latent shape: {validation_latent.shape}")
    print(f"Using latent modes: {modeled_modes}")
    print(f"Device: {device}")
    print(f"Rollout steps K: {args.rollout_steps}")
    print(f"Training windows: {train_inputs.shape[0]}")
    print(f"Validation windows: {val_inputs.shape[0]}")

    model, train_losses, val_losses, lr_history = train_model(
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        rollout_weights=rollout_weights,
        lambda_one_step=args.lambda_one_step,
        lambda_rollout=args.lambda_rollout,
        args=args,
        device=device,
    )

    validation_metrics = compute_validation_rollout_metrics(
        model=model,
        validation_sequence=validation_sequence,
        mean=mean,
        std=std,
        batch_size=args.batch_size,
        device=device,
    )
    summary_metrics: Dict[str, np.ndarray] = {
        "final_train_loss": np.asarray(train_losses[-1], dtype=np.float32),
        "final_val_loss": np.asarray(val_losses[-1], dtype=np.float32),
        **validation_metrics,
    }

    save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        mean=mean,
        std=std,
        t=t,
        x=x,
        modes=modeled_modes,
        args=args,
    )
    save_results(
        output_path=output_path,
        latent_file=args.latent_file,
        t=t,
        x=x,
        modes=modeled_modes,
        mean=mean,
        std=std,
        train_losses=train_losses,
        val_losses=val_losses,
        lr_history=lr_history,
        metrics=summary_metrics,
        args=args,
    )

    autoencoder, ae_feature_mean, ae_feature_std = load_autoencoder_from_latent_file(args.latent_file, device)
    all_case_dirs = list_case_dirs(args.grid_dir)

    start_index, num_steps = resolve_rollout_window(t, args.start_index, args.num_steps, args.t_end)
    evaluation_time = np.asarray(t[start_index : start_index + num_steps + 1], dtype=np.float32)
    case_names: List[str] = []
    case_error_curves: List[np.ndarray] = []
    case_max_errors: List[float] = []

    for case_dir in all_case_dirs:
        error_curve = evaluate_case_rollout(
            case_dir=case_dir,
            autoencoder=autoencoder,
            fno_model=model,
            modeled_modes=modeled_modes,
            latent_dim=nz,
            feature_mean=ae_feature_mean,
            feature_std=ae_feature_std,
            fno_mean=mean,
            fno_std=std,
            start_index=start_index,
            num_steps=num_steps,
            batch_size=args.decode_batch_size,
            fill_unmodeled=args.fill_unmodeled,
            clip_min=args.clip_min,
            device=device,
            t_reference=t,
            x_reference=x,
        )
        max_error_percent = 100.0 * float(np.max(error_curve))
        case_names.append(case_dir.name)
        case_error_curves.append(error_curve)
        case_max_errors.append(max_error_percent)
        print(f"{case_dir.name}: max decoded relative error = {max_error_percent:.3f}%")

    case_error_curves_array = np.asarray(case_error_curves, dtype=np.float32)
    case_max_errors_array = np.asarray(case_max_errors, dtype=np.float32)
    error_grid, t_values, k_values = build_error_grid(case_names, case_max_errors_array)

    save_eval_results(
        output_path=eval_output_path,
        case_names=case_names,
        case_error_curves=case_error_curves_array,
        case_max_errors=case_max_errors_array,
        error_grid=error_grid,
        t_values=t_values,
        k_values=k_values,
        training_case_names=training_case_names,
        validation_case_name=validation_case_name,
        t_eval=evaluation_time,
    )
    plot_error_map(
        output_path=error_map_path,
        error_grid=error_grid,
        t_values=t_values,
        k_values=k_values,
        training_case_names=training_case_names,
        title=args.title,
        annotate_decimals=args.annotate_decimals,
    )

    worst_index = int(np.argmax(case_max_errors_array))
    best_index = int(np.argmin(case_max_errors_array))
    report_lines = [
        f"latent_file: {args.latent_file}",
        f"grid_dir: {args.grid_dir}",
        f"train_latent_shape: {train_latent.shape}",
        f"validation_latent_shape: {validation_latent.shape}",
        f"used_modes: {modeled_modes}",
        f"device: {device}",
        f"rollout_steps: {args.rollout_steps}",
        f"lambda_one_step: {args.lambda_one_step:.8e}",
        f"lambda_rollout: {args.lambda_rollout:.8e}",
        f"rollout_loss_weights: {rollout_weights.tolist()}",
        f"train_windows: {train_inputs.shape[0]}",
        f"validation_windows: {val_inputs.shape[0]}",
        f"hidden_channels: {args.hidden_channels}",
        f"n_layers: {args.n_layers}",
        f"n_modes_fourier: {args.n_modes}",
        f"epochs: {args.epochs}",
        f"batch_size: {args.batch_size}",
        f"decode_batch_size: {args.decode_batch_size}",
        f"learning_rate: {args.learning_rate:.8e}",
        f"weight_decay: {args.weight_decay:.8e}",
        f"evaluation_start_index: {start_index}",
        f"evaluation_num_steps: {num_steps}",
        f"fill_unmodeled: {args.fill_unmodeled}",
        f"final_train_loss: {float(train_losses[-1]):.8e}",
        f"final_val_loss: {float(val_losses[-1]):.8e}",
    ]
    for key in sorted(validation_metrics):
        report_lines.append(f"{key}: {float(validation_metrics[key]):.8e}")
    report_lines.extend(
        [
            f"full_grid_mean_max_relative_error_percent: {float(np.mean(case_max_errors_array)):.6f}",
            f"full_grid_median_max_relative_error_percent: {float(np.median(case_max_errors_array)):.6f}",
            f"full_grid_worst_case: {case_names[worst_index]} ({float(case_max_errors_array[worst_index]):.6f}%)",
            f"full_grid_best_case: {case_names[best_index]} ({float(case_max_errors_array[best_index]):.6f}%)",
            f"checkpoint: {checkpoint_path}",
            f"output: {output_path}",
            f"eval_output: {eval_output_path}",
            f"error_map: {error_map_path}",
        ]
    )
    save_report(report_path, report_lines)

    print(f"Saved checkpoint to: {checkpoint_path}")
    print(f"Saved FNO metrics to: {output_path}")
    print(f"Saved full-grid evaluation to: {eval_output_path}")
    print(f"Saved error map to: {error_map_path}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
