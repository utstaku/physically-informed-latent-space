#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

try:
    import torch
    from neuralop.models import FNO
except ImportError as exc:
    raise SystemExit(
        "PyTorch and neuraloperator are required. Run this script in the 'lasdi' environment, for example:\n"
        "  conda run -n lasdi python simulate_latent_FNO.py ..."
    ) from exc

from Conv_velocity_AE import ConvVelocityAutoencoder
from latent_dynamics import load_latent_data
from reconstruct_e_field import load_distribution as load_case_distribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Roll out a learned latent FNO in time, decode the latent trajectory with the "
            "convolutional autoencoder, and compare the result against Vlasov truth."
        )
    )
    parser.add_argument(
        "--fno-checkpoint",
        type=Path,
        required=True,
        help="Path to the latent FNO checkpoint (.pt) produced by latent_dynamics_FNO.py.",
    )
    parser.add_argument(
        "--latent-file",
        type=Path,
        default=None,
        help="Optional latent autoencoder .npz. Defaults to the path stored in the FNO checkpoint.",
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=None,
        help="Optional case directory containing distribution_full.npz. Defaults to latent metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <fno-checkpoint-stem>_eval next to the checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Torch device used for rollout and decoding.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Batch size used while decoding latent trajectories.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Time index used as the rollout initial condition.",
    )
    parser.add_argument(
        "--initial-source",
        type=str,
        default="encoder",
        choices=("encoder", "latent"),
        help=(
            "How to construct the initial latent state. "
            "'encoder' encodes f(x, v, t_start) with the AE encoder, "
            "'latent' uses the saved latent trajectory directly."
        ),
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Number of one-step FNO transitions to apply. Defaults to all remaining steps.",
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=None,
        help="Optional final time. Overrides --num-steps when provided.",
    )
    parser.add_argument(
        "--fill-unmodeled",
        type=str,
        default="truth",
        choices=("truth", "initial", "zero"),
        help=(
            "How to fill latent channels not covered by the FNO before decoding. "
            "'truth' is recommended unless all latent modes were modeled."
        ),
    )
    parser.add_argument(
        "--clip-min",
        type=float,
        default=None,
        help="Optional lower clip applied to decoded f before error evaluation.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=120,
        help="Maximum number of frames written to the GIF.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=12,
        help="GIF frames per second.",
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="Skip GIF generation.",
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


def infer_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    return (args.fno_checkpoint.resolve().parent / f"{args.fno_checkpoint.stem}_eval").resolve()


def scalar_meta_to_str(value: np.ndarray) -> str:
    array = np.asarray(value)
    return str(array.item()) if array.shape == () else str(array)


def resolve_case_dir(latent_file: Path, latent_meta: Dict[str, np.ndarray], case_dir_override: Path | None) -> Path:
    del latent_file
    if case_dir_override is not None:
        case_dir = case_dir_override.resolve()
        if not case_dir.is_dir():
            raise FileNotFoundError(f"Case directory not found: {case_dir}")
        return case_dir

    case_dir_value = latent_meta.get("case_dir")
    if case_dir_value is None:
        raise ValueError("Could not infer case_dir from the latent file. Please pass --case-dir explicitly.")
    case_dir = Path(scalar_meta_to_str(case_dir_value)).resolve()
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Inferred case directory not found: {case_dir}")
    return case_dir


def load_autoencoder_from_latent_file(
    latent_file: Path,
    device: torch.device,
) -> tuple[ConvVelocityAutoencoder, np.ndarray, np.ndarray]:
    with np.load(latent_file, allow_pickle=False) as data:
        if "conv_channels" not in data or "kernel_size" not in data or "hidden_dim" not in data:
            raise KeyError(f"{latent_file} does not look like a convolutional autoencoder results file.")

        input_dim = int(np.asarray(data["nv"]).item()) if "nv" in data else int(np.asarray(data["v"]).shape[0])
        hidden_dim = int(np.asarray(data["hidden_dim"]).item())
        latent_dim = int(np.asarray(data["nz"]).item())
        conv_channels = tuple(int(item) for item in np.asarray(data["conv_channels"], dtype=np.int32))
        kernel_size = int(np.asarray(data["kernel_size"]).item())
        padding_mode = str(np.asarray(data["padding_mode"]).item()) if "padding_mode" in data else "zeros"
        feature_mean = np.asarray(data["feature_mean"], dtype=np.float32)
        feature_std = np.asarray(data["feature_std"], dtype=np.float32)

        state_dict = {}
        for key in data.files:
            if key.startswith("param_"):
                state_key = key[len("param_") :].replace("__", ".")
                state_dict[state_key] = torch.from_numpy(np.asarray(data[key], dtype=np.float32))

    model = ConvVelocityAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        conv_channels=conv_channels,
        kernel_size=kernel_size,
        padding_mode=padding_mode,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, feature_mean, feature_std


def resolve_spectral_modes(requested_modes: int, nx: int) -> int:
    return min(requested_modes, max(nx // 2, 1))


def load_fno_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[FNO, np.ndarray, np.ndarray, list[int], Path, np.ndarray, np.ndarray, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "state_dict" not in checkpoint or "feature_mean" not in checkpoint or "feature_std" not in checkpoint:
        raise KeyError(f"{checkpoint_path} does not look like a latent FNO checkpoint.")

    config = dict(checkpoint.get("config", {}))
    hidden_channels = int(config["hidden_channels"])
    n_layers = int(config["n_layers"])
    requested_modes = int(config["n_modes"])
    lifting_ratio = float(config.get("lifting_channel_ratio", 2.0))
    projection_ratio = float(config.get("projection_channel_ratio", 2.0))
    dropout = float(config.get("dropout", 0.0))
    norm = config.get("norm", "instance_norm")
    norm = None if norm == "none" else norm

    x = np.asarray(checkpoint["x"].detach().cpu().numpy(), dtype=np.float32)
    t = np.asarray(checkpoint["t"].detach().cpu().numpy(), dtype=np.float32)
    modes = [int(item) for item in checkpoint["modes"].detach().cpu().numpy().tolist()]
    feature_mean = np.asarray(checkpoint["feature_mean"].detach().cpu().numpy(), dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"].detach().cpu().numpy(), dtype=np.float32)
    if feature_mean.ndim == 3 and feature_mean.shape[0] == 1:
        feature_mean = feature_mean[0]
    if feature_std.ndim == 3 and feature_std.shape[0] == 1:
        feature_std = feature_std[0]

    spectral_modes = resolve_spectral_modes(requested_modes, len(x))
    model = FNO(
        n_modes=(spectral_modes,),
        in_channels=len(modes),
        out_channels=len(modes),
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        lifting_channel_ratio=lifting_ratio,
        projection_channel_ratio=projection_ratio,
        positional_embedding="grid",
        norm=norm,
        channel_mlp_dropout=dropout,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    latent_file_from_config = config.get("latent_file")
    latent_file = Path(latent_file_from_config).resolve() if latent_file_from_config is not None else None
    return model, feature_mean, feature_std, modes, latent_file, t, x, config


@torch.no_grad()
def rollout_fno(
    model: FNO,
    initial_state: np.ndarray,
    num_steps: int,
    device: torch.device,
) -> np.ndarray:
    if num_steps < 0:
        raise ValueError(f"num_steps must be non-negative, got {num_steps}")

    state = torch.from_numpy(initial_state[None, ...]).to(device=device, dtype=torch.float32)
    states = [state.squeeze(0).cpu().numpy().astype(np.float32)]
    for _ in range(num_steps):
        state = model(state)
        states.append(state.squeeze(0).cpu().numpy().astype(np.float32))
    return np.stack(states, axis=0)


@torch.no_grad()
def encode_distribution_snapshot(
    model: ConvVelocityAutoencoder,
    snapshot_x_v: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    snapshot = np.asarray(snapshot_x_v, dtype=np.float32)
    normalized = ((snapshot - feature_mean) / feature_std).astype(np.float32)
    outputs: List[np.ndarray] = []
    for start in range(0, normalized.shape[0], batch_size):
        batch = torch.from_numpy(normalized[start : start + batch_size]).unsqueeze(1).to(
            device=device, dtype=torch.float32
        )
        encoded = model.encode(batch).cpu().numpy().astype(np.float32)
        outputs.append(encoded)
    return np.concatenate(outputs, axis=0)


@torch.no_grad()
def decode_latent_trajectory(
    model: ConvVelocityAutoencoder,
    latent_t_x_z: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    latent = np.asarray(latent_t_x_z, dtype=np.float32)
    nt, nx, nz = latent.shape
    flattened = latent.reshape(nt * nx, nz)
    outputs: List[np.ndarray] = []
    for start in range(0, flattened.shape[0], batch_size):
        batch = torch.from_numpy(flattened[start : start + batch_size]).to(device=device, dtype=torch.float32)
        decoded = model.decode(batch).squeeze(1).cpu().numpy().astype(np.float32)
        outputs.append(decoded)
    decoded = np.concatenate(outputs, axis=0)
    decoded = decoded * feature_std + feature_mean
    return decoded.reshape(nt, nx, -1).astype(np.float32)


def build_full_latent_prediction(
    latent_true: np.ndarray,
    latent_pred_modeled: np.ndarray,
    modeled_modes: Sequence[int],
    fill_unmodeled: str,
) -> np.ndarray:
    full = np.zeros_like(latent_true, dtype=np.float64)
    if fill_unmodeled == "truth":
        full[...] = latent_true
    elif fill_unmodeled == "initial":
        full[...] = latent_true[0:1]
    full[:, :, modeled_modes] = latent_pred_modeled
    return full


def relative_l2_per_time(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    diff = np.asarray(prediction, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    diff_norm = np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1)
    truth_norm = np.linalg.norm(np.asarray(truth, dtype=np.float64).reshape(truth.shape[0], -1), axis=1)
    return np.where(truth_norm > 0.0, diff_norm / truth_norm, 0.0)


def mse_per_time(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    diff = np.asarray(prediction, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return np.mean(diff**2, axis=tuple(range(1, diff.ndim)))


def make_animation(
    t: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    rel_l2: np.ndarray,
    output_path: Path,
    max_frames: int,
    fps: int,
) -> None:
    frame_count = len(t)
    stride = max(1, int(math.ceil(frame_count / max(max_frames, 1))))
    frame_indices = np.arange(0, frame_count, stride, dtype=int)
    if frame_indices[-1] != frame_count - 1:
        frame_indices = np.append(frame_indices, frame_count - 1)

    global_min = float(min(np.min(truth), np.min(prediction)))
    global_max = float(max(np.max(truth), np.max(prediction)))
    error_max = float(np.max(np.abs(prediction - truth)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax_truth, ax_pred = axes[0]
    ax_err, ax_hist = axes[1]

    truth_img = ax_truth.imshow(
        truth[0].T,
        origin="lower",
        aspect="auto",
        extent=[float(x[0]), float(x[-1]), float(v[0]), float(v[-1])],
        vmin=global_min,
        vmax=global_max,
        cmap="viridis",
    )
    pred_img = ax_pred.imshow(
        prediction[0].T,
        origin="lower",
        aspect="auto",
        extent=[float(x[0]), float(x[-1]), float(v[0]), float(v[-1])],
        vmin=global_min,
        vmax=global_max,
        cmap="viridis",
    )
    err_img = ax_err.imshow(
        np.abs(prediction[0] - truth[0]).T,
        origin="lower",
        aspect="auto",
        extent=[float(x[0]), float(x[-1]), float(v[0]), float(v[-1])],
        vmin=0.0,
        vmax=error_max,
        cmap="magma",
    )
    ax_truth.set_title("Vlasov Truth")
    ax_pred.set_title("Decoded FNO Rollout")
    ax_err.set_title("Absolute Error")
    for axis in (ax_truth, ax_pred, ax_err):
        axis.set_xlabel("x")
        axis.set_ylabel("v")

    fig.colorbar(truth_img, ax=ax_truth, shrink=0.85)
    fig.colorbar(pred_img, ax=ax_pred, shrink=0.85)
    fig.colorbar(err_img, ax=ax_err, shrink=0.85)

    ax_hist.plot(t, rel_l2, color="tab:blue", lw=2)
    marker_line = ax_hist.axvline(t[0], color="tab:red", lw=2)
    ax_hist.set_xlabel("t")
    ax_hist.set_ylabel("Relative L2 Error")
    ax_hist.set_title("Decoded FNO Error vs Truth")
    if t.size == 1:
        margin = 0.5
        ax_hist.set_xlim(float(t[0] - margin), float(t[0] + margin))
    else:
        ax_hist.set_xlim(float(t[0]), float(t[-1]))
    ax_hist.set_ylim(0.0, float(max(np.max(rel_l2) * 1.05, 1e-8)))

    def update(frame_slot: int) -> Iterable:
        frame_index = int(frame_indices[frame_slot])
        truth_img.set_data(truth[frame_index].T)
        pred_img.set_data(prediction[frame_index].T)
        err_img.set_data(np.abs(prediction[frame_index] - truth[frame_index]).T)
        marker_line.set_xdata([t[frame_index], t[frame_index]])
        fig.suptitle(f"t = {t[frame_index]:.3f}, relative L2 = {rel_l2[frame_index]:.3e}")
        return truth_img, pred_img, err_img, marker_line

    animation = FuncAnimation(fig, update, frames=len(frame_indices), interval=1000 / max(fps, 1), blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def save_error_plot(
    t: np.ndarray,
    rollout_error: np.ndarray,
    ae_error: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(t, rollout_error, label="decoded FNO vs truth", lw=2)
    ax.plot(t, ae_error, label="decoded latent truth vs truth", lw=2)
    ax.set_xlabel("t")
    ax.set_ylabel("Relative L2 Error")
    ax.set_title("Error Against Vlasov Truth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def resolve_rollout_window(
    t: np.ndarray,
    start_index: int,
    num_steps: int | None,
    t_end: float | None,
) -> tuple[int, int]:
    if start_index < 0 or start_index >= len(t):
        raise ValueError(f"--start-index must be in [0, {len(t) - 1}], got {start_index}")

    max_steps = len(t) - 1 - start_index
    if max_steps < 0:
        raise ValueError("No rollout steps available.")

    if t_end is not None:
        valid = np.where(t <= t_end + 1e-12)[0]
        valid = valid[valid >= start_index]
        if valid.size == 0:
            raise ValueError(f"--t-end={t_end} produced an empty rollout window.")
        final_index = int(valid[-1])
        return start_index, final_index - start_index

    if num_steps is None:
        return start_index, max_steps
    if num_steps < 0:
        raise ValueError(f"--num-steps must be non-negative, got {num_steps}")
    return start_index, min(num_steps, max_steps)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = infer_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.fno_checkpoint.resolve()
    (
        fno_model,
        fno_mean,
        fno_std,
        modeled_modes,
        latent_file_from_ckpt,
        checkpoint_t,
        checkpoint_x,
        config,
    ) = load_fno_checkpoint(checkpoint_path, device)

    latent_file = args.latent_file.resolve() if args.latent_file is not None else latent_file_from_ckpt
    if latent_file is None:
        raise ValueError("Could not infer latent file from the FNO checkpoint. Please pass --latent-file explicitly.")

    latent_true, latent_t, latent_x, latent_meta = load_latent_data(latent_file)
    case_dir = resolve_case_dir(latent_file, latent_meta, args.case_dir)
    truth_payload = load_case_distribution(case_dir)

    f_true = np.asarray(truth_payload["f"], dtype=np.float64)
    t_true = np.asarray(truth_payload["t"], dtype=np.float64)
    x_true = np.asarray(truth_payload["x"], dtype=np.float64)
    v_true = np.asarray(truth_payload["v"], dtype=np.float64)

    if latent_true.shape[2] <= max(modeled_modes):
        raise ValueError(
            f"Latent file has Nz={latent_true.shape[2]}, but the FNO checkpoint requires modes {modeled_modes}."
        )
    if checkpoint_t.shape != latent_t.shape or not np.allclose(checkpoint_t, latent_t, rtol=1e-6, atol=1e-8):
        raise ValueError("FNO checkpoint time grid does not match the latent file.")
    if checkpoint_x.shape != latent_x.shape or not np.allclose(checkpoint_x, latent_x, rtol=1e-6, atol=1e-8):
        raise ValueError("FNO checkpoint x grid does not match the latent file.")
    if not np.allclose(latent_t, t_true, rtol=1e-6, atol=1e-8):
        raise ValueError("Latent file time grid does not match distribution_full.npz.")
    if not np.allclose(latent_x, x_true, rtol=1e-6, atol=1e-8):
        raise ValueError("Latent file x grid does not match distribution_full.npz.")

    start_index, num_steps = resolve_rollout_window(t_true, args.start_index, args.num_steps, args.t_end)
    stop_index = start_index + num_steps

    latent_window = latent_true[start_index : stop_index + 1]
    f_true_window = f_true[start_index : stop_index + 1]
    t_window = t_true[start_index : stop_index + 1]

    autoencoder, feature_mean, feature_std = load_autoencoder_from_latent_file(latent_file, device)

    if args.initial_source == "encoder":
        initial_latent_full = encode_distribution_snapshot(
            model=autoencoder,
            snapshot_x_v=f_true[start_index],
            feature_mean=feature_mean,
            feature_std=feature_std,
            batch_size=args.batch_size,
            device=device,
        )
    else:
        initial_latent_full = latent_true[start_index].astype(np.float32)

    initial_modeled = initial_latent_full[:, modeled_modes].T.astype(np.float32)
    initial_modeled_norm = ((initial_modeled - fno_mean) / fno_std).astype(np.float32)
    latent_modeled_norm = rollout_fno(
        model=fno_model,
        initial_state=initial_modeled_norm,
        num_steps=num_steps,
        device=device,
    )
    latent_modeled = (latent_modeled_norm * fno_std + fno_mean).astype(np.float32)
    latent_modeled = np.transpose(latent_modeled, (0, 2, 1))

    latent_pred_full = build_full_latent_prediction(
        latent_true=np.asarray(latent_window, dtype=np.float64),
        latent_pred_modeled=np.asarray(latent_modeled, dtype=np.float64),
        modeled_modes=modeled_modes,
        fill_unmodeled=args.fill_unmodeled,
    )

    f_pred = decode_latent_trajectory(
        model=autoencoder,
        latent_t_x_z=latent_pred_full,
        feature_mean=feature_mean,
        feature_std=feature_std,
        batch_size=args.batch_size,
        device=device,
    )
    f_latent_truth = decode_latent_trajectory(
        model=autoencoder,
        latent_t_x_z=np.asarray(latent_window, dtype=np.float64),
        feature_mean=feature_mean,
        feature_std=feature_std,
        batch_size=args.batch_size,
        device=device,
    )

    if args.clip_min is not None:
        f_pred = np.maximum(f_pred, args.clip_min)
        f_latent_truth = np.maximum(f_latent_truth, args.clip_min)

    latent_rel_l2 = relative_l2_per_time(latent_pred_full, latent_window)
    decoded_rel_l2 = relative_l2_per_time(f_pred, f_true_window)
    ae_rel_l2 = relative_l2_per_time(f_latent_truth, f_true_window)
    decoded_mse = mse_per_time(f_pred, f_true_window)
    ae_mse = mse_per_time(f_latent_truth, f_true_window)

    np.savez_compressed(
        output_dir / "fno_evaluation.npz",
        fno_checkpoint=np.asarray(str(checkpoint_path)),
        latent_file=np.asarray(str(latent_file)),
        case_dir=np.asarray(str(case_dir)),
        t=t_window.astype(np.float64),
        x=x_true.astype(np.float64),
        v=v_true.astype(np.float64),
        modeled_modes=np.asarray(modeled_modes, dtype=np.int32),
        start_index=np.asarray(start_index, dtype=np.int32),
        num_steps=np.asarray(num_steps, dtype=np.int32),
        initial_source=np.asarray(args.initial_source),
        fill_unmodeled=np.asarray(args.fill_unmodeled),
        latent_prediction_modeled=latent_modeled.astype(np.float32),
        latent_prediction_full=latent_pred_full.astype(np.float32),
        latent_truth=latent_window.astype(np.float32),
        decoded_prediction=f_pred.astype(np.float32),
        decoded_latent_truth=f_latent_truth.astype(np.float32),
        truth=f_true_window.astype(np.float32),
        latent_relative_l2=latent_rel_l2.astype(np.float64),
        decoded_relative_l2=decoded_rel_l2.astype(np.float64),
        decoded_mse=decoded_mse.astype(np.float64),
        ae_relative_l2=ae_rel_l2.astype(np.float64),
        ae_mse=ae_mse.astype(np.float64),
    )

    save_error_plot(t_window, decoded_rel_l2, ae_rel_l2, output_dir / "error_over_time.png")
    if not args.no_animation:
        make_animation(
            t=t_window,
            x=x_true,
            v=v_true,
            truth=f_true_window,
            prediction=f_pred,
            rel_l2=decoded_rel_l2,
            output_path=output_dir / "fno_vs_truth.gif",
            max_frames=args.max_frames,
            fps=args.fps,
        )

    summary_lines = [
        f"fno_checkpoint: {checkpoint_path}",
        f"latent_file: {latent_file}",
        f"case_dir: {case_dir}",
        f"device: {device}",
        f"modeled_modes: {modeled_modes}",
        f"initial_source: {args.initial_source}",
        f"fill_unmodeled: {args.fill_unmodeled}",
        f"start_index: {start_index}",
        f"num_steps: {num_steps}",
        f"t_start: {t_window[0]:.12e}",
        f"t_end: {t_window[-1]:.12e}",
        f"hidden_channels: {config.get('hidden_channels')}",
        f"n_layers: {config.get('n_layers')}",
        f"n_modes_fourier_requested: {config.get('n_modes')}",
        f"latent_mean_relative_l2: {np.mean(latent_rel_l2):.12e}",
        f"latent_max_relative_l2: {np.max(latent_rel_l2):.12e}",
        f"latent_final_relative_l2: {latent_rel_l2[-1]:.12e}",
        f"decoded_mean_relative_l2: {np.mean(decoded_rel_l2):.12e}",
        f"decoded_max_relative_l2: {np.max(decoded_rel_l2):.12e}",
        f"decoded_final_relative_l2: {decoded_rel_l2[-1]:.12e}",
        f"decoded_mean_mse: {np.mean(decoded_mse):.12e}",
        f"decoded_final_mse: {decoded_mse[-1]:.12e}",
        f"ae_mean_relative_l2: {np.mean(ae_rel_l2):.12e}",
        f"ae_max_relative_l2: {np.max(ae_rel_l2):.12e}",
        f"ae_final_relative_l2: {ae_rel_l2[-1]:.12e}",
        f"ae_mean_mse: {np.mean(ae_mse):.12e}",
        f"ae_final_mse: {ae_mse[-1]:.12e}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Loaded FNO checkpoint from: {checkpoint_path}")
    print(f"Loaded latent autoencoder results from: {latent_file}")
    print(f"Using case directory: {case_dir}")
    print(f"Rollout window: start_index={start_index}, num_steps={num_steps}, t=[{t_window[0]:.3f}, {t_window[-1]:.3f}]")
    print(f"Saved evaluation arrays to: {output_dir / 'fno_evaluation.npz'}")
    print(f"Saved error plot to: {output_dir / 'error_over_time.png'}")
    if not args.no_animation:
        print(f"Saved animation to: {output_dir / 'fno_vs_truth.gif'}")
    print(f"Final decoded relative L2 error: {decoded_rel_l2[-1]:.6e}")
    print(f"Final decoder-only relative L2 error: {ae_rel_l2[-1]:.6e}")


if __name__ == "__main__":
    main()
