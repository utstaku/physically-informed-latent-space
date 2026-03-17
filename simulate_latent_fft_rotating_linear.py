#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required. Run this script in the 'lasdi' environment, for example:\n"
        "  conda run -n lasdi python simulate_latent_fft_rotating_linear.py --latent-file <path>"
    ) from exc

from reconstruct_e_field import load_distribution as load_case_distribution
from simulate_latent_pde_rk45 import (
    decode_latent_trajectory,
    electric_energy_per_time,
    load_autoencoder_from_latent_file,
    make_animation,
    mse_per_time,
    relative_l2_per_time,
    resolve_case_dir,
    save_electric_energy_plot,
    save_error_plot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "FFT the latent field in x, estimate a dominant phase frequency, remove that rotation, "
            "fit a linear ODE in the rotating frame, and compare damping and phase against truth."
        )
    )
    parser.add_argument(
        "--latent-file",
        type=Path,
        required=True,
        help="Path to conv_velocity_autoencoder_results*.npz containing latent(t, x, z).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <latent-file-stem>_fft_rotating_linear_eval.",
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=None,
        help="Optional case directory containing distribution_full.npz. Defaults to latent metadata.",
    )
    parser.add_argument(
        "--modes",
        type=int,
        nargs="*",
        default=None,
        help="Latent mode indices to include. Default is all modes.",
    )
    parser.add_argument(
        "--k-index",
        type=int,
        default=1,
        help="Discrete FFT index to analyze. Default: 1.",
    )
    parser.add_argument(
        "--fit-fraction",
        type=float,
        default=1.0,
        help="Fraction of the time history used to estimate omega and fit the linear ODE.",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1e-8,
        help="Ridge regularization for the linear regression on y_dot = A y in the rotating frame.",
    )
    parser.add_argument(
        "--omega-mode",
        type=int,
        default=None,
        help=(
            "Latent mode index used to estimate the dominant phase rate omega. "
            "Default: choose the mode with the largest mean |z_hat(k)| on the fit interval."
        ),
    )
    parser.add_argument(
        "--rotation-sign",
        type=str,
        default="auto",
        choices=("auto", "minus", "plus"),
        help=(
            "Phase-removal factor. 'minus' uses exp(-i omega t), 'plus' uses exp(+i omega t), "
            "and 'auto' picks the one that minimizes the residual phase rate of the reference mode."
        ),
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=None,
        help="Optional final time. If provided, truncate the latent trajectory to t <= t_end.",
    )
    parser.add_argument(
        "--fill-unmodeled",
        type=str,
        default="truth",
        choices=("truth", "initial", "zero"),
        help="How to fill latent content outside the modeled (mode, k) coefficients before decode.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size used while decoding latent trajectories.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=("cpu", "cuda"),
        help="Torch device used for decoding.",
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="Skip the decoded prediction animation.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=200,
        help="Maximum number of frames saved in the animation.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Animation frames per second.",
    )
    return parser.parse_args()


def infer_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    return args.latent_file.with_name(f"{args.latent_file.stem}_fft_rotating_linear_eval").resolve()


def load_latent_data(latent_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with np.load(latent_file, allow_pickle=False) as data:
        if "latent" not in data or "t" not in data or "x" not in data:
            raise KeyError(f"{latent_file} must contain 'latent', 't', and 'x'.")
        latent = np.asarray(data["latent"], dtype=np.float64)
        t = np.asarray(data["t"], dtype=np.float64)
        x = np.asarray(data["x"], dtype=np.float64)
        meta = {key: np.asarray(data[key]) for key in data.files}

    if latent.ndim != 3:
        raise ValueError(f"Expected latent shape (Nt, Nx, Nz), got {latent.shape}")
    if latent.shape[0] != t.size or latent.shape[1] != x.size:
        raise ValueError(
            f"Inconsistent latent/t/x shapes: latent={latent.shape}, len(t)={t.size}, len(x)={x.size}"
        )
    return latent, t, x, meta


def resolve_modes(requested_modes: Sequence[int] | None, nz: int) -> list[int]:
    if requested_modes is None or len(requested_modes) == 0:
        return list(range(nz))
    modes = sorted({int(mode) for mode in requested_modes})
    bad = [mode for mode in modes if mode < 0 or mode >= nz]
    if bad:
        raise ValueError(f"Requested latent modes out of range for Nz={nz}: {bad}")
    return modes


def validate_uniform_grid(values: np.ndarray, name: str) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"{name} must be a 1D array with at least two points.")
    diffs = np.diff(values)
    spacing = float(np.mean(diffs))
    if not np.allclose(diffs, spacing, rtol=1e-5, atol=1e-6):
        raise ValueError(f"{name} grid must be uniformly spaced.")
    return spacing


def validate_k_index(k_index: int, nx: int) -> int:
    k_index = int(k_index) % nx
    if k_index == 0:
        raise ValueError("k=0 is not supported here. Use a nonzero Fourier mode such as --k-index 1.")
    return k_index


def compute_fit_steps(nt: int, fit_fraction: float) -> int:
    if not (0.0 < fit_fraction <= 1.0):
        raise ValueError(f"--fit-fraction must lie in (0, 1], got {fit_fraction}")
    return max(3, min(nt, int(math.ceil(fit_fraction * nt))))


def estimate_phase_line(
    coefficient_t: np.ndarray,
    t: np.ndarray,
    fit_steps: int,
    amplitude_floor_scale: float = 1e-10,
) -> tuple[float, float, np.ndarray]:
    coefficient_t = np.asarray(coefficient_t, dtype=np.complex128)
    t = np.asarray(t, dtype=np.float64)
    fit_t = t[:fit_steps]
    fit_coeff = coefficient_t[:fit_steps]
    amplitude = np.abs(fit_coeff)
    phase = np.unwrap(np.angle(fit_coeff))

    reference = max(float(np.max(amplitude)), 1e-30)
    valid = amplitude > amplitude_floor_scale * reference
    if np.count_nonzero(valid) < 3:
        raise ValueError("Not enough non-negligible samples to estimate the dominant phase line.")

    omega, phase0 = np.polyfit(fit_t[valid], phase[valid], 1)
    return float(omega), float(phase0), valid


def choose_reference_mode_slot(
    coefficient_t_mode: np.ndarray,
    fit_steps: int,
    mode_indices: Sequence[int],
    requested_mode: int | None,
) -> int:
    if requested_mode is not None:
        if requested_mode not in mode_indices:
            raise ValueError(f"--omega-mode={requested_mode} is not in the selected modes {list(mode_indices)}")
        return list(mode_indices).index(requested_mode)

    fit_amplitude = np.abs(np.asarray(coefficient_t_mode[:fit_steps], dtype=np.complex128))
    mean_amplitude = np.mean(fit_amplitude, axis=0)
    return int(np.argmax(mean_amplitude))


def rotation_factor(t: np.ndarray, omega: float, sign: str) -> np.ndarray:
    sign_value = -1.0 if sign == "minus" else 1.0
    return np.exp(1j * sign_value * float(omega) * np.asarray(t, dtype=np.float64))


def choose_rotation_sign(
    coefficient_t: np.ndarray,
    t: np.ndarray,
    fit_steps: int,
    omega: float,
    sign_mode: str,
) -> tuple[str, np.ndarray, float]:
    if sign_mode in ("minus", "plus"):
        factor = rotation_factor(t, omega, sign_mode)
        rotated = np.asarray(coefficient_t, dtype=np.complex128) * factor
        residual_omega, _phase0, _valid = estimate_phase_line(rotated, t, fit_steps)
        return sign_mode, factor, residual_omega

    best_sign = None
    best_factor = None
    best_residual = None
    for candidate in ("minus", "plus"):
        factor = rotation_factor(t, omega, candidate)
        rotated = np.asarray(coefficient_t, dtype=np.complex128) * factor
        residual_omega, _phase0, _valid = estimate_phase_line(rotated, t, fit_steps)
        score = abs(residual_omega)
        if best_residual is None or score < best_residual:
            best_sign = candidate
            best_factor = factor
            best_residual = score

    assert best_sign is not None and best_factor is not None and best_residual is not None
    rotated = np.asarray(coefficient_t, dtype=np.complex128) * best_factor
    residual_omega, _phase0, _valid = estimate_phase_line(rotated, t, fit_steps)
    return best_sign, best_factor, residual_omega


def build_real_imag_state(coefficients_t_mode: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(coefficients_t_mode, dtype=np.complex128)
    nt, n_modes = coefficients.shape
    state = np.empty((nt, 2 * n_modes), dtype=np.float64)
    state[:, 0::2] = coefficients.real
    state[:, 1::2] = coefficients.imag
    return state


def state_to_complex_coefficients(state_t_feature: np.ndarray) -> np.ndarray:
    state = np.asarray(state_t_feature, dtype=np.float64)
    return state[:, 0::2] + 1j * state[:, 1::2]


def fit_linear_ode(
    state_t_feature: np.ndarray,
    t: np.ndarray,
    ridge_alpha: float,
    fit_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit_state = np.asarray(state_t_feature[:fit_steps], dtype=np.float64)
    fit_t = np.asarray(t[:fit_steps], dtype=np.float64)
    edge_order = 2 if fit_steps >= 3 else 1
    state_dt = np.gradient(fit_state, fit_t, axis=0, edge_order=edge_order)

    lhs = fit_state.T @ fit_state
    if ridge_alpha > 0.0:
        lhs = lhs + ridge_alpha * np.eye(lhs.shape[0], dtype=np.float64)
    rhs = fit_state.T @ state_dt
    regression_matrix = np.linalg.solve(lhs, rhs)
    a_matrix = regression_matrix.T
    fitted_dt = fit_state @ regression_matrix
    return a_matrix, state_dt, fitted_dt


def rollout_linear_system(a_matrix: np.ndarray, y0: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    current = np.asarray(y0, dtype=np.float64).copy()
    trajectory = np.empty((t.size, current.size), dtype=np.float64)
    trajectory[0] = current

    for index in range(1, t.size):
        dt = float(t[index] - t[index - 1])
        current = expm(a_matrix * dt) @ current
        trajectory[index] = current

    return trajectory


def build_fourier_baseline(latent_hat_true: np.ndarray, fill_unmodeled: str) -> np.ndarray:
    if fill_unmodeled == "truth":
        return np.asarray(latent_hat_true, dtype=np.complex128).copy()
    if fill_unmodeled == "initial":
        return np.repeat(np.asarray(latent_hat_true[0:1], dtype=np.complex128), latent_hat_true.shape[0], axis=0)
    if fill_unmodeled == "zero":
        return np.zeros_like(latent_hat_true, dtype=np.complex128)
    raise ValueError(f"Unsupported fill strategy: {fill_unmodeled}")


def inject_mode_coefficients(
    latent_hat_base: np.ndarray,
    coefficients_t_mode: np.ndarray,
    mode_indices: Sequence[int],
    k_index: int,
) -> np.ndarray:
    latent_hat = np.asarray(latent_hat_base, dtype=np.complex128).copy()
    coefficients = np.asarray(coefficients_t_mode, dtype=np.complex128)
    pair_index = (-k_index) % latent_hat.shape[1]

    latent_hat[:, k_index, :][:, mode_indices] = coefficients
    if pair_index == k_index:
        latent_hat[:, k_index, :][:, mode_indices] = coefficients.real
    else:
        latent_hat[:, pair_index, :][:, mode_indices] = np.conjugate(coefficients)
    return latent_hat


def inverse_fft_latent(latent_hat_t_x_z: np.ndarray) -> np.ndarray:
    latent = np.fft.ifft(np.asarray(latent_hat_t_x_z, dtype=np.complex128), axis=1)
    if np.max(np.abs(latent.imag)) > 1e-8:
        raise ValueError("Inverse FFT produced a non-negligible imaginary latent field.")
    return np.asarray(latent.real, dtype=np.float64)


def coefficient_relative_error_per_time(predicted_t_mode: np.ndarray, truth_t_mode: np.ndarray) -> np.ndarray:
    diff = np.asarray(predicted_t_mode, dtype=np.complex128) - np.asarray(truth_t_mode, dtype=np.complex128)
    diff_norm = np.linalg.norm(diff, axis=1)
    truth_norm = np.linalg.norm(np.asarray(truth_t_mode, dtype=np.complex128), axis=1)
    return np.where(truth_norm > 0.0, diff_norm / truth_norm, 0.0)


def analyze_mode_envelope_phase(
    coefficients_t_mode: np.ndarray,
    t: np.ndarray,
    fit_steps: int,
    amplitude_floor_scale: float = 1e-10,
) -> dict[str, np.ndarray]:
    coefficients = np.asarray(coefficients_t_mode, dtype=np.complex128)
    t = np.asarray(t, dtype=np.float64)

    amplitude = np.abs(coefficients)
    phase = np.unwrap(np.angle(coefficients), axis=0)
    n_modes = coefficients.shape[1]

    damping_rate = np.full(n_modes, np.nan, dtype=np.float64)
    phase_rate = np.full(n_modes, np.nan, dtype=np.float64)
    fit_mask = np.zeros((fit_steps, n_modes), dtype=bool)

    fit_t = t[:fit_steps]
    for mode_slot in range(n_modes):
        amp_fit = amplitude[:fit_steps, mode_slot]
        phase_fit = phase[:fit_steps, mode_slot]
        reference = max(float(np.max(amp_fit)), 1e-30)
        valid = np.isfinite(amp_fit) & np.isfinite(phase_fit) & (amp_fit > amplitude_floor_scale * reference)
        fit_mask[:, mode_slot] = valid
        if np.count_nonzero(valid) < 3:
            continue

        gamma, _intercept = np.polyfit(fit_t[valid], np.log(amp_fit[valid]), 1)
        omega, _phase0 = np.polyfit(fit_t[valid], phase_fit[valid], 1)
        damping_rate[mode_slot] = float(gamma)
        phase_rate[mode_slot] = float(omega)

    return {
        "amplitude": amplitude,
        "phase": phase,
        "damping_rate": damping_rate,
        "phase_rate": phase_rate,
        "fit_mask": fit_mask,
    }


def save_mode_amplitude_phase_plot(
    t: np.ndarray,
    truth_analysis: dict[str, np.ndarray],
    pred_analysis: dict[str, np.ndarray],
    mode_indices: Sequence[int],
    output_path: Path,
    title_prefix: str,
) -> None:
    n_modes = len(mode_indices)
    fig, axes = plt.subplots(
        n_modes,
        2,
        figsize=(12, max(3.0 * n_modes, 4.0)),
        constrained_layout=True,
        squeeze=False,
        sharex=True,
    )

    for row, mode_index in enumerate(mode_indices):
        ax_amp = axes[row, 0]
        ax_phase = axes[row, 1]

        truth_amp = np.asarray(truth_analysis["amplitude"][:, row], dtype=np.float64)
        pred_amp = np.asarray(pred_analysis["amplitude"][:, row], dtype=np.float64)
        truth_phase = np.asarray(truth_analysis["phase"][:, row], dtype=np.float64)
        pred_phase = np.asarray(pred_analysis["phase"][:, row], dtype=np.float64)

        ax_amp.plot(t, np.maximum(truth_amp, 1e-30), label="truth", lw=2)
        ax_amp.plot(t, np.maximum(pred_amp, 1e-30), label="linear rollout", lw=2)
        ax_amp.set_yscale("log")
        ax_amp.set_ylabel(f"mode {mode_index}\n|z|")
        ax_amp.grid(True, alpha=0.3, which="both")
        if row == 0:
            ax_amp.set_title(f"{title_prefix} amplitude")
            ax_amp.legend()

        ax_phase.plot(t, truth_phase, label="truth", lw=2)
        ax_phase.plot(t, pred_phase, label="linear rollout", lw=2)
        ax_phase.set_ylabel(f"mode {mode_index}\nphase [rad]")
        ax_phase.grid(True, alpha=0.3)
        if row == 0:
            ax_phase.set_title(f"{title_prefix} phase")
            ax_phase.legend()

    axes[-1, 0].set_xlabel("t")
    axes[-1, 1].set_xlabel("t")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_state_labels(mode_indices: Sequence[int], prefix: str, k_index: int) -> list[str]:
    labels: list[str] = []
    for mode_index in mode_indices:
        labels.append(f"Re({prefix}_{mode_index}[k_index={k_index}])")
        labels.append(f"Im({prefix}_{mode_index}[k_index={k_index}])")
    return labels


def format_linear_equations(
    a_matrix: np.ndarray,
    state_labels: Sequence[str],
    coefficient_tol: float = 1e-12,
) -> list[str]:
    equations: list[str] = []
    for row_index, lhs_label in enumerate(state_labels):
        terms: list[str] = []
        for col_index, rhs_label in enumerate(state_labels):
            coefficient = float(a_matrix[row_index, col_index])
            if abs(coefficient) < coefficient_tol:
                continue
            terms.append(f"{coefficient:.12e} * {rhs_label}")
        rhs_text = " + ".join(terms) if terms else "0"
        equations.append(f"d/dt {lhs_label} = {rhs_text}")
    return equations


def main() -> None:
    args = parse_args()
    args.latent_file = args.latent_file.resolve()
    output_dir = infer_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    latent_true, latent_t, latent_x, latent_meta = load_latent_data(args.latent_file)
    case_dir = resolve_case_dir(args.latent_file, latent_meta, args.case_dir)
    case_payload = load_case_distribution(case_dir)

    f_true = np.asarray(case_payload["f"], dtype=np.float64)
    t_true = np.asarray(case_payload["t"], dtype=np.float64)
    x_true = np.asarray(case_payload["x"], dtype=np.float64)
    v_true = np.asarray(case_payload["v"], dtype=np.float64)

    if not np.allclose(latent_t, t_true, rtol=1e-6, atol=1e-8):
        raise ValueError("Latent file time grid does not match distribution_full.npz.")
    if not np.allclose(latent_x, x_true, rtol=1e-6, atol=1e-8):
        raise ValueError("Latent file x grid does not match distribution_full.npz.")

    if args.t_end is not None:
        selected_mask = latent_t <= args.t_end + 1e-12
        if not np.any(selected_mask):
            raise ValueError(f"--t-end={args.t_end} produced an empty time window.")
        latent_true = latent_true[selected_mask]
        latent_t = latent_t[selected_mask]
        f_true = f_true[selected_mask]
        t_true = t_true[selected_mask]

    dx = validate_uniform_grid(latent_x, "x")
    k_values = 2.0 * np.pi * np.fft.fftfreq(latent_x.size, d=dx)
    k_index = validate_k_index(args.k_index, latent_x.size)
    k_physical = float(k_values[k_index])
    fit_steps = compute_fit_steps(latent_true.shape[0], args.fit_fraction)

    mode_indices = resolve_modes(args.modes, latent_true.shape[2])
    latent_selected = np.asarray(latent_true[:, :, mode_indices], dtype=np.float64)
    coefficient_truth = np.asarray(np.fft.fft(latent_selected, axis=1)[:, k_index, :], dtype=np.complex128)

    reference_mode_slot = choose_reference_mode_slot(coefficient_truth, fit_steps, mode_indices, args.omega_mode)
    reference_mode_index = mode_indices[reference_mode_slot]
    omega_raw, phase0_raw, _fit_valid = estimate_phase_line(coefficient_truth[:, reference_mode_slot], latent_t, fit_steps)
    rotation_sign, rot_factor, residual_ref_phase_rate = choose_rotation_sign(
        coefficient_truth[:, reference_mode_slot],
        latent_t,
        fit_steps,
        omega_raw,
        args.rotation_sign,
    )

    coefficient_rot_truth = coefficient_truth * rot_factor[:, None]
    state_rot_truth = build_real_imag_state(coefficient_rot_truth)
    a_matrix, fit_state_dt, fit_state_dt_pred = fit_linear_ode(
        state_t_feature=state_rot_truth,
        t=latent_t,
        ridge_alpha=args.ridge_alpha,
        fit_steps=fit_steps,
    )
    state_rot_pred = rollout_linear_system(a_matrix, state_rot_truth[0], latent_t)
    coefficient_rot_pred = state_to_complex_coefficients(state_rot_pred)
    coefficient_pred = coefficient_rot_pred / rot_factor[:, None]

    latent_hat_base = build_fourier_baseline(np.fft.fft(np.asarray(latent_true, dtype=np.float64), axis=1), args.fill_unmodeled)
    latent_pred = inverse_fft_latent(
        inject_mode_coefficients(
            latent_hat_base=latent_hat_base,
            coefficients_t_mode=coefficient_pred,
            mode_indices=mode_indices,
            k_index=k_index,
        )
    )

    fit_state_error = fit_state_dt - fit_state_dt_pred
    fit_state_dt_rel_l2 = np.linalg.norm(fit_state_error) / max(np.linalg.norm(fit_state_dt), 1e-12)
    coefficient_rel_error = coefficient_relative_error_per_time(coefficient_pred, coefficient_truth)

    original_truth = analyze_mode_envelope_phase(coefficient_truth, latent_t, fit_steps)
    original_pred = analyze_mode_envelope_phase(coefficient_pred, latent_t, fit_steps)
    rotated_truth = analyze_mode_envelope_phase(coefficient_rot_truth, latent_t, fit_steps)
    rotated_pred = analyze_mode_envelope_phase(coefficient_rot_pred, latent_t, fit_steps)

    damping_rate_error = original_pred["damping_rate"] - original_truth["damping_rate"]
    phase_rate_error = original_pred["phase_rate"] - original_truth["phase_rate"]
    residual_phase_rate_truth = rotated_truth["phase_rate"]
    residual_phase_rate_pred = rotated_pred["phase_rate"]

    state_labels = build_state_labels(mode_indices, "z_rot", k_index)
    linear_equations = format_linear_equations(a_matrix, state_labels)

    device = torch.device(args.device)
    model, feature_mean, feature_std = load_autoencoder_from_latent_file(args.latent_file, device)
    f_pred = decode_latent_trajectory(
        model=model,
        latent_t_x_z=latent_pred,
        feature_mean=feature_mean,
        feature_std=feature_std,
        batch_size=args.batch_size,
        device=device,
    )
    f_latent_truth = decode_latent_trajectory(
        model=model,
        latent_t_x_z=np.asarray(latent_true, dtype=np.float64),
        feature_mean=feature_mean,
        feature_std=feature_std,
        batch_size=args.batch_size,
        device=device,
    )

    decoded_rel_l2 = relative_l2_per_time(f_pred, f_true)
    ae_rel_l2 = relative_l2_per_time(f_latent_truth, f_true)
    decoded_mse = mse_per_time(f_pred, f_true)
    ae_mse = mse_per_time(f_latent_truth, f_true)
    truth_electric_energy, truth_electric_field = electric_energy_per_time(f_true, x_true, v_true)
    pred_electric_energy, pred_electric_field = electric_energy_per_time(f_pred, x_true, v_true)
    ae_electric_energy, ae_electric_field = electric_energy_per_time(f_latent_truth, x_true, v_true)

    np.savez_compressed(
        output_dir / "fft_rotating_linear_evaluation.npz",
        latent_file=np.asarray(str(args.latent_file)),
        case_dir=np.asarray(str(case_dir)),
        t=latent_t.astype(np.float64),
        x=latent_x.astype(np.float64),
        v=v_true.astype(np.float64),
        mode_indices=np.asarray(mode_indices, dtype=np.int32),
        k_index=np.asarray(k_index, dtype=np.int32),
        k_physical=np.asarray(k_physical, dtype=np.float64),
        fit_steps=np.asarray(fit_steps, dtype=np.int32),
        fit_fraction=np.asarray(args.fit_fraction, dtype=np.float64),
        ridge_alpha=np.asarray(args.ridge_alpha, dtype=np.float64),
        fill_unmodeled=np.asarray(args.fill_unmodeled),
        omega_reference_mode=np.asarray(reference_mode_index, dtype=np.int32),
        omega_raw=np.asarray(omega_raw, dtype=np.float64),
        phase0_raw=np.asarray(phase0_raw, dtype=np.float64),
        rotation_sign=np.asarray(rotation_sign),
        rotation_factor=rot_factor.astype(np.complex64),
        residual_reference_phase_rate=np.asarray(residual_ref_phase_rate, dtype=np.float64),
        linear_matrix_A=a_matrix.astype(np.float64),
        fourier_coeff_truth=coefficient_truth.astype(np.complex64),
        fourier_coeff_pred=coefficient_pred.astype(np.complex64),
        fourier_coeff_rot_truth=coefficient_rot_truth.astype(np.complex64),
        fourier_coeff_rot_pred=coefficient_rot_pred.astype(np.complex64),
        state_rot_truth=state_rot_truth.astype(np.float64),
        state_rot_pred=state_rot_pred.astype(np.float64),
        latent_pred=latent_pred.astype(np.float32),
        decoded_pred=f_pred.astype(np.float32),
        decoded_latent_truth=f_latent_truth.astype(np.float32),
        truth=f_true.astype(np.float32),
        truth_electric=truth_electric_field.astype(np.float32),
        pred_electric=pred_electric_field.astype(np.float32),
        decoded_latent_truth_electric=ae_electric_field.astype(np.float32),
        original_amplitude_truth=original_truth["amplitude"].astype(np.float64),
        original_amplitude_pred=original_pred["amplitude"].astype(np.float64),
        original_phase_truth=original_truth["phase"].astype(np.float64),
        original_phase_pred=original_pred["phase"].astype(np.float64),
        rotated_amplitude_truth=rotated_truth["amplitude"].astype(np.float64),
        rotated_amplitude_pred=rotated_pred["amplitude"].astype(np.float64),
        rotated_phase_truth=rotated_truth["phase"].astype(np.float64),
        rotated_phase_pred=rotated_pred["phase"].astype(np.float64),
        damping_rate_truth=original_truth["damping_rate"].astype(np.float64),
        damping_rate_pred=original_pred["damping_rate"].astype(np.float64),
        damping_rate_error=damping_rate_error.astype(np.float64),
        phase_rate_truth=original_truth["phase_rate"].astype(np.float64),
        phase_rate_pred=original_pred["phase_rate"].astype(np.float64),
        phase_rate_error=phase_rate_error.astype(np.float64),
        residual_phase_rate_truth=residual_phase_rate_truth.astype(np.float64),
        residual_phase_rate_pred=residual_phase_rate_pred.astype(np.float64),
        fourier_coeff_relative_l2=coefficient_rel_error.astype(np.float64),
        fit_state_dt_relative_l2=np.asarray(fit_state_dt_rel_l2, dtype=np.float64),
        decoded_relative_l2=decoded_rel_l2.astype(np.float64),
        ae_relative_l2=ae_rel_l2.astype(np.float64),
        decoded_mse=decoded_mse.astype(np.float64),
        ae_mse=ae_mse.astype(np.float64),
        truth_electric_energy=truth_electric_energy.astype(np.float64),
        pred_electric_energy=pred_electric_energy.astype(np.float64),
        ae_electric_energy=ae_electric_energy.astype(np.float64),
    )

    save_error_plot(latent_t, decoded_rel_l2, ae_rel_l2, "FFT-rotating-linear", output_dir / "error_over_time.png")
    save_electric_energy_plot(
        t=latent_t,
        truth_energy=truth_electric_energy,
        solver_energy=pred_electric_energy,
        ae_energy=ae_electric_energy,
        solver_name="FFT-rotating-linear",
        output_path=output_dir / "electric_energy_over_time.png",
    )
    save_mode_amplitude_phase_plot(
        latent_t,
        original_truth,
        original_pred,
        mode_indices,
        output_dir / "original_k_mode_amplitude_phase.png",
        title_prefix=f"Original k-index={k_index}",
    )
    save_mode_amplitude_phase_plot(
        latent_t,
        rotated_truth,
        rotated_pred,
        mode_indices,
        output_dir / "rotated_k_mode_amplitude_phase.png",
        title_prefix=f"Rotating-frame k-index={k_index}",
    )
    if not args.no_animation:
        make_animation(
            t=latent_t,
            x=latent_x,
            v=v_true,
            truth=f_true,
            prediction=f_pred,
            rel_l2=decoded_rel_l2,
            solver_name="FFT-rotating-linear",
            output_path=output_dir / "fft_rotating_linear_vs_truth.gif",
            max_frames=args.max_frames,
            fps=args.fps,
        )

    summary_lines = [
        "Rotating-frame latent linear regression",
        f"latent_file: {args.latent_file}",
        f"case_dir: {case_dir}",
        f"latent_shape: (Nt={latent_true.shape[0]}, Nx={latent_true.shape[1]}, Nz={latent_true.shape[2]})",
        f"mode_indices: {mode_indices}",
        f"k_index: {k_index}",
        f"k_physical: {k_physical:.12e}",
        f"fit_fraction: {args.fit_fraction:.6f}",
        f"fit_steps: {fit_steps}",
        f"ridge_alpha: {args.ridge_alpha:.12e}",
        f"fill_unmodeled: {args.fill_unmodeled}",
        f"omega_reference_mode: {reference_mode_index}",
        f"omega_raw_from_phase: {omega_raw:.12e}",
        f"phase0_raw: {phase0_raw:.12e}",
        f"rotation_sign: {rotation_sign}",
        f"residual_reference_phase_rate_after_rotation: {residual_ref_phase_rate:.12e}",
        f"linear_matrix_shape: {a_matrix.shape}",
        f"fit_state_dt_relative_l2: {fit_state_dt_rel_l2:.12e}",
        f"fourier_coeff_mean_relative_l2: {np.mean(coefficient_rel_error):.12e}",
        f"fourier_coeff_final_relative_l2: {coefficient_rel_error[-1]:.12e}",
        f"decoded_mean_relative_l2: {np.mean(decoded_rel_l2):.12e}",
        f"decoded_final_relative_l2: {decoded_rel_l2[-1]:.12e}",
        f"decoded_mean_mse: {np.mean(decoded_mse):.12e}",
        f"decoded_final_mse: {decoded_mse[-1]:.12e}",
        f"ae_mean_relative_l2: {np.mean(ae_rel_l2):.12e}",
        f"ae_final_relative_l2: {ae_rel_l2[-1]:.12e}",
        f"truth_initial_electric_energy: {truth_electric_energy[0]:.12e}",
        f"pred_initial_electric_energy: {pred_electric_energy[0]:.12e}",
        f"pred_final_electric_energy: {pred_electric_energy[-1]:.12e}",
        "",
        "Modewise damping and phase comparison:",
    ]
    for mode_slot, mode_index in enumerate(mode_indices):
        summary_lines.extend(
            [
                f"mode_{mode_index}_truth_damping_rate: {original_truth['damping_rate'][mode_slot]:.12e}",
                f"mode_{mode_index}_pred_damping_rate: {original_pred['damping_rate'][mode_slot]:.12e}",
                f"mode_{mode_index}_damping_rate_error: {damping_rate_error[mode_slot]:.12e}",
                f"mode_{mode_index}_truth_phase_rate: {original_truth['phase_rate'][mode_slot]:.12e}",
                f"mode_{mode_index}_pred_phase_rate: {original_pred['phase_rate'][mode_slot]:.12e}",
                f"mode_{mode_index}_phase_rate_error: {phase_rate_error[mode_slot]:.12e}",
                f"mode_{mode_index}_truth_residual_phase_rate_rot: {residual_phase_rate_truth[mode_slot]:.12e}",
                f"mode_{mode_index}_pred_residual_phase_rate_rot: {residual_phase_rate_pred[mode_slot]:.12e}",
            ]
        )
    summary_lines.extend(
        [
            "",
            "Learned rotating-frame linear system:",
            *linear_equations,
        ]
    )
    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Loaded latent autoencoder results from: {args.latent_file}")
    print(f"Selected k_index={k_index}, k={k_physical:.6e}")
    print(f"Reference mode for omega: {reference_mode_index}")
    print(f"Estimated omega from phase: {omega_raw:.6e}")
    print(f"Rotation sign: {rotation_sign}")
    print(f"Saved evaluation arrays to: {output_dir / 'fft_rotating_linear_evaluation.npz'}")
    print(f"Saved summary to: {output_dir / 'summary.txt'}")
    print(f"Final decoded relative L2 error: {decoded_rel_l2[-1]:.6e}")


if __name__ == "__main__":
    main()
