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
        "  conda run -n lasdi python simulate_latent_fft_linear.py --latent-file <path>"
    ) from exc

from latent_dynamics import load_latent_data, resolve_modes
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
            "Model dominant Fourier-mode latent dynamics with a linear ODE, decode the rollout, "
            "and compare against the Vlasov truth."
        )
    )
    parser.add_argument(
        "--latent-file",
        type=Path,
        required=True,
        help="Path to conv_velocity_autoencoder_results*.npz containing latent(t, x, z).",
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
        help="Output directory. Defaults to <latent-file-stem>_fft_linear_eval.",
    )
    parser.add_argument(
        "--modes",
        type=int,
        nargs="*",
        default=None,
        help="Latent mode indices to include in the FFT state. Default is all modes.",
    )
    parser.add_argument(
        "--k-index",
        type=int,
        default=None,
        help="Discrete FFT index to model. If omitted, infer from --wave-number, case metadata, or dominant energy.",
    )
    parser.add_argument(
        "--wave-number",
        type=float,
        default=None,
        help="Physical positive wave number to model. The nearest FFT mode is used.",
    )
    parser.add_argument(
        "--fit-fraction",
        type=float,
        default=1.0,
        help="Fraction of the time history used to identify the linear model.",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1e-8,
        help="Ridge regularization for the linear regression on y_dot = A y.",
    )
    parser.add_argument(
        "--fill-unmodeled",
        type=str,
        default="truth",
        choices=("truth", "initial", "zero"),
        help=(
            "How to fill Fourier coefficients outside the modeled (mode, k) pairs before inverse FFT and decode. "
            "'truth' is recommended to isolate the dominant-wave dynamics."
        ),
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=None,
        help="Optional final time. If provided, truncate the latent and truth trajectories to t <= t_end.",
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
    return args.latent_file.with_name(f"{args.latent_file.stem}_fft_linear_eval").resolve()


def validate_uniform_grid(values: np.ndarray, name: str) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"{name} must be a 1D array with at least two points.")
    diffs = np.diff(values)
    spacing = float(np.mean(diffs))
    if not np.allclose(diffs, spacing, rtol=1e-5, atol=1e-6):
        raise ValueError(f"{name} grid must be uniformly spaced.")
    return spacing


def infer_k_index(
    args: argparse.Namespace,
    x: np.ndarray,
    latent_hat: np.ndarray,
    case_payload: dict[str, np.ndarray],
) -> tuple[int, float, str]:
    nx = x.size
    dx = validate_uniform_grid(x, "x")
    k_values = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    positive = np.where(k_values > 0.0)[0]
    if positive.size == 0:
        raise ValueError("Could not find a positive Fourier wave number on the x grid.")

    def normalize_index(index: int) -> int:
        index = int(index) % nx
        if index == 0:
            raise ValueError("k=0 is not supported for the dominant-wave model. Choose a nonzero Fourier mode.")
        return index

    if args.k_index is not None:
        selected = normalize_index(args.k_index)
        return selected, float(abs(k_values[selected])), "user --k-index"

    target_wave_number = None
    source = None
    if args.wave_number is not None:
        target_wave_number = abs(float(args.wave_number))
        source = "user --wave-number"
    elif "k0" in case_payload:
        target_wave_number = abs(float(np.asarray(case_payload["k0"]).item()))
        source = "case metadata k0"
    elif "k" in case_payload:
        target_wave_number = abs(float(np.asarray(case_payload["k"]).item()))
        source = "case metadata k"

    if target_wave_number is not None:
        best_index = int(positive[np.argmin(np.abs(k_values[positive] - target_wave_number))])
        return best_index, float(abs(k_values[best_index])), source or "wave-number"

    initial_energy = np.mean(np.abs(latent_hat[0, positive, :]) ** 2, axis=1)
    best_index = int(positive[np.argmax(initial_energy)])
    return best_index, float(abs(k_values[best_index])), "dominant initial latent spectrum"


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
    fit_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if not (0.0 < fit_fraction <= 1.0):
        raise ValueError(f"--fit-fraction must lie in (0, 1], got {fit_fraction}")

    nt = state_t_feature.shape[0]
    fit_steps = int(math.ceil(fit_fraction * nt))
    fit_steps = max(3, min(nt, fit_steps))

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
    return a_matrix, state_dt, fitted_dt, fit_steps


def rollout_linear_system(a_matrix: np.ndarray, y0: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    y0 = np.asarray(y0, dtype=np.float64)
    trajectory = np.empty((t.size, y0.size), dtype=np.float64)
    trajectory[0] = y0
    current = y0.copy()

    for index in range(1, t.size):
        dt = float(t[index] - t[index - 1])
        current = expm(a_matrix * dt) @ current
        trajectory[index] = current

    return trajectory


def build_fourier_baseline(
    latent_hat_true: np.ndarray,
    fill_unmodeled: str,
) -> np.ndarray:
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


def coefficient_relative_error_per_time(
    predicted_t_mode: np.ndarray,
    truth_t_mode: np.ndarray,
) -> np.ndarray:
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
    fit_steps = max(3, min(int(fit_steps), coefficients.shape[0]))

    amplitude = np.abs(coefficients)
    phase = np.unwrap(np.angle(coefficients), axis=0)
    n_modes = coefficients.shape[1]

    damping_rate = np.full(n_modes, np.nan, dtype=np.float64)
    damping_intercept = np.full(n_modes, np.nan, dtype=np.float64)
    phase_rate = np.full(n_modes, np.nan, dtype=np.float64)
    phase_intercept = np.full(n_modes, np.nan, dtype=np.float64)
    amplitude_fit_mask = np.zeros((fit_steps, n_modes), dtype=bool)

    fit_t = t[:fit_steps]
    for mode_slot in range(n_modes):
        amp_fit = amplitude[:fit_steps, mode_slot]
        phase_fit = phase[:fit_steps, mode_slot]

        reference = max(float(np.max(amp_fit)), 1e-30)
        amplitude_floor = amplitude_floor_scale * reference
        valid = np.isfinite(amp_fit) & np.isfinite(phase_fit) & (amp_fit > amplitude_floor)
        amplitude_fit_mask[:, mode_slot] = valid
        if np.count_nonzero(valid) < 3:
            continue

        gamma, gamma_intercept = np.polyfit(fit_t[valid], np.log(amp_fit[valid]), 1)
        omega, phase0 = np.polyfit(fit_t[valid], phase_fit[valid], 1)
        damping_rate[mode_slot] = float(gamma)
        damping_intercept[mode_slot] = float(gamma_intercept)
        phase_rate[mode_slot] = float(omega)
        phase_intercept[mode_slot] = float(phase0)

    return {
        "amplitude": amplitude,
        "phase": phase,
        "damping_rate": damping_rate,
        "damping_intercept": damping_intercept,
        "phase_rate": phase_rate,
        "phase_intercept": phase_intercept,
        "fit_mask": amplitude_fit_mask,
    }


def save_mode_amplitude_phase_plot(
    t: np.ndarray,
    truth_analysis: dict[str, np.ndarray],
    pred_analysis: dict[str, np.ndarray],
    mode_indices: Sequence[int],
    output_path: Path,
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

        amp_eps = 1e-30
        ax_amp.plot(t, np.maximum(truth_amp, amp_eps), label="truth", lw=2)
        ax_amp.plot(t, np.maximum(pred_amp, amp_eps), label="linear rollout", lw=2)
        ax_amp.set_yscale("log")
        ax_amp.set_ylabel(f"mode {mode_index}\n|z_hat|")
        ax_amp.grid(True, alpha=0.3, which="both")
        if row == 0:
            ax_amp.set_title("Dominant-k Amplitude")
            ax_amp.legend()

        ax_phase.plot(t, truth_phase, label="truth", lw=2)
        ax_phase.plot(t, pred_phase, label="linear rollout", lw=2)
        ax_phase.set_ylabel(f"mode {mode_index}\nphase [rad]")
        ax_phase.grid(True, alpha=0.3)
        if row == 0:
            ax_phase.set_title("Dominant-k Phase")
            ax_phase.legend()

    axes[-1, 0].set_xlabel("t")
    axes[-1, 1].set_xlabel("t")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_state_labels(mode_indices: Sequence[int], k_index: int, k_physical: float) -> list[str]:
    labels: list[str] = []
    for mode_index in mode_indices:
        labels.append(f"Re(z_hat_{mode_index}[k_index={k_index}, k={k_physical:.6e}])")
        labels.append(f"Im(z_hat_{mode_index}[k_index={k_index}, k={k_physical:.6e}])")
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
        f_true = f_true[selected_mask]
        latent_t = latent_t[selected_mask]
        t_true = t_true[selected_mask]

    mode_indices = resolve_modes(args.modes, latent_true.shape[2])
    latent_selected = np.asarray(latent_true[:, :, mode_indices], dtype=np.float64)
    latent_hat_selected = np.fft.fft(latent_selected, axis=1)

    k_index, k_physical, k_source = infer_k_index(args, latent_x, latent_hat_selected, case_payload)
    coefficient_truth = np.asarray(latent_hat_selected[:, k_index, :], dtype=np.complex128)
    state_truth = build_real_imag_state(coefficient_truth)

    a_matrix, fit_state_dt, fit_state_dt_pred, fit_steps = fit_linear_ode(
        state_t_feature=state_truth,
        t=latent_t,
        ridge_alpha=args.ridge_alpha,
        fit_fraction=args.fit_fraction,
    )
    state_pred = rollout_linear_system(a_matrix=a_matrix, y0=state_truth[0], t=latent_t)
    coefficient_pred = state_to_complex_coefficients(state_pred)

    latent_hat_base = build_fourier_baseline(np.fft.fft(np.asarray(latent_true, dtype=np.float64), axis=1), args.fill_unmodeled)
    latent_pred = inverse_fft_latent(
        inject_mode_coefficients(
            latent_hat_base=latent_hat_base,
            coefficients_t_mode=coefficient_pred,
            mode_indices=mode_indices,
            k_index=k_index,
        )
    )

    coefficient_rel_error = coefficient_relative_error_per_time(coefficient_pred, coefficient_truth)
    fit_state_error = fit_state_dt - fit_state_dt_pred
    fit_state_dt_rel_l2 = np.linalg.norm(fit_state_error) / max(np.linalg.norm(fit_state_dt), 1e-12)
    state_labels = build_state_labels(mode_indices=mode_indices, k_index=k_index, k_physical=k_physical)
    linear_equations = format_linear_equations(a_matrix=a_matrix, state_labels=state_labels)
    truth_mode_analysis = analyze_mode_envelope_phase(coefficient_truth, latent_t, fit_steps)
    pred_mode_analysis = analyze_mode_envelope_phase(coefficient_pred, latent_t, fit_steps)
    damping_rate_error = pred_mode_analysis["damping_rate"] - truth_mode_analysis["damping_rate"]
    phase_rate_error = pred_mode_analysis["phase_rate"] - truth_mode_analysis["phase_rate"]

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
        output_dir / "fft_linear_evaluation.npz",
        latent_file=np.asarray(str(args.latent_file)),
        case_dir=np.asarray(str(case_dir)),
        t=latent_t.astype(np.float64),
        x=latent_x.astype(np.float64),
        v=v_true.astype(np.float64),
        mode_indices=np.asarray(mode_indices, dtype=np.int32),
        k_index=np.asarray(k_index, dtype=np.int32),
        k_physical=np.asarray(k_physical, dtype=np.float64),
        k_source=np.asarray(k_source),
        fill_unmodeled=np.asarray(args.fill_unmodeled),
        fit_fraction=np.asarray(args.fit_fraction, dtype=np.float64),
        fit_steps=np.asarray(fit_steps, dtype=np.int32),
        ridge_alpha=np.asarray(args.ridge_alpha, dtype=np.float64),
        linear_matrix_A=a_matrix.astype(np.float64),
        fourier_coeff_truth=coefficient_truth.astype(np.complex64),
        fourier_coeff_pred=coefficient_pred.astype(np.complex64),
        fourier_amplitude_truth=truth_mode_analysis["amplitude"].astype(np.float64),
        fourier_amplitude_pred=pred_mode_analysis["amplitude"].astype(np.float64),
        fourier_phase_truth=truth_mode_analysis["phase"].astype(np.float64),
        fourier_phase_pred=pred_mode_analysis["phase"].astype(np.float64),
        damping_rate_truth=truth_mode_analysis["damping_rate"].astype(np.float64),
        damping_rate_pred=pred_mode_analysis["damping_rate"].astype(np.float64),
        damping_rate_error=damping_rate_error.astype(np.float64),
        phase_rate_truth=truth_mode_analysis["phase_rate"].astype(np.float64),
        phase_rate_pred=pred_mode_analysis["phase_rate"].astype(np.float64),
        phase_rate_error=phase_rate_error.astype(np.float64),
        state_truth=state_truth.astype(np.float64),
        state_pred=state_pred.astype(np.float64),
        latent_pred=latent_pred.astype(np.float32),
        decoded_pred=f_pred.astype(np.float32),
        decoded_latent_truth=f_latent_truth.astype(np.float32),
        truth=f_true.astype(np.float32),
        truth_electric=truth_electric_field.astype(np.float32),
        pred_electric=pred_electric_field.astype(np.float32),
        decoded_latent_truth_electric=ae_electric_field.astype(np.float32),
        fourier_coeff_relative_l2=coefficient_rel_error.astype(np.float64),
        decoded_relative_l2=decoded_rel_l2.astype(np.float64),
        ae_relative_l2=ae_rel_l2.astype(np.float64),
        decoded_mse=decoded_mse.astype(np.float64),
        ae_mse=ae_mse.astype(np.float64),
        truth_electric_energy=truth_electric_energy.astype(np.float64),
        pred_electric_energy=pred_electric_energy.astype(np.float64),
        ae_electric_energy=ae_electric_energy.astype(np.float64),
    )

    save_error_plot(latent_t, decoded_rel_l2, ae_rel_l2, "FFT-linear", output_dir / "error_over_time.png")
    save_electric_energy_plot(
        t=latent_t,
        truth_energy=truth_electric_energy,
        solver_energy=pred_electric_energy,
        ae_energy=ae_electric_energy,
        solver_name="FFT-linear",
        output_path=output_dir / "electric_energy_over_time.png",
    )
    save_mode_amplitude_phase_plot(
        t=latent_t,
        truth_analysis=truth_mode_analysis,
        pred_analysis=pred_mode_analysis,
        mode_indices=mode_indices,
        output_path=output_dir / "dominant_k_mode_amplitude_phase.png",
    )
    if not args.no_animation:
        make_animation(
            t=latent_t,
            x=latent_x,
            v=v_true,
            truth=f_true,
            prediction=f_pred,
            rel_l2=decoded_rel_l2,
            solver_name="FFT-linear",
            output_path=output_dir / "fft_linear_vs_truth.gif",
            max_frames=args.max_frames,
            fps=args.fps,
        )

    summary_lines = [
        "Dominant-wave latent linear regression",
        f"latent_file: {args.latent_file}",
        f"case_dir: {case_dir}",
        f"latent_shape: (Nt={latent_true.shape[0]}, Nx={latent_true.shape[1]}, Nz={latent_true.shape[2]})",
        f"mode_indices: {mode_indices}",
        f"k_index: {k_index}",
        f"k_physical: {k_physical:.12e}",
        f"k_source: {k_source}",
        f"fill_unmodeled: {args.fill_unmodeled}",
        f"fit_fraction: {args.fit_fraction:.6f}",
        f"fit_steps: {fit_steps}",
        f"ridge_alpha: {args.ridge_alpha:.12e}",
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
        "Modewise damping and phase-rate comparison:",
    ]
    for mode_slot, mode_index in enumerate(mode_indices):
        summary_lines.extend(
            [
                f"mode_{mode_index}_truth_damping_rate: {truth_mode_analysis['damping_rate'][mode_slot]:.12e}",
                f"mode_{mode_index}_pred_damping_rate: {pred_mode_analysis['damping_rate'][mode_slot]:.12e}",
                f"mode_{mode_index}_damping_rate_error: {damping_rate_error[mode_slot]:.12e}",
                f"mode_{mode_index}_truth_phase_rate: {truth_mode_analysis['phase_rate'][mode_slot]:.12e}",
                f"mode_{mode_index}_pred_phase_rate: {pred_mode_analysis['phase_rate'][mode_slot]:.12e}",
                f"mode_{mode_index}_phase_rate_error: {phase_rate_error[mode_slot]:.12e}",
                f"mode_{mode_index}_truth_initial_phase: {truth_mode_analysis['phase'][0, mode_slot]:.12e}",
                f"mode_{mode_index}_pred_initial_phase: {pred_mode_analysis['phase'][0, mode_slot]:.12e}",
                f"mode_{mode_index}_truth_final_phase: {truth_mode_analysis['phase'][-1, mode_slot]:.12e}",
                f"mode_{mode_index}_pred_final_phase: {pred_mode_analysis['phase'][-1, mode_slot]:.12e}",
            ]
        )
    summary_lines.extend(
        [
            "",
        "Learned linear system:",
        *linear_equations,
        ]
    )
    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Loaded latent autoencoder results from: {args.latent_file}")
    print(f"Using case directory: {case_dir}")
    print(f"Selected k_index={k_index}, k={k_physical:.6e} ({k_source})")
    print(f"Saved evaluation arrays to: {output_dir / 'fft_linear_evaluation.npz'}")
    print(f"Saved summary to: {output_dir / 'summary.txt'}")
    print(f"Final Fourier relative L2 error: {coefficient_rel_error[-1]:.6e}")
    print(f"Final decoded relative L2 error: {decoded_rel_l2[-1]:.6e}")


if __name__ == "__main__":
    main()
