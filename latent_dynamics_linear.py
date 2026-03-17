#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
from scipy.signal import savgol_filter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a coupled linear latent transport model Z_t = c + A Z + B Z_x "
            "from an autoencoder latent trajectory."
        )
    )
    parser.add_argument(
        "--latent-file",
        type=Path,
        required=True,
        help="Path to conv_velocity_autoencoder_results*.npz containing latent(t, x, z).",
    )
    parser.add_argument(
        "--modes",
        type=int,
        nargs="*",
        default=None,
        help="Latent mode indices to fit. Default is all modes.",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1e-6,
        help="Ridge parameter for the global least-squares fit.",
    )
    parser.add_argument(
        "--fit-fraction",
        type=float,
        default=1.0,
        help="Fraction of the time history used for fitting. Default: 1.0.",
    )
    parser.add_argument(
        "--include-constant",
        action="store_true",
        help="Include a constant forcing term c in Z_t = c + A Z + B Z_x.",
    )
    parser.add_argument(
        "--space-diff",
        type=str,
        default="Fourier",
        choices=("Fourier", "FD", "SG"),
        help="Method used to compute Z_x on the x grid.",
    )
    parser.add_argument(
        "--time-diff",
        type=str,
        default="FD",
        choices=("FD", "SG"),
        help="Method used to compute Z_t on the time grid.",
    )
    parser.add_argument(
        "--sg-window-x",
        type=int,
        default=9,
        help="Savitzky-Golay odd window length in x when --space-diff SG.",
    )
    parser.add_argument(
        "--sg-window-t",
        type=int,
        default=9,
        help="Savitzky-Golay odd window length in t when --time-diff SG.",
    )
    parser.add_argument(
        "--sg-poly-x",
        type=int,
        default=3,
        help="Savitzky-Golay polynomial degree in x when --space-diff SG.",
    )
    parser.add_argument(
        "--sg-poly-t",
        type=int,
        default=3,
        help="Savitzky-Golay polynomial degree in t when --time-diff SG.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npz path. Defaults to <latent-file-stem>_linear_transport.npz.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Text report path. Defaults to <output-stem>.txt.",
    )
    return parser.parse_args()


def infer_output_path(latent_file: Path) -> Path:
    return latent_file.with_name(f"{latent_file.stem}_linear_transport.npz")


def infer_report_path(output_path: Path) -> Path:
    return output_path.with_suffix(".txt")


def load_latent_data(latent_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
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
    if not np.allclose(diffs, spacing, rtol=1e-3, atol=1e-6):
        raise ValueError(f"{name} grid must be uniformly spaced.")
    return spacing


def adjusted_savgol_window(size: int, requested: int, polyorder: int, axis_name: str) -> int:
    if size < 3:
        raise ValueError(f"Need at least 3 samples along {axis_name} for Savitzky-Golay derivatives, got {size}.")

    window = min(int(requested), int(size))
    if window % 2 == 0:
        window -= 1
    if window <= polyorder:
        window = polyorder + 1
        if window % 2 == 0:
            window += 1
    if window > size:
        window = size if size % 2 == 1 else size - 1
    if window <= polyorder or window < 3:
        raise ValueError(
            f"Could not choose a valid Savitzky-Golay window for {axis_name}: "
            f"size={size}, requested={requested}, polyorder={polyorder}."
        )
    return window


def time_derivative(
    latent_t_x_z: np.ndarray,
    dt: float,
    method: str,
    sg_window_t: int,
    sg_poly_t: int,
) -> np.ndarray:
    values = np.asarray(latent_t_x_z, dtype=np.float64)
    method_key = method.upper()
    if method_key == "FD":
        edge_order = 2 if values.shape[0] >= 3 else 1
        return np.gradient(values, dt, axis=0, edge_order=edge_order)
    if method_key == "SG":
        window = adjusted_savgol_window(values.shape[0], sg_window_t, sg_poly_t, "t")
        return np.asarray(
            savgol_filter(
                values,
                window_length=window,
                polyorder=sg_poly_t,
                deriv=1,
                delta=dt,
                axis=0,
                mode="interp",
            ),
            dtype=np.float64,
        )
    raise ValueError(f"Unsupported time derivative method: {method}")


def space_derivative(
    latent_t_x_z: np.ndarray,
    dx: float,
    method: str,
    sg_window_x: int,
    sg_poly_x: int,
) -> np.ndarray:
    values = np.asarray(latent_t_x_z, dtype=np.float64)
    method_key = method.upper()
    if method_key == "FOURIER":
        n_x = values.shape[1]
        k = 2.0 * np.pi * np.fft.fftfreq(n_x, d=dx)
        spectrum = np.fft.fft(values, axis=1)
        return np.asarray(np.fft.ifft((1j * k)[None, :, None] * spectrum, axis=1).real, dtype=np.float64)
    if method_key == "FD":
        edge_order = 2 if values.shape[1] >= 3 else 1
        return np.gradient(values, dx, axis=1, edge_order=edge_order)
    if method_key == "SG":
        window = adjusted_savgol_window(values.shape[1], sg_window_x, sg_poly_x, "x")
        return np.asarray(
            savgol_filter(
                values,
                window_length=window,
                polyorder=sg_poly_x,
                deriv=1,
                delta=dx,
                axis=1,
                mode="wrap",
            ),
            dtype=np.float64,
        )
    raise ValueError(f"Unsupported space derivative method: {method}")


def build_feature_tensor(
    latent_t_x_m: np.ndarray,
    z_x_t_x_m: np.ndarray,
    mode_indices: Sequence[int],
    include_constant: bool,
) -> tuple[np.ndarray, list[str]]:
    features: list[np.ndarray] = []
    labels: list[str] = []

    if include_constant:
        features.append(np.ones(latent_t_x_m.shape[:2] + (1,), dtype=np.float64))
        labels.append("1")

    for local_slot, mode_index in enumerate(mode_indices):
        features.append(latent_t_x_m[:, :, local_slot : local_slot + 1])
        labels.append(f"z_{mode_index}")

    for local_slot, mode_index in enumerate(mode_indices):
        features.append(z_x_t_x_m[:, :, local_slot : local_slot + 1])
        labels.append(f"z_{mode_index}_x")

    return np.concatenate(features, axis=2), labels


def solve_ridge(
    design_matrix: np.ndarray,
    targets: np.ndarray,
    ridge_alpha: float,
    include_constant: bool,
) -> np.ndarray:
    x = np.asarray(design_matrix, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)

    if ridge_alpha > 0.0:
        lhs = x.T @ x
        penalty = ridge_alpha * np.eye(lhs.shape[0], dtype=np.float64)
        if include_constant:
            penalty[0, 0] = 0.0
        rhs = x.T @ y
        return np.linalg.solve(lhs + penalty, rhs)

    return np.linalg.lstsq(x, y, rcond=None)[0]


def predict_rhs(feature_tensor: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    nt, nx, n_features = feature_tensor.shape
    prediction = feature_tensor.reshape(nt * nx, n_features) @ coefficients.T
    return prediction.reshape(nt, nx, coefficients.shape[0])


def relative_l2_per_mode(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    diff = np.asarray(prediction, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    diff_norm = np.linalg.norm(diff.reshape(-1, diff.shape[-1]), axis=0)
    truth_norm = np.linalg.norm(np.asarray(truth, dtype=np.float64).reshape(-1, truth.shape[-1]), axis=0)
    return np.where(truth_norm > 0.0, diff_norm / truth_norm, 0.0)


def mse_per_mode(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    diff = np.asarray(prediction, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return np.mean(diff * diff, axis=(0, 1))


def fit_linear_transport(
    latent_t_x_m: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    mode_indices: Sequence[int],
    ridge_alpha: float,
    fit_fraction: float,
    include_constant: bool,
    space_diff_method: str,
    time_diff_method: str,
    sg_window_x: int,
    sg_window_t: int,
    sg_poly_x: int,
    sg_poly_t: int,
) -> dict[str, np.ndarray | list[str] | int]:
    if not (0.0 < fit_fraction <= 1.0):
        raise ValueError(f"--fit-fraction must lie in (0, 1], got {fit_fraction}")

    dt = validate_uniform_grid(t, "t")
    dx = validate_uniform_grid(x, "x")
    nt = latent_t_x_m.shape[0]
    n_modes = latent_t_x_m.shape[2]
    fit_steps = max(3, min(nt, int(math.ceil(fit_fraction * nt))))

    z_t_full = time_derivative(
        latent_t_x_m,
        dt=dt,
        method=time_diff_method,
        sg_window_t=sg_window_t,
        sg_poly_t=sg_poly_t,
    )
    z_x_full = space_derivative(
        latent_t_x_m,
        dx=dx,
        method=space_diff_method,
        sg_window_x=sg_window_x,
        sg_poly_x=sg_poly_x,
    )
    feature_tensor_full, labels = build_feature_tensor(
        latent_t_x_m=latent_t_x_m,
        z_x_t_x_m=z_x_full,
        mode_indices=mode_indices,
        include_constant=include_constant,
    )

    feature_fit = feature_tensor_full[:fit_steps]
    target_fit = z_t_full[:fit_steps]

    design_fit = feature_fit.reshape(-1, feature_fit.shape[-1])
    rhs_fit = target_fit.reshape(-1, n_modes)
    beta = solve_ridge(
        design_matrix=design_fit,
        targets=rhs_fit,
        ridge_alpha=ridge_alpha,
        include_constant=include_constant,
    )
    coefficients = beta.T

    rhs_fit_pred = predict_rhs(feature_fit, coefficients)
    rhs_full_pred = predict_rhs(feature_tensor_full, coefficients)

    offset = 1 if include_constant else 0
    bias = coefficients[:, 0] if include_constant else np.zeros(n_modes, dtype=np.float64)
    a_matrix = coefficients[:, offset : offset + n_modes]
    b_matrix = coefficients[:, offset + n_modes : offset + 2 * n_modes]

    return {
        "dt": np.asarray(dt, dtype=np.float64),
        "dx": np.asarray(dx, dtype=np.float64),
        "fit_steps": int(fit_steps),
        "library_labels": labels,
        "coefficients": coefficients.astype(np.float64),
        "linear_bias": bias.astype(np.float64),
        "linear_matrix_A": a_matrix.astype(np.float64),
        "linear_matrix_B": b_matrix.astype(np.float64),
        "z_t_full": z_t_full.astype(np.float64),
        "z_x_full": z_x_full.astype(np.float64),
        "rhs_fit_pred": rhs_fit_pred.astype(np.float64),
        "rhs_full_pred": rhs_full_pred.astype(np.float64),
        "fit_mse_per_mode": mse_per_mode(rhs_fit_pred, target_fit).astype(np.float64),
        "full_mse_per_mode": mse_per_mode(rhs_full_pred, z_t_full).astype(np.float64),
        "fit_relative_l2_per_mode": relative_l2_per_mode(rhs_fit_pred, target_fit).astype(np.float64),
        "full_relative_l2_per_mode": relative_l2_per_mode(rhs_full_pred, z_t_full).astype(np.float64),
    }


def format_equation(mode_index: int, labels: Sequence[str], coeffs: np.ndarray, tol: float = 1e-12) -> str:
    terms: list[str] = []
    for label, coeff in zip(labels, coeffs):
        value = float(coeff)
        if abs(value) <= tol:
            continue
        terms.append(f"{value:+.6e}*{label}")
    rhs = " ".join(terms) if terms else "0"
    return f"z_{mode_index}_t = {rhs}"


def write_report(
    report_path: Path,
    latent_file: Path,
    output_path: Path,
    mode_indices: Sequence[int],
    args: argparse.Namespace,
    fit_result: dict[str, np.ndarray | list[str] | int],
) -> None:
    labels = fit_result["library_labels"]
    coefficients = np.asarray(fit_result["coefficients"], dtype=np.float64)
    equations = [
        format_equation(mode_index=mode_index, labels=labels, coeffs=coefficients[row_index])
        for row_index, mode_index in enumerate(mode_indices)
    ]

    fit_rel = np.asarray(fit_result["fit_relative_l2_per_mode"], dtype=np.float64)
    full_rel = np.asarray(fit_result["full_relative_l2_per_mode"], dtype=np.float64)
    fit_mse = np.asarray(fit_result["fit_mse_per_mode"], dtype=np.float64)
    full_mse = np.asarray(fit_result["full_mse_per_mode"], dtype=np.float64)

    lines = [
        "Linear coupled latent transport fit",
        f"latent_file: {latent_file.resolve()}",
        f"output_file: {output_path.resolve()}",
        f"system: coupled",
        f"fit_equation: Z_t = c + A Z + B Z_x" if args.include_constant else "fit_equation: Z_t = A Z + B Z_x",
        f"ridge_alpha: {args.ridge_alpha:.12e}",
        f"fit_fraction: {args.fit_fraction:.12e}",
        f"fit_steps: {int(fit_result['fit_steps'])}",
        f"time_diff: {args.time_diff}",
        f"space_diff: {args.space_diff}",
        f"mode_indices: {' '.join(str(mode) for mode in mode_indices)}",
        "library_labels:",
    ]
    lines.extend(f"  {idx:02d}: {label}" for idx, label in enumerate(labels))
    lines.append("per_mode_metrics:")
    for row_index, mode_index in enumerate(mode_indices):
        lines.extend(
            [
                f"- mode {mode_index}:",
                f"  fit_relative_l2: {fit_rel[row_index]:.12e}",
                f"  full_relative_l2: {full_rel[row_index]:.12e}",
                f"  fit_mse: {fit_mse[row_index]:.12e}",
                f"  full_mse: {full_mse[row_index]:.12e}",
                f"  equation: {equations[row_index]}",
            ]
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.latent_file = args.latent_file.resolve()
    args.output = infer_output_path(args.latent_file) if args.output is None else args.output.resolve()
    args.report = infer_report_path(args.output) if args.report is None else args.report.resolve()

    latent, t, x, meta = load_latent_data(args.latent_file)
    mode_indices = resolve_modes(args.modes, latent.shape[2])
    latent_selected = np.asarray(latent[:, :, mode_indices], dtype=np.float64)

    fit_result = fit_linear_transport(
        latent_t_x_m=latent_selected,
        t=t,
        x=x,
        mode_indices=mode_indices,
        ridge_alpha=args.ridge_alpha,
        fit_fraction=args.fit_fraction,
        include_constant=args.include_constant,
        space_diff_method=args.space_diff,
        time_diff_method=args.time_diff,
        sg_window_x=args.sg_window_x,
        sg_window_t=args.sg_window_t,
        sg_poly_x=args.sg_poly_x,
        sg_poly_t=args.sg_poly_t,
    )

    labels = fit_result["library_labels"]
    coefficients = np.asarray(fit_result["coefficients"], dtype=np.float64)
    equations = np.asarray(
        [
            format_equation(mode_index=mode_index, labels=labels, coeffs=coefficients[row_index])
            for row_index, mode_index in enumerate(mode_indices)
        ]
    )

    payload: Dict[str, np.ndarray] = {
        "latent_file": np.asarray(str(args.latent_file)),
        "system_mode": np.asarray("coupled"),
        "model_type": np.asarray("linear_coupled_transport"),
        "mode_indices": np.asarray(mode_indices, dtype=np.int32),
        "library_labels": np.asarray(labels),
        "coefficients": coefficients.astype(np.float64),
        "equations": equations,
        "linear_bias": np.asarray(fit_result["linear_bias"], dtype=np.float64),
        "linear_matrix_A": np.asarray(fit_result["linear_matrix_A"], dtype=np.float64),
        "linear_matrix_B": np.asarray(fit_result["linear_matrix_B"], dtype=np.float64),
        "fit_relative_l2_per_mode": np.asarray(fit_result["fit_relative_l2_per_mode"], dtype=np.float64),
        "full_relative_l2_per_mode": np.asarray(fit_result["full_relative_l2_per_mode"], dtype=np.float64),
        "fit_mse_per_mode": np.asarray(fit_result["fit_mse_per_mode"], dtype=np.float64),
        "full_mse_per_mode": np.asarray(fit_result["full_mse_per_mode"], dtype=np.float64),
        "ridge_alpha": np.asarray(args.ridge_alpha, dtype=np.float64),
        "fit_fraction": np.asarray(args.fit_fraction, dtype=np.float64),
        "fit_steps": np.asarray(fit_result["fit_steps"], dtype=np.int32),
        "include_constant": np.asarray(args.include_constant),
        "time_diff": np.asarray(args.time_diff),
        "space_diff": np.asarray(args.space_diff),
        "used_electric_field": np.asarray(False),
        "t": np.asarray(t, dtype=np.float64),
        "x": np.asarray(x, dtype=np.float64),
        "dt": np.asarray(fit_result["dt"], dtype=np.float64),
        "dx": np.asarray(fit_result["dx"], dtype=np.float64),
    }
    for key in ("case_name", "case_dir", "nt", "nx", "nz"):
        if key in meta:
            payload[key] = np.asarray(meta[key])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)

    write_report(
        report_path=args.report,
        latent_file=args.latent_file,
        output_path=args.output,
        mode_indices=mode_indices,
        args=args,
        fit_result=fit_result,
    )

    print(f"Loaded latent trajectory from: {args.latent_file}")
    print(f"Selected modes: {mode_indices}")
    print(f"Fit steps: {int(fit_result['fit_steps'])} / {t.size}")
    print(f"Saved linear transport coefficients to: {args.output}")
    print(f"Saved report to: {args.report}")
    print(
        "Mean fit relative L2: "
        f"{float(np.mean(np.asarray(fit_result['fit_relative_l2_per_mode'], dtype=np.float64))):.6e}"
    )
    print(
        "Mean full relative L2: "
        f"{float(np.mean(np.asarray(fit_result['full_relative_l2_per_mode'], dtype=np.float64))):.6e}"
    )


if __name__ == "__main__":
    main()
