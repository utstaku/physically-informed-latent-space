#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the full 1D electric-field history E(t, x) from "
            "distribution_full.npz for a Vlasov two-stream case."
        )
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        required=True,
        help="Case directory that contains distribution_full.npz.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npz path. Defaults to <case-dir>/electric_field_full.npz.",
    )
    parser.add_argument(
        "--verify-animation",
        action="store_true",
        help="Compare every frame_stride-th reconstructed frame with animation_data.npz if it exists.",
    )
    return parser.parse_args()


def load_distribution(case_dir: Path) -> dict[str, np.ndarray]:
    path = case_dir / "distribution_full.npz"
    if not path.exists():
        raise FileNotFoundError(f"distribution_full.npz not found: {path}")

    with np.load(path) as data:
        payload = {key: data[key] for key in data.files}

    f = np.asarray(payload["f"])
    t = np.asarray(payload["t"])
    x = np.asarray(payload["x"])
    v = np.asarray(payload["v"])

    expected = (len(t), len(x), len(v))
    if f.shape == expected:
        payload["f"] = f
    elif f.shape == (len(x), len(v), len(t)):
        payload["f"] = np.transpose(f, (2, 0, 1))
    else:
        raise ValueError(
            "Unsupported f shape. Expected (Nt, Nx, Nv) or (Nx, Nv, Nt), "
            f"got {f.shape}."
        )

    return payload


def reconstruct_electric_field(
    f_t_x_v: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x64 = np.asarray(x, dtype=np.float64)
    v64 = np.asarray(v, dtype=np.float64)
    f64 = np.asarray(f_t_x_v, dtype=np.float64)

    # Match the saved animation field: integrate in v with trapezoidal rule,
    # then solve periodic Poisson in Fourier space with zero-mean E.
    n_t_x = np.trapezoid(f64, v64, axis=-1)
    rho_t_x = n_t_x - n_t_x.mean(axis=1, keepdims=True)

    nx = len(x64)
    dx = float(x64[1] - x64[0])
    k = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)

    e_t_x = np.empty_like(rho_t_x)
    nonzero = k != 0.0
    for time_index in range(rho_t_x.shape[0]):
        rho_hat = np.fft.fft(rho_t_x[time_index])
        e_hat = np.zeros_like(rho_hat, dtype=np.complex128)
        e_hat[nonzero] = -rho_hat[nonzero] / (1j * k[nonzero])
        e_t_x[time_index] = np.fft.ifft(e_hat).real

    return (
        n_t_x.astype(np.float32),
        rho_t_x.astype(np.float32),
        e_t_x.astype(np.float32),
    )


def verify_animation(case_dir: Path, e_t_x: np.ndarray) -> None:
    path = case_dir / "animation_data.npz"
    if not path.exists():
        print(f"animation_data.npz not found: {path}")
        return

    with np.load(path) as data:
        saved_e = np.asarray(data["E"], dtype=np.float32)
        frame_stride = int(data["frame_stride"])

    reconstructed = e_t_x[::frame_stride]
    if reconstructed.shape != saved_e.shape:
        raise ValueError(
            "Stride-matched reconstructed E shape does not match animation_data.npz: "
            f"{reconstructed.shape} vs {saved_e.shape}"
        )

    abs_error = np.abs(reconstructed - saved_e)
    print(f"frame_stride={frame_stride}")
    print(f"verify_max_abs_error={float(abs_error.max()):.6e}")
    print(f"verify_mean_abs_error={float(abs_error.mean()):.6e}")


def main() -> None:
    args = parse_args()
    case_dir = args.case_dir.resolve()
    output_path = args.output.resolve() if args.output is not None else case_dir / "electric_field_full.npz"

    payload = load_distribution(case_dir)
    n_t_x, rho_t_x, e_t_x = reconstruct_electric_field(payload["f"], payload["x"], payload["v"])

    save_payload = {
        "t": payload["t"],
        "x": payload["x"],
        "v": payload["v"],
        "n": n_t_x,
        "rho": rho_t_x,
        "E": e_t_x,
    }
    for key in ("T", "k", "dt", "tmax"):
        if key in payload:
            save_payload[key] = payload[key]

    np.savez_compressed(output_path, **save_payload)
    print(f"saved={output_path}")
    print(f"E_shape={e_t_x.shape}")

    if args.verify_animation:
        verify_animation(case_dir, e_t_x)


if __name__ == "__main__":
    main()
