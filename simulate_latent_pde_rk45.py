#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter

try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required. Run this script in the 'lasdi' environment, for example:\n"
        "  conda run -n lasdi python simulate_latent_pde_rk45.py ..."
    ) from exc

from Conv_velocity_AE import ConvVelocityAutoencoder
from latent_dynamics import load_latent_data, resolve_electric_data
from reconstruct_e_field import load_distribution as load_case_distribution


TOKEN_PATTERN = re.compile(r"(inv_(?:z\d+|z|E)|(?:z\d+|z|E))(?:\^(\d+))?")
DERIV_PATTERN = re.compile(r"((?:z\d+|z|E)_\{(x+)\})$")


@dataclass(frozen=True)
class PolynomialFactor:
    name: str
    power: int


@dataclass(frozen=True)
class DerivativeFactor:
    name: str
    order: int


@dataclass(frozen=True)
class TermSpec:
    raw: str
    polynomial_factors: tuple[PolynomialFactor, ...]
    derivative_factor: DerivativeFactor | None


@dataclass
class PDEPayload:
    pde_file: Path
    latent_file: Path
    coefficients: np.ndarray
    term_descriptions: list[str]
    mode_indices: list[int]
    t: np.ndarray
    x: np.ndarray
    dt: float
    dx: float
    space_diff: str
    system_mode: str
    used_electric_field: bool


@dataclass
class IntegrationResult:
    success: bool
    t: np.ndarray
    y: np.ndarray
    status: int
    message: str
    nfev: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Integrate a learned latent PDE with RK45, decode it back to f(x, v, t), "
            "and compare it against the Vlasov ground truth."
        )
    )
    parser.add_argument(
        "--pde-file",
        type=Path,
        required=True,
        help="Path to latent_pde_find*.npz or latent_compact_pde_find*.npz.",
    )
    parser.add_argument(
        "--latent-file",
        type=Path,
        default=None,
        help="Optional latent autoencoder .npz. Defaults to the path stored in the PDE file.",
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
        help="Output directory. Defaults to <pde-file-stem>_rk45_eval next to the PDE file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=("cpu", "cuda"),
        help="Torch device used for decoding.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Batch size used while decoding latent trajectories.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="RK45",
        choices=("RK45", "DOP853", "Radau", "BDF", "LSODA", "RK4"),
        help="Time integrator passed to scipy.integrate.solve_ivp.",
    )
    parser.add_argument(
        "--rk4-substeps",
        type=int,
        default=1,
        help="Number of fixed RK4 substeps inside each adjacent pair of t_eval points.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-6,
        help="Relative tolerance for solve_ivp(..., method='RK45').",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-8,
        help="Absolute tolerance for solve_ivp(..., method='RK45').",
    )
    parser.add_argument(
        "--max-step",
        type=float,
        default=None,
        help="Optional RK45 max_step. Defaults to the truth dt stored in the PDE file.",
    )
    parser.add_argument(
        "--initial-step",
        type=float,
        default=None,
        help="Optional initial_step passed to solve_ivp.",
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=None,
        help="Optional final time. Defaults to the end of the truth trajectory.",
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
        "--fill-unmodeled",
        type=str,
        default="truth",
        choices=("truth", "initial", "zero"),
        help=(
            "How to fill latent channels not covered by the PDE file before decoding. "
            "'truth' is recommended unless all latent modes are modeled."
        ),
    )
    parser.add_argument(
        "--clip-min",
        type=float,
        default=None,
        help="Optional lower clip applied to decoded f before error evaluation.",
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="Skip GIF generation.",
    )
    return parser.parse_args()


def load_pde_payload(pde_file: Path, latent_file_override: Path | None) -> PDEPayload:
    with np.load(pde_file, allow_pickle=False) as data:
        if "coefficients_real" not in data or "coefficients_imag" not in data:
            raise KeyError(f"{pde_file} must contain 'coefficients_real' and 'coefficients_imag'.")
        if "mode_indices" not in data:
            raise KeyError(f"{pde_file} must contain 'mode_indices'.")

        latent_path = (
            latent_file_override.resolve()
            if latent_file_override is not None
            else Path(np.asarray(data["latent_file"]).item()).resolve()
        )

        if "rhs_description" in data:
            term_descriptions = [str(item) for item in data["rhs_description"]]
        elif "library_description" in data:
            term_descriptions = [str(item) for item in data["library_description"]]
        else:
            raise KeyError(f"{pde_file} must contain 'rhs_description' or 'library_description'.")

        coefficients = np.asarray(data["coefficients_real"], dtype=np.float64) + 1j * np.asarray(
            data["coefficients_imag"], dtype=np.float64
        )
        system_mode = str(np.asarray(data["system_mode"]).item()) if "system_mode" in data else "compact"
        used_electric_field = bool(np.asarray(data["used_electric_field"]).item()) if "used_electric_field" in data else False
        t = np.asarray(data["t"], dtype=np.float64)
        x = np.asarray(data["x"], dtype=np.float64)
        dt = float(np.asarray(data["dt"], dtype=np.float64).item()) if "dt" in data else float(np.mean(np.diff(t)))
        dx = float(np.asarray(data["dx"], dtype=np.float64).item()) if "dx" in data else float(np.mean(np.diff(x)))
        space_diff = str(np.asarray(data["space_diff"]).item()) if "space_diff" in data else "FD"
        mode_indices = [int(mode) for mode in np.asarray(data["mode_indices"], dtype=np.int32)]

    if not latent_path.exists():
        raise FileNotFoundError(f"Latent autoencoder file not found: {latent_path}")

    return PDEPayload(
        pde_file=pde_file.resolve(),
        latent_file=latent_path,
        coefficients=coefficients,
        term_descriptions=term_descriptions,
        mode_indices=mode_indices,
        t=t,
        x=x,
        dt=dt,
        dx=dx,
        space_diff=space_diff,
        system_mode=system_mode,
        used_electric_field=used_electric_field,
    )


def infer_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    return (args.pde_file.resolve().parent / f"{args.pde_file.stem}_rk45_eval").resolve()


def parse_term_description(description: str) -> TermSpec:
    if description == "":
        return TermSpec(raw=description, polynomial_factors=tuple(), derivative_factor=None)

    derivative_factor = None
    polynomial_part = description
    derivative_match = DERIV_PATTERN.search(description)
    if derivative_match is not None:
        derivative_factor = DerivativeFactor(
            name=derivative_match.group(1).split("_{", 1)[0],
            order=len(derivative_match.group(2)),
        )
        polynomial_part = description[: derivative_match.start()]

    factors: list[PolynomialFactor] = []
    cursor = 0
    while cursor < len(polynomial_part):
        match = TOKEN_PATTERN.match(polynomial_part, cursor)
        if match is None:
            raise ValueError(f"Could not parse PDE term description: {description!r}")
        factors.append(PolynomialFactor(name=match.group(1), power=int(match.group(2) or "1")))
        cursor = match.end()

    return TermSpec(raw=description, polynomial_factors=tuple(factors), derivative_factor=derivative_factor)


def max_derivative_order(term_specs: Sequence[TermSpec]) -> int:
    max_order = 0
    for spec in term_specs:
        if spec.derivative_factor is not None:
            max_order = max(max_order, spec.derivative_factor.order)
    return max_order


def finite_difference_stack(values: np.ndarray, dx: float, max_order: int) -> dict[int, np.ndarray]:
    derivatives: dict[int, np.ndarray] = {}
    if max_order <= 0:
        return derivatives

    current = np.asarray(values, dtype=np.float64)
    n_fields, n_x = current.shape

    def first_derivative(u: np.ndarray) -> np.ndarray:
        out = np.empty_like(u)
        out[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * dx)
        out[:, 0] = (-1.5 * u[:, 0] + 2.0 * u[:, 1] - 0.5 * u[:, 2]) / dx
        out[:, -1] = (1.5 * u[:, -1] - 2.0 * u[:, -2] + 0.5 * u[:, -3]) / dx
        return out

    def second_derivative(u: np.ndarray) -> np.ndarray:
        out = np.empty_like(u)
        out[:, 1:-1] = (u[:, 2:] - 2.0 * u[:, 1:-1] + u[:, :-2]) / (dx * dx)
        out[:, 0] = (2.0 * u[:, 0] - 5.0 * u[:, 1] + 4.0 * u[:, 2] - u[:, 3]) / (dx * dx)
        out[:, -1] = (2.0 * u[:, -1] - 5.0 * u[:, -2] + 4.0 * u[:, -3] - u[:, -4]) / (dx * dx)
        return out

    def third_derivative(u: np.ndarray) -> np.ndarray:
        out = np.zeros_like(u)
        out[:, 2:-2] = (0.5 * u[:, 4:] - u[:, 3:-1] + u[:, 1:-3] - 0.5 * u[:, :-4]) / (dx**3)
        out[:, 0] = (-2.5 * u[:, 0] + 9.0 * u[:, 1] - 12.0 * u[:, 2] + 7.0 * u[:, 3] - 1.5 * u[:, 4]) / (dx**3)
        out[:, 1] = (-2.5 * u[:, 1] + 9.0 * u[:, 2] - 12.0 * u[:, 3] + 7.0 * u[:, 4] - 1.5 * u[:, 5]) / (dx**3)
        out[:, -1] = (2.5 * u[:, -1] - 9.0 * u[:, -2] + 12.0 * u[:, -3] - 7.0 * u[:, -4] + 1.5 * u[:, -5]) / (dx**3)
        out[:, -2] = (2.5 * u[:, -2] - 9.0 * u[:, -3] + 12.0 * u[:, -4] - 7.0 * u[:, -5] + 1.5 * u[:, -6]) / (dx**3)
        return out

    if n_x < 6 and max_order >= 3:
        raise ValueError(f"Need at least 6 spatial points for third derivatives, got {n_x}.")

    for order in range(1, max_order + 1):
        if order == 1:
            derivatives[order] = first_derivative(current)
        elif order == 2:
            derivatives[order] = second_derivative(current)
        elif order == 3:
            derivatives[order] = third_derivative(current)
        else:
            derivatives[order] = first_derivative(derivatives[order - 1])

    return derivatives


def central_difference_stack(values: np.ndarray, dx: float, max_order: int) -> dict[int, np.ndarray]:
    return finite_difference_stack(values, dx, max_order)


def fourier_derivative_stack(values: np.ndarray, dx: float, max_order: int) -> dict[int, np.ndarray]:
    derivatives: dict[int, np.ndarray] = {}
    if max_order <= 0:
        return derivatives

    n_x = values.shape[1]
    k = 2.0 * np.pi * np.fft.fftfreq(n_x, d=dx)
    spectrum = np.fft.fft(values, axis=1)
    for order in range(1, max_order + 1):
        derivatives[order] = np.fft.ifft(((1j * k) ** order)[None, :] * spectrum, axis=1).real
    return derivatives


def savgol_derivative_stack(
    values: np.ndarray,
    dx: float,
    max_order: int,
    window: int = 7,
    polyorder: int = 3,
) -> dict[int, np.ndarray]:
    derivatives: dict[int, np.ndarray] = {}
    if max_order <= 0:
        return derivatives
    for order in range(1, max_order + 1):
        derivatives[order] = np.asarray(
            savgol_filter(values, window_length=window, polyorder=polyorder, deriv=order, delta=dx, axis=1, mode="interp"),
            dtype=np.float64,
        )
    return derivatives


def derivative_stack(values: np.ndarray, dx: float, method: str, max_order: int) -> dict[int, np.ndarray]:
    method_key = method.upper()
    if method_key == "FOURIER":
        return fourier_derivative_stack(values, dx, max_order)
    if method_key == "SG":
        return savgol_derivative_stack(values, dx, max_order)
    if method_key in {"FD", "FDCONV", "POLY", "TIK"}:
        return finite_difference_stack(values, dx, max_order)
    if method_key == "CD":
        return central_difference_stack(values, dx, max_order)
    return finite_difference_stack(values, dx, max_order)


class TimeSeriesForcing:
    def __init__(self, t: np.ndarray, values_t_x: np.ndarray) -> None:
        self.t = np.asarray(t, dtype=np.float64)
        self.values = np.asarray(values_t_x, dtype=np.float64)
        if self.values.shape[0] != self.t.size:
            raise ValueError(f"Forcing shape mismatch: len(t)={self.t.size}, values={self.values.shape}")

    def __call__(self, time_value: float) -> np.ndarray:
        if time_value <= self.t[0]:
            return self.values[0]
        if time_value >= self.t[-1]:
            return self.values[-1]
        upper = int(np.searchsorted(self.t, time_value, side="right"))
        lower = upper - 1
        alpha = (time_value - self.t[lower]) / (self.t[upper] - self.t[lower])
        return (1.0 - alpha) * self.values[lower] + alpha * self.values[upper]


class LatentPDERHS:
    def __init__(
        self,
        payload: PDEPayload,
        term_specs: Sequence[TermSpec],
        electric_forcing: TimeSeriesForcing | None,
    ) -> None:
        self.payload = payload
        self.term_specs = list(term_specs)
        self.electric_forcing = electric_forcing
        self.n_modes = len(payload.mode_indices)
        self.n_x = payload.x.size
        self.max_order = max_derivative_order(term_specs)
        self.coefficients = np.asarray(payload.coefficients, dtype=np.complex128)

        explicit_mode_names = any(re.search(r"(?:^|[^a-zA-Z0-9_])z\d+", term.raw) for term in term_specs)
        self.generic_single_mode = payload.system_mode == "independent" or not explicit_mode_names

    def _evaluate_common_terms(self, state_vars: Dict[str, np.ndarray]) -> np.ndarray:
        base_fields: dict[str, np.ndarray] = {}
        for name, values in state_vars.items():
            base_fields[name] = np.asarray(values, dtype=np.float64)
            if not name.startswith("inv_"):
                safe_sign = np.where(base_fields[name] < 0.0, -1.0, 1.0)
                safe = np.where(np.abs(base_fields[name]) >= 1e-10, base_fields[name], safe_sign * 1e-10)
                base_fields[f"inv_{name}"] = 1.0 / safe

        derivative_sources: dict[str, np.ndarray] = {name: values for name, values in state_vars.items() if not name.startswith("inv_")}
        stacked_names = list(derivative_sources)
        stacked_values = np.stack([derivative_sources[name] for name in stacked_names], axis=0)
        derivatives = derivative_stack(stacked_values, self.payload.dx, self.payload.space_diff, self.max_order)
        derivative_lookup = {
            (stacked_names[idx], order): derivative_values[idx]
            for order, derivative_values in derivatives.items()
            for idx in range(len(stacked_names))
        }

        features = np.empty((len(self.term_specs), self.n_x), dtype=np.complex128)
        ones = np.ones(self.n_x, dtype=np.complex128)
        for term_index, spec in enumerate(self.term_specs):
            value = ones.copy()
            for factor in spec.polynomial_factors:
                value *= np.power(base_fields[factor.name], factor.power)
            if spec.derivative_factor is not None:
                value *= derivative_lookup[(spec.derivative_factor.name, spec.derivative_factor.order)]
            features[term_index] = value
        return features

    def __call__(self, time_value: float, state_flat: np.ndarray) -> np.ndarray:
        state = np.asarray(state_flat, dtype=np.float64).reshape(self.n_modes, self.n_x)
        electric_now = self.electric_forcing(time_value) if self.electric_forcing is not None else None

        if self.generic_single_mode:
            rhs = np.empty_like(state, dtype=np.complex128)
            for local_mode_index in range(self.n_modes):
                vars_for_mode = {"z": state[local_mode_index]}
                if electric_now is not None:
                    vars_for_mode["E"] = electric_now
                features = self._evaluate_common_terms(vars_for_mode)
                rhs[local_mode_index] = self.coefficients[local_mode_index] @ features
        else:
            vars_for_system = {
                f"z{mode}": state[local_mode_index]
                for local_mode_index, mode in enumerate(self.payload.mode_indices)
            }
            if electric_now is not None:
                vars_for_system["E"] = electric_now
            features = self._evaluate_common_terms(vars_for_system)
            rhs = self.coefficients @ features

        if np.max(np.abs(rhs.imag)) > 1e-9:
            raise ValueError(f"RHS evaluation produced a non-negligible imaginary part: {np.max(np.abs(rhs.imag)):.3e}")
        return np.asarray(rhs.real, dtype=np.float64).reshape(-1)


def resolve_case_dir(latent_file: Path, latent_meta: Dict[str, np.ndarray], case_dir_override: Path | None) -> Path:
    if case_dir_override is not None:
        case_dir = case_dir_override.resolve()
        if not case_dir.is_dir():
            raise FileNotFoundError(f"Case directory not found: {case_dir}")
        return case_dir

    case_dir_value = latent_meta.get("case_dir")
    if case_dir_value is None:
        raise ValueError("Could not infer case_dir from the latent file. Please pass --case-dir explicitly.")
    case_dir = Path(np.asarray(case_dir_value).item()).resolve()
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Inferred case directory not found: {case_dir}")
    return case_dir


def load_autoencoder_from_latent_file(latent_file: Path, device: torch.device) -> tuple[ConvVelocityAutoencoder, np.ndarray, np.ndarray]:
    with np.load(latent_file, allow_pickle=False) as data:
        if "conv_channels" not in data or "kernel_size" not in data or "hidden_dim" not in data:
            raise KeyError(f"{latent_file} does not look like a convolutional autoencoder results file.")

        input_dim = int(np.asarray(data["nv"]).item()) if "nv" in data else int(np.asarray(data["v"]).shape[0])
        hidden_dim = int(np.asarray(data["hidden_dim"]).item())
        latent_dim = int(np.asarray(data["nz"]).item())
        conv_channels = tuple(int(item) for item in np.asarray(data["conv_channels"], dtype=np.int32))
        kernel_size = int(np.asarray(data["kernel_size"]).item())
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
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, feature_mean, feature_std


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
    outputs: list[np.ndarray] = []
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
    return np.mean(diff**2, axis=(1, 2))


def make_animation(
    t: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    rel_l2: np.ndarray,
    solver_name: str,
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
    ax_pred.set_title("Decoded RK45 Prediction")
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
    ax_hist.set_title(f"{solver_name} Decoded Error vs Truth")
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
    solver_error: np.ndarray,
    ae_error: np.ndarray,
    solver_name: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(t, solver_error, label=f"decoded {solver_name} vs truth", lw=2)
    ax.plot(t, ae_error, label="decoded latent truth vs truth", lw=2)
    ax.set_xlabel("t")
    ax.set_ylabel("Relative L2 Error")
    ax.set_title("Error Against Vlasov Truth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def integrate_with_rk4(
    rhs,
    t_eval: np.ndarray,
    y0: np.ndarray,
    rk4_substeps: int,
) -> IntegrationResult:
    if rk4_substeps <= 0:
        raise ValueError(f"--rk4-substeps must be positive, got {rk4_substeps}")

    t_eval = np.asarray(t_eval, dtype=np.float64)
    y0 = np.asarray(y0, dtype=np.float64)
    states = np.empty((y0.size, t_eval.size), dtype=np.float64)
    states[:, 0] = y0

    current_t = float(t_eval[0])
    current_y = y0.copy()
    nfev = 0

    for step_index in range(1, t_eval.size):
        target_t = float(t_eval[step_index])
        dt_total = target_t - current_t
        dt_sub = dt_total / float(rk4_substeps)

        for _ in range(rk4_substeps):
            k1 = np.asarray(rhs(current_t, current_y), dtype=np.float64)
            k2 = np.asarray(rhs(current_t + 0.5 * dt_sub, current_y + 0.5 * dt_sub * k1), dtype=np.float64)
            k3 = np.asarray(rhs(current_t + 0.5 * dt_sub, current_y + 0.5 * dt_sub * k2), dtype=np.float64)
            k4 = np.asarray(rhs(current_t + dt_sub, current_y + dt_sub * k3), dtype=np.float64)
            nfev += 4

            if not (
                np.all(np.isfinite(k1))
                and np.all(np.isfinite(k2))
                and np.all(np.isfinite(k3))
                and np.all(np.isfinite(k4))
            ):
                return IntegrationResult(
                    success=False,
                    t=t_eval[:step_index],
                    y=states[:, :step_index],
                    status=-1,
                    message="RK4 produced non-finite stage values.",
                    nfev=nfev,
                )

            current_y = current_y + (dt_sub / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            current_t += dt_sub

            if not np.all(np.isfinite(current_y)):
                return IntegrationResult(
                    success=False,
                    t=t_eval[:step_index],
                    y=states[:, :step_index],
                    status=-1,
                    message="RK4 produced a non-finite state.",
                    nfev=nfev,
                )

        states[:, step_index] = current_y

    return IntegrationResult(
        success=True,
        t=t_eval,
        y=states,
        status=0,
        message="The solver successfully reached the end of the integration interval.",
        nfev=nfev,
    )


def main() -> None:
    args = parse_args()
    output_dir = infer_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_pde_payload(args.pde_file.resolve(), args.latent_file)
    latent_true, latent_t, latent_x, latent_meta = load_latent_data(payload.latent_file)
    case_dir = resolve_case_dir(payload.latent_file, latent_meta, args.case_dir)
    truth_payload = load_case_distribution(case_dir)

    f_true = np.asarray(truth_payload["f"], dtype=np.float64)
    t_true = np.asarray(truth_payload["t"], dtype=np.float64)
    x_true = np.asarray(truth_payload["x"], dtype=np.float64)
    v_true = np.asarray(truth_payload["v"], dtype=np.float64)

    if not np.allclose(latent_t, t_true, rtol=1e-6, atol=1e-8):
        raise ValueError("Latent file time grid does not match distribution_full.npz.")
    if not np.allclose(latent_x, x_true, rtol=1e-6, atol=1e-8):
        raise ValueError("Latent file x grid does not match distribution_full.npz.")

    if args.t_end is not None:
        selected_mask = t_true <= args.t_end + 1e-12
        if not np.any(selected_mask):
            raise ValueError(f"--t-end={args.t_end} produced an empty time window.")
        latent_true = latent_true[selected_mask]
        f_true = f_true[selected_mask]
        t_true = t_true[selected_mask]

    electric_forcing = None
    if payload.used_electric_field:
        electric_field, electric_source = resolve_electric_data(
            latent_file=payload.latent_file,
            meta=latent_meta,
            expected_t=latent_t,
            expected_x=latent_x,
            requested_electric_file=None,
            disable_electric_field=False,
        )
        if electric_field is None:
            raise ValueError("The PDE file requires E(t, x), but no electric field source could be resolved.")
        electric_forcing = TimeSeriesForcing(latent_t, np.asarray(electric_field, dtype=np.float64))
        print(f"Using electric forcing from: {electric_source}")

    term_specs = [parse_term_description(description) for description in payload.term_descriptions]
    rhs = LatentPDERHS(payload=payload, term_specs=term_specs, electric_forcing=electric_forcing)

    y0 = latent_true[0, :, payload.mode_indices].T.reshape(-1)
    if args.solver == "RK4":
        solution = integrate_with_rk4(
            rhs=rhs,
            t_eval=t_true,
            y0=y0,
            rk4_substeps=args.rk4_substeps,
        )
    else:
        scipy_solution = solve_ivp(
            fun=rhs,
            t_span=(float(t_true[0]), float(t_true[-1])),
            y0=y0,
            method=args.solver,
            t_eval=t_true,
            rtol=args.rtol,
            atol=args.atol,
            max_step=payload.dt if args.max_step is None else args.max_step,
            first_step=args.initial_step,
            vectorized=False,
        )
        solution = IntegrationResult(
            success=bool(scipy_solution.success),
            t=np.asarray(scipy_solution.t, dtype=np.float64),
            y=np.asarray(scipy_solution.y, dtype=np.float64),
            status=int(scipy_solution.status),
            message=str(scipy_solution.message),
            nfev=int(scipy_solution.nfev),
        )

    if not solution.success:
        print(f"{args.solver} stopped early: {solution.message}")

    if solution.t.size == 0:
        solution_t = np.asarray([t_true[0]], dtype=np.float64)
        solution_y = y0.reshape(-1, 1)
    else:
        solution_t = np.asarray(solution.t, dtype=np.float64)
        solution_y = np.asarray(solution.y, dtype=np.float64)

    valid_steps = solution_t.size
    t_true = t_true[:valid_steps]
    latent_true = latent_true[:valid_steps]
    f_true = f_true[:valid_steps]

    latent_modeled = solution_y.T.reshape(valid_steps, len(payload.mode_indices), len(x_true)).transpose(0, 2, 1)
    latent_pred_full = build_full_latent_prediction(
        latent_true=np.asarray(latent_true, dtype=np.float64),
        latent_pred_modeled=latent_modeled,
        modeled_modes=payload.mode_indices,
        fill_unmodeled=args.fill_unmodeled,
    )

    device = torch.device(args.device)
    model, feature_mean, feature_std = load_autoencoder_from_latent_file(payload.latent_file, device)
    f_pred = decode_latent_trajectory(
        model=model,
        latent_t_x_z=latent_pred_full,
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

    if args.clip_min is not None:
        f_pred = np.maximum(f_pred, args.clip_min)
        f_latent_truth = np.maximum(f_latent_truth, args.clip_min)

    rk45_rel_l2 = relative_l2_per_time(f_pred, f_true)
    ae_rel_l2 = relative_l2_per_time(f_latent_truth, f_true)
    rk45_mse = mse_per_time(f_pred, f_true)
    ae_mse = mse_per_time(f_latent_truth, f_true)

    np.savez_compressed(
        output_dir / "rk45_evaluation.npz",
        pde_file=np.asarray(str(payload.pde_file)),
        latent_file=np.asarray(str(payload.latent_file)),
        case_dir=np.asarray(str(case_dir)),
        t=t_true.astype(np.float64),
        x=x_true.astype(np.float64),
        v=v_true.astype(np.float64),
        modeled_modes=np.asarray(payload.mode_indices, dtype=np.int32),
        rk45_latent=latent_pred_full.astype(np.float32),
        rk45_decoded=f_pred.astype(np.float32),
        decoded_latent_truth=f_latent_truth.astype(np.float32),
        truth=f_true.astype(np.float32),
        rk45_relative_l2=rk45_rel_l2.astype(np.float64),
        ae_relative_l2=ae_rel_l2.astype(np.float64),
        rk45_mse=rk45_mse.astype(np.float64),
        ae_mse=ae_mse.astype(np.float64),
        fill_unmodeled=np.asarray(args.fill_unmodeled),
        solver=np.asarray(args.solver),
        rk4_substeps=np.asarray(args.rk4_substeps, dtype=np.int32),
        rk45_nfev=np.asarray(solution.nfev, dtype=np.int32),
        rk45_status=np.asarray(solution.status, dtype=np.int32),
        rk45_message=np.asarray(solution.message),
    )

    save_error_plot(t_true, rk45_rel_l2, ae_rel_l2, args.solver, output_dir / "error_over_time.png")
    if not args.no_animation:
        make_animation(
            t=t_true,
            x=x_true,
            v=v_true,
            truth=f_true,
            prediction=f_pred,
            rel_l2=rk45_rel_l2,
            solver_name=args.solver,
            output_path=output_dir / "rk45_vs_truth.gif",
            max_frames=args.max_frames,
            fps=args.fps,
        )

    summary_lines = [
        f"pde_file: {payload.pde_file}",
        f"latent_file: {payload.latent_file}",
        f"case_dir: {case_dir}",
        f"system_mode: {payload.system_mode}",
        f"space_diff: {payload.space_diff}",
        f"used_electric_field: {payload.used_electric_field}",
        f"mode_indices: {payload.mode_indices}",
        f"solver: {args.solver}",
        f"rk4_substeps: {args.rk4_substeps}",
        f"fill_unmodeled: {args.fill_unmodeled}",
        f"solver_success: {solution.success}",
        f"solver_message: {solution.message}",
        f"solver_nfev: {solution.nfev}",
        f"solver_mean_relative_l2: {np.mean(rk45_rel_l2):.12e}",
        f"solver_max_relative_l2: {np.max(rk45_rel_l2):.12e}",
        f"solver_final_relative_l2: {rk45_rel_l2[-1]:.12e}",
        f"ae_mean_relative_l2: {np.mean(ae_rel_l2):.12e}",
        f"ae_max_relative_l2: {np.max(ae_rel_l2):.12e}",
        f"ae_final_relative_l2: {ae_rel_l2[-1]:.12e}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Saved evaluation arrays to: {output_dir / 'rk45_evaluation.npz'}")
    print(f"Saved error plot to: {output_dir / 'error_over_time.png'}")
    if not args.no_animation:
        print(f"Saved animation to: {output_dir / 'rk45_vs_truth.gif'}")
    print(f"Final {args.solver} decoded relative L2 error: {rk45_rel_l2[-1]:.6e}")
    print(f"Final decoder-only relative L2 error: {ae_rel_l2[-1]:.6e}")


if __name__ == "__main__":
    main()
