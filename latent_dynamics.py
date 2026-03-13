#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.signal import savgol_filter

REPO_ROOT = Path(__file__).resolve().parent
TUTORIALS_DIR = REPO_ROOT / "tutorials"
if str(TUTORIALS_DIR) not in sys.path:
    sys.path.insert(0, str(TUTORIALS_DIR))

import tutorials.PDE_FIND as pde_find
from reconstruct_e_field import load_distribution as load_case_distribution
from reconstruct_e_field import reconstruct_electric_field


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover PDEs for latent variables saved by velocity_AE.py "
            "using tutorials/PDE_FIND.py without modifying it."
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
        help="Latent mode indices to analyze. Default is all modes.",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="independent",
        choices=("independent", "coupled"),
        help="Whether to fit one PDE per mode independently or a coupled latent PDE system.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npz file. Defaults to <latent-file-stem>_pde_find.npz.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Text report path. Defaults to <output-stem>.txt.",
    )
    parser.add_argument(
        "--electric-file",
        type=Path,
        default=None,
        help=(
            "Optional path to electric_field_full.npz. If omitted, the script tries to infer it "
            "from the latent metadata or reconstruct E(t, x) from distribution_full.npz."
        ),
    )
    parser.add_argument(
        "--no-electric-field",
        action="store_true",
        help="Disable electric-field loading entirely and use a latent-only PDE library.",
    )
    parser.add_argument(
        "--include-reciprocal",
        action="store_true",
        help="Augment the PDE library with reciprocal features such as 1/z_k and 1/E.",
    )
    parser.add_argument(
        "--reciprocal-eps",
        type=float,
        default=1e-6,
        help="Small positive regularization used in reciprocal features to avoid division by zero.",
    )
    parser.add_argument("--D", type=int, default=1, help="Maximum spatial derivative order.")
    parser.add_argument("--P", type=int, default=1, help="Maximum polynomial power.")
    parser.add_argument(
        "--time-diff",
        type=str,
        default="FD",
        choices=("poly", "FD", "FDconv", "Tik", "SG"),
        help="Time-derivative method passed to PDE-FIND.",
    )
    parser.add_argument(
        "--space-diff",
        type=str,
        default="FD",
        choices=("poly", "FD", "CD", "FDconv", "Tik", "Fourier", "SG"),
        help="Space-derivative method passed to PDE-FIND.",
    )
    parser.add_argument("--lam", type=float, default=1e-2, help="Ridge parameter for STRidge.")
    parser.add_argument("--d-tol", type=float, default=0.5, help="Tolerance increment for STRidge.")
    parser.add_argument(
        "--l0-penalty",
        type=float,
        default=None,
        help="Optional penalty on the number of nonzero terms used by TrainSTRidge.",
    )
    parser.add_argument("--maxit", type=int, default=25, help="Max tolerance search iterations.")
    parser.add_argument(
        "--str-iters",
        type=int,
        default=10,
        help="Inner STRidge iterations.",
    )
    parser.add_argument(
        "--normalize",
        type=int,
        default=2,
        help="Normalization passed to TrainSTRidge.",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.8,
        help="Train/validation split passed to TrainSTRidge.",
    )
    parser.add_argument(
        "--width-x",
        type=int,
        default=None,
        help="Polynomial/convolution stencil width in x.",
    )
    parser.add_argument(
        "--width-t",
        type=int,
        default=None,
        help="Polynomial/convolution stencil width in t.",
    )
    parser.add_argument("--deg-x", type=int, default=5, help="Polynomial degree in x.")
    parser.add_argument("--deg-t", type=int, default=None, help="Polynomial degree in t.")
    parser.add_argument("--sigma", type=float, default=2.0, help="Gaussian smoother sigma for FDconv.")
    parser.add_argument(
        "--sg-window-x",
        type=int,
        default=7,
        help="Savitzky-Golay odd window length in x when using space-diff=SG.",
    )
    parser.add_argument(
        "--sg-window-t",
        type=int,
        default=7,
        help="Savitzky-Golay odd window length in t when using time-diff=SG.",
    )
    parser.add_argument(
        "--sg-poly-x",
        type=int,
        default=3,
        help="Savitzky-Golay polynomial degree in x when using space-diff=SG.",
    )
    parser.add_argument(
        "--sg-poly-t",
        type=int,
        default=3,
        help="Savitzky-Golay polynomial degree in t when using time-diff=SG.",
    )
    parser.add_argument(
        "--print-best-tol",
        action="store_true",
        help="Print PDE-FIND's best tolerance during STRidge search.",
    )
    return parser.parse_args()

def patch_pde_find_compatibility(pde_find) -> None:
    def safe_stridge(X0, y, lam, maxit, tol, normalize=2, print_results=False):
        n, d = X0.shape
        X = np.zeros((n, d), dtype=np.complex64)
        if normalize != 0:
            mreg = np.zeros((d, 1))
            for i in range(d):
                mreg[i] = 1.0 / (np.linalg.norm(X0[:, i], normalize))
                X[:, i] = mreg[i] * X0[:, i]
        else:
            X = X0
            mreg = None

        if lam != 0:
            w = np.linalg.lstsq(X.T.dot(X) + lam * np.eye(d), X.T.dot(y), rcond=None)[0]
        else:
            w = np.linalg.lstsq(X, y, rcond=None)[0]

        num_relevant = d
        biginds: Sequence[int] | np.ndarray = np.where(abs(w) > tol)[0]

        for j in range(maxit):
            smallinds = np.where(abs(w) < tol)[0]
            new_biginds = [i for i in range(d) if i not in smallinds]

            if num_relevant == len(new_biginds):
                break
            num_relevant = len(new_biginds)

            if len(new_biginds) == 0:
                if j == 0:
                    return np.multiply(mreg, w) if normalize != 0 else w
                break

            biginds = new_biginds
            w[smallinds] = 0
            if lam != 0:
                w[biginds] = np.linalg.lstsq(
                    X[:, biginds].T.dot(X[:, biginds]) + lam * np.eye(len(biginds)),
                    X[:, biginds].T.dot(y),
                    rcond=None,
                )[0]
            else:
                w[biginds] = np.linalg.lstsq(X[:, biginds], y, rcond=None)[0]

        if len(biginds) != 0:
            w[biginds] = np.linalg.lstsq(X[:, biginds], y, rcond=None)[0]

        if normalize != 0:
            return np.multiply(mreg, w)
        return w

    pde_find.STRidge = safe_stridge


def infer_output_path(latent_file: Path) -> Path:
    return latent_file.with_name(f"{latent_file.stem}_pde_find.npz")


def infer_report_path(output_path: Path) -> Path:
    return output_path.with_suffix(".txt")


def load_latent_data(latent_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    with np.load(latent_file, allow_pickle=False) as data:
        if "latent" not in data or "t" not in data or "x" not in data:
            raise KeyError(f"{latent_file} must contain 'latent', 't', and 'x'")
        latent = np.asarray(data["latent"], dtype=np.float64)
        t = np.asarray(data["t"], dtype=np.float64)
        x = np.asarray(data["x"], dtype=np.float64)
        meta = {
            key: np.asarray(data[key])
            for key in ("case_name", "case_dir", "nt", "nx", "nz")
            if key in data
        }

    if latent.ndim != 3:
        raise ValueError(f"Expected latent shape (Nt, Nx, Nz), got {latent.shape}")
    if latent.shape[0] != len(t) or latent.shape[1] != len(x):
        raise ValueError(
            f"Inconsistent latent/t/x shapes: latent={latent.shape}, len(t)={len(t)}, len(x)={len(x)}"
        )

    return latent, t, x, meta


def scalar_meta_to_str(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    return str(array)


def infer_electric_file(latent_file: Path, meta: Dict[str, np.ndarray]) -> Path | None:
    candidates: List[Path] = []
    case_dir_value = meta.get("case_dir")
    if case_dir_value is not None:
        candidates.append(Path(scalar_meta_to_str(case_dir_value)) / "electric_field_full.npz")
    candidates.append(latent_file.with_name("electric_field_full.npz"))
    candidates.append(latent_file.parent / "electric_field_full.npz")

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def load_electric_data_from_file(
    electric_file: Path,
    expected_t: np.ndarray,
    expected_x: np.ndarray,
) -> np.ndarray:
    with np.load(electric_file, allow_pickle=False) as data:
        if "E" not in data or "t" not in data or "x" not in data:
            raise KeyError(f"{electric_file} must contain 'E', 't', and 'x'")
        electric = np.asarray(data["E"], dtype=np.float64)
        t = np.asarray(data["t"], dtype=np.float64)
        x = np.asarray(data["x"], dtype=np.float64)

    expected_shape = (len(expected_t), len(expected_x))
    if electric.shape == (len(expected_x), len(expected_t)):
        electric = electric.T
    elif electric.shape != expected_shape:
        raise ValueError(
            f"Expected electric field shape {expected_shape} or {(len(expected_x), len(expected_t))}, "
            f"got {electric.shape}"
        )

    if t.shape != expected_t.shape or not np.allclose(t, expected_t, rtol=1e-6, atol=1e-8):
        raise ValueError(f"Electric-field time grid in {electric_file} does not match the latent file.")
    if x.shape != expected_x.shape or not np.allclose(x, expected_x, rtol=1e-6, atol=1e-8):
        raise ValueError(f"Electric-field space grid in {electric_file} does not match the latent file.")

    return electric


def reconstruct_electric_data_from_case(
    case_dir: Path,
    expected_t: np.ndarray,
    expected_x: np.ndarray,
) -> np.ndarray:
    payload = load_case_distribution(case_dir)
    _n_t_x, _rho_t_x, electric = reconstruct_electric_field(payload["f"], payload["x"], payload["v"])
    electric64 = np.asarray(electric, dtype=np.float64)
    t = np.asarray(payload["t"], dtype=np.float64)
    x = np.asarray(payload["x"], dtype=np.float64)

    if electric64.shape != (len(expected_t), len(expected_x)):
        raise ValueError(
            f"Reconstructed electric field shape {(len(expected_t), len(expected_x))} expected, "
            f"got {electric64.shape}"
        )
    if t.shape != expected_t.shape or not np.allclose(t, expected_t, rtol=1e-6, atol=1e-8):
        raise ValueError(f"Reconstructed time grid from {case_dir} does not match the latent file.")
    if x.shape != expected_x.shape or not np.allclose(x, expected_x, rtol=1e-6, atol=1e-8):
        raise ValueError(f"Reconstructed space grid from {case_dir} does not match the latent file.")

    return electric64


def resolve_electric_data(
    latent_file: Path,
    meta: Dict[str, np.ndarray],
    expected_t: np.ndarray,
    expected_x: np.ndarray,
    requested_electric_file: Path | None,
    disable_electric_field: bool,
) -> Tuple[np.ndarray | None, str | None]:
    if disable_electric_field:
        return None, None

    if requested_electric_file is not None:
        electric_file = requested_electric_file.resolve()
        electric = load_electric_data_from_file(electric_file, expected_t, expected_x)
        return electric, str(electric_file)

    inferred_electric_file = infer_electric_file(latent_file, meta)
    if inferred_electric_file is not None:
        electric = load_electric_data_from_file(inferred_electric_file, expected_t, expected_x)
        return electric, str(inferred_electric_file)

    case_dir_value = meta.get("case_dir")
    if case_dir_value is not None:
        case_dir = Path(scalar_meta_to_str(case_dir_value)).resolve()
        distribution_path = case_dir / "distribution_full.npz"
        if distribution_path.exists():
            electric = reconstruct_electric_data_from_case(case_dir, expected_t, expected_x)
            return electric, f"reconstructed:{distribution_path}"

    return None, None


def check_uniform_spacing(coords: np.ndarray, name: str) -> float:
    if coords.ndim != 1 or len(coords) < 2:
        raise ValueError(f"{name} must be a 1D array with at least two points.")
    diffs = np.diff(coords)
    step = float(np.mean(diffs))
    if not np.allclose(diffs, step, rtol=1e-3, atol=1e-6):
        raise ValueError(f"{name} must be uniformly spaced for PDE-FIND. Got diffs {diffs[:5]}")
    return step


def resolve_modes(requested_modes: Sequence[int] | None, nz: int) -> List[int]:
    if requested_modes is None or len(requested_modes) == 0:
        return list(range(nz))
    modes = sorted(set(requested_modes))
    bad = [mode for mode in modes if mode < 0 or mode >= nz]
    if bad:
        raise ValueError(f"Requested latent modes out of range for Nz={nz}: {bad}")
    return modes


def resolve_diff_params(
    nx: int,
    nt: int,
    args: argparse.Namespace,
) -> Tuple[int, int, int, int, int, float, float]:
    width_x = args.width_x if args.width_x is not None else max(nx // 10, 1)
    width_t = args.width_t if args.width_t is not None else max(nt // 10, 1)
    deg_t = args.deg_t if args.deg_t is not None else args.deg_x

    offset_t = width_t if args.time_diff == "poly" else 0
    if args.space_diff == "poly":
        offset_x = width_x
    elif args.space_diff == "CD":
        if args.D <= 2:
            offset_x = 1
        elif args.D == 3:
            offset_x = 2
        else:
            raise ValueError("space_diff='CD' currently supports D <= 3.")
    else:
        offset_x = 0
    n2 = nx - 2 * offset_x
    m2 = nt - 2 * offset_t
    if n2 <= 0 or m2 <= 0:
        raise ValueError(
            f"Invalid cropped size for PDE-FIND: n2={n2}, m2={m2}. "
            "Reduce width_x/width_t or change differentiation method."
        )

    lam_t = 1.0 / nt
    lam_x = 1.0 / nx
    return width_x, width_t, deg_t, offset_x, offset_t, lam_x, lam_t


def central_difference(u: np.ndarray, dx: float, order: int) -> np.ndarray:
    n = u.size
    derivative = np.zeros(n, dtype=np.complex64)

    if order == 1:
        derivative[1 : n - 1] = (u[2:] - u[:-2]) / (2.0 * dx)
        return derivative
    if order == 2:
        derivative[1 : n - 1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx**2)
        return derivative
    if order == 3:
        derivative[2 : n - 2] = (0.5 * u[4:] - u[3:-1] + u[1:-3] - 0.5 * u[:-4]) / (dx**3)
        return derivative

    raise ValueError("central_difference currently supports orders 1, 2, and 3.")


def stabilized_reciprocal(values: np.ndarray, eps: float) -> np.ndarray:
    if eps <= 0.0:
        raise ValueError(f"reciprocal-eps must be positive, got {eps}")
    magnitude = np.abs(values)
    phase = np.where(magnitude > 0.0, values / magnitude, 1.0)
    safe = np.where(magnitude >= eps, values, phase * eps)
    return 1.0 / safe


def validate_savgol_params(window: int, polyorder: int, axis_name: str) -> None:
    if window <= 0 or window % 2 == 0:
        raise ValueError(f"SG window for {axis_name} must be a positive odd integer, got {window}")
    if polyorder < 0:
        raise ValueError(f"SG polyorder for {axis_name} must be nonnegative, got {polyorder}")
    if polyorder >= window:
        raise ValueError(
            f"SG polyorder for {axis_name} must be smaller than the window length, "
            f"got polyorder={polyorder}, window={window}"
        )


def savgol_derivative(u: np.ndarray, delta: float, order: int, window: int, polyorder: int) -> np.ndarray:
    validate_savgol_params(window, polyorder, "derivative")
    if order < 1:
        raise ValueError(f"SG derivative order must be positive, got {order}")
    if order > polyorder:
        raise ValueError(
            f"SG derivative order must be <= polyorder, got order={order}, polyorder={polyorder}"
        )
    if window > u.size:
        raise ValueError(f"SG window length {window} exceeds available samples {u.size}")
    return np.asarray(
        savgol_filter(u, window_length=window, polyorder=polyorder, deriv=order, delta=delta, mode="interp"),
        dtype=np.complex64,
    )


def compute_time_derivative(
    field_x_t: np.ndarray,
    dt: float,
    pde_find,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, int, int]:
    nx, nt = field_x_t.shape
    width_x, width_t, deg_t, offset_x, offset_t, _lam_x, lam_t = resolve_diff_params(nx, nt, args)
    n2 = nx - 2 * offset_x
    m2 = nt - 2 * offset_t

    ut = np.zeros((n2, m2), dtype=np.complex64)

    if args.time_diff == "FDconv":
        smooth = np.zeros((nx, nt), dtype=np.complex64)
        for j in range(nt):
            smooth[:, j] = pde_find.ConvSmoother(field_x_t[:, j], width_t, args.sigma)
        for i in range(n2):
            ut[i, :] = pde_find.FiniteDiff(smooth[i + offset_x, :], dt, 1)
    elif args.time_diff == "poly":
        time_grid = np.linspace(0.0, (nt - 1) * dt, nt)
        for i in range(n2):
            ut[i, :] = pde_find.PolyDiff(
                field_x_t[i + offset_x, :],
                time_grid,
                diff=1,
                width=width_t,
                deg=deg_t,
            )[:, 0]
    elif args.time_diff == "Tik":
        for i in range(n2):
            ut[i, :] = pde_find.TikhonovDiff(field_x_t[i + offset_x, :], dt, lam_t)
    elif args.time_diff == "SG":
        validate_savgol_params(args.sg_window_t, args.sg_poly_t, "t")
        for i in range(n2):
            ut[i, :] = savgol_derivative(
                field_x_t[i + offset_x, :],
                delta=dt,
                order=1,
                window=args.sg_window_t,
                polyorder=args.sg_poly_t,
            )
    else:
        for i in range(n2):
            ut[i, :] = pde_find.FiniteDiff(field_x_t[i + offset_x, :], dt, 1)

    return ut, offset_x, offset_t


def compute_spatial_derivative_stack(
    field_x_t: np.ndarray,
    dx: float,
    pde_find,
    args: argparse.Namespace,
) -> Tuple[List[np.ndarray], int, int]:
    nx, nt = field_x_t.shape
    width_x, _width_t, _deg_t, offset_x, offset_t, lam_x, _lam_t = resolve_diff_params(nx, nt, args)
    n2 = nx - 2 * offset_x
    m2 = nt - 2 * offset_t
    derivatives: List[np.ndarray] = []

    poly_cache = None
    if args.space_diff == "poly":
        x_grid = np.linspace(0.0, (nx - 1) * dx, nx)
        poly_cache = {
            i: pde_find.PolyDiff(field_x_t[:, i + offset_t], x_grid, diff=args.D, width=width_x, deg=args.deg_x)
            for i in range(m2)
        }
    if args.space_diff == "Fourier":
        ik = 1j * np.fft.fftfreq(nx) * nx

    for order in range(1, args.D + 1):
        ux = np.zeros((n2, m2), dtype=np.complex64)
        for i in range(m2):
            if args.space_diff == "Tik":
                deriv = pde_find.TikhonovDiff(field_x_t[:, i + offset_t], dx, lam_x, d=order)
                ux[:, i] = deriv[offset_x : nx - offset_x]
            elif args.space_diff == "FDconv":
                smooth = pde_find.ConvSmoother(field_x_t[:, i + offset_t], width_x, args.sigma)
                deriv = pde_find.FiniteDiff(smooth, dx, order)
                ux[:, i] = deriv[offset_x : nx - offset_x]
            elif args.space_diff == "FD":
                deriv = pde_find.FiniteDiff(field_x_t[:, i + offset_t], dx, order)
                ux[:, i] = deriv[offset_x : nx - offset_x]
            elif args.space_diff == "CD":
                deriv = central_difference(field_x_t[:, i + offset_t], dx, order)
                ux[:, i] = deriv[offset_x : nx - offset_x]
            elif args.space_diff == "poly":
                ux[:, i] = poly_cache[i][:, order - 1]
            elif args.space_diff == "SG":
                validate_savgol_params(args.sg_window_x, args.sg_poly_x, "x")
                ux[:, i] = savgol_derivative(
                    field_x_t[:, i + offset_t],
                    delta=dx,
                    order=order,
                    window=args.sg_window_x,
                    polyorder=args.sg_poly_x,
                )[offset_x : nx - offset_x]
            elif args.space_diff == "Fourier":
                deriv = np.fft.ifft((ik**order) * np.fft.fft(field_x_t[:, i + offset_t]))
                ux[:, i] = deriv[offset_x : nx - offset_x]
            else:
                raise ValueError(f"Unsupported space_diff for coupled system: {args.space_diff}")
        derivatives.append(ux)

    return derivatives, offset_x, offset_t


def build_library_system(
    target_name: str,
    target_field_x_t: np.ndarray,
    library_fields: Sequence[Tuple[str, np.ndarray]],
    dt: float,
    dx: float,
    pde_find,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, Sequence[str], str]:
    ut_grid, offset_x, offset_t = compute_time_derivative(target_field_x_t, dt, pde_find, args)

    data_columns = []
    derivative_columns = [np.ones((ut_grid.size, 1), dtype=np.complex64)]
    derivative_descriptions = [""]
    data_descriptions = []

    for field_name, field_x_t in library_fields:
        cropped = field_x_t[offset_x : field_x_t.shape[0] - offset_x, offset_t : field_x_t.shape[1] - offset_t]
        data_columns.append(np.reshape(cropped, (-1, 1), order="F"))
        data_descriptions.append(field_name)

        if args.include_reciprocal:
            reciprocal = stabilized_reciprocal(cropped, args.reciprocal_eps)
            data_columns.append(np.reshape(reciprocal, (-1, 1), order="F"))
            data_descriptions.append(f"inv_{field_name}")

        derivative_stack, deriv_offset_x, deriv_offset_t = compute_spatial_derivative_stack(
            field_x_t,
            dx,
            pde_find,
            args,
        )
        if deriv_offset_x != offset_x or deriv_offset_t != offset_t:
            raise RuntimeError("Inconsistent cropping between time and space derivatives.")
        for order, deriv_grid in enumerate(derivative_stack, start=1):
            derivative_columns.append(np.reshape(deriv_grid, (-1, 1), order="F"))
            derivative_descriptions.append(f"{field_name}_{{{'x' * order}}}")

    data_matrix = np.hstack(data_columns).astype(np.complex64)
    derivatives_matrix = np.hstack(derivative_columns).astype(np.complex64)
    ut = np.reshape(ut_grid, (-1, 1), order="F")

    theta, rhs_description = pde_find.build_Theta(
        data_matrix,
        derivatives_matrix,
        derivative_descriptions,
        args.P,
        data_description=data_descriptions,
    )
    return ut, theta, rhs_description, f"{target_name}_t"


def build_coupled_system(
    latent: np.ndarray,
    modes: Sequence[int],
    target_mode: int,
    electric_field: np.ndarray | None,
    dt: float,
    dx: float,
    pde_find,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, Sequence[str], str]:
    library_fields = [(f"z{mode}", latent[:, :, mode].T.astype(np.float64)) for mode in modes]
    if electric_field is not None:
        library_fields.append(("E", electric_field.T.astype(np.float64)))

    return build_library_system(
        target_name=f"z{target_mode}",
        target_field_x_t=latent[:, :, target_mode].T.astype(np.float64),
        library_fields=library_fields,
        dt=dt,
        dx=dx,
        pde_find=pde_find,
        args=args,
    )


def complex_to_pairs(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return values.real.astype(np.float64), values.imag.astype(np.float64)


def format_pde(weights: np.ndarray, rhs_description: Sequence[str], lhs_name: str) -> str:
    terms: List[str] = []
    flat_weights = np.asarray(weights).reshape(-1)
    for coeff, description in zip(flat_weights, rhs_description):
        if coeff != 0:
            terms.append(f"({coeff.real:0.6f} {coeff.imag:+0.6f}i){description}")
    rhs = " + ".join(terms) if terms else "0"
    return f"{lhs_name} = {rhs}"


def nonzero_term_count(weights: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(weights).reshape(-1)))


def discover_mode_pde(
    mode_index: int,
    latent: np.ndarray,
    electric_field: np.ndarray | None,
    dt: float,
    dx: float,
    pde_find,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Sequence[str], Dict[str, float], str]:
    library_fields = [("z", latent[:, :, mode_index].T.astype(np.float64))]
    if electric_field is not None:
        library_fields.append(("E", electric_field.T.astype(np.float64)))

    ut, theta, rhs_description, lhs_name = build_library_system(
        target_name=f"z{mode_index}",
        target_field_x_t=latent[:, :, mode_index].T.astype(np.float64),
        library_fields=library_fields,
        dt=dt,
        dx=dx,
        pde_find=pde_find,
        args=args,
    )

    weights = pde_find.TrainSTRidge(
        theta,
        ut,
        args.lam,
        args.d_tol,
        maxit=args.maxit,
        STR_iters=args.str_iters,
        l0_penalty=args.l0_penalty,
        normalize=args.normalize,
        split=args.split,
        print_best_tol=args.print_best_tol,
    )

    residual = ut - theta.dot(weights)
    residual_l2 = float(np.linalg.norm(residual, 2))
    ut_l2 = float(np.linalg.norm(ut, 2))
    relative_residual = residual_l2 / ut_l2 if ut_l2 > 0.0 else 0.0
    metrics = {
        "residual_l2": residual_l2,
        "ut_l2": ut_l2,
        "relative_residual_l2": relative_residual,
        "nonzero_terms": float(nonzero_term_count(weights)),
        "num_rows": float(theta.shape[0]),
        "num_terms": float(theta.shape[1]),
    }

    equation = format_pde(weights, rhs_description, lhs_name=lhs_name)
    return np.asarray(weights).reshape(-1), rhs_description, metrics, equation


def discover_coupled_mode_pde(
    mode_index: int,
    latent: np.ndarray,
    modes: Sequence[int],
    electric_field: np.ndarray | None,
    dt: float,
    dx: float,
    pde_find,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Sequence[str], Dict[str, float], str]:
    ut, theta, rhs_description, lhs_name = build_coupled_system(
        latent=latent,
        modes=modes,
        target_mode=mode_index,
        electric_field=electric_field,
        dt=dt,
        dx=dx,
        pde_find=pde_find,
        args=args,
    )

    weights = pde_find.TrainSTRidge(
        theta,
        ut,
        args.lam,
        args.d_tol,
        maxit=args.maxit,
        STR_iters=args.str_iters,
        l0_penalty=args.l0_penalty,
        normalize=args.normalize,
        split=args.split,
        print_best_tol=args.print_best_tol,
    )

    residual = ut - theta.dot(weights)
    residual_l2 = float(np.linalg.norm(residual, 2))
    ut_l2 = float(np.linalg.norm(ut, 2))
    relative_residual = residual_l2 / ut_l2 if ut_l2 > 0.0 else 0.0
    metrics = {
        "residual_l2": residual_l2,
        "ut_l2": ut_l2,
        "relative_residual_l2": relative_residual,
        "nonzero_terms": float(nonzero_term_count(weights)),
        "num_rows": float(theta.shape[0]),
        "num_terms": float(theta.shape[1]),
    }
    equation = format_pde(weights, rhs_description, lhs_name=lhs_name)
    return np.asarray(weights).reshape(-1), rhs_description, metrics, equation


def discover_electric_field_pde(
    latent: np.ndarray,
    modes: Sequence[int],
    electric_field: np.ndarray | None,
    dt: float,
    dx: float,
    pde_find,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Sequence[str], Dict[str, float], str]:
    if electric_field is None:
        raise ValueError("Electric field data is required to discover an E_t PDE.")

    library_fields = [(f"z{mode}", latent[:, :, mode].T.astype(np.float64)) for mode in modes]
    library_fields.append(("E", electric_field.T.astype(np.float64)))

    ut, theta, rhs_description, lhs_name = build_library_system(
        target_name="E",
        target_field_x_t=electric_field.T.astype(np.float64),
        library_fields=library_fields,
        dt=dt,
        dx=dx,
        pde_find=pde_find,
        args=args,
    )

    weights = pde_find.TrainSTRidge(
        theta,
        ut,
        args.lam,
        args.d_tol,
        maxit=args.maxit,
        STR_iters=args.str_iters,
        l0_penalty=args.l0_penalty,
        normalize=args.normalize,
        split=args.split,
        print_best_tol=args.print_best_tol,
    )

    residual = ut - theta.dot(weights)
    residual_l2 = float(np.linalg.norm(residual, 2))
    ut_l2 = float(np.linalg.norm(ut, 2))
    relative_residual = residual_l2 / ut_l2 if ut_l2 > 0.0 else 0.0
    metrics = {
        "residual_l2": residual_l2,
        "ut_l2": ut_l2,
        "relative_residual_l2": relative_residual,
        "nonzero_terms": float(nonzero_term_count(weights)),
        "num_rows": float(theta.shape[0]),
        "num_terms": float(theta.shape[1]),
    }
    equation = format_pde(weights, rhs_description, lhs_name=lhs_name)
    return np.asarray(weights).reshape(-1), rhs_description, metrics, equation


def save_report(report_path: Path, lines: Iterable[str]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_results(
    output_path: Path,
    latent_file: Path,
    electric_file: str | None,
    t: np.ndarray,
    x: np.ndarray,
    modes: Sequence[int],
    rhs_description: Sequence[str],
    coefficients: np.ndarray,
    metrics: Dict[str, np.ndarray],
    equations: Sequence[str],
    meta: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> None:
    coeff_real, coeff_imag = complex_to_pairs(coefficients)
    payload: Dict[str, np.ndarray] = {
        "latent_file": np.asarray(str(latent_file)),
        "electric_file": np.asarray("" if electric_file is None else electric_file),
        "used_electric_field": np.asarray(electric_file is not None),
        "system_mode": np.asarray(args.system),
        "t": t.astype(np.float64),
        "x": x.astype(np.float64),
        "mode_indices": np.asarray(modes, dtype=np.int32),
        "rhs_description": np.asarray(list(rhs_description)),
        "coefficients_real": coeff_real,
        "coefficients_imag": coeff_imag,
        "equations": np.asarray(list(equations)),
        "D": np.asarray(args.D, dtype=np.int32),
        "P": np.asarray(args.P, dtype=np.int32),
        "dt": np.asarray(check_uniform_spacing(t, "t"), dtype=np.float64),
        "dx": np.asarray(check_uniform_spacing(x, "x"), dtype=np.float64),
        "time_diff": np.asarray(args.time_diff),
        "space_diff": np.asarray(args.space_diff),
        "sg_window_x": np.asarray(args.sg_window_x, dtype=np.int32),
        "sg_window_t": np.asarray(args.sg_window_t, dtype=np.int32),
        "sg_poly_x": np.asarray(args.sg_poly_x, dtype=np.int32),
        "sg_poly_t": np.asarray(args.sg_poly_t, dtype=np.int32),
        "include_reciprocal": np.asarray(args.include_reciprocal),
        "reciprocal_eps": np.asarray(args.reciprocal_eps, dtype=np.float64),
        "lam": np.asarray(args.lam, dtype=np.float64),
        "d_tol": np.asarray(args.d_tol, dtype=np.float64),
        "l0_penalty": np.asarray(np.nan if args.l0_penalty is None else args.l0_penalty, dtype=np.float64),
        "maxit": np.asarray(args.maxit, dtype=np.int32),
        "str_iters": np.asarray(args.str_iters, dtype=np.int32),
        "normalize": np.asarray(args.normalize, dtype=np.int32),
        "split": np.asarray(args.split, dtype=np.float64),
    }
    payload.update(meta)
    payload.update(metrics)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)


def main() -> None:
    args = parse_args()
    patch_pde_find_compatibility(pde_find)

    latent_file = args.latent_file.resolve()
    latent, t, x, meta = load_latent_data(latent_file)
    electric_field, electric_source = resolve_electric_data(
        latent_file=latent_file,
        meta=meta,
        expected_t=t,
        expected_x=x,
        requested_electric_file=args.electric_file,
        disable_electric_field=args.no_electric_field,
    )
    dt = check_uniform_spacing(t, "t")
    dx = check_uniform_spacing(x, "x")

    nt, nx, nz = latent.shape
    modes = resolve_modes(args.modes, nz)
    output_path = args.output if args.output is not None else infer_output_path(latent_file)
    report_path = args.report if args.report is not None else infer_report_path(output_path)

    print(f"Loaded latent data from: {latent_file}")
    if electric_source is None:
        print("Electric field source: none (latent-only library)")
    else:
        print(f"Electric field source: {electric_source}")
        print(f"Electric field shape (Nt, Nx): {electric_field.shape}")
    print(f"Latent shape (Nt, Nx, Nz): ({nt}, {nx}, {nz})")
    print(f"Using dt={dt:.6e}, dx={dx:.6e}")
    print(f"System mode: {args.system}")
    print(f"Analyzing latent modes: {modes}")

    rhs_description: Sequence[str] | None = None
    coefficients = []
    equations = []
    residual_l2 = []
    ut_l2 = []
    relative_residual_l2 = []
    nonzero_terms = []
    num_rows = []
    num_terms = []
    report_lines = [
        f"latent_file: {latent_file}",
        f"electric_file: {electric_source}",
        f"latent_shape: (Nt={nt}, Nx={nx}, Nz={nz})",
        f"electric_shape: {None if electric_field is None else electric_field.shape}",
        f"electric_target_in_report: {electric_field is not None}",
        f"dt: {dt:.12e}",
        f"dx: {dx:.12e}",
        f"system: {args.system}",
        f"include_reciprocal: {args.include_reciprocal}",
        f"reciprocal_eps: {args.reciprocal_eps:.12e}",
        f"D: {args.D}",
        f"P: {args.P}",
        f"l0_penalty: {args.l0_penalty}",
        f"time_diff: {args.time_diff}",
        f"space_diff: {args.space_diff}",
        f"sg_window_x: {args.sg_window_x}",
        f"sg_window_t: {args.sg_window_t}",
        f"sg_poly_x: {args.sg_poly_x}",
        f"sg_poly_t: {args.sg_poly_t}",
        f"modes: {modes}",
        "",
    ]

    for mode in modes:
        if args.system == "coupled":
            weights, rhs_description_mode, metrics, equation = discover_coupled_mode_pde(
                mode_index=mode,
                latent=latent,
                modes=modes,
                electric_field=electric_field,
                dt=dt,
                dx=dx,
                pde_find=pde_find,
                args=args,
            )
        else:
            weights, rhs_description_mode, metrics, equation = discover_mode_pde(
                mode_index=mode,
                latent=latent,
                electric_field=electric_field,
                dt=dt,
                dx=dx,
                pde_find=pde_find,
                args=args,
            )
        if rhs_description is None:
            rhs_description = rhs_description_mode
        elif list(rhs_description) != list(rhs_description_mode):
            raise RuntimeError("Inconsistent rhs_description across analyzed modes.")

        coefficients.append(weights)
        equations.append(equation)
        residual_l2.append(metrics["residual_l2"])
        ut_l2.append(metrics["ut_l2"])
        relative_residual_l2.append(metrics["relative_residual_l2"])
        nonzero_terms.append(metrics["nonzero_terms"])
        num_rows.append(metrics["num_rows"])
        num_terms.append(metrics["num_terms"])

        print("")
        print(f"Mode z{mode}:")
        print(f"  residual_l2={metrics['residual_l2']:.6e}")
        print(f"  relative_residual_l2={metrics['relative_residual_l2']:.6e}")
        print(f"  nonzero_terms={int(metrics['nonzero_terms'])}")
        pde_find.print_pde(weights.reshape(-1, 1), rhs_description_mode, ut=f"z{mode}_t")

        report_lines.extend(
            [
                f"Mode z{mode}",
                f"residual_l2: {metrics['residual_l2']:.12e}",
                f"relative_residual_l2: {metrics['relative_residual_l2']:.12e}",
                f"nonzero_terms: {int(metrics['nonzero_terms'])}",
                equation,
                "",
            ]
        )

    if rhs_description is None:
        raise RuntimeError("No latent modes were analyzed.")

    if electric_field is not None:
        e_weights, e_rhs_description, e_metrics, e_equation = discover_electric_field_pde(
            latent=latent,
            modes=modes,
            electric_field=electric_field,
            dt=dt,
            dx=dx,
            pde_find=pde_find,
            args=args,
        )

        print("")
        print("Electric field E:")
        print(f"  residual_l2={e_metrics['residual_l2']:.6e}")
        print(f"  relative_residual_l2={e_metrics['relative_residual_l2']:.6e}")
        print(f"  nonzero_terms={int(e_metrics['nonzero_terms'])}")
        pde_find.print_pde(e_weights.reshape(-1, 1), e_rhs_description, ut="E_t")

        report_lines.extend(
            [
                "Mode E",
                f"residual_l2: {e_metrics['residual_l2']:.12e}",
                f"relative_residual_l2: {e_metrics['relative_residual_l2']:.12e}",
                f"nonzero_terms: {int(e_metrics['nonzero_terms'])}",
                e_equation,
                "",
            ]
        )

    coefficients_array = np.vstack(coefficients).astype(np.complex128)
    metrics_arrays = {
        "residual_l2": np.asarray(residual_l2, dtype=np.float64),
        "ut_l2": np.asarray(ut_l2, dtype=np.float64),
        "relative_residual_l2": np.asarray(relative_residual_l2, dtype=np.float64),
        "nonzero_terms": np.asarray(nonzero_terms, dtype=np.int32),
        "num_rows": np.asarray(num_rows, dtype=np.int32),
        "num_terms": np.asarray(num_terms, dtype=np.int32),
    }

    save_results(
        output_path=output_path,
        latent_file=latent_file,
        electric_file=electric_source,
        t=t,
        x=x,
        modes=modes,
        rhs_description=rhs_description,
        coefficients=coefficients_array,
        metrics=metrics_arrays,
        equations=equations,
        meta=meta,
        args=args,
    )
    save_report(report_path, report_lines)
    print("")
    print(f"Saved PDE-FIND results to: {output_path}")
    print(f"Saved PDE-FIND report to: {report_path}")


if __name__ == "__main__":
    main()
