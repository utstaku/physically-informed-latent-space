#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from latent_dynamics import (
    check_uniform_spacing,
    complex_to_pairs,
    compute_spatial_derivative_stack,
    compute_time_derivative,
    format_pde,
    load_latent_data,
    nonzero_term_count,
    patch_pde_find_compatibility,
    resolve_electric_data,
    resolve_modes,
)

import tutorials.PDE_FIND as pde_find


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover compact latent dynamics using the fixed coupled library "
            "[z_i, z_{i,x}, E, z_i*E, z_i*z_{i,x}] over all selected modes."
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
        "--output",
        type=Path,
        default=None,
        help="Output .npz file. Defaults to <latent-file-stem>_compact_pde_find.npz.",
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
        "--time-diff",
        type=str,
        default="FD",
        choices=("poly", "FD", "FDconv", "Tik"),
        help="Time-derivative method passed to PDE-FIND.",
    )
    parser.add_argument(
        "--space-diff",
        type=str,
        default="FD",
        choices=("poly", "FD", "CD", "FDconv", "Tik", "Fourier"),
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
        "--print-best-tol",
        action="store_true",
        help="Print PDE-FIND's best tolerance during STRidge search.",
    )
    args = parser.parse_args()
    args.D = 1
    return args


def infer_output_path(latent_file: Path) -> Path:
    return latent_file.with_name(f"{latent_file.stem}_compact_pde_find.npz")


def infer_report_path(output_path: Path) -> Path:
    return output_path.with_suffix(".txt")


def crop_field(field_x_t: np.ndarray, offset_x: int, offset_t: int) -> np.ndarray:
    x_stop = None if offset_x == 0 else -offset_x
    t_stop = None if offset_t == 0 else -offset_t
    return field_x_t[offset_x:x_stop, offset_t:t_stop]


def compact_rhs_description(modes: Sequence[int]) -> List[str]:
    descriptions: List[str] = []
    descriptions.extend(f"z{mode}" for mode in modes)
    descriptions.extend(f"z{mode}_{{x}}" for mode in modes)
    descriptions.append("E")
    descriptions.extend(f"z{mode}E" for mode in modes)
    descriptions.extend(f"z{mode}z{mode}_{{x}}" for mode in modes)
    return descriptions


def compact_library_summary(modes: Sequence[int]) -> List[str]:
    return [
        f"z ({len(modes)})",
        f"z_x ({len(modes)})",
        "E (1)",
        f"z*E ({len(modes)})",
        f"z*z_x ({len(modes)})",
        f"total ({4 * len(modes) + 1})",
    ]


def discover_mode_compact_pde(
    mode_index: int,
    latent: np.ndarray,
    modes: Sequence[int],
    electric_field: np.ndarray,
    dt: float,
    dx: float,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Sequence[str], Dict[str, float], str]:
    z_field_x_t = latent[:, :, mode_index].T.astype(np.float64)
    e_field_x_t = electric_field.T.astype(np.float64)

    ut_grid, offset_x, offset_t = compute_time_derivative(z_field_x_t, dt, pde_find, args)
    e_grid = crop_field(e_field_x_t, offset_x, offset_t)

    z_grids: List[np.ndarray] = []
    z_x_grids: List[np.ndarray] = []
    for source_mode in modes:
        source_field_x_t = latent[:, :, source_mode].T.astype(np.float64)
        derivative_stack, deriv_offset_x, deriv_offset_t = compute_spatial_derivative_stack(
            source_field_x_t,
            dx,
            pde_find,
            args,
        )
        if deriv_offset_x != offset_x or deriv_offset_t != offset_t:
            raise RuntimeError("Inconsistent cropping between time and space derivatives.")
        z_grids.append(crop_field(source_field_x_t, offset_x, offset_t))
        z_x_grids.append(derivative_stack[0])

    theta_columns: List[np.ndarray] = []
    theta_columns.extend(np.reshape(z_grid, (-1,), order="F") for z_grid in z_grids)
    theta_columns.extend(np.reshape(z_x_grid, (-1,), order="F") for z_x_grid in z_x_grids)
    theta_columns.append(np.reshape(e_grid, (-1,), order="F"))
    theta_columns.extend(np.reshape(z_grid * e_grid, (-1,), order="F") for z_grid in z_grids)
    theta_columns.extend(
        np.reshape(z_grid * z_x_grid, (-1,), order="F")
        for z_grid, z_x_grid in zip(z_grids, z_x_grids)
    )

    theta = np.column_stack(theta_columns).astype(np.complex64)
    ut = np.reshape(ut_grid, (-1, 1), order="F")

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

    rhs_description = compact_rhs_description(modes)
    equation = format_pde(weights, rhs_description, lhs_name=f"z{mode_index}_t")
    return np.asarray(weights).reshape(-1), rhs_description, metrics, equation


def save_report(report_path: Path, lines: Iterable[str]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_results(
    output_path: Path,
    latent_file: Path,
    electric_file: str,
    t: np.ndarray,
    x: np.ndarray,
    modes: Sequence[int],
    coefficients: np.ndarray,
    metrics: Dict[str, np.ndarray],
    equations: Sequence[str],
    meta: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> None:
    coeff_real, coeff_imag = complex_to_pairs(coefficients)
    payload: Dict[str, np.ndarray] = {
        "latent_file": np.asarray(str(latent_file)),
        "electric_file": np.asarray(electric_file),
        "used_electric_field": np.asarray(True),
        "library_description": np.asarray(compact_rhs_description(modes)),
        "t": t.astype(np.float64),
        "x": x.astype(np.float64),
        "mode_indices": np.asarray(modes, dtype=np.int32),
        "coefficients_real": coeff_real,
        "coefficients_imag": coeff_imag,
        "equations": np.asarray(list(equations)),
        "D": np.asarray(1, dtype=np.int32),
        "dt": np.asarray(check_uniform_spacing(t, "t"), dtype=np.float64),
        "dx": np.asarray(check_uniform_spacing(x, "x"), dtype=np.float64),
        "time_diff": np.asarray(args.time_diff),
        "space_diff": np.asarray(args.space_diff),
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
        disable_electric_field=False,
    )
    if electric_field is None or electric_source is None:
        raise FileNotFoundError(
            "Electric field data is required for the compact library "
            "[z_i, z_{i,x}, E, z_i*E, z_i*z_{i,x}]. "
            "Provide --electric-file or ensure electric_field_full.npz can be inferred."
        )

    dt = check_uniform_spacing(t, "t")
    dx = check_uniform_spacing(x, "x")
    nt, nx, nz = latent.shape
    modes = resolve_modes(args.modes, nz)
    output_path = args.output if args.output is not None else infer_output_path(latent_file)
    report_path = args.report if args.report is not None else infer_report_path(output_path)

    print(f"Loaded latent data from: {latent_file}")
    print(f"Electric field source: {electric_source}")
    print(f"Latent shape (Nt, Nx, Nz): ({nt}, {nx}, {nz})")
    print(f"Electric field shape (Nt, Nx): {electric_field.shape}")
    print(f"Using dt={dt:.6e}, dx={dx:.6e}")
    print(f"Compact library summary: {compact_library_summary(modes)}")
    print(f"Analyzing latent modes: {modes}")

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
        f"electric_shape: {electric_field.shape}",
        f"dt: {dt:.12e}",
        f"dx: {dx:.12e}",
        f"library: {compact_rhs_description(modes)}",
        f"library_summary: {compact_library_summary(modes)}",
        f"time_diff: {args.time_diff}",
        f"space_diff: {args.space_diff}",
        f"modes: {modes}",
        "",
    ]

    for mode in modes:
        weights, rhs_description, metrics, equation = discover_mode_compact_pde(
            mode_index=mode,
            latent=latent,
            modes=modes,
            electric_field=electric_field,
            dt=dt,
            dx=dx,
            args=args,
        )
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
        pde_find.print_pde(weights.reshape(-1, 1), rhs_description, ut=f"z{mode}_t")

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
        coefficients=coefficients_array,
        metrics=metrics_arrays,
        equations=equations,
        meta=meta,
        args=args,
    )
    save_report(report_path, report_lines)

    print("")
    print(f"Saved compact PDE-FIND results to: {output_path}")
    print(f"Saved compact PDE-FIND report to: {report_path}")


if __name__ == "__main__":
    main()
