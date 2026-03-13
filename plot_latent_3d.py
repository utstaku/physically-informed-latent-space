#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot latent variables z_k(x, t) and optional electric field E(x, t) as 3D surfaces."
    )
    parser.add_argument(
        "--latent-file",
        type=Path,
        required=True,
        help="Path to a .npz file containing latent, x, and t arrays.",
    )
    parser.add_argument(
        "--electric-file",
        type=Path,
        default=None,
        help=(
            "Optional path to electric_field_full.npz. If omitted, the script tries to infer it "
            "from the latent metadata."
        ),
    )
    parser.add_argument(
        "--modes",
        type=int,
        nargs="*",
        default=None,
        help="Latent mode indices to plot. Default is all modes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to <latent-file-stem>_latent_3d.png.",
    )
    parser.add_argument(
        "--t-stride",
        type=int,
        default=10,
        help="Subsampling stride along time for plotting.",
    )
    parser.add_argument(
        "--x-stride",
        type=int,
        default=2,
        help="Subsampling stride along x for plotting.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap name.",
    )
    parser.add_argument("--elev", type=float, default=30.0, help="3D view elevation.")
    parser.add_argument("--azim", type=float, default=-130.0, help="3D view azimuth.")
    parser.add_argument(
        "--figscale",
        type=float,
        default=4.8,
        help="Base subplot size in inches.",
    )
    return parser.parse_args()


def infer_output_path(latent_file: Path) -> Path:
    return latent_file.with_name(f"{latent_file.stem}_latent_3d.png")


def scalar_meta_to_str(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    return str(array)


def infer_electric_file(latent_file: Path, case_dir: Path | None) -> Path | None:
    candidates = []
    if case_dir is not None:
        candidates.append(case_dir / "electric_field_full.npz")
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


def load_electric_data(
    electric_file: Path,
    expected_t: np.ndarray,
    expected_x: np.ndarray,
) -> np.ndarray:
    with np.load(electric_file, allow_pickle=False) as data:
        electric = np.asarray(data["E"], dtype=np.float64)
        t = np.asarray(data["t"], dtype=np.float64)
        x = np.asarray(data["x"], dtype=np.float64)

    expected_shape = (len(expected_t), len(expected_x))
    if electric.shape == (len(expected_x), len(expected_t)):
        electric = electric.T
    elif electric.shape != expected_shape:
        raise ValueError(
            f"Expected electric field shape {expected_shape} or {(len(expected_x), len(expected_t))}, got {electric.shape}"
        )

    if t.shape != expected_t.shape or not np.allclose(t, expected_t, rtol=1e-6, atol=1e-8):
        raise ValueError(f"Electric-field time grid in {electric_file} does not match the latent file.")
    if x.shape != expected_x.shape or not np.allclose(x, expected_x, rtol=1e-6, atol=1e-8):
        raise ValueError(f"Electric-field space grid in {electric_file} does not match the latent file.")

    return electric


def resolve_modes(requested: Sequence[int] | None, nz: int) -> list[int]:
    if requested is None or len(requested) == 0:
        return list(range(nz))
    modes = sorted(set(requested))
    invalid = [mode for mode in modes if mode < 0 or mode >= nz]
    if invalid:
        raise ValueError(f"Requested modes out of range for Nz={nz}: {invalid}")
    return modes


def subplot_grid(num_plots: int) -> tuple[int, int]:
    if num_plots <= 2:
        return 1, num_plots
    if num_plots <= 4:
        return 2, 2
    cols = 3
    rows = int(np.ceil(num_plots / cols))
    return rows, cols


def main() -> None:
    args = parse_args()
    with np.load(args.latent_file, allow_pickle=False) as data:
        latent = np.asarray(data["latent"], dtype=np.float64)
        x = np.asarray(data["x"], dtype=np.float64)
        t = np.asarray(data["t"], dtype=np.float64)
        case_dir = Path(scalar_meta_to_str(data["case_dir"])) if "case_dir" in data else None

    if latent.ndim != 3:
        raise ValueError(f"Expected latent shape (Nt, Nx, Nz), got {latent.shape}")
    nt, nx, nz = latent.shape
    modes = resolve_modes(args.modes, nz)

    if args.t_stride <= 0 or args.x_stride <= 0:
        raise ValueError("t-stride and x-stride must be positive integers.")

    t_plot = t[:: args.t_stride]
    x_plot = x[:: args.x_stride]
    tt, xx = np.meshgrid(t_plot, x_plot, indexing="ij")

    electric = None
    electric_source = None
    if args.electric_file is not None:
        electric_source = args.electric_file.resolve()
    else:
        electric_source = infer_electric_file(args.latent_file.resolve(), case_dir)
    if electric_source is not None:
        electric = load_electric_data(electric_source, t, x)

    plot_specs = [(f"z{mode}(x, t)", f"z{mode}", latent[:: args.t_stride, :: args.x_stride, mode]) for mode in modes]
    if electric is not None:
        plot_specs.append(("E(x, t)", "E", electric[:: args.t_stride, :: args.x_stride]))

    rows, cols = subplot_grid(len(plot_specs))
    fig = plt.figure(figsize=(cols * args.figscale, rows * args.figscale))

    for plot_idx, (title, zlabel, surface) in enumerate(plot_specs, start=1):
        ax = fig.add_subplot(rows, cols, plot_idx, projection="3d")
        ax.plot_surface(
            xx,
            tt,
            surface,
            cmap=args.cmap,
            linewidth=0.0,
            antialiased=True,
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_zlabel(zlabel)
        ax.view_init(elev=args.elev, azim=args.azim)

    fig.suptitle(args.latent_file.name)
    fig.tight_layout()

    output = args.output if args.output is not None else infer_output_path(args.latent_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 3D latent plot to: {output}")
    print(f"Plotted modes: {modes}")
    print(f"Plotted electric field: {electric is not None}")
    if electric_source is not None:
        print(f"Electric field source: {electric_source}")
    print(f"Latent shape: (Nt={nt}, Nx={nx}, Nz={nz})")


if __name__ == "__main__":
    main()
