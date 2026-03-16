#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from latent_dynamics import load_latent_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot latent variables z_i(x, t) as heatmaps with x on the horizontal axis and t on the vertical axis."
    )
    parser.add_argument(
        "--latent-file",
        type=Path,
        required=True,
        help="Path to a latent .npz file containing latent, x, and t arrays.",
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
        help="Optional output image path. Defaults to <latent-file-stem>_heatmap.png.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="RdBu_r",
        help="Matplotlib colormap name.",
    )
    parser.add_argument(
        "--center-zero",
        action="store_true",
        help="Force a symmetric color scale around zero for each mode.",
    )
    parser.add_argument(
        "--figscale",
        type=float,
        default=4.8,
        help="Base subplot size in inches.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure with plt.show() after plotting.",
    )
    return parser.parse_args()


def infer_output_path(latent_file: Path) -> Path:
    return latent_file.with_name(f"{latent_file.stem}_heatmap.png")


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
    latent, t, x, _meta = load_latent_data(args.latent_file.resolve())
    nt, nx, nz = latent.shape
    modes = resolve_modes(args.modes, nz)

    rows, cols = subplot_grid(len(modes))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * args.figscale, rows * args.figscale),
        constrained_layout=True,
        squeeze=False,
    )

    for ax, mode in zip(axes.flat, modes):
        field_t_x = np.asarray(latent[:, :, mode], dtype=np.float64)
        imshow_kwargs = {
            "aspect": "auto",
            "origin": "lower",
            "extent": [float(x.min()), float(x.max()), float(t.min()), float(t.max())],
            "cmap": args.cmap,
        }
        if args.center_zero:
            vmax = float(np.max(np.abs(field_t_x)))
            imshow_kwargs["vmin"] = -vmax
            imshow_kwargs["vmax"] = vmax

        image = ax.imshow(field_t_x, **imshow_kwargs)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title(f"Latent mode z_{mode}")
        fig.colorbar(image, ax=ax, shrink=0.9, label=f"z_{mode}")

    for ax in axes.flat[len(modes) :]:
        ax.axis("off")

    fig.suptitle(args.latent_file.name)

    output_path = args.output if args.output is not None else infer_output_path(args.latent_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved latent heatmap to: {output_path}")
    print(f"Latent shape: (Nt={nt}, Nx={nx}, Nz={nz})")
    print(f"Plotted modes: {modes}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
