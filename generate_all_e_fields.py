#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from reconstruct_e_field import load_distribution, reconstruct_electric_field


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct and save electric_field_full.npz for every case under "
            "vlasov_twostream_param_grid."
        )
    )
    parser.add_argument(
        "--grid-dir",
        type=Path,
        default=Path("vlasov_twostream_param_grid"),
        help="Root directory that contains all case subdirectories.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing electric_field_full.npz files.",
    )
    parser.add_argument(
        "--verify-animation",
        action="store_true",
        help="Verify each generated field against animation_data.npz when present.",
    )
    return parser.parse_args()


def iter_case_dirs(grid_dir: Path) -> list[Path]:
    return sorted(path.parent for path in grid_dir.glob("*/distribution_full.npz"))


def save_case(case_dir: Path, verify: bool) -> tuple[str, float | None, float | None]:
    payload = load_distribution(case_dir)
    n_t_x, rho_t_x, e_t_x = reconstruct_electric_field(payload["f"], payload["x"], payload["v"])

    output_path = case_dir / "electric_field_full.npz"
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

    if verify:
        animation_path = case_dir / "animation_data.npz"
        if animation_path.exists():
            with np.load(animation_path) as data:
                saved_e = np.asarray(data["E"], dtype=np.float32)
                frame_stride = int(data["frame_stride"])
            reconstructed = e_t_x[::frame_stride]
            abs_error = np.abs(reconstructed - saved_e)
            return str(case_dir), float(abs_error.max()), float(abs_error.mean())

    return str(case_dir), None, None


def main() -> None:
    args = parse_args()
    grid_dir = args.grid_dir.resolve()
    case_dirs = iter_case_dirs(grid_dir)
    if not case_dirs:
        raise FileNotFoundError(f"No distribution_full.npz cases found under {grid_dir}")

    if args.workers < 1:
        raise ValueError(f"--workers must be >= 1, got {args.workers}")

    to_process: list[Path] = []
    skipped = 0
    for case_dir in case_dirs:
        output_path = case_dir / "electric_field_full.npz"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        to_process.append(case_dir)

    print(f"grid_dir={grid_dir}")
    print(f"total_cases={len(case_dirs)}")
    print(f"workers={args.workers}")
    print(f"to_process={len(to_process)}")
    print(f"skipped_existing={skipped}")

    completed = 0
    verify_errors: list[tuple[str, float, float]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(save_case, case_dir, args.verify_animation): case_dir
            for case_dir in to_process
        }
        for future in as_completed(futures):
            case_dir = futures[future]
            completed += 1
            try:
                case_name, max_err, mean_err = future.result()
            except Exception as exc:
                print(f"[{completed}/{len(to_process)}] FAILED {case_dir}: {exc}")
                raise

            if max_err is None:
                print(f"[{completed}/{len(to_process)}] saved {case_name}")
            else:
                verify_errors.append((case_name, max_err, mean_err or 0.0))
                print(
                    f"[{completed}/{len(to_process)}] saved {case_name} "
                    f"verify_max_abs_error={max_err:.6e}"
                )

    print("done")
    print(f"generated={len(to_process)}")
    print(f"skipped_existing={skipped}")
    if verify_errors:
        worst_case, worst_max, worst_mean = max(verify_errors, key=lambda item: item[1])
        print(f"verify_worst_case={worst_case}")
        print(f"verify_worst_max_abs_error={worst_max:.6e}")
        print(f"verify_worst_mean_abs_error={worst_mean:.6e}")


if __name__ == "__main__":
    main()
