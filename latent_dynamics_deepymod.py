#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
from itertools import combinations_with_replacement
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from latent_dynamics import check_uniform_spacing, load_latent_data, resolve_modes

try:
    import torch
    from deepymod import DeepMoD
    from deepymod.model.constraint import LeastSquares, Ridge
    from deepymod.model.deepmod import Estimator, Library
    from deepymod.model.func_approx import NN, Siren
    from deepymod.model.library import Library1D, library_deriv
    from deepymod.model.sparse_estimators import PDEFIND, STLSQ, Threshold
    from deepymod.training import train
    from deepymod.training.sparsity_scheduler import Periodic
except ImportError as exc:
    raise SystemExit(
        "PyTorch and deepymod are required. Run this script in the 'lasdi' environment, for example:\n"
        "  conda run -n lasdi python latent_dynamics_deepymod.py --latent-file <path>"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Identify latent spatiotemporal dynamics with DeepMoD from a latent autoencoder .npz file. "
            "Selected latent modes can be fit independently or as a coupled system with mode interactions."
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
        help="Fit each selected mode independently or discover a coupled latent system.",
    )
    parser.add_argument(
        "--poly-order",
        type=int,
        default=2,
        help="Maximum polynomial power in the DeepMoD library.",
    )
    parser.add_argument(
        "--diff-order",
        type=int,
        default=2,
        help="Maximum spatial derivative order in the DeepMoD library.",
    )
    parser.add_argument(
        "--library-type",
        type=str,
        default="full",
        choices=("full", "state-polynomial"),
        help=(
            "Library family. 'full' keeps the current tensor-product library over states "
            "and derivatives. 'state-polynomial' keeps derivative terms linear and only "
            "allows nonlinear products among latent states."
        ),
    )
    parser.add_argument(
        "--network",
        type=str,
        default="tanh",
        choices=("tanh", "siren"),
        help="Function approximator used by DeepMoD.",
    )
    parser.add_argument(
        "--hidden-layers",
        type=str,
        default="64,64,64",
        help="Comma-separated hidden widths for the network.",
    )
    parser.add_argument(
        "--first-omega-0",
        type=float,
        default=30.0,
        help="First-layer omega_0 for SIREN networks.",
    )
    parser.add_argument(
        "--hidden-omega-0",
        type=float,
        default=30.0,
        help="Hidden-layer omega_0 for SIREN networks.",
    )
    parser.add_argument(
        "--sparse-estimator",
        type=str,
        default="stlsq",
        choices=("stlsq", "threshold", "pdefind"),
        help="Sparse-regression backend used by DeepMoD.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Threshold used by STLSQ/Threshold estimators.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Ridge parameter for STLSQ.",
    )
    parser.add_argument(
        "--stlsq-max-iter",
        type=int,
        default=20,
        help="Maximum iterations for STLSQ.",
    )
    parser.add_argument(
        "--pdefind-lam",
        type=float,
        default=1e-3,
        help="Lambda used by the DeepMoD PDEFIND estimator.",
    )
    parser.add_argument(
        "--pdefind-dtol",
        type=float,
        default=1e-1,
        help="Tolerance increment used by the DeepMoD PDEFIND estimator.",
    )
    parser.add_argument(
        "--constraint",
        type=str,
        default="least-squares",
        choices=("least-squares", "ridge"),
        help="Constraint solver used to estimate coefficients from active library terms.",
    )
    parser.add_argument(
        "--constraint-ridge-lambda",
        type=float,
        default=1e-3,
        help="Ridge parameter used when --constraint ridge.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-3,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5000,
        help="Maximum DeepMoD training iterations for each independent mode or for the coupled run.",
    )
    parser.add_argument(
        "--write-iterations",
        type=int,
        default=25,
        help="How often DeepMoD logs metrics and checks sparsity/convergence.",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help=(
            "Random train/test split fraction over sampled space-time points. "
            "Passing 1.0 reuses the training points for the test/validation loader."
        ),
    )
    parser.add_argument(
        "--sparsity-warmup",
        type=int,
        default=1000,
        help="Iteration before sparsity updates start.",
    )
    parser.add_argument(
        "--sparsity-update-interval",
        type=int,
        default=100,
        help="Iterations between sparsity-mask updates after warmup.",
    )
    parser.add_argument(
        "--convergence-patience",
        type=int,
        default=200,
        help="DeepMoD convergence patience on the L1 coefficient norm.",
    )
    parser.add_argument(
        "--convergence-delta",
        type=float,
        default=1e-3,
        help="DeepMoD convergence tolerance on the L1 coefficient norm.",
    )
    parser.add_argument(
        "--time-stride",
        type=int,
        default=1,
        help="Optional striding in time before flattening the data.",
    )
    parser.add_argument(
        "--space-stride",
        type=int,
        default=1,
        help="Optional striding in space before flattening the data.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Optional random cap on the number of sampled (t, x) points used in each run.",
    )
    parser.add_argument(
        "--eval-max-points",
        type=int,
        default=20000,
        help="Maximum number of points used for coefficient/residual evaluation after training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Torch device used for training.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npz path. Defaults to <latent-file-stem>_deepymod.npz.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Text report path. Defaults to <output-stem>.txt.",
    )
    return parser.parse_args()


def parse_hidden_layers(spec: str) -> List[int]:
    layers = [int(item.strip()) for item in spec.split(",") if item.strip()]
    if not layers:
        raise ValueError("hidden-layers must contain at least one positive integer.")
    if any(width <= 0 for width in layers):
        raise ValueError(f"hidden-layers must be positive, got {layers}")
    return layers


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def infer_output_path(latent_file: Path, system: str) -> Path:
    suffix = "_deepymod_coupled" if system == "coupled" else "_deepymod"
    return latent_file.with_name(f"{latent_file.stem}{suffix}.npz")


def infer_report_path(output_path: Path) -> Path:
    return output_path.with_suffix(".txt")


def scalar_meta_to_str(value: np.ndarray) -> str:
    array = np.asarray(value)
    return str(array.item()) if array.shape == () else str(array)


def extract_metadata(meta: Dict[str, np.ndarray]) -> Dict[str, object]:
    payload: Dict[str, object] = {}
    for key, value in meta.items():
        array = np.asarray(value)
        payload[key] = array.item() if array.shape == () else array.tolist()
    return payload


def derivative_label(mode_index: int, order: int, generic: bool = False) -> str:
    prefix = "z" if generic else f"z_{mode_index}"
    if order == 1:
        return f"{prefix}_x"
    return f"{prefix}_x{'x' * (order - 1)}"


def build_full_library_labels(poly_order: int, diff_order: int) -> List[str]:
    labels: List[str] = []
    deriv_terms = ["1"]
    for order in range(1, diff_order + 1):
        deriv_terms.append(derivative_label(mode_index=0, order=order, generic=True))

    poly_terms = ["1"]
    for power in range(1, poly_order + 1):
        poly_terms.append("z" if power == 1 else f"z^{power}")

    for poly_term in poly_terms:
        for deriv_term in deriv_terms:
            if poly_term == "1" and deriv_term == "1":
                labels.append("1")
            elif poly_term == "1":
                labels.append(deriv_term)
            elif deriv_term == "1":
                labels.append(poly_term)
            else:
                labels.append(f"{poly_term} {deriv_term}")
    return labels


def build_state_polynomial_labels(poly_order: int, diff_order: int) -> List[str]:
    labels = ["1", "z"]
    for order in range(1, diff_order + 1):
        labels.append(derivative_label(mode_index=0, order=order, generic=True))
    for power in range(2, poly_order + 1):
        labels.append(f"z^{power}")
    return labels


def build_coupled_base_labels(mode_indices: Sequence[int], diff_order: int) -> List[str]:
    labels: List[str] = []
    for mode_index in mode_indices:
        labels.append(f"z_{mode_index}")
        for order in range(1, diff_order + 1):
            labels.append(derivative_label(mode_index=mode_index, order=order, generic=False))
    return labels


def build_full_coupled_library_labels(mode_indices: Sequence[int], poly_order: int, diff_order: int) -> List[str]:
    base_labels = build_coupled_base_labels(mode_indices, diff_order)
    labels = ["1"]
    for degree in range(1, poly_order + 1):
        for combo in combinations_with_replacement(base_labels, degree):
            labels.append(" ".join(combo))
    return labels


def build_coupled_state_polynomial_labels(
    mode_indices: Sequence[int],
    poly_order: int,
    diff_order: int,
) -> List[str]:
    state_labels = [f"z_{mode_index}" for mode_index in mode_indices]
    derivative_labels: List[str] = []
    for mode_index in mode_indices:
        for order in range(1, diff_order + 1):
            derivative_labels.append(derivative_label(mode_index=mode_index, order=order, generic=False))

    labels = ["1", *state_labels, *derivative_labels]
    for degree in range(2, poly_order + 1):
        for combo in combinations_with_replacement(state_labels, degree):
            labels.append(" ".join(combo))
    return labels


def build_library_labels(poly_order: int, diff_order: int, library_type: str) -> List[str]:
    if library_type == "state-polynomial":
        return build_state_polynomial_labels(poly_order, diff_order)
    return build_full_library_labels(poly_order, diff_order)


def build_coupled_library_labels(
    mode_indices: Sequence[int],
    poly_order: int,
    diff_order: int,
    library_type: str,
) -> List[str]:
    if library_type == "state-polynomial":
        return build_coupled_state_polynomial_labels(mode_indices, poly_order, diff_order)
    return build_full_coupled_library_labels(mode_indices, poly_order, diff_order)


def format_equation(mode_index: int, labels: Sequence[str], coeffs: np.ndarray, tol: float = 1e-12) -> str:
    terms: List[str] = []
    for label, coeff in zip(labels, coeffs):
        if abs(float(coeff)) <= tol:
            continue
        terms.append(f"{coeff:+.6e}*{label}")
    rhs = " ".join(terms) if terms else "0"
    return f"z_{mode_index}_t = {rhs}"


def sample_indices(total_points: int, max_points: int | None, rng: np.random.Generator) -> np.ndarray:
    if max_points is None or max_points >= total_points:
        return np.arange(total_points, dtype=np.int64)
    return np.sort(rng.choice(total_points, size=max_points, replace=False))


def make_coordinate_grid(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    tt, xx = np.meshgrid(t, x, indexing="ij")
    return np.stack((tt.reshape(-1), xx.reshape(-1)), axis=1)


@dataclass
class FullBatchLoader:
    coords: torch.Tensor
    values: torch.Tensor
    device: torch.device

    def __len__(self) -> int:
        return 1

    def __iter__(self) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        yield self.coords, self.values

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx != 0:
            raise IndexError(idx)
        return self.coords, self.values


def split_dataset(
    coords: np.ndarray,
    values: np.ndarray,
    train_fraction: float,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[FullBatchLoader, FullBatchLoader, np.ndarray, np.ndarray]:
    if not 0.0 < train_fraction <= 1.0:
        raise ValueError(f"train-split must be between 0 and 1 inclusive, got {train_fraction}")
    n_samples = coords.shape[0]
    order = rng.permutation(n_samples)
    if train_fraction == 1.0:
        train_idx = np.sort(order)
        test_idx = train_idx.copy()
    else:
        split = int(math.floor(train_fraction * n_samples))
        split = min(max(split, 1), n_samples - 1)
        train_idx = np.sort(order[:split])
        test_idx = np.sort(order[split:])

    train_loader = FullBatchLoader(
        coords=torch.as_tensor(coords[train_idx], dtype=torch.float32, device=device),
        values=torch.as_tensor(values[train_idx], dtype=torch.float32, device=device),
        device=device,
    )
    test_loader = FullBatchLoader(
        coords=torch.as_tensor(coords[test_idx], dtype=torch.float32, device=device),
        values=torch.as_tensor(values[test_idx], dtype=torch.float32, device=device),
        device=device,
    )
    return train_loader, test_loader, train_idx, test_idx


def build_function_approximator(
    args: argparse.Namespace,
    hidden_layers: Sequence[int],
    n_outputs: int,
) -> torch.nn.Module:
    if args.network == "tanh":
        return NN(2, list(hidden_layers), n_outputs)
    return Siren(
        n_in=2,
        n_hidden=list(hidden_layers),
        n_out=n_outputs,
        first_omega_0=args.first_omega_0,
        hidden_omega_0=args.hidden_omega_0,
    )


class CoupledModeLibrary1D(Library):
    def __init__(self, n_outputs: int, poly_order: int, diff_order: int) -> None:
        super().__init__()
        self.n_outputs = n_outputs
        self.poly_order = poly_order
        self.diff_order = diff_order

    def library(self, input: tuple[torch.Tensor, torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        prediction, data = input
        if prediction.shape[1] != self.n_outputs:
            raise ValueError(
                f"CoupledModeLibrary1D expected {self.n_outputs} outputs, got {prediction.shape[1]}"
            )

        time_derivs: list[torch.Tensor] = []
        base_features: list[torch.Tensor] = []

        for output in range(self.n_outputs):
            mode_prediction = prediction[:, output : output + 1]
            time_deriv, derivatives = library_deriv(data, mode_prediction, self.diff_order)
            time_derivs.append(time_deriv)
            base_features.append(mode_prediction)
            for order in range(1, self.diff_order + 1):
                base_features.append(derivatives[:, order : order + 1])

        theta_columns = [torch.ones_like(prediction[:, 0:1])]
        for degree in range(1, self.poly_order + 1):
            for combo in combinations_with_replacement(range(len(base_features)), degree):
                column = base_features[combo[0]]
                for idx in combo[1:]:
                    column = column * base_features[idx]
                theta_columns.append(column)

        theta = torch.cat(theta_columns, dim=1)
        return time_derivs, [theta] * self.n_outputs


def build_state_polynomial_theta(
    state_features: Sequence[torch.Tensor],
    derivative_features: Sequence[torch.Tensor],
    poly_order: int,
) -> torch.Tensor:
    theta_columns = [torch.ones_like(state_features[0])]
    theta_columns.extend(state_features)
    theta_columns.extend(derivative_features)

    for degree in range(2, poly_order + 1):
        for combo in combinations_with_replacement(range(len(state_features)), degree):
            column = state_features[combo[0]]
            for idx in combo[1:]:
                column = column * state_features[idx]
            theta_columns.append(column)
    return torch.cat(theta_columns, dim=1)


class StatePolynomialLibrary1D(Library):
    def __init__(self, poly_order: int, diff_order: int) -> None:
        super().__init__()
        self.poly_order = poly_order
        self.diff_order = diff_order

    def library(self, input: tuple[torch.Tensor, torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        prediction, data = input
        if prediction.shape[1] != 1:
            raise ValueError(f"StatePolynomialLibrary1D expected one output, got {prediction.shape[1]}")

        time_deriv, derivatives = library_deriv(data, prediction, self.diff_order)
        derivative_features = [derivatives[:, order : order + 1] for order in range(1, self.diff_order + 1)]
        theta = build_state_polynomial_theta(
            state_features=[prediction],
            derivative_features=derivative_features,
            poly_order=self.poly_order,
        )
        return [time_deriv], [theta]


class CoupledStatePolynomialLibrary1D(Library):
    def __init__(self, n_outputs: int, poly_order: int, diff_order: int) -> None:
        super().__init__()
        self.n_outputs = n_outputs
        self.poly_order = poly_order
        self.diff_order = diff_order

    def library(self, input: tuple[torch.Tensor, torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        prediction, data = input
        if prediction.shape[1] != self.n_outputs:
            raise ValueError(
                f"CoupledStatePolynomialLibrary1D expected {self.n_outputs} outputs, got {prediction.shape[1]}"
            )

        time_derivs: list[torch.Tensor] = []
        state_features: list[torch.Tensor] = []
        derivative_features: list[torch.Tensor] = []

        for output in range(self.n_outputs):
            mode_prediction = prediction[:, output : output + 1]
            time_deriv, derivatives = library_deriv(data, mode_prediction, self.diff_order)
            time_derivs.append(time_deriv)
            state_features.append(mode_prediction)
            for order in range(1, self.diff_order + 1):
                derivative_features.append(derivatives[:, order : order + 1])

        theta = build_state_polynomial_theta(
            state_features=state_features,
            derivative_features=derivative_features,
            poly_order=self.poly_order,
        )
        return time_derivs, [theta] * self.n_outputs


class STLSQEstimator(Estimator):
    def __init__(self, threshold: float, alpha: float, max_iter: int) -> None:
        super().__init__()
        self.estimator = STLSQ(
            threshold=threshold,
            alpha=alpha,
            max_iter=max_iter,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        coeffs = np.asarray(self.estimator.fit(X, y).coef_, dtype=np.float64)
        return coeffs.reshape(-1)


def build_sparse_estimator(args: argparse.Namespace):
    if args.sparse_estimator == "stlsq":
        return STLSQEstimator(
            threshold=args.threshold,
            alpha=args.alpha,
            max_iter=args.stlsq_max_iter,
        )
    if args.sparse_estimator == "threshold":
        return Threshold(threshold=args.threshold)
    return PDEFIND(lam=args.pdefind_lam, dtol=args.pdefind_dtol)


def build_constraint(args: argparse.Namespace):
    if args.constraint == "ridge":
        return Ridge(l=args.constraint_ridge_lambda)
    return LeastSquares()


def build_independent_library(args: argparse.Namespace):
    if args.library_type == "state-polynomial":
        return StatePolynomialLibrary1D(poly_order=args.poly_order, diff_order=args.diff_order)
    return Library1D(poly_order=args.poly_order, diff_order=args.diff_order)


def build_coupled_library(args: argparse.Namespace, n_outputs: int):
    if args.library_type == "state-polynomial":
        return CoupledStatePolynomialLibrary1D(
            n_outputs=n_outputs,
            poly_order=args.poly_order,
            diff_order=args.diff_order,
        )
    return CoupledModeLibrary1D(
        n_outputs=n_outputs,
        poly_order=args.poly_order,
        diff_order=args.diff_order,
    )


def subsample_latent(
    latent: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if args.time_stride <= 0 or args.space_stride <= 0:
        raise ValueError("time-stride and space-stride must be positive integers.")
    return (
        latent[:: args.time_stride, :: args.space_stride, :],
        t[:: args.time_stride],
        x[:: args.space_stride],
    )


def evaluate_model_predictions(
    model: DeepMoD,
    coords: np.ndarray,
    values: np.ndarray,
    device: torch.device,
    chunk_size: int = 16384,
) -> np.ndarray:
    preds: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, coords.shape[0], chunk_size):
            stop = min(start + chunk_size, coords.shape[0])
            batch_coords = torch.as_tensor(coords[start:stop], dtype=torch.float32, device=device)
            batch_pred = model.func_approx(batch_coords)[0]
            preds.append(batch_pred.cpu())
    prediction = torch.cat(preds, dim=0).numpy()
    return np.mean((prediction - values) ** 2, axis=0).astype(np.float64)


def evaluate_discovered_system(
    model: DeepMoD,
    coords: np.ndarray,
    values: np.ndarray,
    device: torch.device,
    max_points: int,
    rng: np.random.Generator,
) -> list[dict]:
    eval_indices = sample_indices(coords.shape[0], max_points, rng)
    coords_eval = torch.as_tensor(coords[eval_indices], dtype=torch.float32, device=device)
    values_eval = torch.as_tensor(values[eval_indices], dtype=torch.float32, device=device)

    model.eval()
    prediction, time_derivs, thetas = model(coords_eval)
    model.constraint.sparsity_masks = model.sparse_estimator(thetas, time_derivs)
    _ = model.constraint((time_derivs, thetas))

    prediction_np = prediction.detach().cpu().numpy()
    values_np = values_eval.detach().cpu().numpy()
    sample_mse = np.mean((prediction_np - values_np) ** 2, axis=0)

    sparse_coeffs = [
        coeff.detach().cpu().numpy().reshape(-1)
        for coeff in model.constraint_coeffs(scaled=True, sparse=True)
    ]
    sparse_masks = [mask.detach().cpu().numpy().astype(bool) for mask in model.sparsity_masks]

    results: list[dict] = []
    for output_idx, (theta_tensor, ut_tensor, coeffs, mask) in enumerate(
        zip(thetas, time_derivs, sparse_coeffs, sparse_masks)
    ):
        theta = theta_tensor.detach().cpu().numpy()
        ut = ut_tensor.detach().cpu().numpy().reshape(-1)
        residual = ut - theta @ coeffs
        results.append(
            {
                "output_index": output_idx,
                "coefficients": coeffs,
                "mask": mask,
                "residual_l2": float(np.linalg.norm(residual)),
                "ut_l2": float(np.linalg.norm(ut)),
                "relative_residual_l2": float(np.linalg.norm(residual) / (np.linalg.norm(ut) + 1e-12)),
                "sample_mse": float(sample_mse[output_idx]),
            }
        )
    return results


def train_independent_mode(
    mode_index: int,
    coords: np.ndarray,
    values: np.ndarray,
    args: argparse.Namespace,
    hidden_layers: Sequence[int],
    device: torch.device,
    output_dir: Path,
) -> dict:
    rng = np.random.default_rng(args.seed + mode_index)
    selected = sample_indices(coords.shape[0], args.max_points, rng)
    coords_selected = coords[selected]
    values_selected = values[selected]

    train_loader, test_loader, train_idx_local, test_idx_local = split_dataset(
        coords_selected,
        values_selected,
        train_fraction=args.train_split,
        rng=rng,
        device=device,
    )

    model = DeepMoD(
        function_approximator=build_function_approximator(args, hidden_layers, n_outputs=1),
        library=build_independent_library(args),
        sparsity_estimator=build_sparse_estimator(args),
        constraint=build_constraint(args),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = Periodic(
        periodicity=args.sparsity_update_interval,
        initial_iteration=args.sparsity_warmup,
    )

    mode_log_dir = (output_dir / "deepymod_logs" / f"mode_{mode_index:02d}").resolve()
    mode_log_dir.mkdir(parents=True, exist_ok=True)

    train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        sparsity_scheduler=scheduler,
        split=args.train_split,
        exp_ID=f"latent_mode_{mode_index}",
        log_dir=str(mode_log_dir),
        max_iterations=args.max_iterations,
        write_iterations=args.write_iterations,
        patience=args.convergence_patience,
        delta=args.convergence_delta,
    )

    equation_eval = evaluate_discovered_system(
        model=model,
        coords=coords_selected,
        values=values_selected,
        device=device,
        max_points=args.eval_max_points,
        rng=np.random.default_rng(args.seed + 10000 + mode_index),
    )[0]
    full_fit_mse = evaluate_model_predictions(model, coords, values, device=device)[0]

    return {
        "mode_index": mode_index,
        "num_selected_points": int(selected.size),
        "num_train_points": int(train_idx_local.size),
        "num_test_points": int(test_idx_local.size),
        "coefficients": equation_eval["coefficients"],
        "mask": equation_eval["mask"],
        "residual_l2": equation_eval["residual_l2"],
        "ut_l2": equation_eval["ut_l2"],
        "relative_residual_l2": equation_eval["relative_residual_l2"],
        "sample_fit_mse": equation_eval["sample_mse"],
        "full_fit_mse": full_fit_mse,
        "log_dir": str(mode_log_dir),
    }


def train_coupled_system(
    mode_indices: Sequence[int],
    coords: np.ndarray,
    values: np.ndarray,
    args: argparse.Namespace,
    hidden_layers: Sequence[int],
    device: torch.device,
    output_dir: Path,
) -> list[dict]:
    rng = np.random.default_rng(args.seed)
    selected = sample_indices(coords.shape[0], args.max_points, rng)
    coords_selected = coords[selected]
    values_selected = values[selected]

    train_loader, test_loader, train_idx_local, test_idx_local = split_dataset(
        coords_selected,
        values_selected,
        train_fraction=args.train_split,
        rng=rng,
        device=device,
    )

    model = DeepMoD(
        function_approximator=build_function_approximator(args, hidden_layers, n_outputs=len(mode_indices)),
        library=build_coupled_library(args, n_outputs=len(mode_indices)),
        sparsity_estimator=build_sparse_estimator(args),
        constraint=build_constraint(args),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = Periodic(
        periodicity=args.sparsity_update_interval,
        initial_iteration=args.sparsity_warmup,
    )

    mode_tag = "_".join(f"{mode:02d}" for mode in mode_indices)
    mode_log_dir = (output_dir / "deepymod_logs" / f"coupled_modes_{mode_tag}").resolve()
    mode_log_dir.mkdir(parents=True, exist_ok=True)

    train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        sparsity_scheduler=scheduler,
        split=args.train_split,
        exp_ID=f"latent_coupled_{mode_tag}",
        log_dir=str(mode_log_dir),
        max_iterations=args.max_iterations,
        write_iterations=args.write_iterations,
        patience=args.convergence_patience,
        delta=args.convergence_delta,
    )

    equation_eval = evaluate_discovered_system(
        model=model,
        coords=coords_selected,
        values=values_selected,
        device=device,
        max_points=args.eval_max_points,
        rng=np.random.default_rng(args.seed + 10000),
    )
    full_fit_mse = evaluate_model_predictions(model, coords, values, device=device)

    mode_results: list[dict] = []
    for result, mode_index, mse in zip(equation_eval, mode_indices, full_fit_mse):
        mode_results.append(
            {
                "mode_index": int(mode_index),
                "num_selected_points": int(selected.size),
                "num_train_points": int(train_idx_local.size),
                "num_test_points": int(test_idx_local.size),
                "coefficients": result["coefficients"],
                "mask": result["mask"],
                "residual_l2": result["residual_l2"],
                "ut_l2": result["ut_l2"],
                "relative_residual_l2": result["relative_residual_l2"],
                "sample_fit_mse": result["sample_mse"],
                "full_fit_mse": float(mse),
                "log_dir": str(mode_log_dir),
            }
        )
    return mode_results


def write_report(
    report_path: Path,
    args: argparse.Namespace,
    latent_file: Path,
    sampled_t: np.ndarray,
    sampled_x: np.ndarray,
    mode_results: Sequence[dict],
    library_labels: Sequence[str],
    meta: Dict[str, np.ndarray],
) -> None:
    lines = [
        "DeepMoD latent dynamics discovery",
        f"latent_file: {latent_file.resolve()}",
        f"output_file: {args.output.resolve() if args.output is not None else infer_output_path(latent_file, args.system).resolve()}",
        f"system: {args.system}",
        f"network: {args.network}",
        f"hidden_layers: {args.hidden_layers}",
        f"library_type: {args.library_type}",
        f"poly_order: {args.poly_order}",
        f"diff_order: {args.diff_order}",
        f"sparse_estimator: {args.sparse_estimator}",
        f"constraint: {args.constraint}",
        f"device: {args.device}",
        f"sampled_nt: {len(sampled_t)}",
        f"sampled_nx: {len(sampled_x)}",
        f"sampled_dt: {sampled_t[1] - sampled_t[0]:.6e}" if len(sampled_t) > 1 else "sampled_dt: n/a",
        f"sampled_dx: {sampled_x[1] - sampled_x[0]:.6e}" if len(sampled_x) > 1 else "sampled_dx: n/a",
        f"case_name: {meta.get('case_name', 'n/a')}",
        f"case_dir: {meta.get('case_dir', 'n/a')}",
        "",
        "Library terms:",
    ]
    lines.extend(f"  {idx:02d}: {label}" for idx, label in enumerate(library_labels))
    lines.append("")

    for result in mode_results:
        lines.extend(
            [
                f"Mode {result['mode_index']}",
                f"  log_dir: {result['log_dir']}",
                f"  sample_fit_mse: {result['sample_fit_mse']:.6e}",
                f"  full_fit_mse: {result['full_fit_mse']:.6e}",
                f"  residual_l2: {result['residual_l2']:.6e}",
                f"  ut_l2: {result['ut_l2']:.6e}",
                f"  relative_residual_l2: {result['relative_residual_l2']:.6e}",
                f"  selected_points: {result['num_selected_points']}",
                f"  train_points: {result['num_train_points']}",
                f"  test_points: {result['num_test_points']}",
                f"  nonzero_terms: {int(np.count_nonzero(result['mask']))}",
                f"  equation: {format_equation(result['mode_index'], library_labels, result['coefficients'])}",
                "",
            ]
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.latent_file = args.latent_file.resolve()
    args.output = infer_output_path(args.latent_file, args.system) if args.output is None else args.output.resolve()
    args.report = infer_report_path(args.output) if args.report is None else args.report.resolve()

    hidden_layers = parse_hidden_layers(args.hidden_layers)
    set_seed(args.seed)
    device = resolve_device(args.device)

    latent, t, x, meta = load_latent_data(args.latent_file)
    sampled_latent, sampled_t, sampled_x = subsample_latent(latent, t, x, args)
    _dt = check_uniform_spacing(sampled_t, "t")
    _dx = check_uniform_spacing(sampled_x, "x")

    mode_indices = resolve_modes(args.modes, sampled_latent.shape[-1])
    coords = make_coordinate_grid(sampled_t, sampled_x)

    output_dir = args.output.parent.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    coords = coords.astype(np.float32)
    if args.system == "coupled":
        library_labels = build_coupled_library_labels(
            mode_indices,
            args.poly_order,
            args.diff_order,
            args.library_type,
        )
        coupled_values = sampled_latent[:, :, mode_indices].reshape(-1, len(mode_indices)).astype(np.float32)
        mode_results = train_coupled_system(
            mode_indices=mode_indices,
            coords=coords,
            values=coupled_values,
            args=args,
            hidden_layers=hidden_layers,
            device=device,
            output_dir=output_dir,
        )
    else:
        library_labels = build_library_labels(args.poly_order, args.diff_order, args.library_type)
        mode_results = []
        for mode_index in mode_indices:
            values = sampled_latent[:, :, mode_index].reshape(-1, 1).astype(np.float32)
            result = train_independent_mode(
                mode_index=mode_index,
                coords=coords,
                values=values,
                args=args,
                hidden_layers=hidden_layers,
                device=device,
                output_dir=output_dir,
            )
            mode_results.append(result)

    coeff_matrix = np.stack([result["coefficients"] for result in mode_results], axis=0)
    mask_matrix = np.stack([result["mask"] for result in mode_results], axis=0)

    np.savez(
        args.output,
        latent_file=str(args.latent_file),
        t=sampled_t,
        x=sampled_x,
        mode_indices=np.asarray(mode_indices, dtype=np.int32),
        library_labels=np.asarray(library_labels),
        coefficients=coeff_matrix.astype(np.float64),
        active_mask=mask_matrix.astype(bool),
        equations=np.asarray(
            [format_equation(result["mode_index"], library_labels, result["coefficients"]) for result in mode_results]
        ),
        residual_l2=np.asarray([result["residual_l2"] for result in mode_results], dtype=np.float64),
        ut_l2=np.asarray([result["ut_l2"] for result in mode_results], dtype=np.float64),
        relative_residual_l2=np.asarray(
            [result["relative_residual_l2"] for result in mode_results], dtype=np.float64
        ),
        sample_fit_mse=np.asarray([result["sample_fit_mse"] for result in mode_results], dtype=np.float64),
        full_fit_mse=np.asarray([result["full_fit_mse"] for result in mode_results], dtype=np.float64),
        num_selected_points=np.asarray([result["num_selected_points"] for result in mode_results], dtype=np.int32),
        num_train_points=np.asarray([result["num_train_points"] for result in mode_results], dtype=np.int32),
        num_test_points=np.asarray([result["num_test_points"] for result in mode_results], dtype=np.int32),
        log_dirs=np.asarray([result["log_dir"] for result in mode_results]),
        system=np.asarray(args.system),
        library_type=np.asarray(args.library_type),
        poly_order=np.asarray(args.poly_order),
        diff_order=np.asarray(args.diff_order),
        network=np.asarray(args.network),
        hidden_layers=np.asarray(hidden_layers, dtype=np.int32),
        sparse_estimator=np.asarray(args.sparse_estimator),
        constraint=np.asarray(args.constraint),
        learning_rate=np.asarray(args.learning_rate),
        max_iterations=np.asarray(args.max_iterations),
        write_iterations=np.asarray(args.write_iterations),
        train_split=np.asarray(args.train_split),
        sparsity_warmup=np.asarray(args.sparsity_warmup),
        sparsity_update_interval=np.asarray(args.sparsity_update_interval),
        convergence_patience=np.asarray(args.convergence_patience),
        convergence_delta=np.asarray(args.convergence_delta),
        time_stride=np.asarray(args.time_stride),
        space_stride=np.asarray(args.space_stride),
        max_points=np.asarray(-1 if args.max_points is None else args.max_points),
        eval_max_points=np.asarray(args.eval_max_points),
        seed=np.asarray(args.seed),
        metadata_json=np.asarray(json.dumps(extract_metadata(meta))),
    )

    write_report(
        report_path=args.report,
        args=args,
        latent_file=args.latent_file,
        sampled_t=sampled_t,
        sampled_x=sampled_x,
        mode_results=mode_results,
        library_labels=library_labels,
        meta={
            key: scalar_meta_to_str(value) if np.asarray(value).shape == () else value
            for key, value in meta.items()
        },
    )

    print(f"Saved DeepMoD discovery results to {args.output}")
    print(f"Saved text report to {args.report}")


if __name__ == "__main__":
    main()
