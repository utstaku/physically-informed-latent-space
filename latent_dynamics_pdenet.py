#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Sequence

import numpy as np

from latent_dynamics import check_uniform_spacing, load_latent_data, resolve_electric_data, resolve_modes

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required. Run this script in the 'lasdi' environment, for example:\n"
        "  conda run -n lasdi python latent_dynamics_pdenet.py --latent-file <path>"
    ) from exc


@dataclass(frozen=True)
class LibrarySpec:
    term_indices: tuple[int, ...]
    label: str


@dataclass
class Metrics:
    train_rhs_mse: float
    val_rhs_mse: float
    full_rhs_mse: float
    train_state_mse: float
    val_state_mse: float
    full_state_mse: float
    train_rollout_relative_l2: float
    val_rollout_relative_l2: float
    full_rollout_relative_l2: float
    proxy_relative_l2: float
    rollout_steps: int


@dataclass
class RolloutResult:
    trajectory: np.ndarray
    success: bool
    failure_step: int | None
    message: str


class DivergenceError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover latent dynamics with a 1D PDE-Net-2.0-style model: "
            "learnable spatial derivative filters plus a SymNet right-hand side."
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
        "--validation-latent-file",
        type=Path,
        default=None,
        help=(
            "Optional latent file from a neighboring case used only for validation/model selection. "
            "Training still uses all data from --latent-file."
        ),
    )
    parser.add_argument(
        "--electric-file",
        type=Path,
        default=None,
        help=(
            "Optional electric_field_full.npz used as an external forcing input. "
            "If omitted, the script tries to infer it from the latent metadata or case directory."
        ),
    )
    parser.add_argument(
        "--no-electric-field",
        action="store_true",
        help="Disable electric-field forcing and fit a latent-only PDE-Net.",
    )
    parser.add_argument(
        "--fit-fraction",
        type=float,
        default=1.0,
        help="Fraction of the trajectory used for fitting/evaluation. Default: 1.0.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help=(
            "Deprecated compatibility option. PDE-Net training now uses all fitted time transitions; "
            "validation metrics are reported on the same full trajectory."
        ),
    )
    parser.add_argument(
        "--poly-order",
        type=int,
        default=2,
        help=(
            "Maximum polynomial degree used only for the post-hoc symbolic proxy. "
            "If --symnet-num-layers is omitted, this also sets the SymNet depth."
        ),
    )
    parser.add_argument(
        "--diff-order",
        type=int,
        default=2,
        help="Maximum spatial derivative order represented by learned convolution kernels.",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=5,
        help="Odd convolution width used by each learned derivative kernel.",
    )
    parser.add_argument(
        "--symnet-hidden-dim",
        type=int,
        default=None,
        help="Number of multiplicative channels created by each SymNet layer. Default: max(input_dim, 8).",
    )
    parser.add_argument(
        "--symnet-num-layers",
        type=int,
        default=None,
        help="Number of multiplicative SymNet layers. Default: --poly-order.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size over time windows.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1500,
        help="Number of optimization epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--learning-rate-gamma",
        type=float,
        default=0.995,
        help="Exponential decay applied to the optimizer learning rate every epoch.",
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=1.0,
        help="Max gradient norm applied before each optimizer step. Set <= 0 to disable clipping.",
    )
    parser.add_argument(
        "--sparsity-weight",
        type=float,
        default=1e-5,
        help="L1 penalty weight applied to SymNet parameters.",
    )
    parser.add_argument(
        "--kernel-reg-weight",
        type=float,
        default=1e-3,
        help="L2 penalty keeping learned derivative kernels near their finite-difference initialization.",
    )
    parser.add_argument(
        "--moment-reg-weight",
        type=float,
        default=1e-2,
        help="Weight for moment constraints that keep learned kernels close to derivative operators.",
    )
    parser.add_argument(
        "--state-loss-weight",
        type=float,
        default=1.0,
        help="Weight for the 1-step state prediction loss.",
    )
    parser.add_argument(
        "--rollout-loss-weight",
        type=float,
        default=5.0,
        help="Weight for the multi-step rollout loss.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=4,
        help="Maximum rollout length used in the training loss.",
    )
    parser.add_argument(
        "--rollout-loss-weights",
        type=str,
        default=None,
        help=(
            "Optional comma-separated weights for rollout steps 2..K. "
            "Defaults to uniform weights, matching the FNO script style."
        ),
    )
    parser.add_argument(
        "--rollout-curriculum-start",
        type=float,
        default=0.0,
        help="Training fraction after which rollout length starts increasing beyond 1 step.",
    )
    parser.add_argument(
        "--rollout-curriculum-end",
        type=float,
        default=0.25,
        help="Training fraction by which rollout length reaches --rollout-steps.",
    )
    parser.add_argument(
        "--divergence-threshold",
        type=float,
        default=1e6,
        help="Abort training or rollout when normalized latent magnitudes exceed this threshold.",
    )
    parser.add_argument(
        "--coef-threshold",
        type=float,
        default=5e-4,
        help="Absolute threshold used to prune the post-hoc symbolic proxy coefficients.",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1e-6,
        help="Ridge parameter used when fitting the post-hoc symbolic proxy.",
    )
    parser.add_argument(
        "--rollout-method",
        type=str,
        default="rk4",
        choices=("euler", "rk4"),
        help="Integrator used when computing latent rollout diagnostics.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Torch device selection.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npz path. Defaults to <latent-file-stem>_pdenet.npz.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Text report path. Defaults to <output-stem>.txt.",
    )
    return parser.parse_args()


def infer_output_path(latent_file: Path) -> Path:
    return latent_file.with_name(f"{latent_file.stem}_pdenet.npz")


def infer_report_path(output_path: Path) -> Path:
    return output_path.with_suffix(".txt")


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
            raise RuntimeError("CUDA was requested but is not available in the current environment.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def finite_difference_stencil(order: int, kernel_size: int) -> np.ndarray:
    if order < 1:
        raise ValueError(f"Derivative order must be positive, got {order}")
    if kernel_size <= order:
        raise ValueError(f"kernel_size must be greater than derivative order {order}, got {kernel_size}")
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")

    radius = kernel_size // 2
    coords = np.arange(-radius, radius + 1, dtype=np.float64)
    vandermonde = np.vstack([coords**power for power in range(kernel_size)])
    rhs = np.zeros(kernel_size, dtype=np.float64)
    rhs[order] = math.factorial(order)
    weights = np.linalg.solve(vandermonde, rhs)
    return weights.astype(np.float32)


def adjusted_fit_length(num_times: int, fit_fraction: float) -> int:
    if not (0.0 < fit_fraction <= 1.0):
        raise ValueError(f"fit_fraction must be in (0, 1], got {fit_fraction}")
    fit_length = max(2, int(round(num_times * fit_fraction)))
    return min(num_times, fit_length)


def validate_train_fraction(train_fraction: float) -> None:
    if not (0.0 < train_fraction <= 1.0):
        raise ValueError(f"train_fraction must be in (0, 1], got {train_fraction}")


def build_library_specs(primary_names: Sequence[str], poly_order: int) -> list[LibrarySpec]:
    if poly_order < 1:
        raise ValueError(f"poly_order must be >= 1, got {poly_order}")

    specs = [LibrarySpec(term_indices=tuple(), label="1")]
    for degree in range(1, poly_order + 1):
        for combo in combinations_with_replacement(range(len(primary_names)), degree):
            label = " ".join(primary_names[index] for index in combo)
            specs.append(LibrarySpec(term_indices=tuple(combo), label=label))
    return specs


def resolve_rollout_weights(num_steps: int, spec: str | None) -> np.ndarray:
    if num_steps <= 0:
        raise ValueError(f"rollout_steps must be positive, got {num_steps}")
    if num_steps == 1:
        return np.empty((0,), dtype=np.float32)
    if spec is None:
        return np.ones(num_steps - 1, dtype=np.float32)

    weights = np.asarray([float(token.strip()) for token in spec.split(",") if token.strip()], dtype=np.float32)
    if weights.size == num_steps:
        weights = weights[1:]
    if weights.size != num_steps - 1:
        raise ValueError(
            f"--rollout-loss-weights must contain exactly {num_steps - 1} comma-separated values "
            f"for steps 2..{num_steps}, got {weights.size}."
        )
    if np.any(weights < 0.0):
        raise ValueError(f"--rollout-loss-weights must be non-negative, got {weights}.")
    if not np.any(weights > 0.0):
        raise ValueError("--rollout-loss-weights must contain at least one positive value.")
    return weights


def compute_modewise_normalization(sequence_t_x_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(sequence_t_x_m, dtype=np.float32).mean(axis=(0, 1), keepdims=True)
    std = np.asarray(sequence_t_x_m, dtype=np.float32).std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_modewise_normalization(sequence_t_x_m: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((np.asarray(sequence_t_x_m, dtype=np.float32) - mean) / std).astype(np.float32)


def maybe_normalize_feature(
    feature_t_x_f: np.ndarray | None,
    mean: np.ndarray | None,
    std: np.ndarray | None,
) -> np.ndarray | None:
    if feature_t_x_f is None:
        return None
    if mean is None or std is None:
        return np.asarray(feature_t_x_f, dtype=np.float32)
    return apply_modewise_normalization(feature_t_x_f, mean, std)


def einsum_library(primary_features: torch.Tensor, specs: Sequence[LibrarySpec]) -> torch.Tensor:
    library_terms: list[torch.Tensor] = [torch.ones_like(primary_features[..., :1])]
    for spec in specs[1:]:
        term = primary_features[..., spec.term_indices[0] : spec.term_indices[0] + 1]
        for feature_index in spec.term_indices[1:]:
            term = term * primary_features[..., feature_index : feature_index + 1]
        library_terms.append(term)
    return torch.cat(library_terms, dim=-1)


def extract_state_dict_numpy(module: nn.Module) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    for name, tensor in module.state_dict().items():
        payload[f"param_{name.replace('.', '__')}"] = tensor.detach().cpu().numpy().astype(np.float32)
    return payload


class SharedDerivative1D(nn.Module):
    def __init__(self, order: int, kernel_size: int) -> None:
        super().__init__()
        initial_kernel = finite_difference_stencil(order=order, kernel_size=kernel_size)
        self.order = order
        self.kernel_size = kernel_size
        self.kernel = nn.Parameter(torch.from_numpy(initial_kernel.copy()))
        self.register_buffer("initial_kernel", torch.from_numpy(initial_kernel.copy()))

    def forward(self, state_b_m_x: torch.Tensor, dx: float) -> torch.Tensor:
        pad = self.kernel_size // 2
        padded = F.pad(state_b_m_x, (pad, pad), mode="circular")
        weight = self.kernel.view(1, 1, self.kernel_size).repeat(state_b_m_x.shape[1], 1, 1)
        derivative = F.conv1d(padded, weight, groups=state_b_m_x.shape[1])
        return derivative / (dx**self.order)

    def regularization_loss(self) -> torch.Tensor:
        return torch.mean((self.kernel - self.initial_kernel) ** 2)

    def moment_regularization_loss(self) -> torch.Tensor:
        radius = self.kernel_size // 2
        coords = torch.arange(-radius, radius + 1, device=self.kernel.device, dtype=self.kernel.dtype)
        loss = self.kernel.new_zeros(())
        for moment_order in range(self.kernel_size):
            target = math.factorial(self.order) if moment_order == self.order else 0.0
            moment = torch.sum(self.kernel * (coords**moment_order))
            loss = loss + (moment - target) ** 2
        return loss


class SymNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"symnet_hidden_dim must be positive, got {hidden_dim}")
        if num_layers <= 0:
            raise ValueError(f"symnet_num_layers must be positive, got {num_layers}")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)

        left_layers: list[nn.Linear] = []
        right_layers: list[nn.Linear] = []
        current_dim = input_dim
        for _ in range(num_layers):
            left_layers.append(nn.Linear(current_dim, hidden_dim))
            right_layers.append(nn.Linear(current_dim, hidden_dim))
            current_dim += hidden_dim
        self.left_layers = nn.ModuleList(left_layers)
        self.right_layers = nn.ModuleList(right_layers)
        self.output_layer = nn.Linear(current_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in list(self.left_layers) + list(self.right_layers):
            nn.init.xavier_uniform_(layer.weight, gain=0.1)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.05)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x_bxf: torch.Tensor) -> torch.Tensor:
        features = x_bxf
        for left_layer, right_layer in zip(self.left_layers, self.right_layers):
            left = left_layer(features)
            right = right_layer(features)
            product = left * right
            features = torch.cat([features, product], dim=-1)
        return self.output_layer(features)

    def sparsity_regularization_loss(self) -> torch.Tensor:
        total = torch.zeros((), dtype=self.output_layer.weight.dtype, device=self.output_layer.weight.device)
        for parameter in self.parameters():
            total = total + torch.sum(torch.abs(parameter))
        return total


class LatentPDEModel(nn.Module):
    def __init__(
        self,
        mode_indices: Sequence[int],
        dx: float,
        diff_order: int,
        kernel_size: int,
        poly_order: int,
        symnet_hidden_dim: int | None = None,
        symnet_num_layers: int | None = None,
        external_feature_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.mode_indices = tuple(int(mode) for mode in mode_indices)
        self.dx = float(dx)
        self.diff_order = int(diff_order)
        self.kernel_size = int(kernel_size)
        self.poly_order = int(poly_order)
        self.external_feature_names = tuple(external_feature_names or ())

        self.derivative_layers = nn.ModuleList(
            [SharedDerivative1D(order=order, kernel_size=kernel_size) for order in range(1, diff_order + 1)]
        )

        primary_names = [f"z{mode}" for mode in self.mode_indices]
        for order in range(1, diff_order + 1):
            for mode in self.mode_indices:
                primary_names.append(f"z{mode}_{{{'x' * order}}}")
        primary_names.extend(self.external_feature_names)
        self.primary_names = tuple(primary_names)
        self.input_dim = len(self.primary_names)

        self.symnet_hidden_dim = int(symnet_hidden_dim) if symnet_hidden_dim is not None else max(self.input_dim, 8)
        self.symnet_num_layers = int(symnet_num_layers) if symnet_num_layers is not None else max(self.poly_order, 1)
        self.symnet = SymNet(
            input_dim=self.input_dim,
            output_dim=len(self.mode_indices),
            hidden_dim=self.symnet_hidden_dim,
            num_layers=self.symnet_num_layers,
        )

        self.library_specs = tuple(build_library_specs(self.primary_names, poly_order=self.poly_order))
        self.library_labels = tuple(spec.label for spec in self.library_specs)

    def build_primary_features(
        self,
        state_b_x_m: torch.Tensor,
        external_b_x_f: torch.Tensor | None = None,
    ) -> torch.Tensor:
        state_b_m_x = state_b_x_m.permute(0, 2, 1)
        feature_blocks = [state_b_x_m]
        for layer in self.derivative_layers:
            derivative_b_x_m = layer(state_b_m_x, dx=self.dx).permute(0, 2, 1)
            feature_blocks.append(derivative_b_x_m)
        if self.external_feature_names:
            if external_b_x_f is None:
                raise ValueError("External features were configured for the model, but none were provided.")
            feature_blocks.append(external_b_x_f)
        return torch.cat(feature_blocks, dim=-1)

    def build_library(self, state_b_x_m: torch.Tensor, external_b_x_f: torch.Tensor | None = None) -> torch.Tensor:
        primary = self.build_primary_features(state_b_x_m, external_b_x_f=external_b_x_f)
        return einsum_library(primary, self.library_specs)

    def forward(
        self,
        state_b_x_m: torch.Tensor,
        external_b_x_f: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        primary = self.build_primary_features(state_b_x_m, external_b_x_f=external_b_x_f)
        rhs = self.symnet(primary)
        return rhs, primary

    def kernel_regularization_loss(self) -> torch.Tensor:
        if len(self.derivative_layers) == 0:
            return next(self.parameters()).new_zeros(())
        return sum(layer.regularization_loss() for layer in self.derivative_layers)

    def moment_regularization_loss(self) -> torch.Tensor:
        if len(self.derivative_layers) == 0:
            return next(self.parameters()).new_zeros(())
        return sum(layer.moment_regularization_loss() for layer in self.derivative_layers)

    def sparsity_regularization_loss(self) -> torch.Tensor:
        return self.symnet.sparsity_regularization_loss()


def build_transition_tensors(latent_t_x_m: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = np.asarray(latent_t_x_m[:-1], dtype=np.float32)
    next_state = np.asarray(latent_t_x_m[1:], dtype=np.float32)
    rhs_target = np.asarray((next_state - state) / np.float32(dt), dtype=np.float32)
    return state, next_state, rhs_target


def build_rollout_windows(latent_t_x_m: np.ndarray, rollout_steps: int) -> np.ndarray:
    if rollout_steps < 1:
        raise ValueError(f"rollout_steps must be positive, got {rollout_steps}")
    if latent_t_x_m.shape[0] < rollout_steps + 1:
        raise ValueError(
            f"Need at least {rollout_steps + 1} time samples for rollout_steps={rollout_steps}, got {latent_t_x_m.shape[0]}."
        )
    windows = [latent_t_x_m[start : start + rollout_steps + 1] for start in range(latent_t_x_m.shape[0] - rollout_steps)]
    return np.asarray(windows, dtype=np.float32)


def build_loader(
    sequences: np.ndarray,
    batch_size: int,
    shuffle: bool,
    forcing_sequences: np.ndarray | None = None,
) -> DataLoader:
    tensors = [torch.from_numpy(sequences)]
    if forcing_sequences is not None:
        tensors.append(torch.from_numpy(forcing_sequences))
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def has_diverged(tensor: torch.Tensor, threshold: float | None) -> bool:
    if not torch.isfinite(tensor).all():
        return True
    if threshold is None:
        return False
    return bool(torch.amax(torch.abs(tensor)).detach().cpu().item() > threshold)


def midpoint_feature(current: torch.Tensor | None, nxt: torch.Tensor | None) -> torch.Tensor | None:
    if current is None:
        return None
    if nxt is None:
        return current
    return 0.5 * (current + nxt)


def integrate_latent_step(
    model: LatentPDEModel,
    current_state_b_x_m: torch.Tensor,
    dt: float,
    method: str,
    current_external_b_x_f: torch.Tensor | None = None,
    next_external_b_x_f: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    rhs_current, _primary = model(current_state_b_x_m, external_b_x_f=current_external_b_x_f)
    if method == "euler":
        return current_state_b_x_m + dt * rhs_current, rhs_current

    external_mid = midpoint_feature(current_external_b_x_f, next_external_b_x_f)
    k1 = rhs_current
    k2, _primary = model(current_state_b_x_m + 0.5 * dt * k1, external_b_x_f=external_mid)
    k3, _primary = model(current_state_b_x_m + 0.5 * dt * k2, external_b_x_f=external_mid)
    k4, _primary = model(
        current_state_b_x_m + dt * k3,
        external_b_x_f=next_external_b_x_f if next_external_b_x_f is not None else external_mid,
    )
    return current_state_b_x_m + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), rhs_current


def evaluate_rhs_and_state_mse(
    model: LatentPDEModel,
    state: np.ndarray,
    next_state: np.ndarray,
    rhs_target: np.ndarray,
    dt: float,
    device: torch.device,
    batch_size: int,
    rollout_method: str,
    forcing_current: np.ndarray | None = None,
    forcing_next: np.ndarray | None = None,
) -> tuple[float, float]:
    tensors = [torch.from_numpy(state), torch.from_numpy(next_state), torch.from_numpy(rhs_target)]
    if forcing_current is not None:
        tensors.append(torch.from_numpy(forcing_current))
        tensors.append(torch.from_numpy(forcing_next if forcing_next is not None else forcing_current))
    loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=False, drop_last=False)
    mse_rhs = 0.0
    mse_state = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_state = batch[0].to(device=device, dtype=torch.float32)
            batch_next = batch[1].to(device=device, dtype=torch.float32)
            batch_rhs = batch[2].to(device=device, dtype=torch.float32)
            batch_forcing_current = batch[3].to(device=device, dtype=torch.float32) if len(batch) > 3 else None
            batch_forcing_next = batch[4].to(device=device, dtype=torch.float32) if len(batch) > 4 else None
            rhs_pred, _primary = model(batch_state, external_b_x_f=batch_forcing_current)
            next_pred, _rhs_current = integrate_latent_step(
                model=model,
                current_state_b_x_m=batch_state,
                dt=dt,
                method=rollout_method,
                current_external_b_x_f=batch_forcing_current,
                next_external_b_x_f=batch_forcing_next,
            )
            batch_size_now = batch_state.shape[0]
            mse_rhs += float(torch.mean((rhs_pred - batch_rhs) ** 2).item()) * batch_size_now
            mse_state += float(torch.mean((next_pred - batch_next) ** 2).item()) * batch_size_now
            count += batch_size_now
    return mse_rhs / max(count, 1), mse_state / max(count, 1)


@torch.no_grad()
def rollout_latent_dynamics(
    model: LatentPDEModel,
    initial_state_x_m: np.ndarray,
    t: np.ndarray,
    device: torch.device,
    method: str,
    latent_mean: np.ndarray | None = None,
    latent_std: np.ndarray | None = None,
    electric_field_t_x: np.ndarray | None = None,
    electric_mean: np.ndarray | None = None,
    electric_std: np.ndarray | None = None,
    divergence_threshold: float | None = None,
) -> RolloutResult:
    dt = float(np.mean(np.diff(t)))
    current_phys = np.asarray(initial_state_x_m, dtype=np.float32)
    electric_feature = None
    if electric_field_t_x is not None:
        electric_feature = np.asarray(electric_field_t_x, dtype=np.float32)
        if electric_feature.ndim == 2:
            electric_feature = electric_feature[..., None]
        electric_feature = maybe_normalize_feature(electric_feature, electric_mean, electric_std)
    if latent_mean is not None and latent_std is not None:
        mean = np.asarray(latent_mean, dtype=np.float32).reshape(1, 1, -1)
        std = np.asarray(latent_std, dtype=np.float32).reshape(1, 1, -1)
        current_norm = ((current_phys[None, ...] - mean) / std).astype(np.float32)
        current = torch.from_numpy(current_norm).to(device)
        outputs = np.full((len(t),) + current_phys.shape, np.nan, dtype=np.float32)
        outputs[0] = current_phys.astype(np.float32)
    else:
        current = torch.from_numpy(current_phys).unsqueeze(0).to(device)
        outputs = np.full((len(t),) + current_phys.shape, np.nan, dtype=np.float32)
        outputs[0] = current.squeeze(0).detach().cpu().numpy()
    model.eval()

    for step_index in range(len(t) - 1):
        forcing_current = None
        forcing_next = None
        if electric_feature is not None:
            forcing_current = torch.from_numpy(electric_feature[step_index : step_index + 1]).to(device=device, dtype=torch.float32)
            forcing_next = torch.from_numpy(electric_feature[step_index + 1 : step_index + 2]).to(device=device, dtype=torch.float32)
        current, _rhs_current = integrate_latent_step(
            model=model,
            current_state_b_x_m=current,
            dt=dt,
            method=method,
            current_external_b_x_f=forcing_current,
            next_external_b_x_f=forcing_next,
        )
        if has_diverged(current, divergence_threshold):
            reason = "non-finite state" if not torch.isfinite(current).all() else f"|z| exceeded {divergence_threshold:.3e}"
            return RolloutResult(
                trajectory=np.asarray(outputs, dtype=np.float32),
                success=False,
                failure_step=step_index + 1,
                message=f"PDE-Net rollout diverged at step {step_index + 1} ({reason}).",
            )
        current_np = current.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if latent_mean is not None and latent_std is not None:
            outputs[step_index + 1] = (current_np * std.reshape(1, -1) + mean.reshape(1, -1)).astype(np.float32)
        else:
            outputs[step_index + 1] = current_np
    return RolloutResult(
        trajectory=np.asarray(outputs, dtype=np.float32),
        success=True,
        failure_step=None,
        message="PDE-Net rollout completed successfully.",
    )


def relative_l2(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    denom = np.linalg.norm(ref.reshape(-1))
    if denom == 0.0:
        return float(np.linalg.norm(est.reshape(-1)))
    return float(np.linalg.norm((est - ref).reshape(-1)) / denom)


def design_matrix_from_model(
    model: LatentPDEModel,
    state: np.ndarray,
    device: torch.device,
    batch_size: int,
    external_state: np.ndarray | None = None,
) -> np.ndarray:
    tensors = [torch.from_numpy(state)]
    if external_state is not None:
        tensors.append(torch.from_numpy(external_state))
    loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=False, drop_last=False)
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_state = batch[0].to(device=device, dtype=torch.float32)
            batch_external = batch[1].to(device=device, dtype=torch.float32) if len(batch) > 1 else None
            library = model.build_library(batch_state, external_b_x_f=batch_external)
            chunks.append(library.reshape(-1, library.shape[-1]).cpu().numpy().astype(np.float64))
    return np.concatenate(chunks, axis=0)


def prediction_matrix_from_model(
    model: LatentPDEModel,
    state: np.ndarray,
    device: torch.device,
    batch_size: int,
    external_state: np.ndarray | None = None,
) -> np.ndarray:
    tensors = [torch.from_numpy(state)]
    if external_state is not None:
        tensors.append(torch.from_numpy(external_state))
    loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=False, drop_last=False)
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_state = batch[0].to(device=device, dtype=torch.float32)
            batch_external = batch[1].to(device=device, dtype=torch.float32) if len(batch) > 1 else None
            rhs_pred, _primary = model(batch_state, external_b_x_f=batch_external)
            chunks.append(rhs_pred.reshape(-1, rhs_pred.shape[-1]).cpu().numpy().astype(np.float64))
    return np.concatenate(chunks, axis=0)


def ridge_regression_coefficients(design_matrix: np.ndarray, targets: np.ndarray, ridge_alpha: float) -> np.ndarray:
    x = np.asarray(design_matrix, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("design_matrix and targets must both be 2D.")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Row mismatch between design_matrix {x.shape} and targets {y.shape}.")

    xtx = x.T @ x
    xty = x.T @ y
    if ridge_alpha > 0.0:
        solution = np.linalg.solve(xtx + ridge_alpha * np.eye(x.shape[1]), xty)
    else:
        solution = np.linalg.lstsq(x, y, rcond=None)[0]
    return np.asarray(solution, dtype=np.float64)


def sparsify_coefficients(
    design_matrix: np.ndarray,
    targets: np.ndarray,
    raw_coefficients: np.ndarray,
    threshold: float,
    ridge_alpha: float,
) -> np.ndarray:
    n_terms, n_modes = raw_coefficients.shape
    sparse = np.zeros((n_terms, n_modes), dtype=np.float64)

    for mode_slot in range(n_modes):
        raw = raw_coefficients[:, mode_slot]
        active = np.flatnonzero(np.abs(raw) >= threshold)
        if active.size == 0:
            active = np.array([int(np.argmax(np.abs(raw)))], dtype=np.int64)
        x_active = design_matrix[:, active]
        y_active = targets[:, mode_slot]
        if ridge_alpha > 0.0:
            xtx = x_active.T @ x_active
            xty = x_active.T @ y_active
            coeff_active = np.linalg.solve(xtx + ridge_alpha * np.eye(active.size), xty)
        else:
            coeff_active = np.linalg.lstsq(x_active, y_active, rcond=None)[0]
        sparse[active, mode_slot] = coeff_active
    return sparse


def format_coefficient(value: float) -> str:
    if abs(value) >= 1e3 or (0.0 < abs(value) < 1e-3):
        return f"{value:.3e}"
    return f"{value:.6f}"


def equation_strings(
    mode_indices: Sequence[int],
    coefficients: np.ndarray,
    labels: Sequence[str],
    threshold: float,
) -> list[str]:
    equations: list[str] = []
    for mode_slot, mode in enumerate(mode_indices):
        parts: list[str] = []
        for value, label in zip(coefficients[:, mode_slot], labels):
            if abs(value) < threshold:
                continue
            if label == "1":
                parts.append(format_coefficient(float(value)))
            else:
                parts.append(f"{format_coefficient(float(value))} * {label}")
        rhs = " + ".join(parts) if parts else "0"
        equations.append(f"dz{mode}/dt = {rhs}")
    return equations


def report_text(
    latent_file: Path,
    validation_latent_file: Path | None,
    output_path: Path,
    electric_source: str | None,
    mode_indices: Sequence[int],
    model: LatentPDEModel,
    metrics: Metrics,
    proxy_coefficients: np.ndarray,
    threshold: float,
) -> str:
    lines = [
        "PDE-Net-style latent dynamics discovery with SymNet",
        f"latent_file: {latent_file.resolve()}",
        f"validation_latent_file: {validation_latent_file.resolve() if validation_latent_file is not None else 'None'}",
        f"electric_source: {electric_source if electric_source is not None else 'None'}",
        f"output_file: {output_path.resolve()}",
        f"mode_indices: {list(mode_indices)}",
        f"diff_order: {model.diff_order}",
        f"poly_order_proxy: {model.poly_order}",
        f"kernel_size: {model.kernel_size}",
        f"symnet_hidden_dim: {model.symnet_hidden_dim}",
        f"symnet_num_layers: {model.symnet_num_layers}",
        f"proxy_library_terms: {len(model.library_labels)}",
        f"training_rollout_steps: {metrics.rollout_steps}",
        "loss_space: modewise z-score normalized latent/state/rhs",
        f"train_rhs_mse: {metrics.train_rhs_mse:.6e}",
        f"val_rhs_mse: {metrics.val_rhs_mse:.6e}",
        f"full_rhs_mse: {metrics.full_rhs_mse:.6e}",
        f"train_state_mse: {metrics.train_state_mse:.6e}",
        f"val_state_mse: {metrics.val_state_mse:.6e}",
        f"full_state_mse: {metrics.full_state_mse:.6e}",
        f"train_rollout_relative_l2: {metrics.train_rollout_relative_l2:.6e}",
        f"val_rollout_relative_l2: {metrics.val_rollout_relative_l2:.6e}",
        f"full_rollout_relative_l2: {metrics.full_rollout_relative_l2:.6e}",
        f"proxy_relative_l2: {metrics.proxy_relative_l2:.6e}",
        "",
        "Post-hoc symbolic proxy of the learned SymNet (not the trained model itself):",
    ]
    lines.extend(equation_strings(mode_indices, proxy_coefficients, model.library_labels, threshold=threshold))
    lines.append("")
    lines.append("Learned derivative kernels:")
    for layer in model.derivative_layers:
        kernel_values = " ".join(f"{value:.6e}" for value in layer.kernel.detach().cpu().numpy())
        lines.append(f"order {layer.order}: {kernel_values}")
    return "\n".join(lines) + "\n"


def current_rollout_steps(args: argparse.Namespace, epoch_index: int) -> int:
    max_steps = max(1, int(args.rollout_steps))
    if max_steps == 1 or args.epochs <= 1:
        return 1

    start = float(args.rollout_curriculum_start)
    end = float(args.rollout_curriculum_end)
    if not (0.0 <= start <= 1.0 and 0.0 <= end <= 1.0):
        raise ValueError("rollout curriculum fractions must be in [0, 1].")
    if end < start:
        raise ValueError("rollout_curriculum_end must be >= rollout_curriculum_start.")

    progress = epoch_index / max(args.epochs - 1, 1)
    if progress <= start:
        return 1
    if progress >= end:
        return max_steps
    alpha = (progress - start) / max(end - start, 1e-12)
    scheduled = 1.0 + alpha * float(max_steps - 1)
    return max(1, min(max_steps, int(round(scheduled))))


def compute_training_losses(
    model: LatentPDEModel,
    sequence_b_t_x_m: torch.Tensor,
    forcing_sequence_b_t_x_f: torch.Tensor | None,
    dt: float,
    rollout_steps: int,
    rollout_method: str,
    rollout_weights: torch.Tensor,
    lambda_rollout: float,
    divergence_threshold: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if rollout_steps < 1:
        raise ValueError(f"rollout_steps must be positive, got {rollout_steps}")

    seq = sequence_b_t_x_m[:, : rollout_steps + 1]
    forcing_seq = forcing_sequence_b_t_x_f[:, : rollout_steps + 1] if forcing_sequence_b_t_x_f is not None else None
    curr_true = seq[:, :-1]
    next_true = seq[:, 1:]
    batch_size, n_steps, n_x, n_modes = curr_true.shape

    flat_curr_true = curr_true.reshape(batch_size * n_steps, n_x, n_modes)
    flat_next_true = next_true.reshape(batch_size * n_steps, n_x, n_modes)
    rhs_true = (flat_next_true - flat_curr_true) / dt
    flat_forcing_curr = None
    flat_forcing_next = None
    if forcing_seq is not None:
        forcing_curr = forcing_seq[:, :-1]
        forcing_next = forcing_seq[:, 1:]
        flat_forcing_curr = forcing_curr.reshape(batch_size * n_steps, n_x, forcing_curr.shape[-1])
        flat_forcing_next = forcing_next.reshape(batch_size * n_steps, n_x, forcing_next.shape[-1])
    rhs_pred, _primary = model(flat_curr_true, external_b_x_f=flat_forcing_curr)
    if not torch.isfinite(rhs_pred).all():
        raise DivergenceError("SymNet produced a non-finite rhs during teacher forcing.")

    loss_rhs = torch.mean((rhs_pred - rhs_true) ** 2)
    next_pred_teacher, _rhs_current = integrate_latent_step(
        model=model,
        current_state_b_x_m=flat_curr_true,
        dt=dt,
        method=rollout_method,
        current_external_b_x_f=flat_forcing_curr,
        next_external_b_x_f=flat_forcing_next,
    )
    if has_diverged(next_pred_teacher, divergence_threshold):
        raise DivergenceError("One-step latent prediction diverged before the rollout loss was evaluated.")
    loss_state = torch.mean((next_pred_teacher - flat_next_true) ** 2)

    current = seq[:, 0]
    loss_rollout = current.new_zeros(())
    for step_index in range(rollout_steps):
        forcing_now = forcing_seq[:, step_index] if forcing_seq is not None else None
        forcing_next_step = forcing_seq[:, step_index + 1] if forcing_seq is not None else None
        current, _rhs_step = integrate_latent_step(
            model=model,
            current_state_b_x_m=current,
            dt=dt,
            method=rollout_method,
            current_external_b_x_f=forcing_now,
            next_external_b_x_f=forcing_next_step,
        )
        if has_diverged(current, divergence_threshold):
            raise DivergenceError(f"Multi-step rollout diverged at rollout step {step_index + 1}.")
        step_loss = torch.mean((current - seq[:, step_index + 1]) ** 2)
        if step_index >= 1 and rollout_weights.numel() > 0:
            loss_rollout = loss_rollout + lambda_rollout * rollout_weights[step_index - 1] * step_loss

    return loss_rhs, loss_state, loss_rollout


def fit_model(
    model: LatentPDEModel,
    train_sequences: np.ndarray,
    train_state: np.ndarray,
    train_next_state: np.ndarray,
    train_rhs_target: np.ndarray,
    val_state: np.ndarray,
    val_next_state: np.ndarray,
    val_rhs_target: np.ndarray,
    dt: float,
    rollout_weights: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    train_forcing_sequences: np.ndarray | None,
    train_forcing_state: np.ndarray | None,
    train_forcing_next: np.ndarray | None,
    val_forcing_state: np.ndarray | None,
    val_forcing_next: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_loader = build_loader(
        train_sequences,
        batch_size=args.batch_size,
        shuffle=True,
        forcing_sequences=train_forcing_sequences,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.learning_rate_gamma)
    rollout_weights_tensor = torch.as_tensor(rollout_weights, device=device, dtype=torch.float32)
    history_train: list[float] = []
    history_val: list[float] = []
    history_loss_rhs: list[float] = []
    history_loss_state: list[float] = []
    history_loss_rollout: list[float] = []
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        rollout_steps = current_rollout_steps(args, epoch)
        epoch_loss_rhs = 0.0
        epoch_loss_state = 0.0
        epoch_loss_rollout = 0.0
        batch_count = 0
        for batch in train_loader:
            batch_sequence = batch[0].to(device=device, dtype=torch.float32)
            batch_forcing_sequence = batch[1].to(device=device, dtype=torch.float32) if len(batch) > 1 else None
            optimizer.zero_grad(set_to_none=True)
            try:
                loss_rhs, loss_state, loss_rollout = compute_training_losses(
                    model=model,
                    sequence_b_t_x_m=batch_sequence,
                    forcing_sequence_b_t_x_f=batch_forcing_sequence,
                    dt=dt,
                    rollout_steps=rollout_steps,
                    rollout_method=args.rollout_method,
                    rollout_weights=rollout_weights_tensor[: max(rollout_steps - 1, 0)],
                    lambda_rollout=args.rollout_loss_weight,
                    divergence_threshold=args.divergence_threshold,
                )
            except DivergenceError as exc:
                raise RuntimeError(f"Training diverged at epoch {epoch + 1}, rollout_steps={rollout_steps}: {exc}") from exc
            loss_sparse = model.sparsity_regularization_loss()
            loss_kernel = model.kernel_regularization_loss()
            loss_moment = model.moment_regularization_loss()
            loss = (
                loss_rhs
                + args.state_loss_weight * loss_state
                + loss_rollout
                + args.sparsity_weight * loss_sparse
                + args.kernel_reg_weight * loss_kernel
                + args.moment_reg_weight * loss_moment
            )
            loss.backward()
            if args.gradient_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip_norm)
            optimizer.step()
            epoch_loss_rhs += float(loss_rhs.detach().cpu().item())
            epoch_loss_state += float(loss_state.detach().cpu().item())
            epoch_loss_rollout += float(loss_rollout.detach().cpu().item())
            batch_count += 1

        scheduler.step()
        epoch_loss_rhs /= max(batch_count, 1)
        epoch_loss_state /= max(batch_count, 1)
        epoch_loss_rollout /= max(batch_count, 1)
        train_rhs_mse, _train_state_mse = evaluate_rhs_and_state_mse(
            model=model,
            state=train_state,
            next_state=train_next_state,
            rhs_target=train_rhs_target,
            dt=dt,
            device=device,
            batch_size=args.batch_size,
            rollout_method=args.rollout_method,
            forcing_current=train_forcing_state,
            forcing_next=train_forcing_next,
        )
        val_rhs_mse, _val_state_mse = evaluate_rhs_and_state_mse(
            model=model,
            state=val_state,
            next_state=val_next_state,
            rhs_target=val_rhs_target,
            dt=dt,
            device=device,
            batch_size=args.batch_size,
            rollout_method=args.rollout_method,
            forcing_current=val_forcing_state,
            forcing_next=val_forcing_next,
        )
        if not np.isfinite(train_rhs_mse) or not np.isfinite(val_rhs_mse):
            raise RuntimeError(
                f"Training produced non-finite rhs metrics at epoch {epoch + 1}: "
                f"train_rhs_mse={train_rhs_mse}, val_rhs_mse={val_rhs_mse}"
            )
        history_train.append(train_rhs_mse)
        history_val.append(val_rhs_mse)
        history_loss_rhs.append(epoch_loss_rhs)
        history_loss_state.append(epoch_loss_state)
        history_loss_rollout.append(epoch_loss_rollout)
        if val_rhs_mse <= best_val:
            best_val = val_rhs_mse
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if epoch == 0 or (epoch + 1) % max(1, min(100, args.epochs // 10 or 1)) == 0 or epoch + 1 == args.epochs:
            lr_now = float(optimizer.param_groups[0]["lr"])
            print(
                f"Epoch {epoch + 1:04d}/{args.epochs:04d} "
                f"rollout_steps={rollout_steps} "
                f"loss_rhs={epoch_loss_rhs:.6e} "
                f"loss_state={epoch_loss_state:.6e} "
                f"loss_rollout={epoch_loss_rollout:.6e} "
                f"train_rhs_mse={train_rhs_mse:.6e} "
                f"val_rhs_mse={val_rhs_mse:.6e} "
                f"lr={lr_now:.3e}"
            )

    model.load_state_dict(best_state)
    return (
        np.asarray(history_train, dtype=np.float32),
        np.asarray(history_val, dtype=np.float32),
        np.asarray(history_loss_rhs, dtype=np.float32),
        np.asarray(history_loss_state, dtype=np.float32),
        np.asarray(history_loss_rollout, dtype=np.float32),
    )


def save_payload(
    output_path: Path,
    latent_file: Path,
    validation_latent_file: Path | None,
    electric_source: str | None,
    latent_mean: np.ndarray,
    latent_std: np.ndarray,
    electric_mean: np.ndarray | None,
    electric_std: np.ndarray | None,
    mode_indices: Sequence[int],
    fit_t: np.ndarray,
    fit_x: np.ndarray,
    model: LatentPDEModel,
    proxy_coefficients: np.ndarray,
    proxy_coefficients_raw: np.ndarray,
    metrics: Metrics,
    train_history: np.ndarray,
    val_history: np.ndarray,
    loss_rhs_history: np.ndarray,
    loss_state_history: np.ndarray,
    loss_rollout_history: np.ndarray,
    rollout_weights: np.ndarray,
    fit_latent: np.ndarray,
    rollout_prediction: np.ndarray,
    divergence_threshold: float,
) -> None:
    payload: dict[str, np.ndarray] = {
        "model_family": np.asarray("pdenet", dtype=str),
        "pdenet_representation": np.asarray("symnet", dtype=str),
        "latent_file": np.asarray(str(latent_file.resolve()), dtype=str),
        "validation_latent_file": np.asarray(
            str(validation_latent_file.resolve()) if validation_latent_file is not None else "",
            dtype=str,
        ),
        "electric_file": np.asarray("" if electric_source is None else electric_source, dtype=str),
        "mode_indices": np.asarray(mode_indices, dtype=np.int32),
        "latent_mean_modeled": np.asarray(latent_mean, dtype=np.float32),
        "latent_std_modeled": np.asarray(latent_std, dtype=np.float32),
        "t": np.asarray(fit_t, dtype=np.float32),
        "x": np.asarray(fit_x, dtype=np.float32),
        "dt": np.asarray(float(np.mean(np.diff(fit_t))), dtype=np.float32),
        "dx": np.asarray(float(np.mean(np.diff(fit_x))), dtype=np.float32),
        "coefficients": np.asarray(proxy_coefficients, dtype=np.float32),
        "coefficients_raw": np.asarray(proxy_coefficients_raw, dtype=np.float32),
        "library_labels": np.asarray(model.library_labels, dtype=str),
        "rhs_description": np.asarray(model.library_labels, dtype=str),
        "primary_labels": np.asarray(model.primary_names, dtype=str),
        "poly_order": np.asarray(model.poly_order, dtype=np.int32),
        "diff_order": np.asarray(model.diff_order, dtype=np.int32),
        "kernel_size": np.asarray(model.kernel_size, dtype=np.int32),
        "symnet_hidden_dim": np.asarray(model.symnet_hidden_dim, dtype=np.int32),
        "symnet_num_layers": np.asarray(model.symnet_num_layers, dtype=np.int32),
        "training_rollout_steps": np.asarray(metrics.rollout_steps, dtype=np.int32),
        "rollout_loss_weights": np.asarray(rollout_weights, dtype=np.float32),
        "divergence_threshold": np.asarray(divergence_threshold, dtype=np.float32),
        "space_diff": np.asarray("learned-conv", dtype=str),
        "system_mode": np.asarray("coupled", dtype=str),
        "used_electric_field": np.asarray(electric_mean is not None),
        "train_history_rhs_mse": np.asarray(train_history, dtype=np.float32),
        "val_history_rhs_mse": np.asarray(val_history, dtype=np.float32),
        "train_history_loss_rhs": np.asarray(loss_rhs_history, dtype=np.float32),
        "train_history_loss_state": np.asarray(loss_state_history, dtype=np.float32),
        "train_history_loss_rollout": np.asarray(loss_rollout_history, dtype=np.float32),
        "fit_latent": np.asarray(fit_latent, dtype=np.float32),
        "latent_rollout_prediction": np.asarray(rollout_prediction, dtype=np.float32),
        "train_rhs_mse": np.asarray(metrics.train_rhs_mse, dtype=np.float32),
        "val_rhs_mse": np.asarray(metrics.val_rhs_mse, dtype=np.float32),
        "full_rhs_mse": np.asarray(metrics.full_rhs_mse, dtype=np.float32),
        "train_state_mse": np.asarray(metrics.train_state_mse, dtype=np.float32),
        "val_state_mse": np.asarray(metrics.val_state_mse, dtype=np.float32),
        "full_state_mse": np.asarray(metrics.full_state_mse, dtype=np.float32),
        "train_rollout_relative_l2": np.asarray(metrics.train_rollout_relative_l2, dtype=np.float32),
        "val_rollout_relative_l2": np.asarray(metrics.val_rollout_relative_l2, dtype=np.float32),
        "full_rollout_relative_l2": np.asarray(metrics.full_rollout_relative_l2, dtype=np.float32),
        "proxy_relative_l2": np.asarray(metrics.proxy_relative_l2, dtype=np.float32),
    }
    if electric_mean is not None and electric_std is not None:
        payload["electric_mean_modeled"] = np.asarray(electric_mean, dtype=np.float32)
        payload["electric_std_modeled"] = np.asarray(electric_std, dtype=np.float32)
    for layer in model.derivative_layers:
        payload[f"kernel_order_{layer.order}"] = np.asarray(layer.kernel.detach().cpu().numpy(), dtype=np.float32)
        payload[f"kernel_order_{layer.order}_initial"] = np.asarray(
            layer.initial_kernel.detach().cpu().numpy(),
            dtype=np.float32,
        )
    payload.update(extract_state_dict_numpy(model))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    latent, t, x, latent_meta = load_latent_data(args.latent_file)
    dt = check_uniform_spacing(t, "t")
    dx = check_uniform_spacing(x, "x")

    mode_indices = resolve_modes(args.modes, latent.shape[2])
    fit_length = adjusted_fit_length(num_times=latent.shape[0], fit_fraction=args.fit_fraction)
    fit_latent = np.asarray(latent[:fit_length, :, mode_indices], dtype=np.float32)
    fit_t = np.asarray(t[:fit_length], dtype=np.float32)
    fit_x = np.asarray(x, dtype=np.float32)
    electric_field, electric_source = resolve_electric_data(
        latent_file=args.latent_file,
        meta=latent_meta,
        expected_t=t,
        expected_x=x,
        requested_electric_file=args.electric_file,
        disable_electric_field=args.no_electric_field,
    )
    fit_electric = None
    fit_electric_norm = None
    electric_mean_modeled = None
    electric_std_modeled = None
    if electric_field is not None:
        fit_electric = np.asarray(electric_field[:fit_length], dtype=np.float32)[..., None]
        electric_mean_modeled, electric_std_modeled = compute_modewise_normalization(fit_electric)
        fit_electric_norm = apply_modewise_normalization(fit_electric, electric_mean_modeled, electric_std_modeled)
    latent_mean_modeled, latent_std_modeled = compute_modewise_normalization(fit_latent)
    fit_latent_norm = apply_modewise_normalization(fit_latent, latent_mean_modeled, latent_std_modeled)

    state_all, next_state_all, rhs_target_all = build_transition_tensors(fit_latent_norm, dt=dt)
    validate_train_fraction(args.train_fraction)
    max_rollout_steps = min(max(1, int(args.rollout_steps)), fit_latent.shape[0] - 1)
    rollout_weights = resolve_rollout_weights(max_rollout_steps, args.rollout_loss_weights)
    train_sequences = build_rollout_windows(fit_latent_norm, rollout_steps=max_rollout_steps)
    train_forcing_sequences = (
        build_rollout_windows(fit_electric_norm, rollout_steps=max_rollout_steps) if fit_electric_norm is not None else None
    )
    train_state = state_all
    train_next_state = next_state_all
    train_rhs_target = rhs_target_all
    val_state = state_all
    val_next_state = next_state_all
    val_rhs_target = rhs_target_all
    val_fit_latent = fit_latent
    val_fit_t = fit_t
    val_fit_electric = fit_electric
    train_forcing_state = fit_electric_norm[:-1] if fit_electric_norm is not None else None
    train_forcing_next = fit_electric_norm[1:] if fit_electric_norm is not None else None
    val_forcing_state = train_forcing_state
    val_forcing_next = train_forcing_next

    validation_latent_file = args.validation_latent_file.resolve() if args.validation_latent_file is not None else None
    if validation_latent_file is not None:
        validation_latent, validation_t, validation_x, validation_meta = load_latent_data(validation_latent_file)
        validation_dt = check_uniform_spacing(validation_t, "validation t")
        validation_dx = check_uniform_spacing(validation_x, "validation x")
        if not np.isclose(validation_dt, dt, rtol=1e-6, atol=1e-8):
            raise ValueError(
                f"Validation latent dt mismatch: training dt={dt}, validation dt={validation_dt}"
            )
        if not np.isclose(validation_dx, dx, rtol=1e-6, atol=1e-8):
            raise ValueError(
                f"Validation latent dx mismatch: training dx={dx}, validation dx={validation_dx}"
            )
        if validation_latent.shape[2] <= max(mode_indices):
            raise ValueError(
                f"Validation latent has Nz={validation_latent.shape[2]}, cannot select modes {mode_indices}."
            )
        val_fit_length = adjusted_fit_length(num_times=validation_latent.shape[0], fit_fraction=args.fit_fraction)
        val_fit_latent = np.asarray(validation_latent[:val_fit_length, :, mode_indices], dtype=np.float32)
        val_fit_t = np.asarray(validation_t[:val_fit_length], dtype=np.float32)
        val_fit_latent_norm = apply_modewise_normalization(val_fit_latent, latent_mean_modeled, latent_std_modeled)
        val_state, val_next_state, val_rhs_target = build_transition_tensors(val_fit_latent_norm, dt=dt)
        if fit_electric_norm is not None:
            validation_electric_field, _validation_electric_source = resolve_electric_data(
                latent_file=validation_latent_file,
                meta=validation_meta,
                expected_t=validation_t,
                expected_x=validation_x,
                requested_electric_file=None,
                disable_electric_field=False,
            )
            if validation_electric_field is None:
                raise ValueError("Validation latent was provided, but no matching electric field could be resolved.")
            val_fit_electric = np.asarray(validation_electric_field[:val_fit_length], dtype=np.float32)[..., None]
            val_fit_electric_norm = apply_modewise_normalization(
                val_fit_electric,
                electric_mean_modeled,
                electric_std_modeled,
            )
            val_forcing_state = val_fit_electric_norm[:-1]
            val_forcing_next = val_fit_electric_norm[1:]

    model = LatentPDEModel(
        mode_indices=mode_indices,
        dx=dx,
        diff_order=args.diff_order,
        kernel_size=args.kernel_size,
        poly_order=args.poly_order,
        symnet_hidden_dim=args.symnet_hidden_dim,
        symnet_num_layers=args.symnet_num_layers,
        external_feature_names=("E",) if fit_electric_norm is not None else (),
    ).to(device)

    train_history, val_history, loss_rhs_history, loss_state_history, loss_rollout_history = fit_model(
        model=model,
        train_sequences=train_sequences,
        train_state=train_state,
        train_next_state=train_next_state,
        train_rhs_target=train_rhs_target,
        val_state=val_state,
        val_next_state=val_next_state,
        val_rhs_target=val_rhs_target,
        dt=dt,
        rollout_weights=rollout_weights,
        args=args,
        device=device,
        train_forcing_sequences=train_forcing_sequences,
        train_forcing_state=train_forcing_state,
        train_forcing_next=train_forcing_next,
        val_forcing_state=val_forcing_state,
        val_forcing_next=val_forcing_next,
    )

    train_rhs_mse, train_state_mse = evaluate_rhs_and_state_mse(
        model=model,
        state=train_state,
        next_state=train_next_state,
        rhs_target=train_rhs_target,
        dt=dt,
        device=device,
        batch_size=args.batch_size,
        rollout_method=args.rollout_method,
        forcing_current=train_forcing_state,
        forcing_next=train_forcing_next,
    )
    val_rhs_mse, val_state_mse = evaluate_rhs_and_state_mse(
        model=model,
        state=val_state,
        next_state=val_next_state,
        rhs_target=val_rhs_target,
        dt=dt,
        device=device,
        batch_size=args.batch_size,
        rollout_method=args.rollout_method,
        forcing_current=val_forcing_state,
        forcing_next=val_forcing_next,
    )
    full_rhs_mse, full_state_mse = evaluate_rhs_and_state_mse(
        model=model,
        state=state_all,
        next_state=next_state_all,
        rhs_target=rhs_target_all,
        dt=dt,
        device=device,
        batch_size=args.batch_size,
        rollout_method=args.rollout_method,
        forcing_current=train_forcing_state,
        forcing_next=train_forcing_next,
    )

    rollout_result = rollout_latent_dynamics(
        model=model,
        initial_state_x_m=fit_latent[0],
        t=fit_t,
        device=device,
        method=args.rollout_method,
        latent_mean=latent_mean_modeled,
        latent_std=latent_std_modeled,
        electric_field_t_x=fit_electric[:, :, 0] if fit_electric is not None else None,
        electric_mean=electric_mean_modeled,
        electric_std=electric_std_modeled,
        divergence_threshold=args.divergence_threshold,
    )
    if not rollout_result.success:
        raise RuntimeError(rollout_result.message)
    rollout_prediction = rollout_result.trajectory
    val_rollout_prediction = rollout_prediction
    if validation_latent_file is not None:
        val_rollout_result = rollout_latent_dynamics(
            model=model,
            initial_state_x_m=val_fit_latent[0],
            t=val_fit_t,
            device=device,
            method=args.rollout_method,
            latent_mean=latent_mean_modeled,
            latent_std=latent_std_modeled,
            electric_field_t_x=val_fit_electric[:, :, 0] if val_fit_electric is not None else None,
            electric_mean=electric_mean_modeled,
            electric_std=electric_std_modeled,
            divergence_threshold=args.divergence_threshold,
        )
        if not val_rollout_result.success:
            raise RuntimeError(val_rollout_result.message)
        val_rollout_prediction = val_rollout_result.trajectory

    design_matrix = design_matrix_from_model(
        model=model,
        state=state_all,
        device=device,
        batch_size=args.batch_size,
        external_state=train_forcing_state,
    )
    rhs_prediction_matrix = prediction_matrix_from_model(
        model=model,
        state=state_all,
        device=device,
        batch_size=args.batch_size,
        external_state=train_forcing_state,
    )
    proxy_coefficients_raw = ridge_regression_coefficients(
        design_matrix=design_matrix,
        targets=rhs_prediction_matrix,
        ridge_alpha=args.ridge_alpha,
    )
    proxy_coefficients = sparsify_coefficients(
        design_matrix=design_matrix,
        targets=rhs_prediction_matrix,
        raw_coefficients=proxy_coefficients_raw,
        threshold=args.coef_threshold,
        ridge_alpha=args.ridge_alpha,
    )
    proxy_prediction_matrix = design_matrix @ proxy_coefficients

    metrics = Metrics(
        train_rhs_mse=train_rhs_mse,
        val_rhs_mse=val_rhs_mse,
        full_rhs_mse=full_rhs_mse,
        train_state_mse=train_state_mse,
        val_state_mse=val_state_mse,
        full_state_mse=full_state_mse,
        train_rollout_relative_l2=relative_l2(fit_latent, rollout_prediction),
        val_rollout_relative_l2=relative_l2(val_fit_latent, val_rollout_prediction),
        full_rollout_relative_l2=relative_l2(fit_latent, rollout_prediction),
        proxy_relative_l2=relative_l2(rhs_prediction_matrix, proxy_prediction_matrix),
        rollout_steps=max_rollout_steps,
    )

    output_path = args.output.resolve() if args.output is not None else infer_output_path(args.latent_file.resolve())
    report_path = args.report.resolve() if args.report is not None else infer_report_path(output_path)

    save_payload(
        output_path=output_path,
        latent_file=args.latent_file,
        validation_latent_file=validation_latent_file,
        electric_source=electric_source,
        latent_mean=latent_mean_modeled,
        latent_std=latent_std_modeled,
        electric_mean=electric_mean_modeled,
        electric_std=electric_std_modeled,
        mode_indices=mode_indices,
        fit_t=fit_t,
        fit_x=fit_x,
        model=model,
        proxy_coefficients=proxy_coefficients,
        proxy_coefficients_raw=proxy_coefficients_raw,
        metrics=metrics,
        train_history=train_history,
        val_history=val_history,
        loss_rhs_history=loss_rhs_history,
        loss_state_history=loss_state_history,
        loss_rollout_history=loss_rollout_history,
        rollout_weights=rollout_weights,
        fit_latent=fit_latent,
        rollout_prediction=rollout_prediction,
        divergence_threshold=args.divergence_threshold,
    )

    report = report_text(
        latent_file=args.latent_file,
        validation_latent_file=validation_latent_file,
        output_path=output_path,
        electric_source=electric_source,
        mode_indices=mode_indices,
        model=model,
        metrics=metrics,
        proxy_coefficients=proxy_coefficients,
        threshold=args.coef_threshold,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(f"Wrote PDE-Net model to: {output_path}")
    print(f"Wrote report to: {report_path}")


if __name__ == "__main__":
    main()
