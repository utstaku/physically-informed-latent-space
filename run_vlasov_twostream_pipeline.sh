#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/uts22/miniconda3/envs/lasdi/bin/python}"
TRUTH_CASE_DIR=""
OUTPUT_DIR=""
VALIDATION_CASE_DIR=""
LATENT_DIM=8
AE_EPOCHS=50
AE_TRAIN_FRACTION=0.9
AE_F0_EPSILON=1e-3
AE_DENSITY_WEIGHT=0.0
AE_ELECTRIC_WEIGHT=0.0
AE_VLASOV_RESIDUAL_WEIGHT=0.0
DEEPYMOD_ITERATIONS=5000
DEEPYMOD_TRAIN_SPLIT=0.8
DEEPYMOD_LIBRARY_TYPE="full"
DEEPYMOD_SPARSE_ESTIMATOR="stlsq"
DEEPYMOD_THRESHOLD="0.1"
DEEPYMOD_ALPHA="0.05"
DYNAMICS_MODEL="deepymod"
PDEFIND_TIME_DIFF="FD"
PDEFIND_SPACE_DIFF="FD"
PDEFIND_LAM="1e-2"
PDEFIND_D_TOL="0.5"
PDEFIND_PRINT_BEST_TOL=0
PDENET_FIT_FRACTION="1.0"
PDENET_TRAIN_FRACTION="1.0"
PDENET_EPOCHS=1500
PDENET_BATCH_SIZE=32
PDENET_LEARNING_RATE="2e-3"
PDENET_LEARNING_RATE_GAMMA="0.995"
PDENET_SPARSITY_WEIGHT="1e-5"
PDENET_KERNEL_REG_WEIGHT="1e-3"
PDENET_COEF_THRESHOLD="5e-4"
PDENET_RIDGE_ALPHA="1e-6"
PDENET_ROLLOUT_METHOD="rk4"
PDENET_ROLLOUT_LOSS_WEIGHTS=""
LINEAR_RIDGE_ALPHA="1e-6"
LINEAR_FIT_FRACTION="1.0"
LINEAR_INCLUDE_CONSTANT=0
LINEAR_SPACE_DIFF="Fourier"
LINEAR_TIME_DIFF="FD"
LINEAR_SG_WINDOW_X=9
LINEAR_SG_WINDOW_T=9
LINEAR_SG_POLY_X=3
LINEAR_SG_POLY_T=3
SYSTEM_MODE="independent"
DEVICE="auto"
SOLVER="RK45"
NO_ANIMATION=0
PDE_D=1
PDE_P=1
INCLUDE_RECIPROCAL=0
NO_ELECTRIC_FIELD=0
PLOT_LATENT_HEATMAP=1
LATENT_HEATMAP_MODES=""

usage() {
  cat <<EOF
Usage: $(basename "$0") --truth-case-dir PATH [options]

Run the Vlasov two-stream pipeline on an existing truth case:
  Conv_velocity_AE.py -> latent_dynamics_{deepymod|linear|pdefind|pdenet}.py -> simulate_latent_pde_rk45.py

Required:
  --truth-case-dir PATH     Case directory containing distribution_full.npz.

Options:
  --output-dir PATH         Directory for AE/PDE/evaluation outputs. Default: --truth-case-dir
  --validation-case-dir P   Optional neighboring case used for AE/PDE-Net validation. Default: auto-select neighbor in the same grid
  --python PATH             Python executable to use.
  --latent-dim INT          Latent dimension for Conv_velocity_AE.py. Default: ${LATENT_DIM}
  --ae-epochs INT           AE training epochs. Default: ${AE_EPOCHS}
  --ae-train-fraction X     AE train fraction. Default: ${AE_TRAIN_FRACTION}
  --ae-f0-epsilon X         AE epsilon in (f-f0)/(f0+epsilon). Default: ${AE_F0_EPSILON}
  --ae-density-weight X     AE density-loss weight. Default: ${AE_DENSITY_WEIGHT}
  --ae-electric-weight X    AE electric-field loss weight. Default: ${AE_ELECTRIC_WEIGHT}
  --ae-vlasov-residual-weight X
                            AE Vlasov residual loss weight. Default: ${AE_VLASOV_RESIDUAL_WEIGHT}
  --deepymod-iterations INT DeepMoD max-iterations. Default: ${DEEPYMOD_ITERATIONS}
  --deepymod-train-split X  DeepMoD train/test split fraction. Default: ${DEEPYMOD_TRAIN_SPLIT}
  --deepymod-library TYPE   DeepMoD library type: full or state-polynomial. Default: ${DEEPYMOD_LIBRARY_TYPE}
  --sparse-estimator NAME   DeepMoD sparse estimator: stlsq, threshold, or pdefind. Default: ${DEEPYMOD_SPARSE_ESTIMATOR}
  --threshold X             DeepMoD threshold for stlsq/threshold. Default: ${DEEPYMOD_THRESHOLD}
  --alpha X                 DeepMoD ridge alpha for stlsq. Default: ${DEEPYMOD_ALPHA}
  --dynamics-model NAME     Latent dynamics model: deepymod, linear, pdefind, or pdenet. Default: ${DYNAMICS_MODEL}
  --pdefind-time-diff NAME  Time derivative for latent_dynamics.py. Default: ${PDEFIND_TIME_DIFF}
  --pdefind-space-diff NAME Space derivative for latent_dynamics.py. Default: ${PDEFIND_SPACE_DIFF}
  --pdefind-lam X           Ridge parameter for latent_dynamics.py STRidge. Default: ${PDEFIND_LAM}
  --pdefind-d-tol X         Tolerance increment for latent_dynamics.py STRidge. Default: ${PDEFIND_D_TOL}
  --pdefind-print-best-tol  Print best tolerance during latent_dynamics.py STRidge search.
  --pdenet-fit-fraction X   Fit fraction for latent_dynamics_pdenet.py. Default: ${PDENET_FIT_FRACTION}
  --pdenet-train-fraction X Compatibility option passed to latent_dynamics_pdenet.py. Default: ${PDENET_TRAIN_FRACTION}
  --pdenet-epochs INT       Epochs for latent_dynamics_pdenet.py. Default: ${PDENET_EPOCHS}
  --pdenet-batch-size INT   Batch size for latent_dynamics_pdenet.py. Default: ${PDENET_BATCH_SIZE}
  --pdenet-learning-rate X  Learning rate for latent_dynamics_pdenet.py. Default: ${PDENET_LEARNING_RATE}
  --pdenet-lr-gamma X       LR decay gamma for latent_dynamics_pdenet.py. Default: ${PDENET_LEARNING_RATE_GAMMA}
  --pdenet-sparsity-weight X
                            L1 penalty for latent_dynamics_pdenet.py. Default: ${PDENET_SPARSITY_WEIGHT}
  --pdenet-kernel-reg-weight X
                            Kernel regularization for latent_dynamics_pdenet.py. Default: ${PDENET_KERNEL_REG_WEIGHT}
  --pdenet-coef-threshold X Coefficient threshold for latent_dynamics_pdenet.py. Default: ${PDENET_COEF_THRESHOLD}
  --pdenet-ridge-alpha X    Ridge refit alpha for latent_dynamics_pdenet.py. Default: ${PDENET_RIDGE_ALPHA}
  --pdenet-rollout-method M Rollout method for latent_dynamics_pdenet.py: euler or rk4. Default: ${PDENET_ROLLOUT_METHOD}
  --pdenet-rollout-loss-weights S
                            Comma-separated rollout weights for latent_dynamics_pdenet.py.
                            Applied to rollout steps 2..K; default is uniform weighting.
  --linear-ridge-alpha X    Ridge parameter for latent_dynamics_linear.py. Default: ${LINEAR_RIDGE_ALPHA}
  --linear-fit-fraction X   Fit fraction for latent_dynamics_linear.py. Default: ${LINEAR_FIT_FRACTION}
  --linear-include-constant Include a constant forcing term in the linear transport fit.
  --linear-space-diff NAME  Spatial derivative for latent_dynamics_linear.py: Fourier, FD, or SG. Default: ${LINEAR_SPACE_DIFF}
  --linear-time-diff NAME   Time derivative for latent_dynamics_linear.py: FD or SG. Default: ${LINEAR_TIME_DIFF}
  --linear-sg-window-x INT  Savitzky-Golay window in x for linear dynamics. Default: ${LINEAR_SG_WINDOW_X}
  --linear-sg-window-t INT  Savitzky-Golay window in t for linear dynamics. Default: ${LINEAR_SG_WINDOW_T}
  --linear-sg-poly-x INT    Savitzky-Golay poly degree in x for linear dynamics. Default: ${LINEAR_SG_POLY_X}
  --linear-sg-poly-t INT    Savitzky-Golay poly degree in t for linear dynamics. Default: ${LINEAR_SG_POLY_T}
  --system MODE             PDE system mode: independent or coupled. Default: ${SYSTEM_MODE}
  --D INT                   Maximum spatial derivative order for latent_dynamics_deepymod.py or latent_dynamics.py. Default: ${PDE_D}
  --P INT                   Maximum polynomial power for latent_dynamics_deepymod.py or latent_dynamics.py. Default: ${PDE_P}
  --include-reciprocal      Enable reciprocal features in latent_dynamics.py. Ignored by DeepMoD and linear.
  --no-electric-field       Disable E(t,x) in latent_dynamics.py. Ignored by DeepMoD and linear.
  --no-latent-heatmap       Skip latent heatmap generation after AE training.
  --latent-heatmap-modes    Space-separated mode indices passed to plot_latent_heatmap.py.
  --device DEVICE           Torch device for AE and RK45 decode: auto, cpu, or cuda. Default: ${DEVICE}
  --solver NAME             Integrator for simulate_latent_pde_rk45.py. Default: ${SOLVER}
  --no-animation            Skip GIF generation in the RK45 evaluation step.
  -h, --help                Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --truth-case-dir|--case-dir)
      TRUTH_CASE_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --validation-case-dir)
      VALIDATION_CASE_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --latent-dim)
      LATENT_DIM="$2"
      shift 2
      ;;
    --epochs|--ae-epochs)
      AE_EPOCHS="$2"
      shift 2
      ;;
    --ae-train-fraction)
      AE_TRAIN_FRACTION="$2"
      shift 2
      ;;
    --ae-f0-epsilon)
      AE_F0_EPSILON="$2"
      shift 2
      ;;
    --ae-density-weight)
      AE_DENSITY_WEIGHT="$2"
      shift 2
      ;;
    --ae-electric-weight)
      AE_ELECTRIC_WEIGHT="$2"
      shift 2
      ;;
    --ae-vlasov-residual-weight)
      AE_VLASOV_RESIDUAL_WEIGHT="$2"
      shift 2
      ;;
    --deepymod-iterations)
      DEEPYMOD_ITERATIONS="$2"
      shift 2
      ;;
    --deepymod-train-split)
      DEEPYMOD_TRAIN_SPLIT="$2"
      shift 2
      ;;
    --deepymod-library)
      DEEPYMOD_LIBRARY_TYPE="$2"
      shift 2
      ;;
    --sparse-estimator)
      DEEPYMOD_SPARSE_ESTIMATOR="$2"
      shift 2
      ;;
    --threshold)
      DEEPYMOD_THRESHOLD="$2"
      shift 2
      ;;
    --alpha)
      DEEPYMOD_ALPHA="$2"
      shift 2
      ;;
    --dynamics-model)
      DYNAMICS_MODEL="$2"
      shift 2
      ;;
    --pdefind-time-diff)
      PDEFIND_TIME_DIFF="$2"
      shift 2
      ;;
    --pdefind-space-diff)
      PDEFIND_SPACE_DIFF="$2"
      shift 2
      ;;
    --pdefind-lam)
      PDEFIND_LAM="$2"
      shift 2
      ;;
    --pdefind-d-tol)
      PDEFIND_D_TOL="$2"
      shift 2
      ;;
    --pdefind-print-best-tol)
      PDEFIND_PRINT_BEST_TOL=1
      shift
      ;;
    --pdenet-fit-fraction)
      PDENET_FIT_FRACTION="$2"
      shift 2
      ;;
    --pdenet-train-fraction)
      PDENET_TRAIN_FRACTION="$2"
      shift 2
      ;;
    --pdenet-epochs)
      PDENET_EPOCHS="$2"
      shift 2
      ;;
    --pdenet-batch-size)
      PDENET_BATCH_SIZE="$2"
      shift 2
      ;;
    --pdenet-learning-rate)
      PDENET_LEARNING_RATE="$2"
      shift 2
      ;;
    --pdenet-lr-gamma)
      PDENET_LEARNING_RATE_GAMMA="$2"
      shift 2
      ;;
    --pdenet-sparsity-weight)
      PDENET_SPARSITY_WEIGHT="$2"
      shift 2
      ;;
    --pdenet-kernel-reg-weight)
      PDENET_KERNEL_REG_WEIGHT="$2"
      shift 2
      ;;
    --pdenet-coef-threshold)
      PDENET_COEF_THRESHOLD="$2"
      shift 2
      ;;
    --pdenet-ridge-alpha)
      PDENET_RIDGE_ALPHA="$2"
      shift 2
      ;;
    --pdenet-rollout-method)
      PDENET_ROLLOUT_METHOD="$2"
      shift 2
      ;;
    --pdenet-rollout-loss-weights)
      PDENET_ROLLOUT_LOSS_WEIGHTS="$2"
      shift 2
      ;;
    --linear-ridge-alpha)
      LINEAR_RIDGE_ALPHA="$2"
      shift 2
      ;;
    --linear-fit-fraction)
      LINEAR_FIT_FRACTION="$2"
      shift 2
      ;;
    --linear-include-constant)
      LINEAR_INCLUDE_CONSTANT=1
      shift
      ;;
    --linear-space-diff)
      LINEAR_SPACE_DIFF="$2"
      shift 2
      ;;
    --linear-time-diff)
      LINEAR_TIME_DIFF="$2"
      shift 2
      ;;
    --linear-sg-window-x)
      LINEAR_SG_WINDOW_X="$2"
      shift 2
      ;;
    --linear-sg-window-t)
      LINEAR_SG_WINDOW_T="$2"
      shift 2
      ;;
    --linear-sg-poly-x)
      LINEAR_SG_POLY_X="$2"
      shift 2
      ;;
    --linear-sg-poly-t)
      LINEAR_SG_POLY_T="$2"
      shift 2
      ;;
    --system)
      SYSTEM_MODE="$2"
      shift 2
      ;;
    --D)
      PDE_D="$2"
      shift 2
      ;;
    --P)
      PDE_P="$2"
      shift 2
      ;;
    --include-reciprocal)
      INCLUDE_RECIPROCAL=1
      shift
      ;;
    --no-electric-field)
      NO_ELECTRIC_FIELD=1
      shift
      ;;
    --no-latent-heatmap)
      PLOT_LATENT_HEATMAP=0
      shift
      ;;
    --latent-heatmap-modes)
      shift
      while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
        if [[ -n "${LATENT_HEATMAP_MODES}" ]]; then
          LATENT_HEATMAP_MODES+=" "
        fi
        LATENT_HEATMAP_MODES+="$1"
        shift
      done
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --solver)
      SOLVER="$2"
      shift 2
      ;;
    --no-animation)
      NO_ANIMATION=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${TRUTH_CASE_DIR}" ]]; then
  echo "--truth-case-dir is required." >&2
  usage >&2
  exit 1
fi

if [[ "${SYSTEM_MODE}" != "independent" && "${SYSTEM_MODE}" != "coupled" ]]; then
  echo "--system must be 'independent' or 'coupled', got '${SYSTEM_MODE}'." >&2
  exit 1
fi

if [[ "${DYNAMICS_MODEL}" != "deepymod" && "${DYNAMICS_MODEL}" != "linear" && "${DYNAMICS_MODEL}" != "pdefind" && "${DYNAMICS_MODEL}" != "pdenet" ]]; then
  echo "--dynamics-model must be 'deepymod', 'linear', 'pdefind', or 'pdenet', got '${DYNAMICS_MODEL}'." >&2
  exit 1
fi

if [[ "${DEVICE}" != "auto" && "${DEVICE}" != "cpu" && "${DEVICE}" != "cuda" ]]; then
  echo "--device must be 'auto', 'cpu', or 'cuda', got '${DEVICE}'." >&2
  exit 1
fi

if [[ "${DEEPYMOD_LIBRARY_TYPE}" != "full" && "${DEEPYMOD_LIBRARY_TYPE}" != "state-polynomial" ]]; then
  echo "--deepymod-library must be 'full' or 'state-polynomial', got '${DEEPYMOD_LIBRARY_TYPE}'." >&2
  exit 1
fi

if [[ "${DEEPYMOD_SPARSE_ESTIMATOR}" != "stlsq" && "${DEEPYMOD_SPARSE_ESTIMATOR}" != "threshold" && "${DEEPYMOD_SPARSE_ESTIMATOR}" != "pdefind" ]]; then
  echo "--sparse-estimator must be 'stlsq', 'threshold', or 'pdefind', got '${DEEPYMOD_SPARSE_ESTIMATOR}'." >&2
  exit 1
fi

if [[ "${LINEAR_SPACE_DIFF}" != "Fourier" && "${LINEAR_SPACE_DIFF}" != "FD" && "${LINEAR_SPACE_DIFF}" != "SG" ]]; then
  echo "--linear-space-diff must be 'Fourier', 'FD', or 'SG', got '${LINEAR_SPACE_DIFF}'." >&2
  exit 1
fi

if [[ "${LINEAR_TIME_DIFF}" != "FD" && "${LINEAR_TIME_DIFF}" != "SG" ]]; then
  echo "--linear-time-diff must be 'FD' or 'SG', got '${LINEAR_TIME_DIFF}'." >&2
  exit 1
fi

if [[ "${PDEFIND_TIME_DIFF}" != "poly" && "${PDEFIND_TIME_DIFF}" != "FD" && "${PDEFIND_TIME_DIFF}" != "FDconv" && "${PDEFIND_TIME_DIFF}" != "Tik" && "${PDEFIND_TIME_DIFF}" != "SG" ]]; then
  echo "--pdefind-time-diff must be 'poly', 'FD', 'FDconv', 'Tik', or 'SG', got '${PDEFIND_TIME_DIFF}'." >&2
  exit 1
fi

if [[ "${PDEFIND_SPACE_DIFF}" != "poly" && "${PDEFIND_SPACE_DIFF}" != "FD" && "${PDEFIND_SPACE_DIFF}" != "CD" && "${PDEFIND_SPACE_DIFF}" != "FDconv" && "${PDEFIND_SPACE_DIFF}" != "Tik" && "${PDEFIND_SPACE_DIFF}" != "Fourier" && "${PDEFIND_SPACE_DIFF}" != "SG" ]]; then
  echo "--pdefind-space-diff must be 'poly', 'FD', 'CD', 'FDconv', 'Tik', 'Fourier', or 'SG', got '${PDEFIND_SPACE_DIFF}'." >&2
  exit 1
fi

if [[ "${DYNAMICS_MODEL}" == "linear" && "${SYSTEM_MODE}" != "coupled" ]]; then
  echo "--dynamics-model linear currently requires --system coupled." >&2
  exit 1
fi

if [[ "${DYNAMICS_MODEL}" == "pdenet" && "${SYSTEM_MODE}" != "coupled" ]]; then
  echo "--dynamics-model pdenet currently requires --system coupled." >&2
  exit 1
fi

TRUTH_CASE_DIR="$(realpath -m "${TRUTH_CASE_DIR}")"
if [[ ! -f "${TRUTH_CASE_DIR}/distribution_full.npz" ]]; then
  echo "distribution_full.npz not found in truth case directory: ${TRUTH_CASE_DIR}" >&2
  exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${TRUTH_CASE_DIR}"
fi
OUTPUT_DIR="$(realpath -m "${OUTPUT_DIR}")"

LATENT_FILE="${OUTPUT_DIR}/conv_velocity_autoencoder_results.npz"
VALIDATION_LATENT_FILE="${OUTPUT_DIR}/conv_velocity_autoencoder_results_validation.npz"
if [[ "${DYNAMICS_MODEL}" == "linear" ]]; then
  PDE_FILE="${OUTPUT_DIR}/conv_velocity_autoencoder_results_linear_transport.npz"
elif [[ "${DYNAMICS_MODEL}" == "pdefind" ]]; then
  PDE_FILE="${OUTPUT_DIR}/conv_velocity_autoencoder_results_pde_find.npz"
elif [[ "${DYNAMICS_MODEL}" == "pdenet" ]]; then
  PDE_FILE="${OUTPUT_DIR}/conv_velocity_autoencoder_results_pdenet.npz"
elif [[ "${SYSTEM_MODE}" == "coupled" ]]; then
  PDE_FILE="${OUTPUT_DIR}/conv_velocity_autoencoder_results_deepymod_coupled.npz"
else
  PDE_FILE="${OUTPUT_DIR}/conv_velocity_autoencoder_results_deepymod.npz"
fi

resolve_neighbor_case_dir() {
  local truth_case_dir="$1"
  "${PYTHON_BIN}" - "$truth_case_dir" <<'PY'
import math
import re
import sys
from pathlib import Path

truth = Path(sys.argv[1]).resolve()
parent = truth.parent
pattern = re.compile(r"^T_([0-9]+\.[0-9]+)_k_([0-9]+\.[0-9]+)$")
match = pattern.fullmatch(truth.name)
if match is None:
    raise SystemExit("")
truth_t = float(match.group(1))
truth_k = float(match.group(2))

candidates = []
for path in sorted(parent.iterdir()):
    if not path.is_dir() or path == truth:
        continue
    match = pattern.fullmatch(path.name)
    if match is None:
        continue
    t_val = float(match.group(1))
    k_val = float(match.group(2))
    same_k = math.isclose(k_val, truth_k, rel_tol=0.0, abs_tol=1e-12)
    same_t = math.isclose(t_val, truth_t, rel_tol=0.0, abs_tol=1e-12)
    rank = (
        0 if same_k else 1 if same_t else 2,
        abs(t_val - truth_t) if same_k else abs(k_val - truth_k) if same_t else math.hypot(t_val - truth_t, k_val - truth_k),
        0 if t_val > truth_t else 1,
        0 if k_val > truth_k else 1,
        path.name,
    )
    candidates.append((rank, path))

if not candidates:
    raise SystemExit("")

print(min(candidates, key=lambda item: item[0])[1].resolve())
PY
}

if [[ -n "${VALIDATION_CASE_DIR}" ]]; then
  VALIDATION_CASE_DIR="$(realpath -m "${VALIDATION_CASE_DIR}")"
else
  VALIDATION_CASE_DIR="$(resolve_neighbor_case_dir "${TRUTH_CASE_DIR}")"
fi

if [[ -n "${VALIDATION_CASE_DIR}" ]]; then
  if [[ ! -f "${VALIDATION_CASE_DIR}/distribution_full.npz" ]]; then
    echo "validation distribution_full.npz not found: ${VALIDATION_CASE_DIR}" >&2
    exit 1
  fi
fi

resolve_sim_device() {
  if [[ "${DEVICE}" == "cpu" || "${DEVICE}" == "cuda" ]]; then
    printf '%s\n' "${DEVICE}"
    return
  fi

  "${PYTHON_BIN}" - <<'PY'
import sys
try:
    import torch
except Exception:
    print("cpu")
    sys.exit(0)
print("cuda" if torch.cuda.is_available() else "cpu")
PY
}

SIM_DEVICE="$(resolve_sim_device)"

run_cmd() {
  echo
  printf '+'
  printf ' %q' "$@"
  echo
  "$@"
}

mkdir -p "${OUTPUT_DIR}"

AE_ARGS=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/Conv_velocity_AE.py"
  --case-dir "${TRUTH_CASE_DIR}"
  --latent-dim "${LATENT_DIM}"
  --epochs "${AE_EPOCHS}"
  --train-fraction "${AE_TRAIN_FRACTION}"
  --f0-epsilon "${AE_F0_EPSILON}"
  --density-weight "${AE_DENSITY_WEIGHT}"
  --electric-weight "${AE_ELECTRIC_WEIGHT}"
  --vlasov-residual-weight "${AE_VLASOV_RESIDUAL_WEIGHT}"
  --device "${DEVICE}"
  --output "${LATENT_FILE}"
)

if [[ -n "${VALIDATION_CASE_DIR}" ]]; then
  AE_ARGS+=(--validation-case-dir "${VALIDATION_CASE_DIR}" --validation-output "${VALIDATION_LATENT_FILE}")
fi

run_cmd "${AE_ARGS[@]}"

if [[ "${PLOT_LATENT_HEATMAP}" -eq 1 ]]; then
  LATENT_HEATMAP_ARGS=(
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/plot_latent_heatmap.py"
    --latent-file "${LATENT_FILE}"
    --output "${OUTPUT_DIR}/conv_velocity_autoencoder_results_heatmap.png"
  )
  if [[ -n "${LATENT_HEATMAP_MODES}" ]]; then
    read -r -a LATENT_HEATMAP_MODE_ARRAY <<< "${LATENT_HEATMAP_MODES}"
    LATENT_HEATMAP_ARGS+=(--modes "${LATENT_HEATMAP_MODE_ARRAY[@]}")
  fi
  run_cmd "${LATENT_HEATMAP_ARGS[@]}"
fi

if [[ "${DYNAMICS_MODEL}" == "deepymod" ]]; then
  DEEPYMOD_ARGS=(
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/latent_dynamics_deepymod.py"
    --latent-file "${LATENT_FILE}"
    --system "${SYSTEM_MODE}"
    --diff-order "${PDE_D}"
    --poly-order "${PDE_P}"
    --library-type "${DEEPYMOD_LIBRARY_TYPE}"
    --sparse-estimator "${DEEPYMOD_SPARSE_ESTIMATOR}"
    --threshold "${DEEPYMOD_THRESHOLD}"
    --alpha "${DEEPYMOD_ALPHA}"
    --max-iterations "${DEEPYMOD_ITERATIONS}"
    --train-split "${DEEPYMOD_TRAIN_SPLIT}"
    --device "${DEVICE}"
    --output "${PDE_FILE}"
  )

  if [[ "${NO_ELECTRIC_FIELD}" -eq 1 ]]; then
    echo "warning: --no-electric-field is ignored by latent_dynamics_deepymod.py" >&2
  fi

  if [[ "${INCLUDE_RECIPROCAL}" -eq 1 ]]; then
    echo "warning: --include-reciprocal is ignored by latent_dynamics_deepymod.py" >&2
  fi

  run_cmd "${DEEPYMOD_ARGS[@]}"
elif [[ "${DYNAMICS_MODEL}" == "pdefind" ]]; then
  PDEFIND_ARGS=(
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/latent_dynamics.py"
    --latent-file "${LATENT_FILE}"
    --system "${SYSTEM_MODE}"
    --D "${PDE_D}"
    --P "${PDE_P}"
    --time-diff "${PDEFIND_TIME_DIFF}"
    --space-diff "${PDEFIND_SPACE_DIFF}"
    --lam "${PDEFIND_LAM}"
    --d-tol "${PDEFIND_D_TOL}"
    --output "${PDE_FILE}"
  )

  if [[ "${NO_ELECTRIC_FIELD}" -eq 1 ]]; then
    PDEFIND_ARGS+=(--no-electric-field)
  fi

  if [[ "${INCLUDE_RECIPROCAL}" -eq 1 ]]; then
    PDEFIND_ARGS+=(--include-reciprocal)
  fi

  if [[ "${PDEFIND_PRINT_BEST_TOL}" -eq 1 ]]; then
    PDEFIND_ARGS+=(--print-best-tol)
  fi

  run_cmd "${PDEFIND_ARGS[@]}"
elif [[ "${DYNAMICS_MODEL}" == "pdenet" ]]; then
  PDENET_ARGS=(
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/latent_dynamics_pdenet.py"
    --latent-file "${LATENT_FILE}"
    --fit-fraction "${PDENET_FIT_FRACTION}"
    --train-fraction "${PDENET_TRAIN_FRACTION}"
    --poly-order "${PDE_P}"
    --diff-order "${PDE_D}"
    --batch-size "${PDENET_BATCH_SIZE}"
    --epochs "${PDENET_EPOCHS}"
    --learning-rate "${PDENET_LEARNING_RATE}"
    --learning-rate-gamma "${PDENET_LEARNING_RATE_GAMMA}"
    --sparsity-weight "${PDENET_SPARSITY_WEIGHT}"
    --kernel-reg-weight "${PDENET_KERNEL_REG_WEIGHT}"
    --coef-threshold "${PDENET_COEF_THRESHOLD}"
    --ridge-alpha "${PDENET_RIDGE_ALPHA}"
    --rollout-method "${PDENET_ROLLOUT_METHOD}"
    --device "${DEVICE}"
    --output "${PDE_FILE}"
  )
  if [[ -n "${PDENET_ROLLOUT_LOSS_WEIGHTS}" ]]; then
    PDENET_ARGS+=(--rollout-loss-weights "${PDENET_ROLLOUT_LOSS_WEIGHTS}")
  fi

  if [[ -n "${VALIDATION_CASE_DIR}" ]]; then
    PDENET_ARGS+=(--validation-latent-file "${VALIDATION_LATENT_FILE}")
  fi

  if [[ "${NO_ELECTRIC_FIELD}" -eq 1 ]]; then
    PDENET_ARGS+=(--no-electric-field)
  fi

  if [[ "${INCLUDE_RECIPROCAL}" -eq 1 ]]; then
    echo "warning: --include-reciprocal is ignored by latent_dynamics_pdenet.py" >&2
  fi

  run_cmd "${PDENET_ARGS[@]}"
else
  LINEAR_ARGS=(
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/latent_dynamics_linear.py"
    --latent-file "${LATENT_FILE}"
    --ridge-alpha "${LINEAR_RIDGE_ALPHA}"
    --fit-fraction "${LINEAR_FIT_FRACTION}"
    --space-diff "${LINEAR_SPACE_DIFF}"
    --time-diff "${LINEAR_TIME_DIFF}"
    --sg-window-x "${LINEAR_SG_WINDOW_X}"
    --sg-window-t "${LINEAR_SG_WINDOW_T}"
    --sg-poly-x "${LINEAR_SG_POLY_X}"
    --sg-poly-t "${LINEAR_SG_POLY_T}"
    --output "${PDE_FILE}"
  )

  if [[ "${LINEAR_INCLUDE_CONSTANT}" -eq 1 ]]; then
    LINEAR_ARGS+=(--include-constant)
  fi

  if [[ "${NO_ELECTRIC_FIELD}" -eq 1 ]]; then
    echo "warning: --no-electric-field is ignored by latent_dynamics_linear.py" >&2
  fi

  if [[ "${INCLUDE_RECIPROCAL}" -eq 1 ]]; then
    echo "warning: --include-reciprocal is ignored by latent_dynamics_linear.py" >&2
  fi

  run_cmd "${LINEAR_ARGS[@]}"
fi

SIM_ARGS=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/simulate_latent_pde_rk45.py"
  --pde-file "${PDE_FILE}"
  --latent-file "${LATENT_FILE}"
  --case-dir "${TRUTH_CASE_DIR}"
  --solver "${SOLVER}"
  --device "${SIM_DEVICE}"
)

if [[ "${NO_ANIMATION}" -eq 1 ]]; then
  SIM_ARGS+=(--no-animation)
fi

run_cmd "${SIM_ARGS[@]}"

echo
echo "Pipeline finished."
echo "truth_case_dir=${TRUTH_CASE_DIR}"
echo "output_dir=${OUTPUT_DIR}"
echo "latent_file=${LATENT_FILE}"
echo "pde_file=${PDE_FILE}"
echo "dynamics_model=${DYNAMICS_MODEL}"
echo "torch_device=${DEVICE}"
echo "simulate_device=${SIM_DEVICE}"
