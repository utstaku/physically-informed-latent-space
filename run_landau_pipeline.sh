#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/uts22/miniconda3/envs/lasdi/bin/python}"
CASE_DIR=""
LATENT_DIM=8
AE_EPOCHS=50
AE_TRAIN_FRACTION=0.9
AE_DENSITY_WEIGHT=0.0
DEEPYMOD_ITERATIONS=5000
DEEPYMOD_TRAIN_SPLIT=0.8
SAVE_STRIDE=20
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
Usage: $(basename "$0") --case-dir PATH [options]

Run the Landau pipeline:
  vlasov_landau.py -> Conv_velocity_AE.py -> latent_dynamics_deepymod.py -> simulate_latent_pde_rk45.py

Required:
  --case-dir PATH           Output case directory for generated data and model files.

Options:
  --python PATH             Python executable to use.
  --latent-dim INT          Latent dimension for Conv_velocity_AE.py. Default: ${LATENT_DIM}
  --ae-epochs INT           AE training epochs. Default: ${AE_EPOCHS}
  --ae-train-fraction X     AE train fraction. Default: ${AE_TRAIN_FRACTION}
  --ae-density-weight X     AE density-loss weight. Default: ${AE_DENSITY_WEIGHT}
  --deepymod-iterations INT DeepMoD max-iterations. Default: ${DEEPYMOD_ITERATIONS}
  --deepymod-train-split X  DeepMoD train/test split fraction. Default: ${DEEPYMOD_TRAIN_SPLIT}
  --save-stride INT         Snapshot stride used by vlasov_landau.py. Default: ${SAVE_STRIDE}
  --system MODE             PDE system mode: independent or coupled. Default: ${SYSTEM_MODE}
  --D INT                   Maximum spatial derivative order for latent_dynamics_deepymod.py. Default: ${PDE_D}
  --P INT                   Maximum polynomial power for latent_dynamics_deepymod.py. Default: ${PDE_P}
  --include-reciprocal      Ignored in the DeepMoD pipeline.
  --no-electric-field       Ignored in the DeepMoD pipeline.
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
    --case-dir)
      CASE_DIR="$2"
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
    --ae-density-weight)
      AE_DENSITY_WEIGHT="$2"
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
    --save-stride)
      SAVE_STRIDE="$2"
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

if [[ -z "${CASE_DIR}" ]]; then
  echo "--case-dir is required." >&2
  usage >&2
  exit 1
fi

if [[ "${SYSTEM_MODE}" != "independent" && "${SYSTEM_MODE}" != "coupled" ]]; then
  echo "--system must be 'independent' or 'coupled', got '${SYSTEM_MODE}'." >&2
  exit 1
fi

if [[ "${DEVICE}" != "auto" && "${DEVICE}" != "cpu" && "${DEVICE}" != "cuda" ]]; then
  echo "--device must be 'auto', 'cpu', or 'cuda', got '${DEVICE}'." >&2
  exit 1
fi

CASE_DIR="$(realpath -m "${CASE_DIR}")"
LATENT_FILE="${CASE_DIR}/conv_velocity_autoencoder_results.npz"
if [[ "${SYSTEM_MODE}" == "coupled" ]]; then
  PDE_FILE="${CASE_DIR}/conv_velocity_autoencoder_results_deepymod_coupled.npz"
else
  PDE_FILE="${CASE_DIR}/conv_velocity_autoencoder_results_deepymod.npz"
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

mkdir -p "${CASE_DIR}"

run_cmd \
  "${PYTHON_BIN}" \
  "${SCRIPT_DIR}/vlasov_landau.py" \
  "${CASE_DIR}" \
  --save-stride "${SAVE_STRIDE}"

run_cmd \
  "${PYTHON_BIN}" \
  "${SCRIPT_DIR}/Conv_velocity_AE.py" \
  --case-dir "${CASE_DIR}" \
  --latent-dim "${LATENT_DIM}" \
  --epochs "${AE_EPOCHS}" \
  --train-fraction "${AE_TRAIN_FRACTION}" \
  --density-weight "${AE_DENSITY_WEIGHT}" \
  --device "${DEVICE}" \
  --output "${LATENT_FILE}"

if [[ "${PLOT_LATENT_HEATMAP}" -eq 1 ]]; then
  LATENT_HEATMAP_ARGS=(
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/plot_latent_heatmap.py"
    --latent-file "${LATENT_FILE}"
    --output "${CASE_DIR}/conv_velocity_autoencoder_results_heatmap.png"
  )
  if [[ -n "${LATENT_HEATMAP_MODES}" ]]; then
    read -r -a LATENT_HEATMAP_MODE_ARRAY <<< "${LATENT_HEATMAP_MODES}"
    LATENT_HEATMAP_ARGS+=(--modes "${LATENT_HEATMAP_MODE_ARRAY[@]}")
  fi
  run_cmd "${LATENT_HEATMAP_ARGS[@]}"
fi

DEEPYMOD_ARGS=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/latent_dynamics_deepymod.py"
  --latent-file "${LATENT_FILE}"
  --system "${SYSTEM_MODE}"
  --diff-order "${PDE_D}"
  --poly-order "${PDE_P}"
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

SIM_ARGS=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/simulate_latent_pde_rk45.py"
  --pde-file "${PDE_FILE}"
  --latent-file "${LATENT_FILE}"
  --case-dir "${CASE_DIR}"
  --solver "${SOLVER}"
  --device "${SIM_DEVICE}"
)

if [[ "${NO_ANIMATION}" -eq 1 ]]; then
  SIM_ARGS+=(--no-animation)
fi

run_cmd "${SIM_ARGS[@]}"

echo
echo "Pipeline finished."
echo "case_dir=${CASE_DIR}"
echo "latent_file=${LATENT_FILE}"
echo "pde_file=${PDE_FILE}"
echo "torch_device=${DEVICE}"
echo "simulate_device=${SIM_DEVICE}"
