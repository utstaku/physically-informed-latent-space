#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/uts22/miniconda3/envs/lasdi/bin/python}"
GRID_DIR="vlasov_twostream_param_grid"
OUTPUT_DIR="results/twostream_grid_fno_pipeline"
DEVICE="auto"

NUM_T_SAMPLES=5
NUM_K_SAMPLES=5
LATENT_DIM=8
AE_EPOCHS=50
AE_BATCH_SIZE=1024
AE_LEARNING_RATE="1e-3"
AE_HIDDEN_DIM=64
AE_CONV_CHANNELS="8,16,32"
AE_KERNEL_SIZE=5
AE_PADDING_MODE="zeros"
AE_F0_EPSILON="1e-3"
AE_DENSITY_WEIGHT="0.0"
AE_ELECTRIC_WEIGHT="0.0"
AE_VLASOV_RESIDUAL_WEIGHT="0.0"

FNO_EPOCHS=200
FNO_BATCH_SIZE=32
FNO_DECODE_BATCH_SIZE=2048
FNO_LEARNING_RATE="1e-3"
FNO_WEIGHT_DECAY="1e-6"
FNO_ROLLOUT_STEPS=5
FNO_ROLLOUT_LOSS_WEIGHTS=""
FNO_LAMBDA_ONE_STEP="1.0"
FNO_LAMBDA_ROLLOUT="0.1"
FNO_HIDDEN_CHANNELS=64
FNO_N_LAYERS=4
FNO_N_MODES=16
FNO_START_INDEX=0
FNO_NUM_STEPS=""
FNO_T_END=""
FNO_FILL_UNMODELED="truth"
FNO_CLIP_MIN=""
FNO_TITLE="FNO Error Map on Vlasov Full Grid (convae, max relative error %)"
FNO_MODES=()

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Run the full grid pipeline:
  Conv_velocity_AE_grid.py -> latent_dynamics_FNO_grid.py

Options:
  --grid-dir PATH              Full Vlasov parameter grid. Default: ${GRID_DIR}
  --output-dir PATH            Output directory for AE/FNO artifacts. Default: ${OUTPUT_DIR}
  --python PATH                Python executable to use. Default: ${PYTHON_BIN}
  --device DEVICE              Torch device: auto, cpu, cuda. Default: ${DEVICE}

AE options:
  --num-t-samples INT          Number of sampled T values. Default: ${NUM_T_SAMPLES}
  --num-k-samples INT          Number of sampled k values. Default: ${NUM_K_SAMPLES}
  --latent-dim INT             AE latent dimension. Default: ${LATENT_DIM}
  --ae-epochs INT              AE epochs. Default: ${AE_EPOCHS}
  --ae-batch-size INT          AE batch size. Default: ${AE_BATCH_SIZE}
  --ae-learning-rate X         AE learning rate. Default: ${AE_LEARNING_RATE}
  --ae-hidden-dim INT          AE hidden dimension. Default: ${AE_HIDDEN_DIM}
  --ae-conv-channels SPEC      AE conv channels. Default: ${AE_CONV_CHANNELS}
  --ae-kernel-size INT         AE kernel size. Default: ${AE_KERNEL_SIZE}
  --ae-padding-mode MODE       AE padding mode. Default: ${AE_PADDING_MODE}
  --ae-f0-epsilon X            AE epsilon in (f-f0)/(f0+epsilon). Default: ${AE_F0_EPSILON}
  --ae-density-weight X        AE density-loss weight. Default: ${AE_DENSITY_WEIGHT}
  --ae-electric-weight X       AE electric-field-loss weight. Default: ${AE_ELECTRIC_WEIGHT}
  --ae-vlasov-residual-weight X
                               AE Vlasov residual-loss weight. Default: ${AE_VLASOV_RESIDUAL_WEIGHT}

FNO options:
  --fno-epochs INT             FNO epochs. Default: ${FNO_EPOCHS}
  --fno-batch-size INT         FNO batch size. Default: ${FNO_BATCH_SIZE}
  --fno-decode-batch-size INT  Decode batch size for rollout evaluation. Default: ${FNO_DECODE_BATCH_SIZE}
  --fno-learning-rate X        FNO learning rate. Default: ${FNO_LEARNING_RATE}
  --fno-weight-decay X         FNO weight decay. Default: ${FNO_WEIGHT_DECAY}
  --rollout-steps INT          FNO training rollout steps. Default: ${FNO_ROLLOUT_STEPS}
  --rollout-loss-weights SPEC  Comma-separated weights for steps 2..K.
  --lambda-one-step X          FNO one-step loss weight. Default: ${FNO_LAMBDA_ONE_STEP}
  --lambda-rollout X           FNO rollout loss weight. Default: ${FNO_LAMBDA_ROLLOUT}
  --hidden-channels INT        FNO hidden channels. Default: ${FNO_HIDDEN_CHANNELS}
  --n-layers INT               FNO Fourier layers. Default: ${FNO_N_LAYERS}
  --n-modes INT                FNO spectral modes. Default: ${FNO_N_MODES}
  --start-index INT            Full-grid rollout start index. Default: ${FNO_START_INDEX}
  --num-steps INT              Full-grid rollout steps. Default: full horizon
  --t-end X                    Optional final evaluation time.
  --fill-unmodeled MODE        truth, initial, or zero. Default: ${FNO_FILL_UNMODELED}
  --clip-min X                 Optional lower clip for decoded f.
  --title TEXT                 Error-map title.
  --modes I [J ...]            Optional latent modes modeled by FNO. Default: all

  -h, --help                   Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --grid-dir)
      GRID_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --num-t-samples)
      NUM_T_SAMPLES="$2"
      shift 2
      ;;
    --num-k-samples)
      NUM_K_SAMPLES="$2"
      shift 2
      ;;
    --latent-dim)
      LATENT_DIM="$2"
      shift 2
      ;;
    --ae-epochs)
      AE_EPOCHS="$2"
      shift 2
      ;;
    --ae-batch-size)
      AE_BATCH_SIZE="$2"
      shift 2
      ;;
    --ae-learning-rate)
      AE_LEARNING_RATE="$2"
      shift 2
      ;;
    --ae-hidden-dim)
      AE_HIDDEN_DIM="$2"
      shift 2
      ;;
    --ae-conv-channels)
      AE_CONV_CHANNELS="$2"
      shift 2
      ;;
    --ae-kernel-size)
      AE_KERNEL_SIZE="$2"
      shift 2
      ;;
    --ae-padding-mode)
      AE_PADDING_MODE="$2"
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
    --fno-epochs)
      FNO_EPOCHS="$2"
      shift 2
      ;;
    --fno-batch-size)
      FNO_BATCH_SIZE="$2"
      shift 2
      ;;
    --fno-decode-batch-size)
      FNO_DECODE_BATCH_SIZE="$2"
      shift 2
      ;;
    --fno-learning-rate)
      FNO_LEARNING_RATE="$2"
      shift 2
      ;;
    --fno-weight-decay)
      FNO_WEIGHT_DECAY="$2"
      shift 2
      ;;
    --rollout-steps)
      FNO_ROLLOUT_STEPS="$2"
      shift 2
      ;;
    --rollout-loss-weights)
      FNO_ROLLOUT_LOSS_WEIGHTS="$2"
      shift 2
      ;;
    --lambda-one-step)
      FNO_LAMBDA_ONE_STEP="$2"
      shift 2
      ;;
    --lambda-rollout)
      FNO_LAMBDA_ROLLOUT="$2"
      shift 2
      ;;
    --hidden-channels)
      FNO_HIDDEN_CHANNELS="$2"
      shift 2
      ;;
    --n-layers)
      FNO_N_LAYERS="$2"
      shift 2
      ;;
    --n-modes)
      FNO_N_MODES="$2"
      shift 2
      ;;
    --start-index)
      FNO_START_INDEX="$2"
      shift 2
      ;;
    --num-steps)
      FNO_NUM_STEPS="$2"
      shift 2
      ;;
    --t-end)
      FNO_T_END="$2"
      shift 2
      ;;
    --fill-unmodeled)
      FNO_FILL_UNMODELED="$2"
      shift 2
      ;;
    --clip-min)
      FNO_CLIP_MIN="$2"
      shift 2
      ;;
    --title)
      FNO_TITLE="$2"
      shift 2
      ;;
    --modes)
      shift
      FNO_MODES=()
      while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
        FNO_MODES+=("$1")
        shift
      done
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

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ "${DEVICE}" != "auto" && "${DEVICE}" != "cpu" && "${DEVICE}" != "cuda" ]]; then
  echo "--device must be 'auto', 'cpu', or 'cuda', got '${DEVICE}'." >&2
  exit 1
fi

if [[ "${FNO_FILL_UNMODELED}" != "truth" && "${FNO_FILL_UNMODELED}" != "initial" && "${FNO_FILL_UNMODELED}" != "zero" ]]; then
  echo "--fill-unmodeled must be 'truth', 'initial', or 'zero', got '${FNO_FILL_UNMODELED}'." >&2
  exit 1
fi

GRID_DIR="$(realpath -m "${GRID_DIR}")"
OUTPUT_DIR="$(realpath -m "${OUTPUT_DIR}")"

if [[ ! -d "${GRID_DIR}" ]]; then
  echo "Grid directory not found: ${GRID_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

AE_OUTPUT="${OUTPUT_DIR}/vlasov_twostream_5x5_conv_velocity_autoencoder_results.npz"
AE_VALIDATION_OUTPUT="${OUTPUT_DIR}/vlasov_twostream_5x5_conv_velocity_autoencoder_results_validation.npz"
AE_ENCODER_OUTPUT="${OUTPUT_DIR}/vlasov_twostream_5x5_conv_velocity_autoencoder_results_encoder.pt"

FNO_OUTPUT="${OUTPUT_DIR}/vlasov_twostream_5x5_conv_velocity_autoencoder_results_fno_grid_dynamics.npz"
FNO_CHECKPOINT="${OUTPUT_DIR}/vlasov_twostream_5x5_conv_velocity_autoencoder_results_fno_grid_dynamics.pt"
FNO_REPORT="${OUTPUT_DIR}/vlasov_twostream_5x5_conv_velocity_autoencoder_results_fno_grid_dynamics.txt"
FNO_EVAL_OUTPUT="${OUTPUT_DIR}/vlasov_twostream_5x5_conv_velocity_autoencoder_results_fno_grid_dynamics_full_grid_eval.npz"
FNO_ERROR_MAP_OUTPUT="${OUTPUT_DIR}/vlasov_twostream_5x5_conv_velocity_autoencoder_results_fno_grid_dynamics_error_map.png"

run_cmd() {
  echo
  printf '+'
  printf ' %q' "$@"
  echo
  "$@"
}

AE_ARGS=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/Conv_velocity_AE_grid.py"
  --grid-dir "${GRID_DIR}"
  --num-t-samples "${NUM_T_SAMPLES}"
  --num-k-samples "${NUM_K_SAMPLES}"
  --latent-dim "${LATENT_DIM}"
  --hidden-dim "${AE_HIDDEN_DIM}"
  --conv-channels "${AE_CONV_CHANNELS}"
  --kernel-size "${AE_KERNEL_SIZE}"
  --padding-mode "${AE_PADDING_MODE}"
  --f0-epsilon "${AE_F0_EPSILON}"
  --epochs "${AE_EPOCHS}"
  --batch-size "${AE_BATCH_SIZE}"
  --learning-rate "${AE_LEARNING_RATE}"
  --density-weight "${AE_DENSITY_WEIGHT}"
  --electric-weight "${AE_ELECTRIC_WEIGHT}"
  --vlasov-residual-weight "${AE_VLASOV_RESIDUAL_WEIGHT}"
  --device "${DEVICE}"
  --output "${AE_OUTPUT}"
  --validation-output "${AE_VALIDATION_OUTPUT}"
  --encoder-output "${AE_ENCODER_OUTPUT}"
)

FNO_ARGS=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/latent_dynamics_FNO_grid.py"
  --latent-file "${AE_OUTPUT}"
  --grid-dir "${GRID_DIR}"
  --rollout-steps "${FNO_ROLLOUT_STEPS}"
  --lambda-one-step "${FNO_LAMBDA_ONE_STEP}"
  --lambda-rollout "${FNO_LAMBDA_ROLLOUT}"
  --epochs "${FNO_EPOCHS}"
  --batch-size "${FNO_BATCH_SIZE}"
  --decode-batch-size "${FNO_DECODE_BATCH_SIZE}"
  --learning-rate "${FNO_LEARNING_RATE}"
  --weight-decay "${FNO_WEIGHT_DECAY}"
  --hidden-channels "${FNO_HIDDEN_CHANNELS}"
  --n-layers "${FNO_N_LAYERS}"
  --n-modes "${FNO_N_MODES}"
  --start-index "${FNO_START_INDEX}"
  --fill-unmodeled "${FNO_FILL_UNMODELED}"
  --device "${DEVICE}"
  --output "${FNO_OUTPUT}"
  --checkpoint "${FNO_CHECKPOINT}"
  --report "${FNO_REPORT}"
  --eval-output "${FNO_EVAL_OUTPUT}"
  --error-map-output "${FNO_ERROR_MAP_OUTPUT}"
  --title "${FNO_TITLE}"
)

if [[ -n "${FNO_ROLLOUT_LOSS_WEIGHTS}" ]]; then
  FNO_ARGS+=(--rollout-loss-weights "${FNO_ROLLOUT_LOSS_WEIGHTS}")
fi

if [[ -n "${FNO_NUM_STEPS}" ]]; then
  FNO_ARGS+=(--num-steps "${FNO_NUM_STEPS}")
fi

if [[ -n "${FNO_T_END}" ]]; then
  FNO_ARGS+=(--t-end "${FNO_T_END}")
fi

if [[ -n "${FNO_CLIP_MIN}" ]]; then
  FNO_ARGS+=(--clip-min "${FNO_CLIP_MIN}")
fi

if [[ ${#FNO_MODES[@]} -gt 0 ]]; then
  FNO_ARGS+=(--modes "${FNO_MODES[@]}")
fi

run_cmd "${AE_ARGS[@]}"
run_cmd "${FNO_ARGS[@]}"

echo
echo "Pipeline completed."
echo "AE latent file   : ${AE_OUTPUT}"
echo "FNO metrics file : ${FNO_OUTPUT}"
echo "FNO checkpoint   : ${FNO_CHECKPOINT}"
echo "Full-grid eval   : ${FNO_EVAL_OUTPUT}"
echo "Error map image  : ${FNO_ERROR_MAP_OUTPUT}"
