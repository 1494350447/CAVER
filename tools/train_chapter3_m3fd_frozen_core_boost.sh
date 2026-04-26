#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${MODE:-train-all}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/chapter3_m3fd_frozen_core_boost.py}"
MODEL_NAME="${MODEL_NAME:-SAM2PriorAlignmentYOLODetector}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output_chapter3_m3fd_frozen_core}"
BOOSTED_OUTPUT_DIR="${BOOSTED_OUTPUT_DIR:-${OUTPUT_ROOT}/boosted_eval}"
SEEDS="${SEEDS:-42 3407 2026}"
SEED="${SEED:-42}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PREP_TILES="${PREP_TILES:-1}"
SHOW_BAR="${SHOW_BAR:-0}"
CHECKPOINTS="${CHECKPOINTS:-}"
INFO_PREFIX="${INFO_PREFIX:-m3fd_frozencoreboost}"
LIMIT_IMAGES="${LIMIT_IMAGES:-0}"
CALIBRATION_NUM_IMAGES="${CALIBRATION_NUM_IMAGES:-0}"

export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export ALBUMENTATIONS_DISABLE_VERSION_CHECK="${ALBUMENTATIONS_DISABLE_VERSION_CHECK:-1}"

run_prepare_tiles() {
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/prepare_m3fd_tiles.py"
}

run_train_for_seed() {
    local seed="$1"
    local info_tag="${INFO_PREFIX}_seed${seed}"
    "${PYTHON_BIN}" -u main.py \
        --config "${CONFIG}" \
        --model-name "${MODEL_NAME}" \
        --output-root "${OUTPUT_ROOT}" \
        --base-seed "${seed}" \
        --info "${info_tag}"
}

find_checkpoints_for_seeds() {
    local collected=()
    for seed in ${SEEDS}; do
        local path
        path="$(find "${OUTPUT_ROOT}" -type f -path "*/pth/best_state.pth" | grep "seed${seed}" | sort | tail -n 1 || true)"
        if [[ -n "${path}" ]]; then
            collected+=("${path}")
        fi
    done
    printf '%s\n' "${collected[@]}"
}

run_boosted_eval() {
    local checkpoint_args=()
    if [[ -n "${CHECKPOINTS}" ]]; then
        for ckpt in ${CHECKPOINTS}; do
            checkpoint_args+=(--checkpoint "${ckpt}")
        done
    else
        while IFS= read -r ckpt; do
            [[ -n "${ckpt}" ]] && checkpoint_args+=(--checkpoint "${ckpt}")
        done < <(find_checkpoints_for_seeds)
    fi

    if [[ "${#checkpoint_args[@]}" -eq 0 ]]; then
        echo "[ERROR] 没有找到可用于 boosted eval 的 checkpoint。"
        exit 1
    fi

    local show_bar_args=()
    if [[ "${SHOW_BAR}" == "1" ]]; then
        show_bar_args+=(--show-bar)
    fi

    local extra_eval_args=()
    if [[ "${LIMIT_IMAGES}" != "0" ]]; then
        extra_eval_args+=(--limit-images "${LIMIT_IMAGES}")
    fi
    if [[ "${CALIBRATION_NUM_IMAGES}" != "0" ]]; then
        extra_eval_args+=(--calibration-num-images "${CALIBRATION_NUM_IMAGES}")
    fi

    "${PYTHON_BIN}" "${REPO_ROOT}/tools/evaluate_m3fd_boosted.py" \
        --config "${CONFIG}" \
        --model-name "${MODEL_NAME}" \
        --output-dir "${BOOSTED_OUTPUT_DIR}" \
        "${checkpoint_args[@]}" \
        "${extra_eval_args[@]}" \
        "${show_bar_args[@]}"
}

if [[ "${PREP_TILES}" == "1" ]]; then
    run_prepare_tiles
fi

case "${MODE}" in
    prepare)
        ;;
    train-one)
        run_train_for_seed "${SEED}"
        ;;
    train-all)
        for seed in ${SEEDS}; do
            run_train_for_seed "${seed}"
        done
        ;;
    boosted-eval)
        run_boosted_eval
        ;;
    *)
        echo "[ERROR] Unsupported MODE=${MODE}. Use prepare | train-one | train-all | boosted-eval"
        exit 1
        ;;
esac
