#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

print_help() {
    cat <<'EOF'
M3FD 第三章两阶段一键训练脚本

默认行为:
  1. 等待 GPU 空闲
  2. 启动 Stage A 强基线
  3. 自动寻找 Stage A 最新 best_state.pth
  4. 启动 Stage B SAM2 teacher 蒸馏细化

常用方式:
  bash tools/train_chapter3_m3fd_twostage.sh

只跑 Stage A:
  PIPELINE_MODE=stage_a bash tools/train_chapter3_m3fd_twostage.sh

从已有 Stage A best 直接跑 Stage B:
  PIPELINE_MODE=stage_b STAGE_A_BEST=/abs/path/to/best_state.pth bash tools/train_chapter3_m3fd_twostage.sh

直接评估 Stage B:
  PIPELINE_MODE=eval_stage_b STAGE_B_LOAD_FROM=/abs/path/to/best_state.pth bash tools/train_chapter3_m3fd_twostage.sh

跳过等待 GPU:
  WAIT_FOR_IDLE_GPU=0 bash tools/train_chapter3_m3fd_twostage.sh

环境变量:
  PIPELINE_MODE       full | stage_a | stage_b | eval_stage_b
  PYTHON_BIN          Python 可执行文件, 默认 python
  MODEL_NAME          模型名
  STAGE_A_CONFIG      Stage A 配置
  STAGE_B_CONFIG      Stage B 配置
  STAGE_A_OUTPUT_ROOT Stage A 输出目录
  STAGE_B_OUTPUT_ROOT Stage B 输出目录
  PRETRAINED_FILE     本地 backbone 预训练文件
  PRETRAINED_URL      若 PRETRAINED_FILE 不存在时自动下载的 URL
  STAGE_A_BEST        指定 Stage A best checkpoint
  STAGE_B_LOAD_FROM   指定 Stage B checkpoint, eval_stage_b 时优先使用
  CUDA_VISIBLE_DEVICES GPU 编号, 默认 0
  OMP_NUM_THREADS     OpenMP 线程数, 默认 1
  WAIT_FOR_IDLE_GPU   1 表示等待 GPU 空闲, 默认 1
  IDLE_MEM_USED_MB    认为 GPU 空闲时的最大已占显存, 默认 2000
  IDLE_UTIL_PERCENT   认为 GPU 空闲时的最大利用率, 默认 10
  CHECK_INTERVAL_SEC  轮询等待间隔, 默认 30
  STAGE_A_INFO        Stage A 追加实验标签
  STAGE_B_INFO        Stage B 追加实验标签
  STAGE_A_EXTRA_ARGS  透传给 Stage A 的额外参数
  STAGE_B_EXTRA_ARGS  透传给 Stage B 的额外参数
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    print_help
    exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
PIPELINE_MODE="${PIPELINE_MODE:-full}"
MODEL_NAME="${MODEL_NAME:-SAM2PriorAlignmentYOLODetector}"
STAGE_A_CONFIG="${STAGE_A_CONFIG:-${REPO_ROOT}/configs/chapter3_m3fd_strong_base.py}"
STAGE_B_CONFIG="${STAGE_B_CONFIG:-${REPO_ROOT}/configs/chapter3_m3fd_sam2_refine.py}"
STAGE_A_OUTPUT_ROOT="${STAGE_A_OUTPUT_ROOT:-${REPO_ROOT}/output_chapter3_m3fd_strong}"
STAGE_B_OUTPUT_ROOT="${STAGE_B_OUTPUT_ROOT:-${REPO_ROOT}/output_chapter3_m3fd_refine}"
PRETRAINED_DIR="${PRETRAINED_DIR:-${REPO_ROOT}/pretrained}"
PRETRAINED_FILE="${PRETRAINED_FILE:-${PRETRAINED_DIR}/resnet50d_ra2-464e36ba.pth}"
PRETRAINED_URL="${PRETRAINED_URL:-https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/resnet50d_ra2-464e36ba.pth}"
STAGE_A_BEST="${STAGE_A_BEST:-}"
STAGE_B_LOAD_FROM="${STAGE_B_LOAD_FROM:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
WAIT_FOR_IDLE_GPU="${WAIT_FOR_IDLE_GPU:-1}"
IDLE_MEM_USED_MB="${IDLE_MEM_USED_MB:-2000}"
IDLE_UTIL_PERCENT="${IDLE_UTIL_PERCENT:-10}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-30}"
STAGE_A_INFO="${STAGE_A_INFO:-}"
STAGE_B_INFO="${STAGE_B_INFO:-}"
STAGE_A_EXTRA_ARGS="${STAGE_A_EXTRA_ARGS:-}"
STAGE_B_EXTRA_ARGS="${STAGE_B_EXTRA_ARGS:-}"

export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

find_latest_best_checkpoint() {
    local search_root="$1"
    find "${search_root}" -type f -path '*/pth/best_state.pth' -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
}

wait_for_idle_gpu() {
    if [[ "${WAIT_FOR_IDLE_GPU}" != "1" ]]; then
        return 0
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[WARN] 找不到 nvidia-smi，跳过 GPU 空闲等待。"
        return 0
    fi

    echo "[INFO] 等待 GPU 空闲: mem_used<=${IDLE_MEM_USED_MB}MB 且 util<=${IDLE_UTIL_PERCENT}%"
    while true; do
        local query
        query="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -n 1 || true)"
        local used_mem util
        used_mem="$(echo "${query}" | cut -d',' -f1 | xargs)"
        util="$(echo "${query}" | cut -d',' -f2 | xargs)"
        if [[ -n "${used_mem}" && -n "${util}" ]]; then
            echo "[INFO] 当前 GPU: used_mem=${used_mem}MB util=${util}%"
            if (( used_mem <= IDLE_MEM_USED_MB && util <= IDLE_UTIL_PERCENT )); then
                echo "[INFO] GPU 已空闲，开始训练。"
                break
            fi
        else
            echo "[WARN] 未能解析 nvidia-smi 输出，${CHECK_INTERVAL_SEC}s 后重试。"
        fi
        sleep "${CHECK_INTERVAL_SEC}"
    done
}

ensure_pretrained_file() {
    mkdir -p "${PRETRAINED_DIR}"
    if [[ -f "${PRETRAINED_FILE}" ]]; then
        echo "[INFO] 使用本地 backbone 预训练: ${PRETRAINED_FILE}"
        return 0
    fi
    echo "[INFO] 未找到本地 backbone 预训练，开始下载:"
    echo "       ${PRETRAINED_URL}"
    curl -L --fail --retry 3 --retry-delay 2 -o "${PRETRAINED_FILE}" "${PRETRAINED_URL}"
    echo "[INFO] 预训练下载完成: ${PRETRAINED_FILE}"
}

run_stage() {
    local stage_name="$1"
    shift
    local -a cmd=("${PYTHON_BIN}" -u main.py "$@")
    echo "================ ${stage_name} ================"
    printf ' %q' "${cmd[@]}"
    echo
    "${cmd[@]}"
}

append_extra_args() {
    local extra="$1"
    if [[ -z "${extra}" ]]; then
        return 0
    fi
    read -r -a extra_args <<< "${extra}"
    printf '%s\n' "${extra_args[@]}"
}

resolve_stage_a_best() {
    if [[ -n "${STAGE_A_BEST}" ]]; then
        echo "${STAGE_A_BEST}"
        return 0
    fi
    find_latest_best_checkpoint "${STAGE_A_OUTPUT_ROOT}"
}

wait_for_idle_gpu
ensure_pretrained_file

case "${PIPELINE_MODE}" in
    full|stage_a)
        stage_a_cmd=(
            --config "${STAGE_A_CONFIG}"
            --model-name "${MODEL_NAME}"
            --output-root "${STAGE_A_OUTPUT_ROOT}"
            --pretrained "${PRETRAINED_FILE}"
        )
        if [[ -n "${STAGE_A_INFO}" ]]; then
            stage_a_cmd+=(--info "${STAGE_A_INFO}")
        fi
        while IFS= read -r item; do
            stage_a_cmd+=("${item}")
        done < <(append_extra_args "${STAGE_A_EXTRA_ARGS}")
        run_stage "Stage A" "${stage_a_cmd[@]}"
        ;;
    stage_b|eval_stage_b)
        ;;
    *)
        echo "[ERROR] 不支持的 PIPELINE_MODE=${PIPELINE_MODE}"
        exit 1
        ;;
esac

if [[ "${PIPELINE_MODE}" == "stage_a" ]]; then
    exit 0
fi

if [[ -z "${STAGE_B_LOAD_FROM}" ]]; then
    STAGE_B_LOAD_FROM="$(resolve_stage_a_best || true)"
fi

if [[ -z "${STAGE_B_LOAD_FROM}" ]]; then
    echo "[ERROR] 没有找到可用于 Stage B 的 best_state.pth。"
    echo "        请确认 Stage A 已完成，或手动设置 STAGE_A_BEST / STAGE_B_LOAD_FROM。"
    exit 1
fi

stage_b_cmd=(
    --config "${STAGE_B_CONFIG}"
    --model-name "${MODEL_NAME}"
    --output-root "${STAGE_B_OUTPUT_ROOT}"
    --load-from "${STAGE_B_LOAD_FROM}"
    --pretrained "${PRETRAINED_FILE}"
)
if [[ -n "${STAGE_B_INFO}" ]]; then
    stage_b_cmd+=(--info "${STAGE_B_INFO}")
fi
if [[ "${PIPELINE_MODE}" == "eval_stage_b" ]]; then
    stage_b_cmd+=(--evaluate)
fi
while IFS= read -r item; do
    stage_b_cmd+=("${item}")
done < <(append_extra_args "${STAGE_B_EXTRA_ARGS}")

run_stage "Stage B" "${stage_b_cmd[@]}"
