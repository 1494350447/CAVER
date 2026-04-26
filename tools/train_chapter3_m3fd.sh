#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

print_help() {
    cat <<'EOF'
第三章 M3FD 一键训练脚本

常用方式:
  1) 新训练:
     bash tools/train_chapter3_m3fd.sh

  2) 自动寻找最新 checkpoint 后继续训练:
     MODE=resume AUTO_RESUME=1 bash tools/train_chapter3_m3fd.sh

  3) 指定 checkpoint 继续训练:
     MODE=resume LOAD_FROM=/abs/path/to/state.pth bash tools/train_chapter3_m3fd.sh

  4) 只评估:
     MODE=eval LOAD_FROM=/abs/path/to/state.pth bash tools/train_chapter3_m3fd.sh

可用环境变量:
  MODE                 train | resume | eval
  PYTHON_BIN           Python 可执行文件, 默认 python
  CONFIG               配置文件路径
  MODEL_NAME           模型名
  OUTPUT_ROOT          输出目录根路径
  CUDA_VISIBLE_DEVICES GPU 编号, 默认 0
  OMP_NUM_THREADS      OpenMP 线程数, 默认 1
  PRETRAINED           fresh train 时使用的 backbone 预训练
  LOAD_FROM            checkpoint 路径
  AUTO_RESUME          1 表示自动寻找最新 checkpoint
  INFO                 追加到实验名中的标记
  SHOW_BAR             1 表示显示测试进度条
  EXTRA_ARGS           透传给 main.py 的其他参数

示例:
  PRETRAINED=timm:resnet50d.ra4_e3600_r224_in1k bash tools/train_chapter3_m3fd.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    print_help
    exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${MODE:-train}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/chapter3_m3fd_sam2_large.py}"
MODEL_NAME="${MODEL_NAME:-SAM2PriorAlignmentYOLODetector}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output_chapter3_m3fd_full}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
PRETRAINED="${PRETRAINED:-timm:resnet50d.ra4_e3600_r224_in1k}"
LOAD_FROM="${LOAD_FROM:-}"
AUTO_RESUME="${AUTO_RESUME:-1}"
INFO="${INFO:-}"
SHOW_BAR="${SHOW_BAR:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

find_latest_checkpoint() {
    find "${OUTPUT_ROOT}" -type f -path '*/pth/state.pth' -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
}

resolve_load_from() {
    local maybe_path="${LOAD_FROM}"
    if [[ -z "${maybe_path}" && "${AUTO_RESUME}" == "1" ]]; then
        maybe_path="$(find_latest_checkpoint || true)"
    fi
    echo "${maybe_path}"
}

cmd=(
    "${PYTHON_BIN}" -u main.py
    --config "${CONFIG}"
    --model-name "${MODEL_NAME}"
    --output-root "${OUTPUT_ROOT}"
)

case "${MODE}" in
    train)
        ;;
    resume)
        LOAD_FROM="$(resolve_load_from)"
        if [[ -z "${LOAD_FROM}" ]]; then
            echo "[ERROR] MODE=resume 但没有可用 checkpoint。请设置 LOAD_FROM 或打开 AUTO_RESUME=1。"
            exit 1
        fi
        cmd+=(--load-from "${LOAD_FROM}")
        ;;
    eval)
        LOAD_FROM="$(resolve_load_from)"
        if [[ -z "${LOAD_FROM}" ]]; then
            echo "[ERROR] MODE=eval 但没有可用 checkpoint。请设置 LOAD_FROM 或打开 AUTO_RESUME=1。"
            exit 1
        fi
        cmd+=(--load-from "${LOAD_FROM}" --evaluate)
        ;;
    *)
        echo "[ERROR] 不支持的 MODE=${MODE}，只能是 train / resume / eval"
        exit 1
        ;;
esac

if [[ -n "${PRETRAINED}" && "${MODE}" == "train" && -z "${LOAD_FROM}" ]]; then
    cmd+=(--pretrained "${PRETRAINED}")
fi

if [[ -n "${INFO}" ]]; then
    cmd+=(--info "${INFO}")
fi

if [[ "${SHOW_BAR}" == "1" ]]; then
    cmd+=(--show-bar)
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
    read -r -a extra_args <<< "${EXTRA_ARGS}"
    cmd+=("${extra_args[@]}")
fi

echo "================ Resolved Training Launch ================"
echo "REPO_ROOT            : ${REPO_ROOT}"
echo "MODE                 : ${MODE}"
echo "PYTHON_BIN           : ${PYTHON_BIN}"
echo "CONFIG               : ${CONFIG}"
echo "MODEL_NAME           : ${MODEL_NAME}"
echo "OUTPUT_ROOT          : ${OUTPUT_ROOT}"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES}"
echo "OMP_NUM_THREADS      : ${OMP_NUM_THREADS}"
echo "PRETRAINED           : ${PRETRAINED}"
echo "LOAD_FROM            : ${LOAD_FROM:-<empty>}"
echo "AUTO_RESUME          : ${AUTO_RESUME}"
echo "INFO                 : ${INFO:-<empty>}"
echo "SHOW_BAR             : ${SHOW_BAR}"
echo "EXTRA_ARGS           : ${EXTRA_ARGS:-<empty>}"
echo "=========================================================="
echo "Command:"
printf ' %q' "${cmd[@]}"
echo

exec "${cmd[@]}"
