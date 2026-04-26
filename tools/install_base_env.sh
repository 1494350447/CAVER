#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/CAVER"
SAM2_REPO="/root/autodl-fs/third_party/sam2_official"
SAM2_CKPT_DIR="/root/autodl-fs/checkpoints/sam2"
SAM2_CKPT_PATH="${SAM2_CKPT_DIR}/sam2_hiera_large.pt"
SAM2_CKPT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"

python -m pip install --upgrade pip
python -m pip install mmengine timm albumentations opencv-python-headless hydra-core iopath pillow

mkdir -p /root/autodl-fs/third_party
if [ ! -d "${SAM2_REPO}/.git" ]; then
  git clone https://github.com/facebookresearch/sam2 "${SAM2_REPO}"
else
  git -C "${SAM2_REPO}" fetch --all --tags
  git -C "${SAM2_REPO}" pull --ff-only
fi

SAM2_BUILD_CUDA=0 python -m pip install -e "${SAM2_REPO}" --no-deps --no-build-isolation

mkdir -p "${SAM2_CKPT_DIR}"
if [ ! -f "${SAM2_CKPT_PATH}" ]; then
  wget -O "${SAM2_CKPT_PATH}" "${SAM2_CKPT_URL}"
fi

python - <<'PY'
import importlib
mods = ["sam2", "mmengine", "timm", "albumentations", "cv2"]
for name in mods:
    mod = importlib.import_module(name)
    print("OK", name, getattr(mod, "__file__", None))
PY

python "${ROOT_DIR}/tools/prepare_m3fd_detection.py"
