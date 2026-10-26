#!/usr/bin/env bash
# The separate environment for the visual-quality metrics (VBench imaging, aesthetic, motion smoothness): VBench 0.1.5 needs an older
# Transformers and NumPy than the tracker and judge environment (eval/requirements-quality.txt).
#
# Usage:   bash scripts/dgx/setup_quality_env.sh
# aarch64: decord has no wheel there (and FFmpeg 6 is too new to build it), and vbench depends on it. VBench imports decord at the top of
#          vbench/utils.py but only calls it for dimensions PROWBench does not run, so vbench is installed with --no-deps and
#          tools/decord_cv2 (decord's VideoReader on OpenCV) provides the import. On x86_64 the pinned decord is used.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_uv
cd "$PROWBENCH"

[ -d .venv-quality ] || uv venv --python 3.11 .venv-quality
uv pip install --python "$QUALITY_PY" torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130

if [ "$(uname -m)" = "aarch64" ]; then
  REQUIREMENTS="$(mktemp)"
  trap 'rm -f "$REQUIREMENTS"' EXIT
  grep -v -i -E "^(decord|vbench)" eval/requirements-quality.txt > "$REQUIREMENTS"
  uv pip install --python "$QUALITY_PY" -r "$REQUIREMENTS"
  uv pip install --python "$QUALITY_PY" --no-deps ./tools/decord_cv2
  uv pip install --python "$QUALITY_PY" --no-deps vbench==0.1.5
else
  uv pip install --python "$QUALITY_PY" -r eval/requirements-quality.txt
fi

"$QUALITY_PY" -c "import torch, decord, vbench; assert torch.cuda.is_available(), 'CUDA not visible'; print('quality env ok', torch.__version__, 'decord', decord.__version__)"
mkdir -p "$VBENCH_CACHE_DIR"
