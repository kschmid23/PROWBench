#!/usr/bin/env bash
# The separate environment for the visual-quality metrics (VBench imaging, aesthetic, motion smoothness): VBench 0.1.5 needs an older
# Transformers and NumPy than the tracker and judge environment (eval/requirements-quality.txt).
#
# Usage:   bash scripts/dgx/setup_quality_env.sh
# Note:    decord==0.6.0 has no aarch64 wheel. If the install stops there, the script stops with it; nothing is substituted.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_uv
cd "$PROWBENCH"

[ -d .venv-quality ] || uv venv --python 3.11 .venv-quality
uv pip install --python "$QUALITY_PY" torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python "$QUALITY_PY" -r eval/requirements-quality.txt
"$QUALITY_PY" -c "import torch, vbench; assert torch.cuda.is_available(), 'CUDA not visible'; print('quality env ok', torch.__version__)"
mkdir -p "$VBENCH_CACHE_DIR"
