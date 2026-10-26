#!/usr/bin/env bash
# ViPE v1.2.0 for PROWBench's camera estimation (stage 4.4 of run_eval.sh). PROWBench imports vipe.config.parse and other modules of the
# v1.2.0 package layout; the ViPE fork inside the MIND checkout predates it, so this builds the upstream tag next to it and leaves MIND alone.
#
# Usage:   bash scripts/dgx/setup_vipe.sh            (VIPE_ROOT=~/vipe-1.2.0 by default; MAX_JOBS=8 compile jobs)
# Then:    bash scripts/dgx/run_eval.sh <dataset> <method> <representation> --vipe-root ~/vipe-1.2.0
# Notes:   compiles ViPE's CUDA extension for this GPU (about 10 to 30 minutes). ViPE downloads its own model weights on its first run and
#          needs internet then. CUDA 13.2 (/usr/local/cuda-13.2) and torch 2.14.1+cu132: the extension is compiled with nvcc 13.2, so torch is
#          the cu132 build too (older cu130 builds of torch 2.10 have no sm_103 kernels, see setup.sh).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_uv

VIPE_ROOT="${VIPE_ROOT:-$HOME/vipe-1.2.0}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
[ -x "$CUDA_HOME/bin/nvcc" ] || { echo "ABORT: no nvcc under $CUDA_HOME (set CUDA_HOME, for example /usr/local/cuda-13.2)"; exit 1; }
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-8}"

echo "=== 1/4 ViPE v1.2.0 checkout ==="
[ -d "$VIPE_ROOT/.git" ] || git clone --branch v1.2.0 --depth 1 https://github.com/nv-tlabs/vipe.git "$VIPE_ROOT"
echo "    $(git -C "$VIPE_ROOT" describe --tags)"
cd "$VIPE_ROOT"

echo "=== 2/4 environment (Python 3.11, torch 2.14.1+cu132) ==="
[ -d .venv ] || uv venv --python 3.11 .venv
VIPE_PY="$VIPE_ROOT/.venv/bin/python"
uv pip install --python "$VIPE_PY" torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu132
uv pip install --python "$VIPE_PY" setuptools wheel ninja numpy scipy pillow

echo "=== 3/4 build and install ViPE (CUDA extension for this GPU) ==="
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$("$VIPE_PY" -c 'import torch; major, minor = torch.cuda.get_device_capability(); print(f"{major}.{minor}")')}"
echo "    TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST  MAX_JOBS=$MAX_JOBS"
uv pip install --python "$VIPE_PY" --no-build-isolation -e .

echo "=== 4/4 check (the imports PROWBench's camera step uses) ==="
"$VIPE_PY" -c "
import vipe
from vipe.config.parse import parse_typed_config
from vipe.pipeline import _as_dictconfig
from vipe.pipeline.default import DefaultAnnotationPipeline
from vipe.pipeline.processors import TrackAnythingProcessor
from vipe.streams.base import FrameAttribute, ProcessedVideoStream, StreamProcessor
from vipe.streams.raw_mp4_stream import RawMp4Stream
from vipe.utils.cameras import CameraType
import torch
assert torch.cuda.is_available(), 'CUDA not visible'
print('vipe ok', getattr(vipe, '__version__', '?'))
"
PYTHONPATH="$PROWBENCH/eval" "$VIPE_PY" -c "import prowbench_eval.estimate_camera; print('prowbench_eval imports ok in the ViPE environment')"
