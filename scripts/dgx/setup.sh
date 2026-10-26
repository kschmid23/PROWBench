#!/usr/bin/env bash
# PROWBench on the DGX (aarch64, Blackwell): the evaluation environment, the dataset, the prepared evaluation data and the models
# that are public. Safe to run again: every step skips what is already there.
#
# Usage:   bash scripts/dgx/setup.sh
# Needs:   uv on PATH, internet, about 60 GB free (Qwen3.6-27B is 56 GB).
# Gated:   facebook/sam3 needs access approved on Hugging Face. Export HF_TOKEN in this shell (never write it to a file) and run again.
# Not here: the quality environment (setup_quality_env.sh), ViPE 1.2.0 for the camera step, MEt3R (multi-view dataset only).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_uv
cd "$PROWBENCH"

echo "=== 1/6 evaluation environment (Python 3.11) ==="
[ -d .venv ] || uv venv --python 3.11 .venv
# The README pins torch 2.10.0 (cu128). The GB300 is sm_103; torch 2.10.0+cu130 has no sm_103 kernels and a Linear layer with a bias fails
# there (CUBLAS_STATUS_NOT_INITIALIZED in cublasLtMatmulAlgoGetHeuristic), which stops SAM 3 and the judge. 2.13.0+cu130 runs, so it replaces
# the 2.10.0 / 0.25.0 pins here; the quality environment's 2.11.0 runs and keeps its pin.
uv pip install --python "$EVAL_PY" torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
MODEL_REQUIREMENTS="$(mktemp)"
trap 'rm -f "$MODEL_REQUIREMENTS"' EXIT
grep -v -E "^(torch|torchvision)==" eval/requirements-models.txt > "$MODEL_REQUIREMENTS"
uv pip install --python "$EVAL_PY" -r eval/requirements.txt -r "$MODEL_REQUIREMENTS" huggingface_hub
"$EVAL_PY" -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

echo "=== 2/6 dataset (about 2.2 GB) ==="
"$EVAL_HF" download AlayaLab/PROWBench --repo-type dataset --local-dir "$DOWNLOAD"

echo "=== 3/6 prepared evaluation data ==="
if [ -d "$DATA" ]; then
  echo "    $DATA exists, skipping (the tool needs an empty output directory)"
else
  "$EVAL_PY" tools/prepare_eval_data.py --download "$DOWNLOAD" --out "$DATA"
fi

echo "=== 4/6 RAFT and CLIP weights (WBench's copies are reused when present) ==="
WBENCH_WEIGHTS="${WBENCH_WEIGHTS:-$HOME/WBench/weights}"
mkdir -p "$WEIGHTS/models"
if [ ! -f "$WEIGHTS/models/raft-things.pth" ]; then
  if [ -f "$WBENCH_WEIGHTS/raft/raft-things.pth" ]; then
    ln -s "$WBENCH_WEIGHTS/raft/raft-things.pth" "$WEIGHTS/models/raft-things.pth"
  else
    wget -q -O "$WEIGHTS/models.zip" https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip
    "$EVAL_PY" -m zipfile -e "$WEIGHTS/models.zip" "$WEIGHTS"
  fi
fi
if [ ! -f "$WEIGHTS/ViT-B-32.pt" ]; then
  if [ -f "$WBENCH_WEIGHTS/clip/ViT-B-32.pt" ]; then
    ln -s "$WBENCH_WEIGHTS/clip/ViT-B-32.pt" "$WEIGHTS/ViT-B-32.pt"
  else
    wget -q -O "$WEIGHTS/ViT-B-32.pt" https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
  fi
fi

echo "=== 5/6 tracking and judge models (Hugging Face cache) ==="
"$EVAL_HF" download facebook/sam2.1-hiera-large
"$EVAL_HF" download Qwen/Qwen3.6-27B
if [ -n "${HF_TOKEN:-}" ]; then
  "$EVAL_HF" download facebook/sam3
else
  echo "    facebook/sam3 skipped: HF_TOKEN is not set (access must be approved at https://huggingface.co/facebook/sam3)"
fi

echo "=== 6/6 check ==="
ls "$DATA"
ls "$WEIGHTS"
echo "done. next: scripts/dgx/setup_quality_env.sh, then generate videos (scripts/dgx/run_method.sh) and evaluate"
