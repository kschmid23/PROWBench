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
# CUDA 13 build: the GB300 is Blackwell, and the cu128 build the README names is not the one validated here.
uv pip install --python "$EVAL_PY" torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python "$EVAL_PY" -r eval/requirements.txt -r eval/requirements-models.txt huggingface_hub
"$EVAL_PY" -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

echo "=== 2/6 dataset (about 2.2 GB) ==="
"$EVAL_HF" download AlayaLab/PROWBench --repo-type dataset --local-dir "$DOWNLOAD"

echo "=== 3/6 prepared evaluation data ==="
if [ -d "$DATA" ]; then
  echo "    $DATA exists, skipping (the tool needs an empty output directory)"
else
  "$EVAL_PY" tools/prepare_eval_data.py --download "$DOWNLOAD" --out "$DATA"
fi

echo "=== 4/6 RAFT and CLIP weights ==="
mkdir -p "$WEIGHTS"
if [ ! -f "$WEIGHTS/models/raft-things.pth" ]; then
  wget -q -O "$WEIGHTS/models.zip" https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip
  "$EVAL_PY" -m zipfile -e "$WEIGHTS/models.zip" "$WEIGHTS"
fi
[ -f "$WEIGHTS/ViT-B-32.pt" ] || wget -q -O "$WEIGHTS/ViT-B-32.pt" https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt

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
