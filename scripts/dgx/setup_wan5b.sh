#!/usr/bin/env bash
# Build the Wan2.2-Fun-5B-Control folder that scripts/wan5b_worker.py reads ($WAN5B_MODEL, default $WEIGHTS/Wan2.2-Fun-5B-Control), from files already on the DGX
# (the Matrix-Game-4 checkout holds the same files that the alibaba-pai/Wan2.2-Fun-5B-Control repo does): the folder holds symbolic links, so no
# disk space is used besides config.json (301 bytes, downloaded). Each linked file is checked against its size in the repo at the pinned revision.
#
# Usage:   bash scripts/dgx/setup_wan5b.sh
# Needs:   $MATRIX4_ROOT/resources_base_src/diffusion_pytorch_model.safetensors (Matrix-Game-4/scripts/dgx/run_base_model.sh --convert-only --download makes it)
#          and $MATRIX4_ROOT/resources/base/{Wan2.2_VAE.pth,models_t5_umt5-xxl-enc-bf16.pth,google/umt5-xxl} (Matrix-Game-4/scripts/dgx/download_base.sh).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

REPO="alibaba-pai/Wan2.2-Fun-5B-Control"
REVISION="b8bc1a65ab71d054ba4636dc0dac104aa4df2686"
# name in the model folder | source | bytes in the repo at REVISION
FILES=(
  "diffusion_pytorch_model.safetensors|$MATRIX4_ROOT/resources_base_src/diffusion_pytorch_model.safetensors|10003303280"
  "Wan2.2_VAE.pth|$MATRIX4_ROOT/resources/base/Wan2.2_VAE.pth|2818839170"
  "models_t5_umt5-xxl-enc-bf16.pth|$MATRIX4_ROOT/resources/base/models_t5_umt5-xxl-enc-bf16.pth|11361920418"
)

mkdir -p "$WAN5B_MODEL"
for entry in "${FILES[@]}"; do
  IFS='|' read -r name source bytes <<< "$entry"
  [ -f "$source" ] || { echo "ABORT: $source is missing"; exit 1; }
  actual="$(stat -L -c %s "$source")"
  [ "$actual" = "$bytes" ] || { echo "ABORT: $source is $actual bytes, the repo's $name is $bytes"; exit 1; }
  ln -sfn "$source" "$WAN5B_MODEL/$name"
done
[ -d "$MATRIX4_ROOT/resources/base/google/umt5-xxl" ] || { echo "ABORT: $MATRIX4_ROOT/resources/base/google/umt5-xxl is missing"; exit 1; }
mkdir -p "$WAN5B_MODEL/google"
ln -sfn "$MATRIX4_ROOT/resources/base/google/umt5-xxl" "$WAN5B_MODEL/google/umt5-xxl"

"$EVAL_PY" -c "
from huggingface_hub import hf_hub_download
import shutil
shutil.copyfile(hf_hub_download('$REPO', 'config.json', revision='$REVISION'), '$WAN5B_MODEL/config.json')
"
echo "Wan2.2-Fun-5B-Control folder ready: $WAN5B_MODEL"
ls -l "$WAN5B_MODEL"
