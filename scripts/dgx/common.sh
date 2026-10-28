#!/usr/bin/env bash
# Shared settings for the DGX scripts. Source this file; do not run it.
# Every path can be overridden from the environment before sourcing.
set -euo pipefail

PROWBENCH="${PROWBENCH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PROWBENCH
export PATH="$HOME/.local/bin:$PATH"

export DOWNLOAD="${DOWNLOAD:-$PROWBENCH/PROWBench}"                 # the Hugging Face dataset
export DATA="${DATA:-$PROWBENCH/prowbench-eval-data}"               # prepared by tools/prepare_eval_data.py
export WEIGHTS="${WEIGHTS:-$PROWBENCH/weights}"                     # RAFT and CLIP
export VIDEOS="${VIDEOS:-$PROWBENCH/model_output_video}"            # generated videos, USAGE layout
export WORK_ROOT="${WORK_ROOT:-$PROWBENCH/work}"                    # evaluation results, <dataset>/<method>/<representation>
export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-$PROWBENCH/vbench-cache}"
export PYTHONPATH="$PROWBENCH/eval${PYTHONPATH:+:$PYTHONPATH}"

EVAL_PY="$PROWBENCH/.venv/bin/python"
EVAL_HF="$PROWBENCH/.venv/bin/hf"
QUALITY_PY="$PROWBENCH/.venv-quality/bin/python"

# Method checkouts on the DGX.
EDENLIVE_ROOT="${EDENLIVE_ROOT:-$HOME/EdenLive}"
MATRIX4_ROOT="${MATRIX4_ROOT:-$HOME/Matrix-Game/Matrix-Game-4}"
COSMOS3_ROOT="${COSMOS3_ROOT:-$HOME/cosmos}"   # needs .venv-cosmos3-diffusers and the Cosmos3-Nano checkpoint; HF_TOKEN in the shell

require_uv() {
  command -v uv >/dev/null || { echo "ABORT: uv not on PATH (https://astral.sh/uv)"; exit 1; }
}

require_eval_env() {
  [ -x "$EVAL_PY" ] || { echo "ABORT: $EVAL_PY missing; run scripts/dgx/setup.sh first"; exit 1; }
}
