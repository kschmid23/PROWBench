#!/usr/bin/env bash
# Generate PROWBench videos with EdenLive or Matrix-Game 4.0 on the DGX, into the layout the evaluation reads
# ($VIDEOS/<dataset>/<method>/<episode>/<camera>/<representation>/video.mp4). Finished samples are skipped, so it can be started again.
#
# Usage:   bash scripts/dgx/run_method.sh <edenlive|matrix4> <dataset> <white|mixed|colored-obb|all> [options of scripts/generate_videos.py]
# Examples (check the GPU first: nvidia-smi):
#   bash scripts/dgx/run_method.sh edenlive verified_ff_40 mixed --limit 2
#   bash scripts/dgx/run_method.sh matrix4 verified_ff_40 mixed --scene racing --preset baseline --limit 2
# Checkouts: EDENLIVE_ROOT (default ~/EdenLive, needs its dit-5b weights) and MATRIX4_ROOT (default ~/Matrix-Game/Matrix-Game-4).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_eval_env
[ "$#" -ge 3 ] || { sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 2; }

method="$1"
dataset="$2"
representation="$3"
shift 3

"$EVAL_PY" "$PROWBENCH/scripts/generate_videos.py" --method "$method" --dataset "$dataset" --representation "$representation" \
  --data "$DATA" --videos "$VIDEOS" --edenlive-root "$EDENLIVE_ROOT" --matrix4-root "$MATRIX4_ROOT" "$@"
