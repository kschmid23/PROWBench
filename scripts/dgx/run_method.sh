#!/usr/bin/env bash
# Generate PROWBench videos with EdenLive, Matrix-Game 4.0, Cosmos 3 or the Wan2.2-Fun-5B-Control base model on the DGX, into the layout the evaluation reads
# ($VIDEOS/<dataset>/<method>/<episode>/<camera>/<representation>/video.mp4). Finished samples are skipped, so it can be started again.
#
# Usage:   bash scripts/dgx/run_method.sh <edenlive|matrix4|cosmos3|wan5b> <dataset> <white|mixed|colored-obb|all> [options of scripts/generate_videos.py]
# Examples (check the GPU first: nvidia-smi):
#   bash scripts/dgx/run_method.sh edenlive verified_ff_40 mixed --limit 2
#   bash scripts/dgx/run_method.sh matrix4 verified_ff_40 mixed --scene racing --preset baseline --limit 2
#   bash scripts/dgx/run_method.sh cosmos3 verified_ff_40 mixed --limit 2
#   bash scripts/dgx/setup_wan5b.sh                                  (once: builds the Wan 5B model folder)
#   bash scripts/dgx/run_method.sh wan5b verified_ff_40 mixed --episode dv200-153-city-cat-encounter
# Checkouts: EDENLIVE_ROOT (default ~/EdenLive, needs its dit-5b weights), MATRIX4_ROOT (default ~/Matrix-Game/Matrix-Game-4) and
# COSMOS3_ROOT (default ~/cosmos; cosmos3 needs HF_TOKEN exported in the shell, and about 55 GB of free GPU memory); wan5b: WAN5B_ROOT (~/wan22/VideoX-Fun), WAN5B_PY (~/wan22/.venv/bin/python), WAN5B_MODEL.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_eval_env
[ "$#" -ge 3 ] || { sed -n '2,15p' "${BASH_SOURCE[0]}"; exit 2; }

method="$1"
dataset="$2"
representation="$3"
shift 3

"$EVAL_PY" "$PROWBENCH/scripts/generate_videos.py" --method "$method" --dataset "$dataset" --representation "$representation" \
  --data "$DATA" --videos "$VIDEOS" --edenlive-root "$EDENLIVE_ROOT" --matrix4-root "$MATRIX4_ROOT" --cosmos3-root "$COSMOS3_ROOT" \
  --wan5b-root "$WAN5B_ROOT" --wan5b-python "$WAN5B_PY" --wan5b-model "$WAN5B_MODEL" "$@"
