#!/usr/bin/env bash
# Generates the videos for one method (scripts/dgx/run_method.sh) and then evaluates them (scripts/dgx/run_eval.sh). Both steps skip
# finished work, so it can be started again after an interruption. The evaluation only starts when the generation reports no failure.
#
# Usage:   bash scripts/dgx/run_all.sh <edenlive|matrix4> <dataset> <white|mixed|colored-obb> [--vipe-root DIR] [--skip STAGE]... [-- generation options]
#   --vipe-root DIR   ViPE 1.2.0 checkout (default $VIPE_ROOT, else ~/vipe-1.2.0)
#   --skip STAGE      camera | judge | motion, passed to run_eval.sh
#   -- options        everything after -- goes to run_method.sh (for example --limit 2, --scene racing, --hold-last-frame)
# Example (check the GPU first: nvidia-smi):
#   bash scripts/dgx/run_all.sh matrix4 verified_ff_40 mixed
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[ "$#" -ge 3 ] || { sed -n '2,11p' "${BASH_SOURCE[0]}"; exit 2; }

method="$1"
dataset="$2"
representation="$3"
shift 3
vipe_root="${VIPE_ROOT:-$HOME/vipe-1.2.0}"
eval_options=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --skip) eval_options+=(--skip "${2:?--skip needs camera, judge or motion}"); shift 2 ;;
    --vipe-root) vipe_root="${2:?--vipe-root needs a directory}"; shift 2 ;;
    --) shift; break ;;
    *) echo "ERROR: unknown option $1" >&2; exit 2 ;;
  esac
done

here="$(dirname "${BASH_SOURCE[0]}")"
echo "=== generate: $method $dataset $representation"
bash "$here/run_method.sh" "$method" "$dataset" "$representation" "$@" || { echo "ABORT: generation failed, evaluation not started"; exit 1; }
echo "=== evaluate: $method $dataset $representation"
bash "$here/run_eval.sh" "$dataset" "$method" "$representation" --vipe-root "$vipe_root" "${eval_options[@]}"
