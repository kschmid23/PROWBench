#!/usr/bin/env bash
# Runs the PROWBench evaluation (eval/USAGE.md section 4, stages 4.1 to 4.8) for one method and one proxy representation of a
# five-second dataset, on the videos in $VIDEOS/<dataset>/<method>/ (see scripts/dgx/run_method.sh). Results: $WORK/table.json,
# with $WORK = work/<dataset>/<method>/<representation>. Stages that save per-sample results pick up where they stopped.
#
# Usage:   bash scripts/dgx/run_eval.sh <dataset> <method> <representation> [--vipe-root DIR] [--skip STAGE]...
#   dataset         verified_ff_40, unverified_ff_90 or gunfight (long_horizon and multi4 have their own stages in USAGE)
#   --vipe-root     ViPE 1.2.0 checkout with its .venv (camera estimation, stage 4.4)
#   --skip STAGE    camera | judge: leave a stage out. Metrics that depend on it are then missing from the table.
# Needs:   scripts/dgx/setup.sh and setup_quality_env.sh done; facebook/sam3 downloaded (tracking, stage 4.3; needs HF_TOKEN access
#          approved once); the Qwen3.6-27B judge needs about 80 GB of free GPU memory (check with nvidia-smi).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_eval_env
[ "$#" -ge 3 ] || { sed -n '2,11p' "${BASH_SOURCE[0]}"; exit 2; }

dataset="$1"
method="$2"
representation="$3"
shift 3
skipped=" "
vipe_root=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --skip) skipped="$skipped${2:?--skip needs camera or judge} "; shift 2 ;;
    --vipe-root) vipe_root="${2:?--vipe-root needs a directory}"; shift 2 ;;
    *) echo "ERROR: unknown option $1" >&2; exit 2 ;;
  esac
done
skipping() { [[ "$skipped" == *" $1 "* ]]; }

case "$dataset" in
  verified_ff_40|unverified_ff_90|gunfight) ;;
  *) echo "ABORT: $dataset is not a five-second dataset handled here"; exit 1 ;;
esac
[ -d "$VIDEOS/$dataset/$method" ] || { echo "ABORT: no videos in $VIDEOS/$dataset/$method"; exit 1; }
[ -x "$QUALITY_PY" ] || { echo "ABORT: $QUALITY_PY missing; run scripts/dgx/setup_quality_env.sh"; exit 1; }
if ! skipping camera && [ -z "$vipe_root" ]; then
  echo "ABORT: camera estimation needs --vipe-root <ViPE 1.2.0 checkout>, or leave it out with --skip camera"
  exit 1
fi

# Evaluated length in seconds. 4.8 s fits the 117-frame (4.875 s at 24 fps) renders; export it first to use another value.
export PROWBENCH_EVAL_DURATION="${PROWBENCH_EVAL_DURATION:-4.8}"
export WORK="$PROWBENCH/work/$dataset/$method/$representation"
mkdir -p "$WORK/metrics"
cd "$PROWBENCH/eval"

echo "=== 4.1 ground truth"
"$EVAL_PY" -m prowbench_eval.gt_export --dataset "$DATA/$dataset" --out "$WORK/gt" --first-person-filter auto

echo "=== 4.2 samples"
"$EVAL_PY" -m prowbench_eval.make_samples --dataset "$DATA/$dataset" --videos "$VIDEOS/$dataset/$method" --method "$method" \
  --representations "$representation" --out "$WORK/samples.jsonl" --gt-dir "$WORK/gt" --work "$WORK"

echo "=== 4.3 tracking (SAM 3, SAM 2 as fallback)"
"$EVAL_PY" -m prowbench_eval.detect_track --jobs "$WORK/detect_jobs.json" --tracker sam3 --model facebook/sam3 \
  --sam3-score-thresh 0.2 --sam3-prompts '{"obstacle": "barricade", "cable": "tow cable", "weapon": "rifle"}' \
  --sam3-fallback sam2 --sam2-model facebook/sam2.1-hiera-large --grid-hz 8 --device cuda:0

if skipping camera; then
  echo "=== 4.4 camera estimation: skipped"
else
  echo "=== 4.4 camera estimation (ViPE)"
  "$vipe_root/.venv/bin/python" -m prowbench_eval.estimate_camera --jobs "$WORK/camera_jobs.json" --backend vipe --vipe-root "$vipe_root" \
    --vipe-pipeline pose_only --device cuda:0
fi

echo "=== 4.5 entity and camera metrics"
while read -r row; do
  key=$("$EVAL_PY" -c 'import json,sys; print(json.loads(sys.argv[1])["sample_id"].replace("/", "__"))' "$row")
  ep=$("$EVAL_PY" -c 'import json,sys; r=json.loads(sys.argv[1]); print(r["episode"] + "/" + r["camera"])' "$row")
  fps=$("$EVAL_PY" -c 'import json,sys; print(json.loads(sys.argv[1])["fps"])' "$row")
  "$EVAL_PY" -m prowbench_eval.metric_entity --gt "$WORK/gt/$ep.json" --prediction "$WORK/preds/$key.json" --sample "$row" \
    --out "$WORK/metrics/$key.entity.json"
  if ! skipping camera; then
    "$EVAL_PY" -m prowbench_eval.metric_camera --gt "$WORK/gt/$ep.json" --pred-traj "$WORK/camera_traj/$key.json" --pred-fps "$fps" \
      --protocol cami2v --out "$WORK/metrics/$key.camera.json"
  fi
done < "$WORK/samples.jsonl"

echo "=== 4.6 visual quality, warp error, CLIP-Score (quality environment)"
"$QUALITY_PY" -m prowbench_eval.metric_vbench prepare --samples "$WORK/samples.jsonl" --out-dir "$WORK/vbench_input"
VBENCH="$PROWBENCH/.venv-quality/bin/vbench"
"$VBENCH" evaluate --videos_path "$WORK/vbench_input" --mode=custom_input --output_path "$WORK/vbench_out" \
  --load_ckpt_from_local true --dimension "imaging_quality aesthetic_quality"
"$VBENCH" evaluate --videos_path "$WORK/vbench_input" --mode=custom_input --output_path "$WORK/vbench_ms_out" \
  --load_ckpt_from_local true --dimension "motion_smoothness"
"$QUALITY_PY" -m prowbench_eval.metric_vbench summarise --vbench-json "$(ls "$WORK"/vbench_out/*_eval_results.json | head -1)" \
  --mapping "$WORK/vbench_input/mapping.json" --out-dir "$WORK/vbench"
"$QUALITY_PY" -m prowbench_eval.metric_vbench summarise --vbench-json "$(ls "$WORK"/vbench_ms_out/*_eval_results.json | head -1)" \
  --mapping "$WORK/vbench_input/mapping.json" --out-dir "$WORK/vbench_ms"
"$QUALITY_PY" -m prowbench_eval.metric_temporal --samples "$WORK/samples.jsonl" --out-dir "$WORK/metrics" --raft "$WEIGHTS/models/raft-things.pth" --device cuda:0
"$QUALITY_PY" -m prowbench_eval.metric_clipscore --samples "$WORK/samples.jsonl" --out-dir "$WORK/metrics" --index "$DATA/$dataset" \
  --clip "$WEIGHTS/ViT-B-32.pt" --device cuda:0

if skipping judge; then
  echo "=== 4.7 interaction and alignment (judge): skipped"
else
  echo "=== 4.7 interaction success rate and logic-render alignment (judge, Qwen3.6-27B)"
  "$EVAL_PY" -c 'import json, sys; json.dump(dict(dataset=sys.argv[1], output=sys.argv[2]), open(sys.argv[3], "w"))' \
    "$DATA/$dataset" "$WORK/registry" "$WORK/registry.json"
  "$EVAL_PY" -m prowbench_eval.event_registry --config "$WORK/registry.json"
  "$EVAL_PY" -m prowbench_eval.windowed_judging interaction --samples "$WORK/samples.jsonl" --gt-dir "$WORK/gt" \
    --protocol "protocol/$dataset/settings.json" --registry "$WORK/registry/registry.json" \
    --visual-events "protocol/$dataset/events.json" --out "$WORK/judge"
  "$EVAL_PY" -m prowbench_eval.windowed_judging alignment --samples "$WORK/samples.jsonl" --gt-dir "$WORK/gt" \
    --protocol "protocol/$dataset/settings.json" --descriptions "protocol/$dataset/descriptions.json" --out "$WORK/judge"
fi

echo "=== 4.8 table"
"$EVAL_PY" -m prowbench_eval.report_tables rows --samples "$WORK/samples.jsonl" --metrics-dir "$WORK/metrics" \
  --judge-dir "$WORK/judge" --vbench "$WORK/vbench/per_sample.jsonl" --vbench-motion "$WORK/vbench_ms/per_sample.jsonl" \
  --out "$WORK/rows.jsonl"
"$EVAL_PY" -m prowbench_eval.report_tables table --rows "$WORK/rows.jsonl" --dataset "$DATA/$dataset" --out "$WORK/table.json"
echo "table: $WORK/table.json"
