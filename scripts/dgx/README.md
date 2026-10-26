# PROWBench on the DGX (GB300, aarch64)

Scripts that set up PROWBench, generate videos with EdenLive and Matrix-Game 4.0, and run the evaluation on `pmgb300ws-0304` (one NVIDIA GB300, compute capability 10.3, Python 3.11 environments, CUDA 13.2 toolkit under `/usr/local/cuda-13.2`). All paths are under `~/PROWBench` unless noted. Code reaches the DGX through git (`git pull` on the branch), not by copying files.

## Scripts

| File | Purpose |
|---|---|
| `common.sh` | Shared paths and checks. Sourced by the others; override any path from the environment. |
| `setup.sh` | Evaluation environment, dataset, prepared evaluation data, RAFT and CLIP weights, SAM 2.1 and the judge model. Safe to run again. |
| `setup_quality_env.sh` | The separate VBench environment (`.venv-quality`). |
| `setup_vipe.sh` | ViPE v1.2.0 build for the camera metrics (`~/vipe-1.2.0`). |
| `run_method.sh` | Generates videos with `edenlive` or `matrix4` (wraps `scripts/generate_videos.py`). |
| `run_eval.sh` | Runs evaluation stages 4.1 to 4.8 of `eval/USAGE.md` for one dataset, method and representation. |

## Folders

| Path | Content |
|---|---|
| `PROWBench/` (inside the repo) | the Hugging Face dataset, about 2.3 GB |
| `prowbench-eval-data/` | prepared evaluation data: `verified_ff_40` (120 samples), `unverified_ff_90` (270), `gunfight` (60), `long_horizon` (30), `multi4` (120) |
| `weights/` | RAFT and CLIP (links to the WBench copies when they exist) |
| `model_output_video/<dataset>/<method>/<episode>/<camera>/<representation>/video.mp4` | generated videos, the layout the evaluation reads |
| `work/<dataset>/<method>/<representation>/` | evaluation results; the table is `table.json` |
| `.venv`, `.venv-quality` | evaluation and visual-quality environments |

## Order of use

1. `bash scripts/dgx/setup.sh`
2. `bash scripts/dgx/setup_quality_env.sh`
3. `bash scripts/dgx/setup_vipe.sh` (only for the camera metrics)
4. `bash scripts/dgx/run_method.sh edenlive verified_ff_40 mixed --hold-last-frame`
5. `bash scripts/dgx/run_eval.sh verified_ff_40 edenlive mixed --vipe-root ~/vipe-1.2.0 --skip motion`

Check the GPU first with `nvidia-smi`; the GPU is shared. Finished samples are skipped on a second run, and the evaluation stages that save per-sample results continue where they stopped.

## Generating videos

`run_method.sh <edenlive|matrix4> <dataset> <white|mixed|colored-obb|all> [options]`. Options: `--limit N`, `--episode ID`, `--overwrite`, `--hold-last-frame`, and for Matrix-Game `--scene` (default `racing`) and `--preset` (default `baseline`).

| | EdenLive | Matrix-Game 4.0 |
|---|---|---|
| Input per sample | proxy video, prompt, first frame | proxy video, prompt |
| Skips | samples without a first frame | none |
| Output length for a 5 s clip | 117 frames at 24 fps = 4.875 s | follows the input |
| Speed measured | 5.6 fps without model load, 129 frames requested (1 sample, GB300) | not run |

EdenLive's few-step model needs a length of 12n-3 frames, and a 150-frame, 30 fps proxy resamples to 120 frames at 24 fps, so 117 is the most it can use. The evaluation needs the full 5 s. `--hold-last-frame` repeats the last frame to reach 5.0 s (0.125 s added); without it the run log records a warning per sample. `long_horizon` is not supported by the runner (it uses per-chunk JSON prompts).

## Evaluation stages and skips

| Stage | What | Needs | Skip option |
|---|---|---|---|
| 4.1, 4.2 | ground truth, samples | none | |
| 4.3 | tracking (SAM 3, SAM 2.1 as fallback) | `facebook/sam3` access (gated, approved manually) and `HF_TOKEN` in the shell | |
| 4.4 | camera estimation (ViPE) | `--vipe-root ~/vipe-1.2.0` | `--skip camera` |
| 4.5 | entity and camera metrics (IoU, CErr, Miss, Rot, Trans, CamMC) | 4.3, 4.4 | |
| 4.6 | Img, Aes (VBench), Smooth (VBench motion smoothness), Warp (RAFT), CLIP | quality environment | `--skip motion` leaves out Smooth |
| 4.7 | ISR, LRA (judge Qwen3.6-27B, vision-language, about 80 GB of free GPU memory) | free GPU memory | `--skip judge` |
| 4.8 | table | the above | |

Only the three five-second datasets are handled. `long_horizon` and `multi4` have their own stages in `eval/USAGE.md`.

## Environments

| Environment | Python | torch | Why |
|---|---|---|---|
| `.venv` (evaluation) | 3.11 | 2.13.0+cu130, torchvision 0.28.0 | the README pins 2.10.0; 2.10.0+cu130 has no `sm_103` kernels and a `Linear` layer with a bias fails with `CUBLAS_STATUS_NOT_INITIALIZED` on this GPU |
| `.venv-quality` | 3.11 | 2.11.0+cu130 | pinned for VBench 0.1.5; passes the same bias test; numpy 1.26.4, transformers 4.33.2, opencv-python 4.11.0.86 |
| `~/vipe-1.2.0/.venv` | 3.11 | 2.14.1+cu132 | the extension is compiled with nvcc 13.2, so torch is the cu132 build |

## Constraints that shape the setup

- **decord on aarch64:** no wheel exists and FFmpeg 6.1 is too new to build it. `vbench` depends on decord, so it is installed with `--no-deps`, and `tools/decord_cv2` provides `import decord` (VideoReader on OpenCV, no dependencies, so the pinned numpy and opencv stay). VBench imports decord at the top of `vbench/utils.py` but only calls it in dimensions PROWBench does not run. The shim returns the same frames as OpenCV (tested on a PROWBench episode).
- **vbench command:** it starts its workers with the `python` on `PATH` and exits 0 when they fail to start, so `run_eval.sh` puts `.venv-quality/bin` first on `PATH` and stops when no `*_eval_results.json` appears.
- **VBench dependencies skipped by `--no-deps`:** `omegaconf`, `easydict` and `scikit-learn` are installed (pins frozen). `fvcore`, `fairscale`, `lvis`, `boto3`, `pycocoevalcap`, `cython` are not installed because the three dimensions do not import them (not confirmed for every code path).
- **ViPE:** the ViPE fork inside the MIND checkout predates v1.2.0 (no `vipe.config` package), which PROWBench imports. `setup_vipe.sh` builds the upstream tag (`github.com/nv-tlabs/vipe`) separately and leaves MIND alone.
- **Gated model:** `facebook/sam3` needs access approved on Hugging Face; export `HF_TOKEN` in the shell that runs `run_eval.sh` and never write it to a file.

## Models

| Model | Where | Size |
|---|---|---|
| Qwen/Qwen3.6-27B (judge) | Hugging Face cache | 55.6 GB |
| facebook/sam2.1-hiera-large | Hugging Face cache | 1.8 GB |
| facebook/sam3 | Hugging Face cache, after access | 6.9 GB |
| RAFT, CLIP ViT-B/32 | `weights/` | about 0.5 GB |
| VBench weights | `vbench-cache/` | downloaded on first use |

## Known gaps

- **Motion smoothness (Smooth):** VBench's AMT checkpoint holds a `typing.OrderedDict`, which torch 2.11's weights-only loader refuses (`Weights only load failed`). `tools/vbench_safe_globals/sitecustomize.py` allows that one type and `run_eval.sh` puts it on `PYTHONPATH` for the vbench calls; `weights_only` stays on. Loading was checked on `amt-s.pth`; the full motion-smoothness stage has not been run yet, so `--skip motion` remains available.
- **Untested end to end:** `report_tables` without the camera or judge results; the Matrix-Game runner (scene and preset are guesses; `baseline` is meant to be the unoptimized preset); the judge stage.
- **Scores vs reported:** the torch versions differ from the README pins (see Environments), so numbers may differ slightly from the reported scores.
