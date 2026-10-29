# PROWBench on the DGX (GB300, aarch64)

Scripts that set up PROWBench, generate videos with EdenLive and Matrix-Game 4.0, and run the evaluation on `pmgb300ws-0304` (one NVIDIA GB300, compute capability 10.3, Python 3.11 environments, CUDA 13.2 toolkit under `/usr/local/cuda-13.2`). All paths are under `~/PROWBench` unless noted. Code reaches the DGX through git (`git pull` on the branch), not by copying files.

## Scripts

| File | Purpose |
|---|---|
| `common.sh` | Shared paths and checks. Sourced by the others; override any path from the environment. |
| `setup.sh` | Evaluation environment, dataset, prepared evaluation data, RAFT and CLIP weights, SAM 2.1 and the judge model. Safe to run again. |
| `setup_quality_env.sh` | The separate VBench environment (`.venv-quality`). |
| `setup_vipe.sh` | ViPE v1.2.0 build for the camera metrics (`~/vipe-1.2.0`). |
| `setup_wan5b.sh` | Builds the Wan2.2-Fun-5B-Control folder (`weights/Wan2.2-Fun-5B-Control`, symbolic links to files already in the Matrix-Game-4 checkout, plus `config.json`) for the `wan5b` method. |
| `run_method.sh` | Generates videos with `edenlive`, `matrix4`, `cosmos3` or `wan5b` (wraps `scripts/generate_videos.py`). |
| `run_eval.sh` | Runs evaluation stages 4.1 to 4.8 of `eval/USAGE.md` for one dataset, method and representation. |
| `run_all.sh` | `run_method.sh` then `run_eval.sh` for one method; the evaluation starts only if no video failed. `bash scripts/dgx/run_all.sh matrix4 verified_ff_40 mixed` |

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

`run_method.sh <edenlive|matrix4|cosmos3|wan5b> <dataset> <white|mixed|colored-obb|all> [options]`. Options: `--limit N`, `--episode ID`, `--overwrite`, `--hold-last-frame`, and for Matrix-Game `--scene` (default `racing`; `base` = no scene fine-tune), `--matrix4-steps` and `--preset` (default `baseline`).

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

## Matrix-Game 4.0 checkpoints

The release ([Skywork/Matrix-Game-4.0](https://huggingface.co/Skywork/Matrix-Game-4.0)) has scene checkpoints only and no base video model:

| Folder | `--scene` | Content |
|---|---|---|
| `driving_game/` | `racing` | City Racing: Wan2.2-Fun-5B-Control causal, 864 x 480, fixed prompt about a sedan seen from behind on city streets |
| `fps-castle/` | `castle` | Medieval Castle |
| `fps-egypt/` | `ancient-egypt` | Ancient Egypt |
| `base/` | | shared codec files only (`taew2_2.pth`, 22.9 MB, the streaming TAEHV decoder); the Wan2.2 VAE, the T5 text encoder and the tokenizer in `resources/base` come from `alibaba-pai/Wan2.2-Fun-5B-Control` |

- **What PROWBench ran:** the `racing` scene checkpoint with `--preset baseline`. The preset changes denoising steps and codec, not the checkpoint, so the result is the racing scene model, not a base model. Label the row that way.
- **Lineage:** the Hugging Face model tree lists `Wan-AI/Wan2.2-TI2V-5B` as the base model and `alibaba-pai/Wan2.2-Fun-5B-Control` as a finetuned parent.
- **Base-model row:** use `wan5b`, not `--scene base`. The scene models are distilled, causal 2-step generators (`steps [1000, 500]`, guidance 1); `--scene base` puts the undistilled Wan2.2-Fun-5B-Control weights into that code path and gives noise (checked on 3 of 38 samples, and on an earlier taxi clip). `wan5b` runs the same weights the way VideoX-Fun runs them: 40 steps, guidance 6.0, seed 43 (the values of its example `predict_v2v_control_5b.py`), the proxy video as control at 24 fps padded to 121 frames, the sample prompt plus VideoX-Fun's own negative prompt, no reference image, no TeaCache; the result is trimmed to 120 frames (5.0 s). Run `bash scripts/dgx/setup_wan5b.sh` once, then `bash scripts/dgx/run_method.sh wan5b verified_ff_40 mixed --episode dv200-153-city-cat-encounter`. Options: `--wan5b-steps`, `--wan5b-guidance`, `--wan5b-seed`. Label the row "Wan2.2-Fun-5B-Control (the open base of Matrix-Game 4.0)"; it is not an official Matrix-Game base result. The runner is untested end to end.

## Known gaps

- **Motion smoothness (Smooth):** VBench's AMT checkpoint holds a `typing.OrderedDict`, which torch 2.11's weights-only loader refuses (`Weights only load failed`). `tools/vbench_safe_globals/sitecustomize.py` allows that one type and `run_eval.sh` puts it on `PYTHONPATH` for the vbench calls; `weights_only` stays on. The stage has run end to end for EdenLive (Smooth 0.983).
- **Untested end to end:** `report_tables` without the camera or judge results; the Matrix-Game scene and preset choice (`racing` and `baseline` are guesses; `baseline` is meant to be the unoptimized preset). The runner works: 40 of 40 videos rendered at about 70 s each, 5.0 s long. `matrix-game render` rejects an existing `--output` directory, so the runner removes empty leftovers and lets it create the directory.
- **Leaderboard Average and group scores:** the paper (arXiv 2610.02205) does not define them. The project-page leaderboard shows an Average that equals the plain mean of its six group scores (Entity, Logic / State, Camera, Quality, Temporal, Text), but how raw metrics map to those 0 to 100 group scores is not published in the paper, so no overall score is computed here. Compare methods per metric.
- **Scores vs reported:** the torch versions differ from the README pins (see Environments), so numbers may differ slightly from the reported scores.

## Reading the results

`work/<dataset>/<method>/<representation>/rows.jsonl` has one row per video; `table.json` and the final printed line are the mean over those rows. There is no single overall score. Definitions are in Appendix B of the paper (arXiv 2610.02205) and `eval/README.md`.

| Group | Metric | Better | Aggregation |
|---|---|---|---|
| Entity control | IoU | higher | mean box IoU over evaluated entities per clip, missing boxes count 0, then mean over clips |
| Entity control | CErr | lower | box-center distance divided by the image diagonal, over entities with at least one matched instant |
| Logic / State | ISR, LRA | higher | judged share of events (ISR) or timeline segments (LRA) per sample, mean over samples |
| Logic / State | gISR, gLRA | higher | per sample ISR or LRA times clip IoU, mean over samples |
| Camera control | Rot (degrees), Trans, CamMC | lower | ViPE pose-only camera against the recorded camera; Trans and CamMC use translations divided by their own maximum norm, so they measure path shape, not length |
| Quality | Img, Aes | higher | VBench imaging (MUSIQ) and aesthetic (LAION) |
| Quality | Warp | lower (error) | RAFT warp error between adjacent frames, x1000 |
| Temporal | Smooth | higher | VBench motion smoothness |
| Text | CLIP | higher | CLIP ViT-B/32 cosine similarity over 40 grid frames and the episode title, x100 |

Quality and temporal metrics measure plausibility, not adherence to the engine state.

### Verified-FF (40 pairs): EdenLive and the paper's Table 1

EdenLive is this repository's run on `mixed`; the other rows are the paper's Table 1, Verified-FF.

| Method | IoU | CErr | ISR | LRA | gISR | gLRA | Rot (deg) | Trans | CamMC | Img | Aes | Warp (x1e-3) | Smooth | CLIP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EdenLive (this run, mixed) | 0.477 | 0.038 | 0.588 | 0.668 | 0.288 | 0.323 | 5.3 | 0.239 | 0.290 | 68.0 | 0.607 | 0.483 | 0.983 | 24.9 |
| LynnReal-Omni | 0.518 | 0.037 | 0.621 | 0.832 | 0.327 | 0.429 | 2.1 | 0.074 | 0.101 | 73.8 | 0.625 | 1.258 | 0.982 | 25.2 |
| MiniMax-H3 | 0.494 | 0.038 | 0.669 | 0.748 | 0.330 | 0.372 | 3.4 | 0.080 | 0.133 | 70.1 | 0.633 | 0.444 | 0.991 | 25.2 |
| CWM | 0.455 | 0.040 | 0.637 | 0.795 | 0.290 | 0.360 | 2.0 | 0.130 | 0.154 | 66.7 | 0.629 | 0.398 | 0.991 | 25.3 |
| Seedance 2.5 | 0.433 | 0.043 | 0.652 | 0.832 | 0.280 | 0.358 | 3.6 | 0.101 | 0.152 | 66.9 | 0.629 | 0.313 | 0.993 | 25.3 |

Table 1 also lists EchoWM, SANA-WM, LingBot 2.0, AlayaWorld v1.1, Cosmos 2.5 and C2R. It lists each method once; the input each receives is described in the paper's Section 4.1, so a row is not necessarily the same representation as this run's `mixed`.
