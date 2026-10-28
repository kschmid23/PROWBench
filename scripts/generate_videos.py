"""Generate PROWBench videos with EdenLive, Matrix-Game 4.0 or Cosmos 3 from the prepared evaluation data.

For every sample of `<data>/<dataset>/samples.jsonl` (one view of one episode in one representation) the chosen method renders the
proxy video and the result is written where the evaluation reads it:

  <videos>/<dataset>/<method>/<episode>/<camera>/<representation>/video.mp4

Samples whose output already exists are skipped, so an interrupted run can be started again. Each run appends one line per sample to
<videos>/<dataset>/<method>/run_log.jsonl.

  edenlive  needs the first frame of the sample (the photoreal opening frame): samples without one are skipped and counted.
            One EdenLive sample folder per view is written under <edenlive-root>/examples/prowbench/<dataset>/.
  matrix4   renders the proxy video with one of the scene models (castle, ancient-egypt, racing); the prompt is the sample prompt.
  cosmos3   Cosmos 3 video transfer with the blur control (the sample prompt is adapted for it: --cosmos3-prompt): the proxy video is resampled to 24 fps, padded to the model's 121 frames
            (5.04 s), blurred, and sent with the sample prompt to a worker (scripts/cosmos3_worker.py) that runs in the Cosmos 3
            diffusers environment under <cosmos3-root> and keeps the model loaded between samples. Sampling values come from
            <cosmos3-root>/cookbooks/cosmos3/generator/transfer/specs/blur.json.
"""
import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPRESENTATIONS = ("white", "mixed", "colored-obb")


def read_samples(dataset_dir: Path) -> list[dict]:
    with open(dataset_dir / "samples.jsonl", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_prompt(dataset_dir: Path, sample: dict) -> str:
    path = dataset_dir / sample["prompt"]
    if path.suffix == ".json":
        raise ValueError(f"{path.name}: per-chunk JSON prompts (long-horizon) are not supported by this runner")
    return path.read_text(encoding="utf-8").strip()


def edenlive_frames(duration_seconds: float) -> int:
    """Smallest few-step length 12n-3 whose 24 fps playback covers the clip."""
    n = math.ceil((duration_seconds * 24 + 3) / 12)
    return 12 * n - 3


def sample_id(sample: dict) -> str:
    return f"{sample['episode']}__{sample['camera']}__{sample['representation']}"


def video_duration(path: Path) -> float:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                            capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def hold_last_frame(path: Path, required_seconds: float) -> float:
    """Extends a video that is shorter than the clip by repeating its last frame. Returns the seconds added."""
    missing = required_seconds - video_duration(path)
    if missing <= 0:
        return 0.0
    extended = path.with_suffix(".held.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-vf", f"tpad=stop_mode=clone:stop_duration={missing:.4f}",
                    "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(extended)], check=True)
    extended.replace(path)
    return missing


def render_edenlive(args: argparse.Namespace, dataset_dir: Path, sample: dict, prompt: str, target: Path) -> None:
    if not sample.get("firstFrame") or not (dataset_dir / sample["firstFrame"]).is_file():
        raise FileNotFoundError("no first frame for this sample (EdenLive needs the photoreal opening frame)")
    root = Path(args.edenlive_root)
    identifier = sample_id(sample)
    relative = Path("examples") / "prowbench" / sample["dataset"] / identifier
    folder = root / relative
    recording = folder / identifier
    recording.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(dataset_dir / sample["video"], recording / "recording.mp4")
    shutil.copyfile(dataset_dir / sample["firstFrame"], recording / "cover.png")
    duration_ms = int(round(sample["durationSeconds"] * 1000))
    (recording / "prompts.json").write_text(json.dumps({"durationMs": duration_ms, "staticPrompt": prompt, "dynamicPrompts": []}, indent=2), encoding="utf-8")
    with open(folder / "manifest.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "prompt", "proxy_video", "first_frame"])
        writer.writerow([identifier, prompt, f"{identifier}/recording.mp4", f"{identifier}/cover.png"])

    out_relative = Path("inference") / "prowbench" / sample["dataset"] / identifier
    frames = args.frames if args.frames else edenlive_frames(sample["durationSeconds"])
    subprocess.run(["bash", "render_sample_ar.sh", "--samples", str(relative), "--frames", str(frames), "--out", str(out_relative)], cwd=root, check=True)
    candidates = [p for p in (root / out_relative).rglob("*.mp4") if "_proxy" not in p.name]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one rendered mp4 under {root / out_relative}, found {len(candidates)}")
    shutil.copyfile(candidates[0], target)


def render_matrix4(args: argparse.Namespace, dataset_dir: Path, sample: dict, prompt: str, target: Path) -> None:
    root = Path(args.matrix4_root)
    work = root / "outputs" / "prowbench" / sample["dataset"] / sample_id(sample)
    work.parent.mkdir(parents=True, exist_ok=True)
    rendered = work / "rendered.mp4"
    if work.is_dir() and not rendered.is_file():
        work.rmdir()  # matrix-game refuses an existing output directory; rmdir only removes an empty one left by a failed run
    if not work.exists():
        command = [str(root / ".venv" / "bin" / "matrix-game"), "render", "--scene", args.scene, "--input", str((dataset_dir / sample["video"]).resolve()),
                   "--resources", "resources", "--preset", args.preset, "--prompt-mode", "fixed", f"--prompt={prompt}", "--output", str(work)]
        subprocess.run(command, cwd=root, check=True)
    if not rendered.is_file():
        raise RuntimeError(f"matrix-game did not write {rendered}")
    shutil.copyfile(rendered, target)


COSMOS3_MARKER = "@@COSMOS3@@ "
COSMOS3_FRAMES = 121
COSMOS3_FPS = 24.0
COSMOS3_SPEC = Path("cookbooks") / "cosmos3" / "generator" / "transfer" / "specs" / "blur.json"
COSMOS3_NEGATIVE_PROMPT = Path("cookbooks") / "cosmos3" / "generator" / "transfer" / "assets" / "negative_prompt.json"
cosmos3_worker = None


def cosmos3_read_reply(worker: subprocess.Popen) -> dict:
    """Reads the worker's stdout up to its next reply line; every other line is the library's own output and goes to stderr."""
    for line in worker.stdout:
        if line.startswith(COSMOS3_MARKER):
            return json.loads(line[len(COSMOS3_MARKER):])
        sys.stderr.write(line)
    raise RuntimeError(f"the Cosmos 3 worker exited with code {worker.wait()} before answering")


def cosmos3_start(args: argparse.Namespace) -> subprocess.Popen:
    root = Path(args.cosmos3_root)
    python = root / ".venv-cosmos3-diffusers" / "bin" / "python"
    # The worker lives in another environment: the evaluation venv must not leak into it.
    env = {key: value for key, value in os.environ.items() if key not in ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH")}
    command = [str(python), str(Path(__file__).with_name("cosmos3_worker.py")), "--model", args.cosmos3_model,
               "--negative-prompt-file", str(root / COSMOS3_NEGATIVE_PROMPT)]
    worker = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    answer = cosmos3_read_reply(worker)
    if answer.get("status") != "ready":
        raise RuntimeError(f"the Cosmos 3 worker did not start: {answer}")
    return worker


def cosmos3_blur_control(source: Path, target: Path, width: int, height: int, factor: int, sigma: float) -> None:
    """Blur control video for the diffusers pipeline, which takes a ready-made control (it does not blur an input itself).

    Resamples to 24 fps, repeats the last frame until there are COSMOS3_FRAMES frames, downscales by `factor`, scales back up and
    applies a Gaussian blur. The strength is this runner's own choice: the diffusers path has no `preset_blur_strength`.
    """
    filters = (f"fps={COSMOS3_FPS:g},tpad=stop_mode=clone:stop={COSMOS3_FRAMES},scale=iw/{factor}:ih/{factor}:flags=area,"
               f"scale={width}:{height}:flags=bicubic,gblur=sigma={sigma}")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(source), "-vf", filters, "-frames:v", str(COSMOS3_FRAMES), "-an",
                    "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(target)], check=True)


# Words the Cosmos guardrail's blocklist rejects in this dataset's prompts (the check is a fuzzy word match, not a judgement of meaning): found by
# running the guardrail's own text check over every prompt. Each is replaced by a near synonym in the Cosmos 3 copy of the prompt only.
COSMOS3_REPLACEMENTS = (
    (r"\bsage green\b", "pale green"),
    (r"\bsage\b", "pale green"),
    (r"\bflat-colour\b", "solid-colour"),
    (r"\bflat-shaded\b", "solid-shaded"),
    (r"\bgrain\b", "texture"),
    (r"\bgrand\b", "imposing"),
    (r"\bbrick\b", "masonry"),
    (r"\branger(s?)\b", r"warden\1"),
    (r"\bbrake\b", "stop"),
    (r"\bthe knight\b", "the armoured figure"),
    (r"\bgritty\b", "rough"),
)


def cosmos3_replace(text: str) -> str:
    """Applies COSMOS3_REPLACEMENTS, keeping a leading capital letter."""
    for pattern, replacement in COSMOS3_REPLACEMENTS:
        def swap(found: re.Match) -> str:
            new = found.expand(replacement)
            return new[0].upper() + new[1:] if found.group(0)[0].isupper() else new
        text = re.sub(pattern, swap, text, flags=re.IGNORECASE)
    return text


COSMOS3_SECTIONS = ("subject_definitions", "summary", "retention_analysis", "detailed_description", "overall_soundscape", "non_diegetic_music")


def cosmos3_prompt(prompt: str) -> str:
    """The sample prompt rewritten for Cosmos 3 (this method only; the dataset prompts are not changed).

    The PROWBench prompt is written for a model that gets a proxy video and an anchor picture: it opens with a control block about
    <Video 1> and <Picture 1>, lists proxy and retention rules, and ends with soundscape and music lines. Cosmos 3 transfer gets only
    the blurred control video and this text, so the control block, subject and retention lists, soundscape and music are dropped. What stays is
    the shot description (style line, subjects and timeline), with the <Video 1> and <Picture 1> references reworded and the words the
    guardrail blocklist rejects swapped for synonyms (COSMOS3_REPLACEMENTS).
    """
    parts = re.split(rf"^({'|'.join(COSMOS3_SECTIONS)}):[ \t]*$", prompt, flags=re.M)
    sections = dict(zip(parts[1::2], (text.strip() for text in parts[2::2])))
    description = sections["detailed_description"]
    description = re.sub(r"The shot opens exactly on <Picture 1> and evolves continuously from that frame\.\s*", "", description)
    description = re.sub(r"; no proxy colours[^.]*\.", ".", description)
    description = re.sub(r"\s*Brushstrokes and painted edges belong to the target style; they never trace a proxy[^.]*\.", "", description)
    description = description.replace(", as seen in <Picture 1>", "").replace("<Video 1>", "the control video")
    description = re.sub(r"<Subject (\d+)>", r"Subject \1", description)
    description = description.replace("<Picture 1>", "the reference image")
    return cosmos3_replace(description)


def render_cosmos3(args: argparse.Namespace, dataset_dir: Path, sample: dict, prompt: str, target: Path) -> None:
    global cosmos3_worker
    if cosmos3_worker is None:
        cosmos3_worker = cosmos3_start(args)
    spec = json.loads((Path(args.cosmos3_root) / COSMOS3_SPEC).read_text(encoding="utf-8"))
    prompt = cosmos3_prompt(prompt) if args.cosmos3_prompt == "rewrite" else cosmos3_replace(prompt)
    target.with_name("cosmos3_prompt.txt").write_text(prompt, encoding="utf-8")
    control = target.with_name("cosmos3_control_blur.mp4")
    cosmos3_blur_control(dataset_dir / sample["video"], control, sample["width"], sample["height"], args.cosmos3_blur_factor, args.cosmos3_blur_sigma)
    job = {"control": "blur", "control_video": str(control), "prompt": prompt, "output": str(target), "height": sample["height"], "width": sample["width"],
           "num_frames": COSMOS3_FRAMES, "fps": COSMOS3_FPS, "num_steps": spec["num_steps"], "guidance": spec["guidance"],
           "control_guidance": spec["control_guidance"], "shift": spec["shift"], "seed": spec["seed"],
           "num_video_frames_per_chunk": spec["num_video_frames_per_chunk"], "num_conditional_frames": spec["num_conditional_frames"]}
    cosmos3_worker.stdin.write(json.dumps(job) + "\n")
    cosmos3_worker.stdin.flush()
    answer = cosmos3_read_reply(cosmos3_worker)
    control.unlink(missing_ok=True)
    if answer.get("status") != "ok":
        raise RuntimeError(f"Cosmos 3 failed on {sample_id(sample)}: {answer.get('error')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PROWBench videos with EdenLive, Matrix-Game 4.0 or Cosmos 3")
    parser.add_argument("--method", required=True, choices=("edenlive", "matrix4", "cosmos3"))
    parser.add_argument("--dataset", required=True, help="verified_ff_40, unverified_ff_90, gunfight, multi4 (long_horizon needs chunked prompts)")
    parser.add_argument("--representation", required=True, choices=REPRESENTATIONS + ("all",))
    parser.add_argument("--data", required=True, help="prowbench-eval-data")
    parser.add_argument("--videos", required=True, help="output root: <videos>/<dataset>/<method>/...")
    parser.add_argument("--limit", type=int, help="stop after this many rendered samples")
    parser.add_argument("--episode", help="only this episode id")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hold-last-frame", action="store_true", help="repeat the last frame when a video is shorter than the clip (EdenLive at 24 fps gives 4.875 s of a 5 s clip)")
    parser.add_argument("--edenlive-root", required=True)
    parser.add_argument("--frames", type=int, help="EdenLive length (12n-3); default covers the clip at 24 fps")
    parser.add_argument("--matrix4-root", required=True)
    parser.add_argument("--scene", default="racing", choices=("castle", "ancient-egypt", "racing"), help="Matrix-Game 4.0 scene model")
    parser.add_argument("--preset", default="baseline", choices=("realtime", "quality", "baseline", "rtx5090", "rtx5090-sage"), help="Matrix-Game 4.0 preset")
    parser.add_argument("--cosmos3-root", required=True, help="Cosmos checkout with .venv-cosmos3-diffusers")
    parser.add_argument("--cosmos3-model", default="nvidia/Cosmos3-Nano")
    parser.add_argument("--cosmos3-prompt", choices=("minimal", "rewrite"), default="minimal",
                        help="Cosmos 3 prompt: minimal = the sample prompt with only the guardrail-blocked words swapped; rewrite = also drop the proxy-control block (cosmos3_prompt)")
    parser.add_argument("--cosmos3-blur-factor", type=int, default=4, help="Cosmos 3: downscale factor of the blur control (larger = blurrier)")
    parser.add_argument("--cosmos3-blur-sigma", type=float, default=2.0, help="Cosmos 3: Gaussian sigma applied after scaling back up")
    args = parser.parse_args()

    dataset_dir = Path(args.data) / args.dataset
    samples = read_samples(dataset_dir)
    if args.representation != "all":
        samples = [s for s in samples if s["representation"] == args.representation]
    if args.episode:
        samples = [s for s in samples if s["episode"] == args.episode]
    assert samples, "no samples match"

    render = {"edenlive": render_edenlive, "matrix4": render_matrix4, "cosmos3": render_cosmos3}[args.method]
    log_dir = Path(args.videos) / args.dataset / args.method
    log_dir.mkdir(parents=True, exist_ok=True)
    done = skipped = failed = 0
    with open(log_dir / "run_log.jsonl", "a", encoding="utf-8") as log:
        for sample in samples:
            if args.limit is not None and done >= args.limit:
                break
            target = log_dir / sample["episode"] / sample["camera"] / sample["representation"] / "video.mp4"
            if target.is_file() and not args.overwrite:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            start = time.time()
            record = {"sample": sample_id(sample), "method": args.method}
            try:
                render(args, dataset_dir, sample, read_prompt(dataset_dir, sample), target)
                record.update(status="ok", seconds=round(time.time() - start, 1), video_seconds=round(video_duration(target), 3))
                if video_duration(target) < sample["durationSeconds"] - 1e-3:
                    if args.hold_last_frame:
                        record["held_last_frame_seconds"] = round(hold_last_frame(target, sample["durationSeconds"]), 3)
                        record["video_seconds"] = round(video_duration(target), 3)
                    else:
                        record["warning"] = f"video is shorter than the {sample['durationSeconds']} s clip; the evaluation needs the full clip (use --hold-last-frame)"
                done += 1
            except (FileNotFoundError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
                record.update(status="failed", error=str(error))
                failed += 1
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + "\n")
            log.flush()
    if cosmos3_worker is not None:
        cosmos3_worker.stdin.close()
        cosmos3_worker.wait()
    print(f"{args.method} {args.dataset}: {done} rendered, {skipped} already there, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
