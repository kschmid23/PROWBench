"""Generate PROWBench videos with EdenLive or Matrix-Game 4.0 from the prepared evaluation data.

For every sample of `<data>/<dataset>/samples.jsonl` (one view of one episode in one representation) the chosen method renders the
proxy video and the result is written where the evaluation reads it:

  <videos>/<dataset>/<method>/<episode>/<camera>/<representation>/video.mp4

Samples whose output already exists are skipped, so an interrupted run can be started again. Each run appends one line per sample to
<videos>/<dataset>/<method>/run_log.jsonl.

  edenlive  needs the first frame of the sample (the photoreal opening frame): samples without one are skipped and counted.
            One EdenLive sample folder per view is written under <edenlive-root>/examples/prowbench/<dataset>/.
  matrix4   renders the proxy video with one of the scene models (castle, ancient-egypt, racing); the prompt is the sample prompt.
"""
import argparse
import csv
import json
import math
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PROWBench videos with EdenLive or Matrix-Game 4.0")
    parser.add_argument("--method", required=True, choices=("edenlive", "matrix4"))
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
    args = parser.parse_args()

    dataset_dir = Path(args.data) / args.dataset
    samples = read_samples(dataset_dir)
    if args.representation != "all":
        samples = [s for s in samples if s["representation"] == args.representation]
    if args.episode:
        samples = [s for s in samples if s["episode"] == args.episode]
    assert samples, "no samples match"

    render = render_edenlive if args.method == "edenlive" else render_matrix4
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
    print(f"{args.method} {args.dataset}: {done} rendered, {skipped} already there, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
