"""Cosmos 3 video-transfer worker for scripts/generate_videos.py (method cosmos3).

Runs inside the Cosmos 3 diffusers environment (`<cosmos-root>/.venv-cosmos3-diffusers`), not in the evaluation environment. It loads the
pipeline once, then answers one job per line on stdin. Each job is a JSON object and gets one reply line on stdout, prefixed with
RESULT_MARKER so it can be told apart from the library's own prints:

  job:   {"control": "blur", "control_video": ".../control.mp4", "prompt": "...", "output": ".../out.mp4", "height": 720, "width": 1280,
          "num_frames": 121, "fps": 24.0, "num_steps": 50, "guidance": 3.0, "control_guidance": 1.5, "shift": 10.0, "seed": 2026,
          "num_video_frames_per_chunk": 121, "num_conditional_frames": 1}
  reply: {"status": "ok"} or {"status": "error", "error": "..."}

The sampling code follows `run_transfer` of cookbooks/cosmos3/generator/transfer/run_video_transfer_with_diffusers.ipynb. The safety checker
(guardrail) is off for edge and depth and on for the other controls, as in that notebook.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from diffusers import Cosmos3OmniModularPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_video

RESULT_MARKER = "@@COSMOS3@@ "
GUARDRAIL_DISABLED_CONTROLS = frozenset({"edge", "depth"})


def reply(payload: dict) -> None:
    sys.stdout.write(RESULT_MARKER + json.dumps(payload) + "\n")
    sys.stdout.flush()


def compact_json_file(path: Path) -> str:
    return json.dumps(json.loads(path.read_text()), ensure_ascii=True, separators=(",", ":"))


def run_job(pipe: Cosmos3OmniModularPipeline, negative_prompt: str, job: dict) -> None:
    if job["control"] in GUARDRAIL_DISABLED_CONTROLS:
        pipe.disable_safety_checker()
    else:
        pipe.enable_safety_checker()
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=job["shift"], use_karras_sigmas=False)
    generator = torch.Generator(device="cuda").manual_seed(job["seed"])
    videos = pipe(
        prompt=job["prompt"],
        negative_prompt=negative_prompt,
        control_videos={job["control"]: load_video(job["control_video"])},
        control_guidance=job["control_guidance"],
        num_frames=job["num_frames"],
        num_video_frames_per_chunk=job["num_video_frames_per_chunk"],
        num_conditional_frames=job["num_conditional_frames"],
        height=job["height"],
        width=job["width"],
        fps=job["fps"],
        num_inference_steps=job["num_steps"],
        guidance_scale=job["guidance"],
        generator=generator,
        output="videos",
    )
    export_to_video(videos, job["output"], fps=job["fps"], macro_block_size=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cosmos 3 transfer worker (see the module docstring)")
    parser.add_argument("--model", required=True, help="Hugging Face id, for example nvidia/Cosmos3-Nano")
    parser.add_argument("--negative-prompt-file", required=True)
    args = parser.parse_args()

    negative_prompt = compact_json_file(Path(args.negative_prompt_file))
    pipe = Cosmos3OmniModularPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipe.load_components(torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    reply({"status": "ready"})

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            run_job(pipe, negative_prompt, json.loads(line))
            reply({"status": "ok"})
        except Exception as error:  # a failed sample must not end the worker; the caller records it and goes on
            reply({"status": "error", "error": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    main()
