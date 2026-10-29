"""Worker that renders PROWBench proxy videos with the Wan2.2-Fun-5B-Control base model and keeps the model loaded between samples.

Run it with the Python of a VideoX-Fun environment and the VideoX-Fun checkout on PYTHONPATH (scripts/generate_videos.py starts it that way
for --method wan5b):

  PYTHONPATH=~/wan22/VideoX-Fun python scripts/wan5b_worker.py --videox-fun ~/wan22/VideoX-Fun --model-dir ~/PROWBench/weights/Wan2.2-Fun-5B-Control

It prints one reply line `@@WAN5B@@ {"status": "ready"}` when the model is loaded, then reads one JSON job per line on stdin and answers
`@@WAN5B@@ {"status": "ok"}` or `@@WAN5B@@ {"status": "failed", "error": ...}` after each. Every other line on stdout is the libraries' own output.
A job holds control_video, prompt, negative_prompt, output, width, height, num_frames, fps, steps, guidance, shift and seed.

The model is the full-step bidirectional Wan2.2-Fun-5B-Control (no Matrix-Game fine-tune, no distillation): it follows the control video and
the text. Based on VideoX-Fun examples/wan2.2_fun/predict_v2v_control_5b.py; TeaCache and CFG skipping are left off so the sampling is not approximated.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from videox_fun.models import AutoencoderKLWan3_8, AutoTokenizer, Wan2_2Transformer3DModel, WanT5EncoderModel
from videox_fun.pipeline import Wan2_2FunControlPipeline
from videox_fun.utils import apply_gpu_memory_mode, filter_kwargs, get_image_to_video_latent, get_video_to_video_latent, save_videos_grid

MARKER = "@@WAN5B@@ "
CONFIG = "config/wan2.2/wan_civitai_5b.yaml"


def reply(payload: dict) -> None:
    print(MARKER + json.dumps(payload), flush=True)


def load_pipeline(videox_fun: Path, model_dir: Path, memory_mode: str) -> tuple:
    device, dtype = torch.device("cuda"), torch.bfloat16
    config = OmegaConf.load(videox_fun / CONFIG)
    extra = config["transformer_additional_kwargs"]
    if extra.get("transformer_combination_type", "single") != "single":
        raise RuntimeError(f"{CONFIG} describes a two-transformer model; this worker runs the single-transformer 5B model")
    transformer = Wan2_2Transformer3DModel.from_pretrained(str(model_dir / extra.get("transformer_low_noise_model_subpath", "transformer")),
                                                           transformer_additional_kwargs=OmegaConf.to_container(extra), low_cpu_mem_usage=True, torch_dtype=dtype)
    vae = AutoencoderKLWan3_8.from_pretrained(str(model_dir / config["vae_kwargs"]["vae_subpath"]), additional_kwargs=OmegaConf.to_container(config["vae_kwargs"])).to(dtype)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir / config["text_encoder_kwargs"]["tokenizer_subpath"]))
    text_encoder = WanT5EncoderModel.from_pretrained(str(model_dir / config["text_encoder_kwargs"]["text_encoder_subpath"]),
                                                     additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]), low_cpu_mem_usage=True, torch_dtype=dtype).eval()
    scheduler = FlowMatchEulerDiscreteScheduler(**filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config["scheduler_kwargs"])))
    pipeline = Wan2_2FunControlPipeline(transformer=transformer, transformer_2=None, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=scheduler)
    apply_gpu_memory_mode(pipeline, memory_mode, device, dtype)
    return pipeline, vae, extra.get("boundary", 0.875)


def render(job: dict, pipeline, vae, boundary: float) -> None:
    ratio = vae.config.temporal_compression_ratio
    if (job["num_frames"] - 1) % ratio:
        raise ValueError(f"{job['num_frames']} frames is not {ratio}n + 1")
    size = [job["height"], job["width"]]
    with torch.no_grad():
        inpaint_video, inpaint_mask, _ = get_image_to_video_latent(None, None, video_length=job["num_frames"], sample_size=size)
        control, _, _, _ = get_video_to_video_latent(job["control_video"], video_length=job["num_frames"], sample_size=size, fps=job["fps"], ref_image=None)
        sample = pipeline(job["prompt"], num_frames=job["num_frames"], negative_prompt=job["negative_prompt"], height=size[0], width=size[1],
                          generator=torch.Generator(device="cuda").manual_seed(job["seed"]), guidance_scale=job["guidance"],
                          num_inference_steps=job["steps"], video=inpaint_video, mask_video=inpaint_mask, control_video=control,
                          control_camera_video=None, ref_image=None, boundary=boundary, shift=job["shift"]).videos
    Path(job["output"]).parent.mkdir(parents=True, exist_ok=True)
    save_videos_grid(sample, job["output"], fps=job["fps"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videox-fun", type=Path, required=True, help="VideoX-Fun checkout (its config/wan2.2 is read)")
    parser.add_argument("--model-dir", type=Path, required=True, help="Wan2.2-Fun-5B-Control folder: config.json, diffusion_pytorch_model.safetensors, Wan2.2_VAE.pth, models_t5_umt5-xxl-enc-bf16.pth, google/umt5-xxl")
    parser.add_argument("--memory-mode", default="model_full_load", help="VideoX-Fun GPU memory mode")
    args = parser.parse_args()

    start = time.time()
    pipeline, vae, boundary = load_pipeline(args.videox_fun, args.model_dir, args.memory_mode)
    print(json.dumps({"loaded_seconds": round(time.time() - start)}), file=sys.stderr, flush=True)
    reply({"status": "ready"})
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            render(json.loads(line), pipeline, vae, boundary)
        except Exception as error:  # the model stays loaded; the runner records the failure and goes on
            reply({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        else:
            reply({"status": "ok"})


if __name__ == "__main__":
    main()
