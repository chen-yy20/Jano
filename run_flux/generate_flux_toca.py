
from diffusers import FluxPipeline
from flux.toca_single_block import apply_toca_to_pipeline
import torch
import argparse
import os

from utils.timer import init_timer, print_time_statistics, get_timer

MODEL_PATH = "/home/fit/zhaijdcyy/WORK/models/Flux-1"
PROMPT = "A photorealistic cute cat, wearing a simple blue shirt, standing against a clear sky background."
OUTPUT_DIR = "./toca_flux_result/"
ENABLE_TOCA = True

init_timer()

parser = argparse.ArgumentParser()
# parser.add_argument("--model_path", type=str, required=True)
# parser.add_argument("--prompt", type=str, required=True)
# parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

args.model_path = MODEL_PATH
args.prompt = PROMPT
refine_prompt = PROMPT.replace(" ","_")[:20]
os.makedirs(OUTPUT_DIR, exist_ok=True)
tag = 'toca' if ENABLE_TOCA else 'origin'
args.output = os.path.join(OUTPUT_DIR, f"{tag}_{refine_prompt}.png")


# 加载模型
pipe = FluxPipeline.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16
).to("cuda")

# 启用ToCa
def callback_fn(pipe_obj, step_idx, timestep, callback_kwargs):
    patcher.update_step(step_idx)
    return callback_kwargs

num_steps = 50
patcher = apply_toca_to_pipeline(pipe, num_steps, enable=True)
if not ENABLE_TOCA:
    patcher.disable_toca()

with get_timer("pipeline"):
    image = pipe(
    prompt=args.prompt,
    num_inference_steps=num_steps,
    guidance_scale=3.5,
    callback_on_step_end=callback_fn,
    generator=torch.Generator("cuda").manual_seed(42)  # ← 必需
    ).images[0]

image.save(args.output)

print(f"{ENABLE_TOCA=}, result saved to {args.output}.")


print_time_statistics()