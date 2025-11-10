
from flux.pipeline_flux import FluxPipeline
from flux.transformer_flux import FluxTransformer2DModel
from flux.toca_single_block import apply_toca_to_pipeline
import torch
import argparse
import os
from utils.timer import init_timer, get_timer, print_time_statistics, save_time_statistics_to_file, disable_timing, enable_timing
from utils.quality_metric import evaluate_quality_with_origin
from jano.stuff import get_prompt_id

MODEL_PATH = "/home/fit/zhaijdcyy/WORK/models/Flux-1"
PROMPT = "A photorealistic cute cat, wearing a simple blue shirt, standing against a clear sky background."
ENABLE_TOCA = 1
TAG = f"toca" if ENABLE_TOCA else "ori"
OUTPUT_DIR = f"./results/tokencache_flux_result/{get_prompt_id(PROMPT)}"

init_timer()
warmup = 3

parser = argparse.ArgumentParser()
# parser.add_argument("--model_path", type=str, required=True)
# parser.add_argument("--prompt", type=str, required=True)
# parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

args.model_path = MODEL_PATH
args.prompt = PROMPT
args.output_dir = OUTPUT_DIR

# 加载模型
pipe = FluxPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
pipe.transformer = FluxTransformer2DModel.from_pretrained(f"{MODEL_PATH}/transformer", torch_dtype=torch.bfloat16)

pipe.to("cuda")
print("✓ Model loaded to CUDA", flush=True)

# 启用ToCa
def callback_fn(pipe_obj, step_idx, timestep, callback_kwargs):
    patcher.update_step(step_idx)
    return callback_kwargs

num_steps = 50
patcher = apply_toca_to_pipeline(pipe, num_steps, enable=True)
if not ENABLE_TOCA:
    patcher.disable_toca()

disable_timing()
for _ in range(warmup):
    image = pipe(
        prompt=args.prompt,
        num_inference_steps=num_steps,
        guidance_scale=3.5,
        callback_on_step_end=callback_fn,
        generator=torch.Generator("cuda").manual_seed(42)  # ← 必需
        ).images[0]

enable_timing()
image = pipe(
    prompt=args.prompt,
    num_inference_steps=num_steps,
    guidance_scale=3.5,
    callback_on_step_end=callback_fn,
    generator=torch.Generator("cuda").manual_seed(42)  # ← 必需
    ).images[0]

os.makedirs(OUTPUT_DIR, exist_ok=True)
args.output = os.path.join(OUTPUT_DIR, f"{TAG}_{get_prompt_id(PROMPT)}.png")
image.save(args.output)

print(f"{ENABLE_TOCA=}, result saved to {args.output}.")
print_time_statistics()
save_time_statistics_to_file(f"{OUTPUT_DIR}/{TAG}_time_stats.txt")

if ENABLE_TOCA:
    evaluate_quality_with_origin(args.output, TAG)