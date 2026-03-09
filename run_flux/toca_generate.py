
from flux.pipeline_flux import FluxPipeline
from flux.transformer_flux import FluxTransformer2DModel
from flux.toca_single_block import apply_toca_to_pipeline
import torch
import argparse
import os
from utils.timer import init_timer, get_timer, print_time_statistics, save_time_statistics_to_file, disable_timing, enable_timing, get_time_statistics_dict
from utils.quality_metric import evaluate_quality_with_origin
from utils.results import save_params_and_metrics
from jano.stuff import get_prompt_id

MODEL_PATH = os.getenv("MODEL_PATH", "./Flux-1")
PROMPT = "A photorealistic cute cat, wearing a simple blue shirt, standing against a clear sky background."
ENABLE_TOCA = 1
TAG = f"toca" if ENABLE_TOCA else "ori"
OUTPUT_DIR = f"./results/flux/toca/{get_prompt_id(PROMPT)}"

init_timer()
warmup = 3
N = 4  # 重复生成次数
SEEDS = [42 + i for i in range(N)]

parser = argparse.ArgumentParser()
# parser.add_argument("--model_path", type=str, required=True)
# parser.add_argument("--prompt", type=str, required=True)
# parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

args.model_path = MODEL_PATH
args.prompt = PROMPT
args.output_dir = OUTPUT_DIR
args.seed = 42

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

generator = torch.Generator("cuda").manual_seed(SEEDS[0])
disable_timing()
for _ in range(warmup):
    image = pipe(
        prompt=args.prompt,
        num_inference_steps=num_steps,
        guidance_scale=3.5,
        callback_on_step_end=callback_fn,
        generator=generator,
    ).images[0]

os.makedirs(OUTPUT_DIR, exist_ok=True)
enable_timing()
all_quality_metrics = {}
for seed in SEEDS:
    generator = torch.Generator("cuda").manual_seed(seed)
    image = pipe(
        prompt=args.prompt,
        num_inference_steps=num_steps,
        guidance_scale=3.5,
        callback_on_step_end=callback_fn,
        generator=generator,
    ).images[0]
    args.output = os.path.join(OUTPUT_DIR, f"{TAG}_{get_prompt_id(PROMPT)}_seed{seed}.png")
    image.save(args.output)
    print(f"Stored {args.output}!", flush=True)
    quality_result = None
    if ENABLE_TOCA:
        baseline_path = os.path.abspath(args.output).replace("/toca/", "/ori/").replace(f"{TAG}_", "ori_")
        quality_result = evaluate_quality_with_origin(
            args.output,
            TAG,
            save_metrics=False,
            baseline_path=baseline_path,
        )
    all_quality_metrics[f"seed{seed}"] = quality_result.get("metrics") if quality_result else None

print(f"{ENABLE_TOCA=}, results saved to {OUTPUT_DIR}.")
print_time_statistics()

# 保存参数与指标（统一 JSON）
params = {
    "model": "flux",
    "method": "toca",
    "model_path": MODEL_PATH,
    "prompt": PROMPT,
    "num_inference_steps": num_steps,
    "guidance_scale": 3.5,
    "seeds": SEEDS,
    "enable_toca": ENABLE_TOCA,
}
params_path = save_params_and_metrics(OUTPUT_DIR, TAG, params, get_time_statistics_dict(), all_quality_metrics)
print(f"Params & metrics saved to {params_path}", flush=True)