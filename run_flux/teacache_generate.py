from typing import Any, Dict, Optional, Tuple, Union
# from diffusers import DiffusionPipeline
# from diffusers.models import FluxTransformer2DModel
from flux.pipeline_flux import FluxPipeline
from flux.transformer_flux import FluxTransformer2DModel
from flux.teacache.tea_transformer import teacache_forward
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
import torch
import numpy as np
import argparse
import os
import sys
from contextlib import nullcontext

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.timer import init_timer, get_timer, print_time_statistics, save_time_statistics_to_file, disable_timing, enable_timing, get_time_statistics_dict
from utils.quality_metric import evaluate_quality_with_origin
from utils.results import save_params_and_metrics
from jano.stuff import get_prompt_id

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


MODEL_PATH = os.getenv("MODEL_PATH", "./Flux-1")
PROMPT = "A photorealistic cute cat, wearing a simple blue shirt, standing against a clear sky background."
# Teacache 关键参数 
# # 0.25 for 1.5x speedup, 0.4 for 1.8x speedup, 0.6 for 2.0x speedup, 0.8 for 2.25x speedup
THRESH = 0.2

ENABLE_TEACACHE = 1
TAG = f"TEA{THRESH}" if ENABLE_TEACACHE else "ori"
OUTPUT_DIR = f"./results/flux/teacache/{get_prompt_id(PROMPT)}"

init_timer()
warmup = 3
N = 4  # 重复生成次数
SEEDS = [42 + i for i in range(N)]
    
FluxTransformer2DModel.forward = teacache_forward

num_inference_steps = 50
parser = argparse.ArgumentParser()
# parser.add_argument("--model_path", type=str, required=True)
# parser.add_argument("--prompt", type=str, required=True)
# parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

args.model_path = MODEL_PATH
args.prompt = PROMPT
args.output_dir = OUTPUT_DIR
args.seed = 42

if torch.cuda.is_available() and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
    torch.backends.cuda.enable_cudnn_sdp(False)


def _sdpa_compat_context():
    if torch.cuda.is_available() and hasattr(torch.backends.cuda, "sdp_kernel"):
        return torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=True,
            enable_mem_efficient=True,
            enable_cudnn=False,
        )
    return nullcontext()

pipeline = FluxPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
pipeline.transformer = FluxTransformer2DModel.from_pretrained(f"{MODEL_PATH}/transformer", torch_dtype=torch.bfloat16)
# pipeline.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

# TeaCache
pipeline.transformer.__class__.enable_teacache = ENABLE_TEACACHE
pipeline.transformer.__class__.cnt = 0
pipeline.transformer.__class__.num_steps = num_inference_steps
pipeline.transformer.__class__.rel_l1_thresh = THRESH # 0.25 for 1.5x speedup, 0.4 for 1.8x speedup, 0.6 for 2.0x speedup, 0.8 for 2.25x speedup
pipeline.transformer.__class__.accumulated_rel_l1_distance = 0
pipeline.transformer.__class__.previous_modulated_input = None
pipeline.transformer.__class__.previous_residual = None


pipeline.to("cuda")
warmup_generator = torch.Generator("cuda").manual_seed(SEEDS[0])
disable_timing()
for _ in range(warmup):
    pipeline.transformer.__class__.cnt = 0
    pipeline.transformer.__class__.accumulated_rel_l1_distance = 0
    pipeline.transformer.__class__.previous_modulated_input = None
    pipeline.transformer.__class__.previous_residual = None
    with _sdpa_compat_context():
        img = pipeline(
            args.prompt,
            num_inference_steps=num_inference_steps,
            height=1024,
            width=1024,
            generator=warmup_generator,
            guidance_scale=3.5,
            max_sequence_length=512,
        ).images[0]

os.makedirs(OUTPUT_DIR, exist_ok=True)
enable_timing()
all_quality_metrics = {}
for seed in SEEDS:
    pipeline.transformer.__class__.cnt = 0
    pipeline.transformer.__class__.accumulated_rel_l1_distance = 0
    pipeline.transformer.__class__.previous_modulated_input = None
    pipeline.transformer.__class__.previous_residual = None
    generator = torch.Generator("cuda").manual_seed(seed)
    with _sdpa_compat_context():
        img = pipeline(
            args.prompt,
            num_inference_steps=num_inference_steps,
            height=1024,
            width=1024,
            generator=generator,
            guidance_scale=3.5,
            max_sequence_length=512,
        ).images[0]
    args.output = os.path.join(OUTPUT_DIR, f"{TAG}_{get_prompt_id(PROMPT)}_seed{seed}.png")
    img.save(args.output)
    print(f"Stored {args.output}!", flush=True)
    quality_result = None
    if ENABLE_TEACACHE:
        baseline_path = os.path.abspath(args.output).replace("/teacache/", "/ori/").replace(f"{TAG}_", "ori_")
        quality_result = evaluate_quality_with_origin(
            args.output,
            TAG,
            save_metrics=False,
            baseline_path=baseline_path,
        )
    all_quality_metrics[f"seed{seed}"] = quality_result.get("metrics") if quality_result else None

print_time_statistics()

# 保存参数与指标（统一 JSON）
params = {
    "model": "flux",
    "method": "teacache",
    "model_path": MODEL_PATH,
    "prompt": PROMPT,
    "height": 1024,
    "width": 1024,
    "num_inference_steps": num_inference_steps,
    "guidance_scale": 3.5,
    "seeds": SEEDS,
    "enable_teacache": ENABLE_TEACACHE,
    "thresh": THRESH,
}
params_path = save_params_and_metrics(OUTPUT_DIR, TAG, params, get_time_statistics_dict(), all_quality_metrics)
print(f"Params & metrics saved to {params_path}", flush=True)



