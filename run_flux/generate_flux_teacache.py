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

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.timer import init_timer, get_timer, print_time_statistics, save_time_statistics_to_file, disable_timing, enable_timing
from utils.quality_metric import evaluate_quality_with_origin
from jano.stuff import get_prompt_id

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


MODEL_PATH = "/home/fit/zhaijdcyy/WORK/models/Flux-1"
PROMPT = "A photorealistic cute cat, wearing a simple blue shirt, standing against a clear sky background."
# Teacache 关键参数 
# # 0.25 for 1.5x speedup, 0.4 for 1.8x speedup, 0.6 for 2.0x speedup, 0.8 for 2.25x speedup
THRESH = 0.2

ENABLE_TEACACHE = 1
TAG = f"TEA{THRESH}" if ENABLE_TEACACHE else "ori"
OUTPUT_DIR = f"./flux_results/teacache_flux_result/{get_prompt_id(PROMPT)}"

init_timer()
warmup = 3
    
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
generator = torch.Generator("cuda").manual_seed(args.seed)
disable_timing()
for _ in range(warmup):
    img = pipeline(
        args.prompt, 
        num_inference_steps=num_inference_steps,
        height=1024,
        width=1024,
        generator=generator,
        guidance_scale=3.5,
        max_sequence_length=512,
        ).images[0]
enable_timing()
img = pipeline(
        args.prompt, 
        num_inference_steps=num_inference_steps,
        height=1024,
        width=1024,
        generator=generator,
        guidance_scale=3.5,
        max_sequence_length=512,
        ).images[0]

os.makedirs(OUTPUT_DIR, exist_ok=True)
args.output = os.path.join(OUTPUT_DIR, f"{TAG}_{get_prompt_id(PROMPT)}.png")
img.save(args.output)
print(f"Stored {args.output}!", flush=True)
print_time_statistics()
save_time_statistics_to_file(f"{OUTPUT_DIR}/{TAG}_time_stats.txt")

if ENABLE_TEACACHE:
    evaluate_quality_with_origin(args.output, TAG)



