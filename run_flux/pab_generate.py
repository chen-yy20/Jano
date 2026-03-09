#!/usr/bin/env python3
# Copyright 2024-2025 Flux PAB Implementation

import argparse
import logging
import os
import sys
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
flux_src_path = os.path.join(os.path.dirname(current_dir), 'flux', 'src')
sys.path.insert(0, flux_src_path)
sys.path.insert(0, current_dir)

import torch
# from diffusers import FluxPipeline, DiffusionPipeline
from flux.pab.pab_pipeline_flux import FluxPipeline_pab
from flux.pab.pab_transformer_flux import FluxTransformer2DModel_pab
from flux.pab.pab_manager import init_pab_manger
from flux.pipeline_flux import FluxPipeline
from flux.transformer_flux import FluxTransformer2DModel

from utils.timer import init_timer, get_timer, print_time_statistics, save_time_statistics_to_file, disable_timing, enable_timing, get_time_statistics_dict
from utils.quality_metric import evaluate_quality_with_origin
from utils.results import save_params_and_metrics
from jano.stuff import get_prompt_id

MODEL_PATH = os.getenv("MODEL_PATH", "./Flux-1")
PROMPT = "A photorealistic cute cat, wearing a simple blue shirt, standing against a clear sky background."
WARMUP = 3
SELF_RANGE = 8
ENABLE_PAB = 1
TAG = f"w{WARMUP}s{SELF_RANGE}" if ENABLE_PAB else "ori"
OUTPUT_DIR = f"./results/flux/pab/{get_prompt_id(PROMPT)}"

torch.backends.cuda.enable_cudnn_sdp(False)

init_timer()
warmup = 3
N = 4  # 重复生成次数
SEEDS = [42 + i for i in range(N)]

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Flux Image Generation with PAB Acceleration')
    
    # Basic parameters
    parser.add_argument('--model_path', type=str, default="/home/zlq/diffusion/flux/model",
                        help='Path to Flux model')
    parser.add_argument('--prompt', type=str, default='A cat holding a sign that says hello world',
                        help='Text prompt')
    parser.add_argument('--height', type=int, default=1024,
                        help='Image height')
    parser.add_argument('--width', type=int, default=1024,
                        help='Image width')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=3.5,
                        help='Guidance scale')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--max_sequence_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='Output directory')
    parser.add_argument('--enable_cpu_offload', action='store_true', default=False,
                        help='Enable CPU offload to save VRAM')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR,
                        help='Output directory')
    
    args = parser.parse_args()
    args.model_path = MODEL_PATH
    args.prompt = PROMPT
    args.output_dir = OUTPUT_DIR
    args.seed = 42
    
    return args


def main():
    setup_logging()
    args = parse_args()
    
    logging.info("="*60)
    logging.info("Flux Image Generation with PAB")
    logging.info("="*60)
    
    # ==================== Load Model ====================
    logging.info(f"\nLoading Flux model from: {args.model_path}")
    
    try:
        if ENABLE_PAB:
            init_pab_manger(args.num_inference_steps, SELF_RANGE, WARMUP)
            pipe = FluxPipeline_pab.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
            pipe.transformer = FluxTransformer2DModel_pab.from_pretrained(f"{MODEL_PATH}/transformer", torch_dtype=torch.bfloat16)
        else:
            pipe = FluxPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
            pipe.transformer = FluxTransformer2DModel.from_pretrained(f"{MODEL_PATH}/transformer", torch_dtype=torch.bfloat16)
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        logging.error("Please ensure the model path is correct and the model is downloaded.")
        return
    
    # Enable CPU offload if requested
    if args.enable_cpu_offload:
        pipe.enable_model_cpu_offload()
        logging.info("✓ CPU offload enabled")
    else:
        pipe.to("cuda")
        logging.info("✓ Model loaded to CUDA")
    
    
    # ==================== Generate Image ====================
    logging.info(f"\nGeneration Configuration:")
    logging.info(f"  Prompt: {args.prompt}")
    logging.info(f"  Resolution: {args.width}x{args.height}")
    logging.info(f"  Inference Steps: {args.num_inference_steps}")
    logging.info(f"  Guidance Scale: {args.guidance_scale}")
    logging.info(f"  Seed: {args.seed}")
    
    logging.info(f"\nStarting generation...")
    
    # Generate
    warmup_generator = torch.Generator("cuda").manual_seed(SEEDS[0])
    disable_timing()
    for _ in range(warmup):
        image = pipe(
            args.prompt,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            max_sequence_length=args.max_sequence_length,
            generator=warmup_generator,
        ).images[0]

    # ==================== Save Image ====================
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    enable_timing()
    all_quality_metrics = {}
    for seed in SEEDS:
        generator = torch.Generator("cuda").manual_seed(seed)
        image = pipe(
            args.prompt,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            max_sequence_length=args.max_sequence_length,
            generator=generator,
        ).images[0]
        args.output = os.path.join(OUTPUT_DIR, f"{TAG}_{get_prompt_id(PROMPT)}_seed{seed}.png")
        image.save(args.output)
        logging.info(f"Stored {args.output}!")
        quality_result = None
        if ENABLE_PAB:
            baseline_path = os.path.abspath(args.output).replace("/pab/", "/ori/").replace(f"{TAG}_", "ori_")
            quality_result = evaluate_quality_with_origin(
                os.path.abspath(args.output),
                TAG,
                save_metrics=False,
                baseline_path=baseline_path,
            )
        all_quality_metrics[f"seed{seed}"] = quality_result.get("metrics") if quality_result else None

    # 保存参数与指标（统一 JSON，不再单独保存 time_stats.txt / quality_metrics.json）
    params = {
        "model": "flux",
        "method": "pab",
        "model_path": MODEL_PATH,
        "prompt": PROMPT,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seeds": SEEDS,
        "enable_pab": ENABLE_PAB,
        "warmup": WARMUP,
        "self_range": SELF_RANGE,
    }
    params_path = save_params_and_metrics(OUTPUT_DIR, TAG, params, get_time_statistics_dict(), all_quality_metrics)
    logging.info(f"Params & metrics saved to {params_path}")
    
    # ==================== Output Statistics ====================
    logging.info(f"\n" + "="*60)
    logging.info(f"Generation Complete!")
    logging.info(f"="*60)
    logging.info(f"  Image saved to: {args.output}")
    # logging.info(f"  Generation time: {generation_time:.2f} seconds")


if __name__ == '__main__':
    main()
    print_time_statistics()

