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

from utils.timer import init_timer, get_timer, print_time_statistics, save_time_statistics_to_file, disable_timing, enable_timing
from utils.quality_metric import evaluate_quality_with_origin
from jano.stuff import get_prompt_id

MODEL_PATH = os.getenv("MODEL_PATH", "./Flux-1")
PROMPT = "A photorealistic cute cat, wearing a simple blue shirt, standing against a clear sky background."
WARMUP = 5
SELF_RANGE = 2
ENABLE_PAB = 1
TAG = f"w{WARMUP}s{SELF_RANGE}" if ENABLE_PAB else "ori"
OUTPUT_DIR = f"./flux_results/pab_flux_result/{get_prompt_id(PROMPT)}"

init_timer()
warmup = 3

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
    generator = torch.Generator("cuda").manual_seed(args.seed)
    disable_timing()
    for _ in range(warmup):
        image = pipe(
            args.prompt,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            max_sequence_length=args.max_sequence_length,
            generator=generator
        ).images[0]
        
    enable_timing()
    image = pipe(
            args.prompt,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            max_sequence_length=args.max_sequence_length,
            generator=generator
        ).images[0]
    
    # ==================== Save Image ====================
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    args.output = os.path.join(OUTPUT_DIR, f"{TAG}_{get_prompt_id(PROMPT)}.png")
    image.save(args.output)
    save_time_statistics_to_file(f"{OUTPUT_DIR}/{TAG}_time_stats.txt")
    if ENABLE_PAB:
        abs_path = os.path.abspath(args.output)
        evaluate_quality_with_origin(abs_path, TAG)
    
    # ==================== Output Statistics ====================
    logging.info(f"\n" + "="*60)
    logging.info(f"Generation Complete!")
    logging.info(f"="*60)
    logging.info(f"  Image saved to: {args.output}")
    # logging.info(f"  Generation time: {generation_time:.2f} seconds")


if __name__ == '__main__':
    main()
    print_time_statistics()

