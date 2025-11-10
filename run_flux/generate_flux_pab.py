#!/usr/bin/env python3
# Copyright 2024-2025 Flux PAB Implementation
"""
Flux Image Generation with PAB Acceleration

Usage:
    # Enable PAB (default configuration)
    python generate_flux_pab.py --model_path /path/to/flux --prompt "your prompt" --enable_pab
    
    # Custom PAB configuration
    python generate_flux_pab.py --model_path /path/to/flux --prompt "your prompt" \
        --enable_pab --pab_self_range 2 --pab_cross_range 5
    
    # Without PAB (baseline)
    python generate_flux_pab.py --model_path /path/to/flux --prompt "your prompt"
"""

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
from flux.pipeline_flux import FluxPipeline
from flux.transformer_flux import FluxTransformer2DModel
# Import PAB modules
from flux.pab_manager import PABConfig, set_pab_manager, update_steps, enable_pab as check_pab_enabled
from flux.pab_diffusers import apply_pab_to_model, reset_model_pab_cache, reset_pab_stats, get_pab_stats
from flux.pab_flux_model import wrap_flux_forward_with_pab

from utils.timer import init_timer, get_timer, print_time_statistics, save_time_statistics_to_file, disable_timing, enable_timing
from utils.quality_metric import evaluate_quality_with_origin
from jano.stuff import get_prompt_id

MODEL_PATH = "/home/fit/zhaijdcyy/WORK/models/Flux-1"
PROMPT = "A photorealistic cute cat, wearing a simple blue shirt, standing against a clear sky background."
SELF_RANGE = 2
CROSS_RANGE = 5
ENABLE_PAB = 1
TAG = f"s{SELF_RANGE}c{CROSS_RANGE}" if ENABLE_PAB else "ori"
OUTPUT_DIR = f"./results/pab_flux_result/{get_prompt_id(PROMPT)}"

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
    
    # PAB parameters
    parser.add_argument('--enable_pab', action='store_true',
                        help='Enable PAB acceleration')
    parser.add_argument('--pab_self_broadcast', type=lambda x: x.lower() == 'true', default=True,
                        help='Enable self-attention broadcast')
    parser.add_argument('--pab_self_threshold_min', type=int, default=100,
                        help='Self-attention broadcast minimum timestep (based on 0-1000 scale)')
    parser.add_argument('--pab_self_threshold_max', type=int, default=800,
                        help='Self-attention broadcast maximum timestep (based on 0-1000 scale)')
    parser.add_argument('--pab_self_range', type=int, default=2,
                        help='Self-attention broadcast interval')
    parser.add_argument('--pab_cross_broadcast', type=lambda x: x.lower() == 'true', default=True,
                        help='Enable cross-attention broadcast')
    parser.add_argument('--pab_cross_threshold_min', type=int, default=100,
                        help='Cross-attention broadcast minimum timestep (based on 0-1000 scale)')
    parser.add_argument('--pab_cross_threshold_max', type=int, default=900,
                        help='Cross-attention broadcast maximum timestep (based on 0-1000 scale)')
    parser.add_argument('--pab_cross_range', type=int, default=5,
                        help='Cross-attention broadcast interval')
    
    args = parser.parse_args()
    args.model_path = MODEL_PATH
    args.prompt = PROMPT
    args.enable_pab = ENABLE_PAB
    args.pab_self_range = SELF_RANGE
    args.pab_cross_range = CROSS_RANGE
    args.output_dir = OUTPUT_DIR
    
    return args


def main():
    setup_logging()
    args = parse_args()
    
    logging.info("="*60)
    logging.info("Flux Image Generation with PAB")
    logging.info("="*60)
    
    # ==================== Initialize PAB ====================
    if args.enable_pab:
        pab_config = PABConfig(
            self_broadcast=args.pab_self_broadcast,
            self_threshold=[args.pab_self_threshold_min, args.pab_self_threshold_max],
            self_range=args.pab_self_range,
            cross_broadcast=args.pab_cross_broadcast,
            cross_threshold=[args.pab_cross_threshold_min, args.pab_cross_threshold_max],
            cross_range=args.pab_cross_range,
        )
        set_pab_manager(pab_config)
        logging.info(f"✓ PAB Enabled")
        logging.info(f"  Self-Attention: Range={pab_config.self_range}, Threshold={pab_config.self_threshold}")
        logging.info(f"  Cross-Attention: Range={pab_config.cross_range}, Threshold={pab_config.cross_threshold}")
        update_steps(args.num_inference_steps)
    else:
        logging.info("✗ PAB Disabled (Baseline Mode)")
    
    # ==================== Load Model ====================
    logging.info(f"\nLoading Flux model from: {args.model_path}")
    
    try:
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
    
    # ==================== Apply PAB ====================
    if args.enable_pab:
        # Apply PAB to blocks (monkey-patch forward methods)
        apply_pab_to_model(pipe.transformer)
        
        # Wrap forward method to pass timestep
        wrap_flux_forward_with_pab(pipe.transformer)
        
        # Reset cache and stats
        reset_model_pab_cache(pipe.transformer)
        reset_pab_stats()
        logging.info("✓ PAB applied to Flux transformer")
    
    # ==================== Generate Image ====================
    logging.info(f"\nGeneration Configuration:")
    logging.info(f"  Prompt: {args.prompt}")
    logging.info(f"  Resolution: {args.width}x{args.height}")
    logging.info(f"  Inference Steps: {args.num_inference_steps}")
    logging.info(f"  Guidance Scale: {args.guidance_scale}")
    logging.info(f"  Seed: {args.seed}")
    
    if args.enable_pab:
        expected_pab_ratio = ((args.pab_cross_threshold_max - args.pab_cross_threshold_min) / 1000 * 100)
        logging.info(f"  Expected PAB cache rate: ~{expected_pab_ratio:.0f}%")
    
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
    
    if args.enable_pab:
        logging.info(f"  PAB Acceleration: Enabled")
        # Display PAB statistics
        stats = get_pab_stats()
        logging.info(f"\n  PAB Statistics:")
        logging.info(f"    Self-Attention:")
        logging.info(f"      Computed: {stats['self_compute']}")
        logging.info(f"      Cached: {stats['self_cached']}")
        logging.info(f"      Cache Rate: {stats['self_cache_ratio']:.1f}%")
        logging.info(f"    Cross-Attention:")
        logging.info(f"      Computed: {stats['cross_compute']}")
        logging.info(f"      Cached: {stats['cross_cached']}")
        logging.info(f"      Cache Rate: {stats['cross_cache_ratio']:.1f}%")
    else:
        logging.info(f"  PAB Acceleration: Not used")
    logging.info(f"="*60)


if __name__ == '__main__':
    main()
    print_time_statistics()

