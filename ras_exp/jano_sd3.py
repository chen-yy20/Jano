import argparse
import torch
import os
# from diffusers import StableDiffusion3Pipeline

from jano.modules.sd3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from jano.modules.sd3.transformer_sd3 import SD3Transformer2DModel
from ras.utils.stable_diffusion_3.update_pipeline_sd3 import update_sd3_pipeline
from ras.utils import ras_manager
from ras.utils.ras_argparser import parse_args

from jano import init_jano
from jano.stuff import get_prompt_id
from jano.modules.sd3.stuff import wrap_sd3_model_with_jano
from utils.envs import GlobalEnv
from utils.timer import init_timer, print_time_statistics, disable_timing, enable_timing, save_time_statistics_to_file
from utils.quality_metric import evaluate_quality_with_origin

HEIGHT = 1024
WIDTH = 1024
MODEL_PATH = "/home/fit/zhaijdcyy/WORK/models/sd3"
PROMPT = "A photorealistic cute cat, wearing a simple blue shirt, standing against a clear sky background."


ANALYZE_BLOCK_SIZE = (1, HEIGHT//128,  WIDTH//128)
DIFFUSION_STENGTH = 0.8
DIFFUSION_DISTANCE = 2
STATIC_THRESH = 0.25
MEDIUM_THRESH =0.25
WARMUP = 5

ENABLE_JANO = 0
ENABLE_RAS = 1

OUTPUT_DIR = f"./sd3_results/{get_prompt_id(PROMPT)}"


# TAG = f"W_{WARMUP}_B({ANALYZE_BLOCK_SIZE[0]}*{ANALYZE_BLOCK_SIZE[1]}*{ANALYZE_BLOCK_SIZE[2]})_DS({DIFFUSION_STENGTH}-{DIFFUSION_DISTANCE})_S{STATIC_THRESH}_M{MEDIUM_THRESH}" if ENABLE_JANO else "ori"
if ENABLE_JANO:
    TAG = "jano"
elif ENABLE_RAS:
    TAG = "ras"
else:
    TAG = "ori"
    

init_jano(
    enable=ENABLE_JANO,
    model="sd3",
    analyze_block_size=ANALYZE_BLOCK_SIZE,
    tag = TAG,
    save_dir=OUTPUT_DIR,
    num_inference_steps=50,
    warmup_steps=WARMUP,
    cooldown_steps=2,
    t_weight=0,
    d_strength=DIFFUSION_STENGTH,
    d_distance=DIFFUSION_DISTANCE,
    medium_thresh = MEDIUM_THRESH,
    medium_interval = 4,
    static_thresh = STATIC_THRESH,
    static_interval = 8,
)


init_timer()

def sd3_inf(args):
    pipeline = StableDiffusion3Pipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, use_auth_token=True)
    pipeline.transformer = SD3Transformer2DModel.from_pretrained(f"{MODEL_PATH}/transformer", torch_dtype=torch.float16)
    if ENABLE_JANO:
        wrap_sd3_model_with_jano(pipeline.transformer)
    pipeline.to("cuda")
    if ENABLE_RAS:
        pipeline = update_sd3_pipeline(pipeline)
    generator = torch.Generator("cuda").manual_seed(args.seed) if args.seed is not None else None
    numsteps = args.num_inference_steps
    disable_timing()
    for _ in range(4):
        pipeline(
                generator=generator,
                num_inference_steps=numsteps,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                guidance_scale=7.0,
                ).images[0]
        
    enable_timing()
    image = pipeline(
                    generator=generator,
                    num_inference_steps=numsteps,
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    height=args.height,
                    width=args.width,
                    guidance_scale=7.0,
                    ).images[0]
    image.save(args.output)

if __name__ == "__main__":
    args = parse_args()
    args.seed = 42
    args.num_inference_steps = 50
    args.height = HEIGHT
    args.width = WIDTH
    args.prompt = PROMPT
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    args.output = os.path.join(OUTPUT_DIR, f"{TAG}_{get_prompt_id(PROMPT)}.png")
    if ENABLE_RAS:
        ras_manager.MANAGER.set_parameters(args)
    sd3_inf(args)
    print_time_statistics()
    save_time_statistics_to_file(os.path.join(OUTPUT_DIR, f"{TAG}_timestats.txt"))
    if TAG != "ori":
        evaluate_quality_with_origin(args.output, tag=TAG)
