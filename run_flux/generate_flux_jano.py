import torch
import os

from jano import init_jano
from jano.modules.flux.pipeline_flux import FluxPipeline
from jano.modules.flux.transformer_flux import FluxTransformer2DModel
from jano.stuff import get_prompt_id

from utils.timer import print_time_statistics, enable_timing, disable_timing, save_time_statistics_to_file
from utils.envs import GlobalEnv
from utils.quality_metric import evaluate_quality_with_origin

HEIGHT = 1024
WIDTH = 1024
MODEL_PATH = "/home/fit/zhaijdcyy/WORK/models/Flux-1"
PROMPT = "A photorealistic cute cat, wearing a simple blue shirt, standing against a clear sky background."

ENABLE_JANO = 1
ANALYZE_BLOCK_SIZE = (1, HEIGHT//128,  WIDTH//128)
DIFFUSION_STENGTH = 0.8
DIFFUSION_DISTANCE = 2
STATIC_THRESH = 0.2
MEDIUM_THRESH = 0.4
WARMUP = 7
# TAG = f"W_{WARMUP}_B({ANALYZE_BLOCK_SIZE[0]}*{ANALYZE_BLOCK_SIZE[1]}*{ANALYZE_BLOCK_SIZE[2]})_DS({DIFFUSION_STENGTH}-{DIFFUSION_DISTANCE})_S{STATIC_THRESH}_M{MEDIUM_THRESH}" if ENABLE_JANO else "ori"
# OUTPUT_DIR = f"./flux_results/jano_flux_result/{get_prompt_id(PROMPT)}"
OUTPUT_DIR = f"../evaluation/janox_flux/{get_prompt_id(PROMPT)}"
TAG = "jano"

# Jano+X
JANO_X = "no" # 设置为pab去启用jano_pab
GlobalEnv.set_envs("janox", JANO_X) 
if JANO_X == "pab":
    from flux.pab.pab_manager import init_pab_manger
    SELF_RANGE = 3
    init_pab_manger(50, self_range=SELF_RANGE, warmup=WARMUP)
    TAG = f"pab{SELF_RANGE}_{TAG}"
    
if JANO_X == "teacache":
    from flux.teacache.tea_transformer import jano_teacache_forward
    FluxTransformer2DModel.forward = jano_teacache_forward
    THRESH = 0.3
    TAG = f"tea{THRESH}_{TAG}"

save_dir = OUTPUT_DIR
num_inference_steps = 50

init_jano(
        enable=ENABLE_JANO,
        model="flux",
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
        static_interval = 12,
    )

save_dir = GlobalEnv.get_envs("save_dir")
tag = GlobalEnv.get_envs("tag")

os.makedirs(save_dir, exist_ok=True)

# prompt = "A cat holding a sign that says hello world"
prompt = PROMPT


pipe = FluxPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
pipe.transformer = FluxTransformer2DModel.from_pretrained(f"{MODEL_PATH}/transformer", torch_dtype=torch.bfloat16)
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
if JANO_X == "teacache":
     # TeaCache
    pipe.transformer.__class__.enable_teacache = True
    pipe.transformer.__class__.cnt = 0
    pipe.transformer.__class__.num_steps = num_inference_steps
    pipe.transformer.__class__.rel_l1_thresh = THRESH # 0.25 for 1.5x speedup, 0.4 for 1.8x speedup, 0.6 for 2.0x speedup, 0.8 for 2.25x speedup
    pipe.transformer.__class__.accumulated_rel_l1_distance = 0
    pipe.transformer.__class__.previous_modulated_input = None
    pipe.transformer.__class__.previous_residual = None



pipe = pipe.to('cuda')
print(f"Model loaded, GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB", flush=True)

disable_timing()
warmup = 3 # 大于等于2才有正确计时
generator = torch.Generator("cuda").manual_seed(42)
for _ in range(warmup):
    image = pipe(
        prompt,
        # height=1024,
        # width=1024,
        height = HEIGHT,
        width = WIDTH,
        guidance_scale=3.5,
        num_inference_steps=num_inference_steps,
        max_sequence_length=512,
        generator=generator
    ).images[0]


enable_timing()
image = pipe(
    prompt,
    height=HEIGHT,
    width=WIDTH,
    guidance_scale=3.5,
    num_inference_steps=num_inference_steps,
    max_sequence_length=512,
    generator=generator,
).images[0]

os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, f"{TAG}_{get_prompt_id(PROMPT)}.png")
image.save(output_path)
print(f"Stored {output_path}!", flush=True)
print_time_statistics()
save_time_statistics_to_file(f"{OUTPUT_DIR}/{TAG}_time_stats.txt")

if TAG != "ori":
    evaluate_quality_with_origin(output_path, TAG)
