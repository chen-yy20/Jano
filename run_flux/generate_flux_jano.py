import torch
import os

from jano import init_jano
from jano.modules.flux.pipeline_flux import FluxPipeline
from jano.modules.flux.transformer_flux import FluxTransformer2DModel

from utils.timer import print_time_statistics, enable_timing, disable_timing
from utils.envs import GlobalEnv

MODEL_PATH = "/home/fit/zhaijdcyy/WORK/models/Flux-1"
PROMPT = "A photorealistic cute cat, wearing a simple blue shirt, standing against a clear sky background."
OUTPUT_DIR = "./jano_flux_result/"

pipe = FluxPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
pipe.transformer = FluxTransformer2DModel.from_pretrained(f"{MODEL_PATH}/transformer", torch_dtype=torch.bfloat16)
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
pipe = pipe.to('cuda')
print(f"Model loaded, GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB", flush=True)

save_dir = OUTPUT_DIR
num_inference_steps = 45

HEIGHT = 1024
WIDTH = 1024
ANALYZE_BLOCK_SIZE = (1, HEIGHT//128, WIDTH//128)
DIFFUSION_STENGTH = 0.8
DIFFUSION_DISTANCE = 2

init_jano(
        enable=True,
        model="flux",
        analyze_block_size=ANALYZE_BLOCK_SIZE,
        tag = os.getenv("TAG", 'no_tag'),
        save_dir=OUTPUT_DIR,
        num_inference_steps=50,
        warmup_steps=7,
        cooldown_steps=3,
        t_weight=0,
        d_strength=DIFFUSION_STENGTH,
        d_distance=DIFFUSION_DISTANCE,
        medium_thresh = 0.5,
        medium_interval = 3,
        static_thresh = 0.2,
        static_interval = 10,
    )

save_dir = GlobalEnv.get_envs("save_dir")
tag = GlobalEnv.get_envs("tag")

os.makedirs(save_dir, exist_ok=True)

# prompt = "A cat holding a sign that says hello world"
prompt = PROMPT

disable_timing()

warmup = 3 # 大于等于2才有正确计时
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
        generator=torch.Generator("cpu").manual_seed(42)
    ).images[0]


enable_timing()
image = pipe(
    prompt,
    height=HEIGHT,
    width=WIDTH,
    guidance_scale=3.5,
    num_inference_steps=num_inference_steps,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(42)
).images[0]



print_time_statistics()

prompt_name = prompt[:20].replace(" ", "_")
img_save_path = os.path.join(save_dir, f"{prompt_name}.png")
image.save(img_save_path)
