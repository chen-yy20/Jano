import torch
import os

from jano import init_jano
from jano.modules.flux.pipeline_flux import FluxPipeline
from jano.modules.flux.transformer_flux import FluxTransformer2DModel
from jano.stuff import get_prompt_id

from utils.timer import print_time_statistics, enable_timing, disable_timing, save_time_statistics_to_file, get_time_statistics_dict
from utils.envs import GlobalEnv
from utils.quality_metric import evaluate_quality_with_origin
from utils.results import save_params_and_metrics

HEIGHT = 1024
WIDTH = 1024
MODEL_PATH = os.getenv("MODEL_PATH", "./Flux-1")
PROMPT = "A photorealistic cute cat, wearing a simple blue shirt, standing against a clear sky background."

DEFAULT_ENABLE_JANO = 1
ENABLE_JANO = int(os.getenv("ENABLE_JANO", str(DEFAULT_ENABLE_JANO)))
N = 4  # 重复生成次数，利用4次生成获取不同图片
ANALYZE_BLOCK_SIZE = (1, HEIGHT//128,  WIDTH//128)
STATIC_THRESH = 0.2
MEDIUM_THRESH = 0.4
WARMUP = 7
TAG = f"jano" if ENABLE_JANO else "ori"
METHOD_DIR = "jano" if ENABLE_JANO else "ori"
OUTPUT_DIR = f"./results/flux/{METHOD_DIR}/{get_prompt_id(PROMPT)}"
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



pipe = pipe.to('cuda')
print(f"Model loaded, GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB", flush=True)


SEEDS = [42 + i for i in range(N)]

disable_timing()
warmup = 2  # 大于等于2才有正确计时
warmup_generator = torch.Generator("cuda").manual_seed(SEEDS[0])
for _ in range(warmup):
    image = pipe(
        prompt,
        height=HEIGHT,
        width=WIDTH,
        guidance_scale=3.5,
        num_inference_steps=num_inference_steps,
        max_sequence_length=512,
        generator=warmup_generator,
    ).images[0]

os.makedirs(OUTPUT_DIR, exist_ok=True)
enable_timing()
all_quality_metrics = {}
for seed in SEEDS:
    generator = torch.Generator("cuda").manual_seed(seed)
    image = pipe(
        prompt,
        height=HEIGHT,
        width=WIDTH,
        guidance_scale=3.5,
        num_inference_steps=num_inference_steps,
        max_sequence_length=512,
        generator=generator,
    ).images[0]
    output_path = os.path.join(OUTPUT_DIR, f"{TAG}_{get_prompt_id(PROMPT)}_seed{seed}.png")
    image.save(output_path)
    print(f"Stored {output_path}!", flush=True)
    quality_result = None
    if TAG != "ori":
        baseline_path = os.path.abspath(output_path).replace("/jano/", "/ori/").replace(f"{TAG}_", "ori_")
        quality_result = evaluate_quality_with_origin(
            os.path.abspath(output_path),
            TAG,
            save_metrics=False,
            baseline_path=baseline_path,
        )
    all_quality_metrics[f"seed{seed}"] = quality_result.get("metrics") if quality_result else None

print_time_statistics()
params = {
    "model": "flux",
    "method": "jano" if ENABLE_JANO else "ori",
    "model_path": MODEL_PATH,
    "prompt": PROMPT,
    "height": HEIGHT,
    "width": WIDTH,
    "num_inference_steps": num_inference_steps,
    "guidance_scale": 3.5,
    "seeds": SEEDS,
    "enable_jano": ENABLE_JANO,
    "warmup": WARMUP,
    "static_thresh": STATIC_THRESH,
    "medium_thresh": MEDIUM_THRESH,
}
params_path = save_params_and_metrics(OUTPUT_DIR, TAG, params, get_time_statistics_dict(), all_quality_metrics)
print(f"Params & metrics saved to {params_path}", flush=True)
