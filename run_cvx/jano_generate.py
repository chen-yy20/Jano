"""
This script demonstrates how to generate a video using the CogVideoX model with the Hugging Face `diffusers` pipeline.
The script supports different types of video generation, including text-to-video (t2v), image-to-video (i2v),
and video-to-video (v2v), depending on the input data and different weight.

- text-to-video: THUDM/CogVideoX-5b, THUDM/CogVideoX-2b or THUDM/CogVideoX1.5-5b
- video-to-video: THUDM/CogVideoX-5b, THUDM/CogVideoX-2b or THUDM/CogVideoX1.5-5b
- image-to-video: THUDM/CogVideoX-5b-I2V or THUDM/CogVideoX1.5-5b-I2V

Running the Script:
To run the script, use the following command with appropriate arguments:

```bash
$ python cli_demo.py --prompt "A girl riding a bike." --model_path THUDM/CogVideoX1.5-5b --generate_type "t2v"
```

You can change `pipe.enable_sequential_cpu_offload()` to `pipe.enable_model_cpu_offload()` to speed up inference, but this will use more GPU memory

Additional options are available to specify the model path, guidance scale, number of inference steps, video generation type, and output paths.

"""

import argparse
import logging
from typing import Literal, Optional
import os
import torch

from diffusers import (
    CogVideoXDPMScheduler,
    # CogVideoXImageToVideoPipeline,
    # CogVideoXPipeline,
    # CogVideoXVideoToVideoPipeline,
)
from diffusers.utils import export_to_video, load_image, load_video

from jano import init_jano
from jano.modules.cogvideox.pipeline_cogvideox import CogVideoXPipeline
from jano.modules.cogvideox.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from jano.stuff import get_prompt_id
from utils.timer import init_timer, print_time_statistics, save_time_statistics_to_file
from utils.quality_metric import evaluate_quality_with_origin

PROMPT = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
 
MODEL_PATH = os.getenv("MODEL_PATH", "./cogvideox1.5-5b")
T_WEIGHT = 0.6
DIFFUSION_STENGTH = 0.8
DIFFUSION_DISTANCE = 2
ANALYZE_BLOCK_SIZE = (11,8,17)
STATIC_THRESH = 0.2
MEDIUM_THRESH = 0.4
WARMUP = 10
ENABLE_JANO = 1

TAG = f"W{WARMUP}_B({ANALYZE_BLOCK_SIZE[0]}*{ANALYZE_BLOCK_SIZE[1]}*{ANALYZE_BLOCK_SIZE[2]})_DS({DIFFUSION_STENGTH}-{DIFFUSION_DISTANCE})_S{STATIC_THRESH}_M{MEDIUM_THRESH}" if ENABLE_JANO else "ori"
OUTPUT_DIR = f"./cvx_results/jano_cvx_result/{get_prompt_id(PROMPT)}"

logging.basicConfig(level=logging.INFO)

init_timer()

# Recommended resolution for each model (width, height)
RESOLUTION_MAP = {
    # cogvideox1.5-*
    "cogvideox1.5-5b-i2v": (768, 1360),
    "cogvideox1.5-5b": (768, 1360),
    # cogvideox-*
    "cogvideox-5b-i2v": (480, 720),
    "cogvideox-5b": (480, 720),
    "cogvideox-2b": (480, 720),
}

init_jano(
        enable=ENABLE_JANO,
        model='cvx-5b',
        analyze_block_size=ANALYZE_BLOCK_SIZE,
        tag = TAG,
        save_dir=OUTPUT_DIR,
        num_inference_steps=50,
        warmup_steps=WARMUP,
        cooldown_steps=4,
        t_weight=T_WEIGHT,
        d_strength=DIFFUSION_STENGTH,
        d_distance=DIFFUSION_DISTANCE,
        medium_thresh = MEDIUM_THRESH,
        medium_interval = 3,
        static_thresh = STATIC_THRESH,
        static_interval = 10,
    )

def generate_video(
    prompt: str,
    model_path: str,
    lora_path: str = None,
    lora_rank: int = 128,
    num_frames: int = 81,
    width: Optional[int] = None,
    height: Optional[int] = None,
    output_path: str = "./output.mp4",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v", "i2v", "v2v"],  # i2v: image to video, v2v: video to video
    seed: int = 42,
    fps: int = 16,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - num_frames (int): Number of frames to generate. CogVideoX1.0 generates 49 frames for 6 seconds at 8 fps, while CogVideoX1.5 produces either 81 or 161 frames, corresponding to 5 seconds or 10 seconds at 16 fps.
    - width (int): The width of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - height (int): The height of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - generate_type (str): The type of video generation (e.g., 't2v', 'i2v', 'v2v').·
    - seed (int): The seed for reproducibility.
    - fps (int): The frames per second for the generated video.
    """

    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.

    image = None
    video = None

    # model_name = model_path.split("/")[-1].lower()
    model_name = "cogvideox1.5-5b"
    desired_resolution = RESOLUTION_MAP[model_name]
    if width is None or height is None:
        height, width = desired_resolution
        logging.info(
            f"\033[1mUsing default resolution {desired_resolution} for {model_name}\033[0m"
        )
    elif (height, width) != desired_resolution:
        if generate_type == "i2v":
            # For i2v models, use user-defined width and height
            logging.warning(
                f"\033[1;31mThe width({width}) and height({height}) are not recommended for {model_name}. The best resolution is {desired_resolution}.\033[0m"
            )
        else:
            # Otherwise, use the recommended width and height
            logging.warning(
                f"\033[1;31m{model_name} is not supported for custom resolution. Setting back to default resolution {desired_resolution}.\033[0m"
            )
            height, width = desired_resolution
            
    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    pipe.transformer = CogVideoXTransformer3DModel.from_pretrained(f"{model_path}/transformer", torch_dtype=dtype)

    # If you're using with lora, add this code
    if lora_path:
        pipe.load_lora_weights(
            lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1"
        )
        pipe.fuse_lora(components=["transformer"], lora_scale=1.0)

    # 2. Set Scheduler.
    # Can be changed to `CogVideoXDPMScheduler` or `CogVideoXDDIMScheduler`.
    # We recommend using `CogVideoXDDIMScheduler` for CogVideoX-2B.
    # using `CogVideoXDPMScheduler` for CogVideoX-5B / CogVideoX-5B-I2V.

    # pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    # 3. Enable CPU offload for the model.
    # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to("cuda")
    # pipe.to("cuda")

    # pipe.enable_model_cpu_offload()
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # 4. Generate the video frames based on the prompt.
    # `num_frames` is the Number of frames to generate.
    if generate_type == "i2v":
        video_generate = pipe(
            height=height,
            width=width,
            prompt=prompt,
            image=image,
            # The path of the image, the resolution of video will be the same as the image for CogVideoX1.5-5B-I2V, otherwise it will be 720 * 480
            num_videos_per_prompt=num_videos_per_prompt,  # Number of videos to generate per prompt
            num_inference_steps=num_inference_steps,  # Number of inference steps
            num_frames=num_frames,  # Number of frames to generate
            use_dynamic_cfg=True,  # This id used for DPM scheduler, for DDIM scheduler, it should be False
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
        ).frames[0]
    elif generate_type == "t2v":
        video_generate = pipe(
            height=height,
            width=width,
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
        ).frames[0]
    else:
        video_generate = pipe(
            height=height,
            width=width,
            prompt=prompt,
            video=video,  # The path of the video to be used as the background of the video
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
        ).frames[0]
    export_to_video(video_generate, output_path, fps=fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a video from a text prompt using CogVideoX"
    )
    parser.add_argument(
        "--prompt", type=str, help="The description of the video to be generated"
    )
    parser.add_argument(
        "--image_or_video_path",
        type=str,
        default=None,
        help="The path of the image to be used as the background of the video",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="THUDM/CogVideoX1.5-5B",
        help="Path of the pre-trained model use",
    )
    parser.add_argument(
        "--lora_path", type=str, default=None, help="The path of the LoRA weights to be used"
    )
    parser.add_argument("--lora_rank", type=int, default=128, help="The rank of the LoRA weights")
    parser.add_argument(
        "--output_path", type=str, default="./output.mp4", help="The path save generated video"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance"
    )
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Inference steps")
    parser.add_argument(
        "--num_frames", type=int, default=81, help="Number of steps for the inference process"
    )
    parser.add_argument("--width", type=int, default=None, help="The width of the generated video")
    parser.add_argument(
        "--height", type=int, default=None, help="The height of the generated video"
    )
    parser.add_argument(
        "--fps", type=int, default=16, help="The frames per second for the generated video"
    )
    parser.add_argument(
        "--num_videos_per_prompt",
        type=int,
        default=1,
        help="Number of videos to generate per prompt",
    )
    parser.add_argument(
        "--generate_type", type=str, default="t2v", help="The type of video generation"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation"
    )
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    
    args.prompt = PROMPT
    args.model_path = MODEL_PATH
    filename = f"{TAG}_{get_prompt_id(PROMPT)}" + ".mp4"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    args.output_path = os.path.join(OUTPUT_DIR, filename)
    
    generate_video(
        prompt=args.prompt,
        model_path=args.model_path,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        output_path=args.output_path,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        image_or_video_path=args.image_or_video_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        generate_type=args.generate_type,
        seed=args.seed,
        fps=args.fps,
    )
    print_time_statistics()
    save_time_statistics_to_file(f"{OUTPUT_DIR}/{TAG}_time_stats.txt")

    if TAG != "ori":
        evaluate_quality_with_origin(args.output_path, TAG)