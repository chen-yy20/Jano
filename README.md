# Jano: Adaptive Diffusion Generation with Early-stage Convergence Awareness
[![arXiv](https://img.shields.io/badge/arXiv-2603.00519-b31b1b.svg)](https://arxiv.org/abs/2603.00519)


**Jano** is an inference acceleration framework for diffusion-based video/image generation models. It profiles the spatio-temporal dynamics of the latent space during a short warm-up phase and then selectively skips computations for low-dynamic regions, achieving significant speedup with minimal quality degradation.

## Showcase / 效果展示

The following examples are from `./assets`, with runtime (`generate_e2e`) and quality metrics from the corresponding JSON files.

以下展示来自 `./assets`，时间（`generate_e2e`）与质量指标均读取自对应 JSON 文件。

### FLUX.1-dev (Text-to-Image)

Prompt: `A photorealistic cute cat, wearing a simple blue shirt, standing against a clear sky background.`

<table>
  <tr>
    <th align="center">ORI</th>
    <th align="center">Jano</th>
    <th align="center">PAB</th>
    <th align="center">TeaCache</th>
    <th align="center">ToCA</th>
  </tr>
  <tr>
    <td align="center"><img src="assets/flux/ori/photorealistic_cute_cat,_wearing/ori_photorealistic_cute_cat,_wearing_seed42.png" style="width:180px; height:auto; max-width:180px;"></td>
    <td align="center"><img src="assets/flux/jano/photorealistic_cute_cat,_wearing/jano_photorealistic_cute_cat,_wearing_seed42.png" style="width:180px; height:auto; max-width:180px;"></td>
    <td align="center"><img src="assets/flux/pab/photorealistic_cute_cat,_wearing/w3s8_photorealistic_cute_cat,_wearing_seed42.png" style="width:180px; height:auto; max-width:180px;"></td>
    <td align="center"><img src="assets/flux/teacache/photorealistic_cute_cat,_wearing/TEA0.2_photorealistic_cute_cat,_wearing_seed42.png" style="width:180px; height:auto; max-width:180px;"></td>
    <td align="center"><img src="assets/flux/toca/photorealistic_cute_cat,_wearing/toca_photorealistic_cute_cat,_wearing_seed42.png" style="width:180px; height:auto; max-width:180px;"></td>
  </tr>
  <tr>
    <td align="center">36.14 s<br>1.00×<br>PSNR ----- / SSIM ----- / LPIPS -----</td>
    <td align="center">19.33 s<br>1.87×<br>PSNR 28.19 / SSIM 0.940 / LPIPS 0.088</td>
    <td align="center">23.81 s<br>1.52×<br>PSNR 23.87 / SSIM 0.898 / LPIPS 0.137</td>
    <td align="center">23.36 s<br>1.55×<br>PSNR 23.60 / SSIM 0.908 / LPIPS 0.107</td>
    <td align="center">19.38 s<br>1.86×<br>PSNR 14.79 / SSIM 0.761 / LPIPS 0.379</td>
  </tr>
</table>

### Wan2.1-1.3B (Text-to-Video)

Prompt: `Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.`

<table>
  <tr>
    <td align="center">
      <b>ORI</b><br>
      <video src="https://github.com/user-attachments/assets/6da3c726-d5de-49f5-a96f-d08a54795a5d" controls muted loop style="width:180px; height:auto; max-width:180px;"></video><br>
      165.60 s | 1.00× | N/A
    </td>
    <td align="center">
      <b>Jano</b><br>
      <video src="https://github.com/user-attachments/assets/8caf9810-f7ed-4b28-8240-e3f6364077c8" controls muted loop style="width:180px; height:auto; max-width:180px;"></video><br>
      83.94 s | 1.97× | PSNR 17.96 / SSIM 0.782 / LPIPS 0.184
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>PAB</b><br>
      <video src="https://github.com/user-attachments/assets/8bc47626-a647-4148-bf8c-8bf13f917146" controls muted loop style="width:180px; height:auto; max-width:180px;"></video><br>
      139.94 s | 1.18× | PSNR 16.08 / SSIM 0.734 / LPIPS 0.243
    </td>
    <td align="center">
      <b>TeaCache</b><br>
      <video src="https://github.com/user-attachments/assets/9e66b02a-d568-4aa4-b128-9762d45b2875" controls muted loop style="width:180px; height:auto; max-width:180px;"></video><br>
      106.10 s | 1.56× | PSNR 21.28 / SSIM 0.854 / LPIPS 0.100
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>ToCA</b><br>
      <video src="https://github.com/user-attachments/assets/6aca050f-8b88-4fb6-ac04-5f8705a38dc8" controls muted loop style="width:180px; height:auto; max-width:180px;"></video><br>
      71.39 s | 2.32× | PSNR 13.97 / SSIM 0.670 / LPIPS 0.347
    </td>
    <td></td>
  </tr>
</table>

Jano supports the following workloads:

| Model | Task |
|-------|------|
| [Wan2.1-1.3B-T2V](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) | Text-to-Video |
| [Wan2.1-14B-T2V](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) | Text-to-Video |
| [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) | Text-to-Image |

Baseline implementations included in this repo:
- [TeaCache](https://github.com/ali-vilab/TeaCache)
- [PAB](https://github.com/hao-ai-lab/FastVideo)
- [TokenCache (ToCA)](https://github.com/Shenyi-Z/ToCa)

---

**Jano** 是一个面向扩散模型视频/图像生成推理加速的框架。它在短暂的预热阶段对潜空间的时空动态性进行分析，然后对低动态区域选择性地跳过计算，在几乎不损失质量的情况下显著提升推理速度。

## Project Structure / 项目结构

```
Jano/
├── jano/                   # Core Jano library
│   ├── __init__.py         # init_jano() entry point
│   ├── block_manager.py    # Latent-space block partitioning
│   ├── dynamic_analyzer.py # Spatio-temporal dynamics analysis
│   ├── stuff.py            # Shared utilities (timestep tracking, etc.)
│   ├── mask_manager/       # Per-model cache-mask managers
│   ├── modules/            # Modified model forward passes (Wan, Flux, SD3, CogVideoX)
│   └── dist/               # Distributed (CFG-parallel) utilities
├── wan/                    # Wan2.1 model code + baseline implementations
├── flux/                   # FLUX.1 model code + baseline implementations
├── utils/                  # Shared utilities (timer, logger, quality metrics, envs)
├── run_wan/                # Inference scripts for Wan2.1
├── run_flux/               # Inference scripts for FLUX.1
├── run_cvx/                # Inference scripts for CogVideoX
├── ras_exp/                # Experimental RAS baseline
├── requirements.txt
└── LICENSE
```

## Installation / 安装

### 1. Set up the environment / 配置环境

```bash
conda create -n jano python=3.10
conda activate jano
pip install -r requirements.txt
```

### 2. Download models / 下载模型

```bash
pip install "huggingface_hub[cli]"

# (Optional) Use a mirror in China / 中国用户可使用镜像
export HF_ENDPOINT=https://hf-mirror.com

# Wan2.1 (choose one or both)
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B
huggingface-cli download Wan-AI/Wan2.1-T2V-14B  --local-dir ./Wan2.1-T2V-14B

# FLUX.1-dev
huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir ./Flux-1
```

## Usage / 运行

Please use the unified launcher `launch.py`.
First set `MODEL_PATH`, then run with `--model` and `--method`:

```bash
# Wan2.1
export MODEL_PATH=<your model path>

python launch.py --model <wan/flux>  --method <ori/jano/teacache/pab/toca> --gpus-per-node <1/2> --partition <if specific>
```

> **参数修改说明 / Parameter configuration:**
> Please edit generation parameters directly in the corresponding `*_generate.py` files
> under `run_wan/`, `run_flux/`, and `run_cvx/`.
>
> 请在 `run_wan/`、`run_flux/`、`run_cvx/` 下对应的 `*_generate.py` 文件中修改具体参数。

> **Note:** If you get `ModuleNotFoundError`, add the project root to your Python path first:
> ```bash
> export PYTHONPATH=$PYTHONPATH:$(pwd)
> ```


## Memory Optimization for Wan-14B / Wan-14B 内存优化

| Method | Technique |
|--------|-----------|
| Jano   | Set `KV_OFFLOAD=1` + 2-GPU parallel |
| PAB    | Set `LAYER_INTERVAL=2` (memory ÷ n) + 2-GPU parallel |

## License / 许可证

This project is licensed under the [Apache License 2.0](LICENSE).

The Wan2.1 model weights are subject to their own license.  
The FLUX.1-dev model weights are subject to the [FLUX.1-dev Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).

## Citation / 引用
If you use this code, please cite our paper:
```
@misc{chen2026janoadaptivediffusiongeneration,
      title={Jano: Adaptive Diffusion Generation with Early-stage Convergence Awareness}, 
      author={Yuyang Chen and Linqian Zeng and Yijin ZHou and Hengjie Li and Jidong Zhai},
      year={2026},
      eprint={2603.00519},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.00519}, 
}
```
