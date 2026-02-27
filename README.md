# Jano: Adaptive Diffusion Generation with Early-stage Convergence Awareness

**Jano** is an inference acceleration framework for diffusion-based video/image generation models. It profiles the spatio-temporal dynamics of the latent space during a short warm-up phase and then selectively skips computations for low-dynamic regions, achieving significant speedup with minimal quality degradation.

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

All inference scripts are in `run_wan/`, `run_flux/`, and `run_cvx/`. Key parameters can be modified directly inside each script.

> **Note:** If you get `ModuleNotFoundError`, add the project root to your Python path first:
> ```bash
> export PYTHONPATH=$PYTHONPATH:$(pwd)
> ```

### Wan2.1

```bash
# Set the model path (defaults to ./Wan2.1-T2V-14B)
export MODEL_PATH=./Wan2.1-T2V-14B

# Jano
python run_wan/jano_generate.py

# Baselines
python run_wan/pab_generate.py
python run_wan/teacache_generate.py
python run_wan/toca_generate.py
```

### FLUX.1

```bash
export MODEL_PATH=./Flux-1

python run_flux/generate_flux_jano.py
python run_flux/generate_flux_pab.py
python run_flux/generate_flux_teacache.py
python run_flux/generate_flux_toca.py
```

## Distributed Inference / 分布式运行

Multi-GPU (CFG-parallel) is supported for Jano and PAB on Wan2.1.

**SLURM clusters:**
```bash
# Edit infer.sh to select the script, then:
bash 2gpu_wan_run.sh
```

**Other launchers (e.g., torchrun):**  
Set `RANK`, `WORLD_SIZE`, and `LOCAL_RANK` environment variables accordingly.

## Memory Optimization for Wan-14B / Wan-14B 内存优化

| Method | Technique |
|--------|-----------|
| Jano   | Set `MEMORY_EFFICIENT_CACHE=True` + 2-GPU parallel |
| PAB    | Set `LAYER_INTERVAL=2` (memory ÷ n) + 2-GPU parallel |

## License / 许可证

This project is licensed under the [Apache License 2.0](LICENSE).

The Wan2.1 model weights are subject to their own license.  
The FLUX.1-dev model weights are subject to the [FLUX.1-dev Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).
