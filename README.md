
本仓库包含Jano源代码，以及其baseline代码，包括：
* TeaCache
* PAB
* TokenCache

使用的workload为：
* Flux-1-dev
* Wan2.1-1.3B-T2V
* Wan2.1-14B-T2V

## 下载模型
```
pip install "huggingface_hub[cli]"

export HF_ENDPOINT=https://hf-mirror.com 

huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir ./Flux-1

huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B

huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./Wan2.1-T2V-14B
```

## 初始化

### 配置环境
```
conda create -n jano python=3.10
conda activate jano
pip install -r requirements.txt
```

## 运行
运行代码都在`run_flux\` 和 `run_wan\`中。相关参数在对应启动代码中修改。
> 如果运行报错ModuleNotFoundError，请更新python path:

```
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## 分布式运行

取决与你的集群管理机制。

基于slurm的集群可以直接使用`bash jano_wan_2gpu.sh`来启动两卡。
目前支持`jano`和`pab`的两卡运行，修改`infer.sh`来选择跑哪个。

```
# SCRIPT="./run_wan/jano_generate.py"
SCRIPT="./run_wan/pab_generate.py"
```

如果用torchrun之类的，请自行适配。


## Wan-14B 内存优化
* Jano: `MEMORY_EFFICIENT_CACHE=True` + 2卡并行
* PAB: 设置 `LAYER_INTERVAL = 2`，设置为n，内存消耗就减少为n分之一 + 2卡并行

其他未实现。