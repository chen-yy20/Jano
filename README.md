
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

如果运行报错ModuleNotFoundError，请更新python path:
```
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## 运行
运行代码都在`run_flux\` 和 `run_wan\`中。相关参数在对应启动代码中修改。

