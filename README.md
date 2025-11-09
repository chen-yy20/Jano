
本仓库包含Jano源代码，以及其baseline代码，包括：
* TeaCache
* PAB
* TokenCache

使用的workload为：
* Flux-1-dev
* Wan2.1-1.3B-T2V
* Wan2.1-14B-T2V

如果运行报错找不到utils模块等，请更新python path
```
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

运行代码都在`run_flux\` 和 `run_wan\`中。相关参数在对应启动代码中修改。