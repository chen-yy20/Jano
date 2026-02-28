# run_flux

FLUX 推理脚本目录（Jano + baseline）：

- `generate_flux_jano.py`
- `generate_flux_pab.py`
- `generate_flux_teacache.py`
- `generate_flux_toca.py`

评估工具已统一为共享实现，入口保留在 `flux_evaluate/evaluate.py`。

推荐统一入口（仓库根目录）：

```bash
python launch.py --model flux --method jano
python launch.py --model flux --method teacache
```

