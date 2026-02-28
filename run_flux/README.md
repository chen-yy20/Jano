# run_flux

FLUX inference scripts (Jano + baselines):

- `generate_flux_jano.py`
- `generate_flux_pab.py`
- `generate_flux_teacache.py`
- `generate_flux_toca.py`

The evaluation implementation is shared in `utils/image_eval.py`, with a compatibility entrypoint kept at `flux_evaluate/evaluate.py`.

Recommended unified entrypoint (from repository root):

```bash
python launch.py --model flux --method jano
python launch.py --model flux --method teacache
```
