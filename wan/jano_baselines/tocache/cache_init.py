# tocache/cache_init.py
import torch

def cache_init(model_kwargs, num_steps: int):
    cache_dic = {
        'cache_type':           model_kwargs.get('cache_type', 'attention'),
        'fresh_ratio_schedule': model_kwargs.get('ratio_scheduler', 'ToCa'),
        'fresh_ratio':          float(model_kwargs.get('fresh_ratio', 0.10)),
        'fresh_threshold':      int(model_kwargs.get('fresh_threshold', 3)),
        'force_fresh':          model_kwargs.get('force_fresh', 'global'),
        'soft_fresh_weight':    float(model_kwargs.get('soft_fresh_weight', 0.25)),
        'cache_all_self_attn': model_kwargs.get('cache_all_self_attn', True),
        'cache':          {0:{}, 1:{}},              # per-flag -> per-layer -> per-module: (B,N,C)
        'cache_index':    {0:{}, 1:{}, 'layer_index':{}},   # per-flag -> per-layer -> per-module: (B,N)
        'attn_map':       {0:{}, 1:{}},              # per-flag -> layer: (B,N,N)
        'cross_attn_map': {0:{}, 1:{}},              # per-flag -> layer: (B,N,Nk)
    }

    current = {
        'num_steps': int(num_steps),
        'step': 0,
        'layer': 0,
        'module': '',   # 'attn' | 'cross-attn' | 'ffn'
        'flag': 0,      # 0: cond, 1: uncond
        'is_force_fresh': False,
    }
    return cache_dic, current


def ensure_layer_entries(cache_dic, flag: int, layer_id: int, module: str,
                         B: int, N: int, C: int, device):
    cache_dic['cache'].setdefault(flag, {})
    cache_dic['cache_index'].setdefault(flag, {})
    cache_dic['attn_map'].setdefault(flag, {})
    cache_dic['cross_attn_map'].setdefault(flag, {})

    cache_dic['cache'][flag].setdefault(layer_id, {})
    cache_dic['cache_index'][flag].setdefault(layer_id, {})

    if module not in cache_dic['cache'][flag][layer_id]:
        cache_dic['cache'][flag][layer_id][module] = torch.zeros(B, N, C, device=device)

    if module not in cache_dic['cache_index'][flag][layer_id]:
        cache_dic['cache_index'][flag][layer_id][module] = torch.zeros(B, N, dtype=torch.int, device=device)

    # 注意力图容器在写入时按 Nk 懒创建
    cache_dic['attn_map'][flag].setdefault(layer_id, None)
    cache_dic['cross_attn_map'][flag].setdefault(layer_id, None)
