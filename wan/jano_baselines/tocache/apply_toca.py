# tocache/apply_toca.py
import torch.nn as nn
from .cache_init import cache_init
from .wrappers import ToCaWrapper

def _guess_blocks_and_fields(transformer: nn.Module):
    blocks = None
    for cand in ['blocks', 'transformer_blocks', 'layers', 'module_list', 'net']:
        if hasattr(transformer, cand):
            obj = getattr(transformer, cand)
            if isinstance(obj, (nn.ModuleList, nn.Sequential, list, tuple)) and len(obj) > 0:
                blocks = obj
                break
    if blocks is None:
        raise RuntimeError("apply_toca_to_wan: 未找到 transformer 的 blocks 列表")

    def get_self_name(block):
        for name in ['self_attn', 'attn', 'attention', 'self_attention']:
            if hasattr(block, name): return name
        return None

    def get_cross_name(block):
        for name in ['cross_attn', 'attn2', 'xattn', 'cross_attention']:
            if hasattr(block, name): return name
        return None

    def get_ffn_name(block):
        for name in ['ffn', 'mlp', 'ff', 'feed_forward']:
            if hasattr(block, name): return name
        return None

    def get_flag_fn(block):
        return (lambda: 0)  # Wan2.1：flag 由外部写 0/1

    return blocks, get_self_name, get_cross_name, get_ffn_name, get_flag_fn


def apply_toca_to_wan(transformer: nn.Module, num_steps: int, cache_config: dict):
    cache_dic, current = cache_init(cache_config, num_steps)
    blocks, get_self, get_cross, get_ffn, get_flag_fn = _guess_blocks_and_fields(transformer)

    for i, block in enumerate(list(blocks)):
        flag_fn = get_flag_fn(block)

        name = get_self(block)
        if name is not None:
            inner = getattr(block, name)
            setattr(block, name, ToCaWrapper(inner, i, 'attn', flag_fn, cache_dic, current))

        name = get_cross(block)
        if name is not None:
            inner = getattr(block, name)
            setattr(block, name, ToCaWrapper(inner, i, 'cross-attn', flag_fn, cache_dic, current))

        name = get_ffn(block)
        if name is not None:
            inner = getattr(block, name)
            setattr(block, name, ToCaWrapper(inner, i, 'ffn', flag_fn, cache_dic, current))

    transformer.toca_cache_dic = cache_dic
    transformer.toca_current  = current
    return cache_dic, current
