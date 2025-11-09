# tocache/scores.py
import torch
import torch.nn.functional as F

def attention_score(cache_dic, current, tokens):
    layer  = current['layer']
    flag   = current['flag']
    module = current['module']  # 'attn' | 'cross-attn' | 'ffn'

    if module == 'cross-attn':
        amap_c = cache_dic['cross_attn_map'].get(0, {}).get(layer, None)
        amap_u = cache_dic['cross_attn_map'].get(1, {}).get(layer, None)
        amap = 0.5 * (amap_c + amap_u) if (amap_c is not None and amap_u is not None) \
               else cache_dic['cross_attn_map'].get(flag, {}).get(layer, None)
    else:
        amap_c = cache_dic['attn_map'].get(0, {}).get(layer, None)
        amap_u = cache_dic['attn_map'].get(1, {}).get(layer, None)
        amap = 0.5 * (amap_c + amap_u) if (amap_c is not None and amap_u is not None) \
               else cache_dic['attn_map'].get(flag, {}).get(layer, None)

    if amap is None:
        return F.normalize(tokens.norm(dim=-1, p=2), dim=-1, p=2)

    p = amap.clamp_min(1e-7)             # (B,N,Nk) / (B,N,N)
    ent = -(p * p.log()).sum(dim=-1)     # (B,N)
    score = 1.0 + ent
    return F.normalize(score, dim=-1, p=2)

def similarity_score(cache_dic, current, tokens):
    ref = cache_dic['cache'][current['flag']][current['layer']][current['module']]
    sim = F.cosine_similarity(tokens, ref, dim=-1)
    return F.normalize(1.0 - sim, dim=-1, p=2)

def norm_score(cache_dic, current, tokens):
    return F.normalize(tokens.norm(dim=-1, p=2), dim=-1, p=2)
