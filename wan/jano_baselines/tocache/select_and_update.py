# tocache/select_and_update.py
import torch
from .fresh_ratio_scheduler import fresh_ratio_scheduler
from .scores import attention_score, similarity_score, norm_score

@torch.no_grad()
def score_evaluate(cache_dic, tokens, current):
    ctype = cache_dic['cache_type']
    if ctype == 'attention':
        return attention_score(cache_dic, current, tokens)
    elif ctype == 'similarity':
        return similarity_score(cache_dic, current, tokens)
    elif ctype == 'norm':
        return norm_score(cache_dic, current, tokens)
    else:
        raise ValueError(f"Unknown cache_type: {ctype}")

@torch.no_grad()
def cache_cutfresh(cache_dic, x, current):
    B, N, C = x.shape
    r = float(fresh_ratio_scheduler(cache_dic, current))
    K = max(1, min(N, int(N * r)))

    score = score_evaluate(cache_dic, x, current)  # (B,N)
    idx = torch.topk(score, k=K, dim=1, largest=True, sorted=False).indices
    fresh_tokens = torch.gather(x, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, C))
    return idx, fresh_tokens

@torch.no_grad()
def update_cache(fresh_indices, fresh_tokens, cache_dic, current):
    flag  = current['flag']
    layer = current['layer']
    module= current['module']

    full = cache_dic['cache'][flag][layer][module]
    B, N, C = full.shape

    full.scatter_(dim=1, index=fresh_indices.unsqueeze(-1).expand(-1, -1, C), src=fresh_tokens)

    age = cache_dic['cache_index'][flag][layer][module]
    age += 1
    zeros = torch.zeros_like(fresh_indices, dtype=age.dtype)
    age.scatter_(dim=1, index=fresh_indices, src=zeros)
