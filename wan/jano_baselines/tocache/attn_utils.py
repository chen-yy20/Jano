# tocache/attn_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def _get_attr_any(obj, names):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None

def _infer_heads_and_dim_from_module(module, Dh, model_dim):
    for name in ['num_heads', 'n_heads', 'heads', 'num_head']:
        if hasattr(module, name):
            h = int(getattr(module, name))
            d = Dh // max(1, h)
            return h, d
    if model_dim > 0 and Dh % model_dim == 0:
        h = Dh // model_dim
        return int(h), int(model_dim)
    return 8, max(1, Dh // 8)

@torch.no_grad()
def compute_cross_attn_map(inner_attn: nn.Module,
                           q_tokens: torch.Tensor,      # (B,Nq,Cq)
                           kv_tokens: torch.Tensor):    # (B,Nk,Ck)
    q = _get_attr_any(inner_attn, ['q_proj','query','to_q','q'])
    k = _get_attr_any(inner_attn, ['k_proj','key','to_k','k'])
    if not isinstance(q, nn.Linear) or not isinstance(k, nn.Linear):
        return None
    B, Nq, Cq = q_tokens.shape
    B2, Nk, Ck = kv_tokens.shape
    if B != B2: return None

    Q = q(q_tokens); K = k(kv_tokens)
    Dh = Q.shape[-1]
    H, D = _infer_heads_and_dim_from_module(inner_attn, Dh, Cq)
    if H * D != Dh:
        D = Dh // max(1, H)
        if H * D != Dh: return None

    Q = Q.view(B,Nq,H,D).transpose(1,2).contiguous()
    K = K.view(B,Nk,H,D).transpose(1,2).contiguous()
    logits = torch.matmul(Q * (D**-0.5), K.transpose(-2,-1))
    attn = F.softmax(logits, dim=-1)
    return attn.mean(dim=1)  # (B,Nq,Nk)

@torch.no_grad()
def compute_self_attn_map(inner_attn: nn.Module,
                          tokens: torch.Tensor):        # (B,N,C)
    B, N, C = tokens.shape
    # A: 分投影
    q = _get_attr_any(inner_attn, ['q_proj','query','to_q','q'])
    k = _get_attr_any(inner_attn, ['k_proj','key','to_k','k'])
    if isinstance(q, nn.Linear) and isinstance(k, nn.Linear):
        Q = q(tokens); K = k(tokens)
        Dh = Q.shape[-1]
        H, D = _infer_heads_and_dim_from_module(inner_attn, Dh, C)
        if H * D != Dh:
            D = Dh // max(1, H)
            if H * D != Dh: return None
        Q = Q.view(B,N,H,D).transpose(1,2).contiguous()
        K = K.view(B,N,H,D).transpose(1,2).contiguous()
        logits = torch.matmul(Q * (D**-0.5), K.transpose(-2,-1))
        attn = F.softmax(logits, dim=-1)
        return attn.mean(dim=1)

    # B: qkv 一体
    qkv = _get_attr_any(inner_attn, ['qkv','to_qkv','in_proj'])
    if isinstance(qkv, nn.Linear):
        QKV = qkv(tokens)                  # (B,N,3*Dh)
        Dh = QKV.shape[-1] // 3
        q, k = QKV.split([Dh, Dh], dim=-1)
        H, D = _infer_heads_and_dim_from_module(inner_attn, Dh, C)
        if H * D != Dh:
            D = Dh // max(1, H)
            if H * D != Dh: return None
        q = q.view(B,N,H,D).transpose(1,2).contiguous()
        k = k.view(B,N,H,D).transpose(1,2).contiguous()
        logits = torch.matmul(q * (D**-0.5), k.transpose(-2,-1))
        attn = F.softmax(logits, dim=-1)
        return attn.mean(dim=1)

    return None
