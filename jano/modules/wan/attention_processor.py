# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.cuda.amp as amp
import torch.nn as nn

from .attention import flash_attention

from utils.envs import GlobalEnv
from utils.timer import get_timer
from jano.mask_manager.wan_mask_manager import get_mask_manager
from jano.stuff import get_timestep, get_masked_timer, print_gpu_memory


class RoPEManager:
    _instance = None
    # 用于储存计算完毕的freqs_i和mask_freq_i，避免重复的运算
    
    def __init__(self):
        self.full_freqs_i = None # self.full_freqs_i.shape=torch.Size([32760, 1, 64])
        self.medium_freqs_i = None
        self.active_freqs_i = None
    
    @staticmethod
    def get_instance():
        if RoPEManager._instance is None:
            RoPEManager._instance = RoPEManager()
        return RoPEManager._instance
    
    def get_freqs_i(self, grid_sizes, freqs):
        if self.full_freqs_i is None:
            f, h, w = grid_sizes[0].tolist()
            c = freqs.size(1)
            
            # split freqs
            freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
            
            # compute freqs_i
            self.full_freqs_i = torch.cat([
                freqs_split[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs_split[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs_split[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ], dim=-1).reshape(f * h * w, 1, -1)
            
        mm = get_mask_manager()
        level = mm.step_level
        if level == 3 or level == 0:
            return self.full_freqs_i
        elif level == 2:
            if self.medium_freqs_i is None:
                self.medium_freqs_i = self.full_freqs_i[mm.medium_mask]
            return self.medium_freqs_i
        elif level == 1:
            if self.active_freqs_i is None:
                self.active_freqs_i = self.full_freqs_i[mm.active_mask]
            return self.active_freqs_i
            
@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2
    # seq_len = grid_sizes[0].prod().item()  # f * h * w
    seq_len = x.shape[1]
    
    # 获取缓存的freqs_i
    rope_manager = RoPEManager.get_instance()
    freqs_i = rope_manager.get_freqs_i(grid_sizes, freqs)
    
    assert freqs_i.shape[0] == x.shape[1] # 
    
    # apply rope
    x_i = torch.view_as_complex(x[0, :seq_len].to(torch.float64).reshape(
        seq_len, n, -1, 2))
    x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
    
    # handle remaining sequence if any
    if x.size(1) > seq_len:
        x_i = torch.cat([x_i, x[0, seq_len:]])
    
    return x_i.unsqueeze(0).float()

class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

class WanSelfAttention_masked_KV(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 layer_idx=-1):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.layer_idx = layer_idx

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
        mm = get_mask_manager()
        if mm is not None:
            k, v = mm.process_kv_sequence(
                kv = torch.cat([k, v]),
                name = GlobalEnv.get_envs("cond"),
                layer_idx=self.layer_idx
            ).chunk(2, dim=0)
                

        x = flash_attention(
            q=q,
            k=k,
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x