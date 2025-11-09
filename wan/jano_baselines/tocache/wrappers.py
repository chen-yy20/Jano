# tocache/wrappers.py
import torch
import torch.nn as nn

from .cache_init import ensure_layer_entries
from .force_scheduler import force_scheduler
from .global_force_fresh import global_force_fresh
from .select_and_update import cache_cutfresh, update_cache
from .attn_utils import compute_cross_attn_map, compute_self_attn_map
from .fresh_ratio_scheduler import fresh_ratio_scheduler

class ToCaWrapper(nn.Module):
    """
    对 (B,N,C) 的输入，仅在 top-K token 上调用原模块（self-attn / cross-attn / ffn）。
    """
    def __init__(self, inner, layer_id: int, module_name: str, flag_fn, cache_dic, current):
        super().__init__()
        self.inner = inner
        self.layer_id = int(layer_id)
        self.module_name = module_name
        self.flag_fn = flag_fn
        self.cache_dic = cache_dic
        self.current = current
        force_scheduler(cache_dic, current)

    def forward(self, x, *args, **kwargs):
        assert x.dim() == 3, "ToCaWrapper expects (B,N,C)"
        B, N, C = x.shape
        device = x.device

        # flag（CFG 分桶）：优先外部写入，否则回退 flag_fn()
        ext_flag = self.current.get('flag', None)
        if ext_flag is None:
            flag = self.flag_fn() if callable(self.flag_fn) else 0
        else:
            flag = int(ext_flag)
        self.current['flag'] = flag
        self.current['layer'] = self.layer_id
        self.current['module'] = self.module_name

        ensure_layer_entries(self.cache_dic, flag, self.layer_id, self.module_name, B, N, C, device)

        # r_eff≈1：完全直通原生，实现与不开 ToCa 等价
        r_eff = float(fresh_ratio_scheduler(self.cache_dic, self.current))
        if self.module_name == 'attn' and self.cache_dic.get('cache_all_self_attn', False):
        # 首 3 步/到点强制刷新仍然走 must_full
            force_dict = global_force_fresh(self.cache_dic, self.current)
            if not force_dict.get('attn', False):
            # 直接返回上一步缓存（B,N,C），不做打分/TopK/gather/scatter/复算注意力图
                return self.cache_dic['cache'][flag][self.layer_id]['attn']
        if r_eff >= 1.0 - 1e-8:
            return self.inner(x, *args, **kwargs)

        # cross-attn 需要 context；self-attn/ffn 不需要
        ctx = None
        if self.module_name == 'cross-attn':
            for k in ['context', 'encoder_hidden_states', 'y']:
                if k in kwargs: ctx = kwargs[k]; break
            if ctx is None and len(args) >= 1: ctx = args[0]
            assert ctx is not None and ctx.dim()==3 and ctx.shape[0]==B, \
                "cross-attn wrapper: context must be (B,Nk,C)"

        # 是否强制全量
        must_full = global_force_fresh(self.cache_dic, self.current).get(
            'cross-attn' if self.module_name == 'cross-attn' else self.module_name, False
        )

        if must_full:
            out = self.inner(x, *args, **kwargs)
            out_main = out[0] if isinstance(out, (tuple, list)) else out
            self.cache_dic['cache'][flag][self.layer_id][self.module_name] = out_main.detach()
            self.cache_dic['cache_index'][flag][self.layer_id][self.module_name].zero_()

            # 写全量注意力图
            if self.module_name == 'cross-attn':
                amap = compute_cross_attn_map(self.inner, x, ctx)
                slot = 'cross_attn_map'
            elif self.module_name == 'attn':
                amap = None
                slot = 'attn_map'
            else:
                amap, slot = None, None

            if amap is not None:
                holder = self.cache_dic[slot][flag].get(self.layer_id, None)
                if holder is None or holder.shape[:3] != amap.shape:
                    self.cache_dic[slot][flag][self.layer_id] = torch.zeros_like(amap)
                self.cache_dic[slot][flag][self.layer_id].copy_(amap)

            return out

        # ===== 抽样路径 =====
        fresh_indices, x_fresh = cache_cutfresh(self.cache_dic, x, self.current)
        out_fresh = self.inner(x_fresh, *args, **kwargs)
        out_fresh_main = out_fresh[0] if isinstance(out_fresh, (tuple, list)) else out_fresh
        update_cache(fresh_indices, out_fresh_main.detach(), self.cache_dic, self.current)

        # 注意力图：仅回填 fresh 行
        if self.module_name in ('cross-attn', 'attn'):
            if self.module_name == 'cross-attn':
                amap = compute_cross_attn_map(self.inner, x_fresh, ctx)
                slot = 'cross_attn_map'
            else:
                amap = compute_self_attn_map(self.inner, x_fresh)
                slot = 'attn_map'
            if amap is not None:
                full = self.cache_dic[slot][flag].get(self.layer_id, None)
                Nk = amap.shape[-1]
                if full is None or full.shape[0] != B or full.shape[1] != N or full.shape[2] != Nk:
                    self.cache_dic[slot][flag][self.layer_id] = torch.zeros(B, N, Nk, device=device, dtype=amap.dtype)
                self.cache_dic[slot][flag][self.layer_id].scatter_(
                    dim=1,
                    index=fresh_indices.unsqueeze(-1).expand(-1, -1, amap.shape[-1]),
                    src=amap
                )

        return self.cache_dic['cache'][flag][self.layer_id][self.module_name]
