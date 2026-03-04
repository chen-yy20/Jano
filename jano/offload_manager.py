"""
Pool-Based Dual-Stream Offload Manager for Jano  (Preallocated Edition)
------------------------------------------------------------------------
设计原则
--------
CPU Pinned Memory Pool 在 init 阶段由 preallocate() **同步**分配，
彻底消除推理期间的 cudaHostAlloc 调用（原方案的性能瓶颈）。

每层 pool 布局（flat 化存储，保证 DMA slice 连续性）：
    pool[cond][layer][  0     : s_flat        ]  → static  KV
    pool[cond][layer][ s_flat : s_flat+m_flat ]  → medium  KV

    flat_full = prod(full_shape[:-1])，head_dim = full_shape[-1]
    s_flat、m_flat 在首次 store_async 时记录，fetch 时依此切片。

GPU Staging Buffers（形状 (flat_full, head_dim)，k 槽循环）
    store staging : tensor → staging[:flat] → async D2H → pool[layer][off:off+flat]
    fetch staging : pool[layer][off:off+flat] → async H2D → staging[:flat] → reshape

    所有 DMA 源/目 均为连续 2D slice，效率最优。

Preallocate API
    preallocate(layer_num, full_shape, full_dtype, conds)
        在 init_mask_manager（推理初始化阶段）调用，同步完成所有 cudaHostAlloc。
        推理期间不再触发任何 pinned memory 分配。
"""

from __future__ import annotations

import time
import torch
from math import prod
from typing import Dict, List, Optional

from utils.timer import get_timer


def _fmt(n: int) -> str:
    return f"{n / 1024 ** 3:.2f} GB"


class OffloadManager:
    """
    Pool-based + 滑动窗口预取的 Offload Manager（预分配版）。

    使用流程
    --------
    1. 初始化时调用 preallocate()，同步完成 pinned memory 分配。
    2. step_level==3 时 store_async() 写入 pool。
    3. step_level==1/2 时 begin_fetch_step() + fetch() 滑动窗口回读。
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        prefetch_window: int = 1,
    ):
        self._device_arg = device
        self.device: Optional[torch.device] = None
        self.k = prefetch_window

        # ── CPU Pinned Memory Pool ──────────────────────────────────────────
        # cond → Tensor[L, flat_full, head_dim]  (pinned CPU)
        self._pool: Dict[str, torch.Tensor] = {}

        # ── 形状 / dtype ────────────────────────────────────────────────────
        self._flat_full: int = 0          # prod(full_shape[:-1])
        self._head_dim: int = 0           # full_shape[-1]
        self._pool_dtype: Optional[torch.dtype] = None
        self._layer_num: int = 0

        # ── per-(cond, layer) KV 长度 & 原始形状 ───────────────────────────
        # 以 flat 元素数量计（= prod(orig_shape[:-1])）
        self._s_len:        Dict[str, List[int]]              = {}
        self._m_len:        Dict[str, List[int]]              = {}
        self._s_orig_shape: Dict[str, List[Optional[tuple]]]  = {}
        self._m_orig_shape: Dict[str, List[Optional[tuple]]]  = {}

        # ── GPU Staging Buffers ─────────────────────────────────────────────
        # store staging（s/m 共享，单层内 s→m 串行）
        self._store_staging:   List[torch.Tensor] = []   # k × (flat_full, head_dim)
        # fetch staging（s/m 分开，支持 step_level==1 同时预取）
        self._fetch_s_staging: List[torch.Tensor] = []   # k × (flat_full, head_dim)
        self._fetch_m_staging: List[torch.Tensor] = []   # k × (flat_full, head_dim)
        self._staging_ready: bool = False

        # ── Store 槽位循环 ──────────────────────────────────────────────────
        self._store_slot: int = 0
        self._store_slot_done: List[torch.cuda.Event] = []

        # ── CUDA Streams ────────────────────────────────────────────────────
        self.fetch_stream: Optional[torch.cuda.Stream] = None
        self.store_stream: Optional[torch.cuda.Stream] = None

        # ── per-(cond, layer) CUDA Events ───────────────────────────────────
        self._s_fetch_events: Dict[str, List[torch.cuda.Event]] = {}
        self._m_fetch_events: Dict[str, List[torch.cuda.Event]] = {}
        self._s_store_events: Dict[str, List[torch.cuda.Event]] = {}
        self._m_store_events: Dict[str, List[torch.cuda.Event]] = {}

        # ── Fetch 滑动窗口状态（每 fetch step 重置）─────────────────────────
        self._fetch_cond:       str = "0"
        self._fetch_step_level: int = 0
        self._fetch_num_layers: int = 0
        self._fetch_next_layer: int = 0
        self._fetch_s_slot:     int = 0
        self._fetch_m_slot:     int = 0
        self._fetch_s_slot_map: Dict[int, int] = {}   # layer_idx → staging slot
        self._fetch_m_slot_map: Dict[int, int] = {}

    # =========================================================================
    # 内部工具
    # =========================================================================

    def _ensure_streams(self) -> None:
        if self.fetch_stream is not None:
            return
        self.device = self._device_arg or torch.device(
            f"cuda:{torch.cuda.current_device()}"
        )
        self.fetch_stream = torch.cuda.Stream(device=self.device)
        self.store_stream = torch.cuda.Stream(device=self.device)

    def _ensure_staging(self) -> None:
        """分配 GPU staging buffers（快，μs 级；由 preallocate 之后首次 store/fetch 触发）。"""
        if self._staging_ready:
            return
        assert self._flat_full > 0, "preallocate() 必须先于 _ensure_staging() 调用"
        k = self.k
        shape = (self._flat_full, self._head_dim)
        dtype = self._pool_dtype
        for _ in range(k):
            self._store_staging.append(
                torch.empty(shape, dtype=dtype, device=self.device)
            )
            self._fetch_s_staging.append(
                torch.empty(shape, dtype=dtype, device=self.device)
            )
            self._fetch_m_staging.append(
                torch.empty(shape, dtype=dtype, device=self.device)
            )
            ev = torch.cuda.Event()
            ev.record()   # 初始化为"已完成"状态
            self._store_slot_done.append(ev)
        self._staging_ready = True

    def _ensure_cond_events(self, cond: str) -> None:
        """为该 cond 分配 per-layer CUDA events（μs 级）。"""
        L = self._layer_num
        if cond not in self._s_fetch_events:
            self._s_fetch_events[cond] = [torch.cuda.Event() for _ in range(L)]
            self._m_fetch_events[cond] = [torch.cuda.Event() for _ in range(L)]
            self._s_store_events[cond] = [torch.cuda.Event() for _ in range(L)]
            self._m_store_events[cond] = [torch.cuda.Event() for _ in range(L)]

    def _ensure_cond_ready(self, cond: str) -> None:
        """确保 cond 的 pool 已分配，并完成 staging / events 初始化。"""
        assert cond in self._pool, (
            f"[OffloadManager] cond={cond!r} 的 pool 尚未分配，"
            "请在推理初始化时调用 preallocate()。"
        )
        self._ensure_staging()
        self._ensure_cond_events(cond)

    # =========================================================================
    # 预分配（init 阶段同步调用）
    # =========================================================================

    def preallocate(
        self,
        layer_num: int,
        full_shape: tuple,
        full_dtype: torch.dtype,
        conds: List[str],
    ) -> None:
        """
        在推理初始化阶段同步分配所有 cond 的 CPU pinned memory pool。

        Parameters
        ----------
        layer_num : int
            Transformer 层数。
        full_shape : tuple
            单层 KV 的最大形状，例如 (2, 32760, 1536)。
            flat_full = prod(full_shape[:-1])，head_dim = full_shape[-1]。
            static + medium 的 flat token 数之和不超过 flat_full。
        full_dtype : torch.dtype
            KV 数据类型，例如 torch.bfloat16。
        conds : list of str
            需要预分配的 cond 标识列表，例如 ["0", "1"]。
        """
        self._ensure_streams()

        flat_full = prod(full_shape[:-1])
        head_dim  = full_shape[-1]

        # 记录形状参数（staging 由 _ensure_staging 在首次使用时懒分配）
        self._flat_full   = flat_full
        self._head_dim    = head_dim
        self._pool_dtype  = full_dtype
        self._layer_num   = layer_num

        t0 = time.perf_counter()
        for cond in conds:
            if cond in self._pool:
                continue   # 幂等
            pool = torch.empty(
                (layer_num, flat_full, head_dim),
                dtype=full_dtype,
                pin_memory=True,
            )
            self._pool[cond] = pool
            self._s_len[cond]        = [0] * layer_num
            self._m_len[cond]        = [0] * layer_num
            self._s_orig_shape[cond] = [None] * layer_num
            self._m_orig_shape[cond] = [None] * layer_num
            print(
                f"[OffloadManager] preallocate cond={cond}  "
                f"pool={pool.nbytes / 1024**2:.1f}MB  "
                f"shape=[{layer_num}, {flat_full}, {head_dim}]",
                flush=True,
            )
        elapsed = (time.perf_counter() - t0) * 1e3
        print(
            f"[OffloadManager] preallocate done  "
            f"conds={conds}  elapsed={elapsed:.2f}ms",
            flush=True,
        )

    def is_ready(self, cond: Optional[str] = None) -> bool:
        """检查 pool 是否就绪（已通过 preallocate 分配）。"""
        if cond is not None:
            return cond in self._pool
        return bool(self._pool)

    # =========================================================================
    # Store（GPU → CPU Pool，双流流水线）
    # =========================================================================

    def store_async(
        self,
        tensor: torch.Tensor,
        layer_idx: int,
        kv_type: str,
        cond: str,
    ) -> None:
        """
        异步将 GPU tensor 存入 pool[cond][layer_idx] 的对应区段。

        存储布局（flat 化后均为连续 slice）：
          kv_type='s'：pool[layer][0 : s_flat]
          kv_type='m'：pool[layer][s_flat : s_flat+m_flat]

        kv_type: 's'（static）或 'm'（medium）
        """
        self._ensure_cond_ready(cond)

        # ── flat 化源 tensor ────────────────────────────────────────────────
        orig_shape  = tensor.shape
        tensor_flat = tensor.detach().reshape(-1, orig_shape[-1])  # (flat, head_dim)
        flat_len    = tensor_flat.shape[0]

        assert flat_len <= self._flat_full, (
            f"[OffloadManager] tensor flat_len={flat_len} > flat_full={self._flat_full}，"
            "请检查 full_shape 是否足够大。"
        )

        # ── 计算 pool 内偏移 & 记录元信息 ───────────────────────────────────
        if kv_type == "s":
            offset   = 0
            store_ev = self._s_store_events[cond][layer_idx]
            self._s_len[cond][layer_idx]        = flat_len
            self._s_orig_shape[cond][layer_idx] = orig_shape
        else:
            # m 跟在 s 之后（s 必须先于 m 存入）
            offset   = self._s_len[cond][layer_idx]
            store_ev = self._m_store_events[cond][layer_idx]
            self._m_len[cond][layer_idx]        = flat_len
            self._m_orig_shape[cond][layer_idx] = orig_shape

        # ── 申请 store staging 槽位 ─────────────────────────────────────────
        slot = self._store_slot % self.k
        self._store_slot += 1
        # 确保此槽位前一条 DMA 已完成（通常已完成，~0μs）
        self._store_slot_done[slot].synchronize()

        staging = self._store_staging[slot]

        # step1: 同步 GPU→GPU 拷贝到 staging slice，隔离源 tensor 生命期
        staging[:flat_len].copy_(tensor_flat, non_blocking=False)

        # step2: staging slice → pinned pool slice 异步 D2H DMA（均为连续 2D 视图）
        pool_slice    = self._pool[cond][layer_idx][offset : offset + flat_len]
        staging_slice = staging[:flat_len]

        self.store_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.store_stream):
            pool_slice.copy_(staging_slice, non_blocking=True)
            self._store_slot_done[slot].record(self.store_stream)
            store_ev.record(self.store_stream)

    def wait_store(self, cond: str, layer_idx: int, kv_type: str) -> None:
        """阻塞等待指定 (cond, layer_idx, kv_type) 的 store DMA 完成。"""
        ev = (
            self._s_store_events[cond][layer_idx]
            if kv_type == "s"
            else self._m_store_events[cond][layer_idx]
        )
        ev.synchronize()

    # =========================================================================
    # Fetch（CPU Pool → GPU，滑动窗口预取）
    # =========================================================================

    def begin_fetch_step(
        self,
        cond: str,
        num_layers: int,
        step_level: int,
    ) -> None:
        """
        每个 fetch step 开始时调用，重置窗口并预取前 k 层。

        step_level==2：仅预取 s_kv（static）
        step_level==1：预取 s_kv + m_kv（static + medium）
        """
        self._ensure_cond_ready(cond)
        self._fetch_cond       = cond
        self._fetch_step_level = step_level
        self._fetch_num_layers = num_layers
        self._fetch_next_layer = 0
        self._fetch_s_slot     = 0
        self._fetch_m_slot     = 0
        self._fetch_s_slot_map.clear()
        self._fetch_m_slot_map.clear()

        # 确保 store_stream 的所有 D2H 写入先于 fetch_stream 的 H2D 读取，
        # 防止数据竞争。
        self.fetch_stream.wait_stream(self.store_stream)

        for _ in range(min(self.k, num_layers)):
            self._issue_prefetch_layer(self._fetch_next_layer)
            self._fetch_next_layer += 1

    def _issue_prefetch_layer(self, layer_idx: int) -> None:
        """在 fetch_stream 上为 layer_idx 发起异步 H2D DMA（连续 slice）。"""
        if layer_idx < 0 or layer_idx >= self._fetch_num_layers:
            return
        cond = self._fetch_cond
        if cond not in self._pool:
            return

        self.fetch_stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self.fetch_stream):
            if self._fetch_step_level in (1, 2):
                s_flat = self._s_len[cond][layer_idx]
                slot   = self._fetch_s_slot % self.k
                self._fetch_s_slot += 1
                self._fetch_s_staging[slot][:s_flat].copy_(
                    self._pool[cond][layer_idx][:s_flat], non_blocking=True
                )
                self._s_fetch_events[cond][layer_idx].record(self.fetch_stream)
                self._fetch_s_slot_map[layer_idx] = slot

            if self._fetch_step_level == 1:
                s_flat = self._s_len[cond][layer_idx]
                m_flat = self._m_len[cond][layer_idx]
                slot   = self._fetch_m_slot % self.k
                self._fetch_m_slot += 1
                self._fetch_m_staging[slot][:m_flat].copy_(
                    self._pool[cond][layer_idx][s_flat : s_flat + m_flat],
                    non_blocking=True,
                )
                self._m_fetch_events[cond][layer_idx].record(self.fetch_stream)
                self._fetch_m_slot_map[layer_idx] = slot

    def _issue_prefetch_fallback(self, layer_idx: int, kv_type: str) -> None:
        """Fallback：layer 未在预取窗口内时同步补发。"""
        print(
            f"[OffloadManager] Fallback prefetch: {kv_type}_kv layer={layer_idx}",
            flush=True,
        )
        cond = self._fetch_cond
        self.fetch_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.fetch_stream):
            if kv_type == "s":
                s_flat = self._s_len[cond][layer_idx]
                slot   = self._fetch_s_slot % self.k
                self._fetch_s_slot += 1
                self._fetch_s_staging[slot][:s_flat].copy_(
                    self._pool[cond][layer_idx][:s_flat], non_blocking=True
                )
                self._s_fetch_events[cond][layer_idx].record(self.fetch_stream)
                self._fetch_s_slot_map[layer_idx] = slot
            else:
                s_flat = self._s_len[cond][layer_idx]
                m_flat = self._m_len[cond][layer_idx]
                slot   = self._fetch_m_slot % self.k
                self._fetch_m_slot += 1
                self._fetch_m_staging[slot][:m_flat].copy_(
                    self._pool[cond][layer_idx][s_flat : s_flat + m_flat],
                    non_blocking=True,
                )
                self._m_fetch_events[cond][layer_idx].record(self.fetch_stream)
                self._fetch_m_slot_map[layer_idx] = slot

    def fetch(self, layer_idx: int, kv_type: str, cond: str) -> torch.Tensor:
        """
        等待 layer_idx 的预取完成，返回 GPU tensor（已还原为原始形状）。

        kv_type: 's'（static）或 'm'（medium）
        """
        self._ensure_cond_ready(cond)

        if kv_type == "s":
            if layer_idx not in self._fetch_s_slot_map:
                self._issue_prefetch_fallback(layer_idx, "s")
            torch.cuda.current_stream().wait_event(
                self._s_fetch_events[cond][layer_idx]
            )
            # ⚠ 必须先 clone，再触发下一层预取。
            # 若先触发预取，fetch_stream.wait_stream(current_stream) 只能看到
            # wait_event 为止的快照，可能提前启动 H2D 并覆盖同一 staging slot。
            flat   = self._s_len[cond][layer_idx]
            shape  = self._s_orig_shape[cond][layer_idx]
            slot   = self._fetch_s_slot_map[layer_idx]
            result = self._fetch_s_staging[slot][:flat].clone().reshape(shape)
        else:
            if layer_idx not in self._fetch_m_slot_map:
                self._issue_prefetch_fallback(layer_idx, "m")
            torch.cuda.current_stream().wait_event(
                self._m_fetch_events[cond][layer_idx]
            )
            flat   = self._m_len[cond][layer_idx]
            shape  = self._m_orig_shape[cond][layer_idx]
            slot   = self._fetch_m_slot_map[layer_idx]
            result = self._fetch_m_staging[slot][:flat].clone().reshape(shape)

        # 触发下一层预取（每层仅在 's' 时驱动一次，避免重复）
        # clone() 之后 staging slot 与返回张量已解耦，可安全复用。
        if kv_type == "s":
            self._issue_prefetch_layer(self._fetch_next_layer)
            self._fetch_next_layer += 1

        return result

    # =========================================================================
    # 废弃接口（向后兼容，空操作）
    # =========================================================================

    def start_preallocate(
        self,
        cond: str,
        layer_num: int,
        s_shape: tuple,
        s_dtype: torch.dtype,
        m_shape: tuple,
        m_dtype: torch.dtype,
    ) -> None:
        """[已废弃] pool 现在由 preallocate() 在 init 阶段统一分配，此方法为空操作。"""
        pass

    def register_shape(self, key: str, shape: tuple, dtype: torch.dtype) -> None:
        """[已废弃] 兼容旧调用，空操作。"""
        pass

    def build_pinned_buffers(self) -> None:
        """[已废弃] 兼容旧调用，空操作。"""
        pass

    # =========================================================================
    # Memory Stats
    # =========================================================================

    def memory_stats(self) -> dict:
        cpu_bytes = sum(t.nbytes for t in self._pool.values())
        gpu_store = sum(t.nbytes for t in self._store_staging)
        gpu_fetch = sum(t.nbytes for t in self._fetch_s_staging) + \
                    sum(t.nbytes for t in self._fetch_m_staging)
        if self.device is not None:
            gpu_allocated = torch.cuda.memory_allocated(self.device)
            gpu_peak      = torch.cuda.max_memory_allocated(self.device)
        else:
            gpu_allocated = gpu_peak = 0
        return {
            "cpu_pinned_bytes":        cpu_bytes,
            "gpu_fetch_staging_bytes": gpu_fetch,
            "gpu_store_staging_bytes": gpu_store,
            "gpu_allocated_bytes":     gpu_allocated,
            "gpu_peak_bytes":          gpu_peak,
        }

    def print_stats(self) -> None:
        s = self.memory_stats()
        print(
            f"[OffloadManager] "
            f"CPU pinned: {_fmt(s['cpu_pinned_bytes'])} | "
            f"GPU fetch staging: {_fmt(s['gpu_fetch_staging_bytes'])} | "
            f"GPU store staging: {_fmt(s['gpu_store_staging_bytes'])} | "
            f"GPU allocated: {_fmt(s['gpu_allocated_bytes'])} | "
            f"GPU peak: {_fmt(s['gpu_peak_bytes'])}",
            flush=True,
        )

    def reset_lens(self) -> None:
        """
        重置 per-layer 长度 & 形状记录（不释放 pinned pool）。
        在同一视频内更新 frozen 区域后调用，避免重新 cudaHostAlloc。
        """
        for cond in self._pool:
            L = self._layer_num
            self._s_len[cond]        = [0] * L
            self._m_len[cond]        = [0] * L
            self._s_orig_shape[cond] = [None] * L
            self._m_orig_shape[cond] = [None] * L
        self._fetch_s_slot_map.clear()
        self._fetch_m_slot_map.clear()
        self._store_slot   = 0
        self._fetch_s_slot = 0
        self._fetch_m_slot = 0

    def clear(self) -> None:
        """释放所有资源（推理结束后调用）。"""
        self._pool.clear()
        self._store_staging.clear()
        self._fetch_s_staging.clear()
        self._fetch_m_staging.clear()
        self._store_slot_done.clear()
        self._s_fetch_events.clear()
        self._m_fetch_events.clear()
        self._s_store_events.clear()
        self._m_store_events.clear()
        self._fetch_s_slot_map.clear()
        self._fetch_m_slot_map.clear()
        self._s_len.clear()
        self._m_len.clear()
        self._s_orig_shape.clear()
        self._m_orig_shape.clear()
        self._staging_ready  = False
        self._flat_full      = 0
        self._head_dim       = 0
        self._layer_num      = 0
        self._store_slot     = 0
        self._fetch_s_slot   = 0
        self._fetch_m_slot   = 0
        self.device          = None
        self.fetch_stream    = None
        self.store_stream    = None
        self._pool_dtype     = None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_OFFLOAD_MANAGER: Optional[OffloadManager] = None


def init_offload_manager(
    device: Optional[torch.device] = None,
    prefetch_window: int = 1,
    layer_num: int = 0,   # 兼容旧调用，内部不使用
) -> OffloadManager:
    global _OFFLOAD_MANAGER
    _OFFLOAD_MANAGER = OffloadManager(device=device, prefetch_window=prefetch_window)
    return _OFFLOAD_MANAGER


def get_offload_manager() -> Optional[OffloadManager]:
    return _OFFLOAD_MANAGER
