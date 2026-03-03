"""
Pool-Based Dual-Stream Offload Manager for Jano
-------------------------------------------------
设计原则
--------
所有 KV 的 pinned CPU memory 以 [layer_num, *shape] 的池形式一次性分配，
pinned 分配在后台线程中执行，与第一个 step_level==3 的 GPU layer-0 计算重叠。

CPU Pinned Memory Pool
    _s_pool[cond]: Tensor[L, *s_shape]  — 所有层 static  KV 共享一块连续内存
    _m_pool[cond]: Tensor[L, *m_shape]  — 所有层 medium  KV 共享一块连续内存
    索引方式: pool[layer_idx]  → 对应层的 pinned view，零拷贝

GPU Staging Buffers（循环复用，固定 k 个，store/fetch 各一组）
    形状与 s/m_shape 对应，多个 cond 共享（cond 之间串行）
    store staging: GPU→GPU 同步隔离源 tensor → 异步 DMA 到 pinned pool
    fetch staging: 从 pinned pool 异步 H2D → 滑动窗口预取

Background Thread
    start_preallocate() 在 step_level==3 的 layer_idx==0 时启动后台线程
    主线程在 layer_idx>=0 的 store_async / begin_fetch_step 调用前 join
    两次 cudaHostAlloc（s_pool + m_pool）与 GPU layer-0 计算并行执行
"""

from __future__ import annotations

import threading
import time
import torch
from typing import Dict, List, Optional, Tuple

from utils.timer import get_timer


def _fmt(n: int) -> str:
    return f"{n / 1024 ** 3:.2f} GB"


class OffloadManager:
    """
    Pool-based + 滑动窗口预取的 Offload Manager。

    使用流程
    --------
    1. step_level==3, layer_idx==0 时调用 start_preallocate()，
       在后台线程完成两次 cudaHostAlloc，与 GPU layer-0 计算重叠。
    2. 同一 step 后续 store_async(tensor, layer_idx, kv_type, cond) 写入 pool。
    3. step_level==1/2 时，begin_fetch_step() + fetch() 滑动窗口回读。
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        prefetch_window: int = 1,
    ):
        self._device_arg = device
        self.device: Optional[torch.device] = None
        self.k = prefetch_window

        # ── 后台分配线程 ────────────────────────────────────────────────────
        # cond → Thread（pool 分配线程）
        self._pool_threads: Dict[str, threading.Thread] = {}

        # ── CPU Pinned Memory Pools ─────────────────────────────────────────
        # cond → Tensor[layer_num, *shape]  (pinned CPU)
        self._s_pool: Dict[str, torch.Tensor] = {}
        self._m_pool: Dict[str, torch.Tensor] = {}

        # ── GPU Staging Buffers（所有 cond 共享，因 cond 之间串行）──────────
        # Store staging（GPU 端临时缓冲，隔离源 tensor 生命期）
        self._s_store_staging: List[torch.Tensor] = []   # k × s_shape
        self._m_store_staging: List[torch.Tensor] = []   # k × m_shape
        # Fetch staging（H2D DMA 目标，滑动窗口）
        self._s_fetch_staging: List[torch.Tensor] = []
        self._m_fetch_staging: List[torch.Tensor] = []
        # staging 已初始化标志（首次 _ensure_cond_ready 时分配）
        self._staging_ready: bool = False

        # ── Store staging 槽位循环 ─────────────────────────────────────────
        self._s_store_slot: int = 0
        self._m_store_slot: int = 0
        # 每个 store staging 槽位的 DMA 完成事件（用于复用前等待）
        self._s_store_slot_done: List[torch.cuda.Event] = []
        self._m_store_slot_done: List[torch.cuda.Event] = []

        # ── CUDA Streams & Events ──────────────────────────────────────────
        self.fetch_stream: Optional[torch.cuda.Stream] = None
        self.store_stream: Optional[torch.cuda.Stream] = None
        # per-(cond, layer) 事件
        self._s_fetch_events: Dict[str, List[torch.cuda.Event]] = {}
        self._m_fetch_events: Dict[str, List[torch.cuda.Event]] = {}
        self._s_store_events: Dict[str, List[torch.cuda.Event]] = {}
        self._m_store_events: Dict[str, List[torch.cuda.Event]] = {}

        # ── Fetch 滑动窗口状态（每 fetch step 重置）─────────────────────────
        self._fetch_cond: str = "0"
        self._fetch_step_level: int = 0
        self._fetch_num_layers: int = 0
        self._fetch_next_layer: int = 0
        self._s_fetch_slot: int = 0
        self._m_fetch_slot: int = 0
        self._s_fetch_slot_map: Dict[int, int] = {}   # layer_idx → staging slot
        self._m_fetch_slot_map: Dict[int, int] = {}

        # ── 形状缓存（首次 start_preallocate 后填充）────────────────────────
        self._s_shape: Optional[tuple] = None
        self._m_shape: Optional[tuple] = None
        self._s_dtype: Optional[torch.dtype] = None
        self._m_dtype: Optional[torch.dtype] = None
        self._layer_num: int = 0

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
        """在主线程完成 GPU staging buffer 和 slot events 的分配（快，μs 级）。"""
        if self._staging_ready:
            return
        assert self._s_shape is not None, "start_preallocate 必须在此之前调用"
        k = self.k
        with get_timer("offload.ensure_staging"):
            for _ in range(k):
                self._s_store_staging.append(
                    torch.empty(self._s_shape, dtype=self._s_dtype, device=self.device)
                )
                self._m_store_staging.append(
                    torch.empty(self._m_shape, dtype=self._m_dtype, device=self.device)
                )
                self._s_fetch_staging.append(
                    torch.empty(self._s_shape, dtype=self._s_dtype, device=self.device)
                )
                self._m_fetch_staging.append(
                    torch.empty(self._m_shape, dtype=self._m_dtype, device=self.device)
                )
                # store staging 槽位完成事件（初始 record 一次让 query 不阻塞）
                ev_s = torch.cuda.Event()
                ev_s.record()   # 初始化为"已完成"
                self._s_store_slot_done.append(ev_s)
                ev_m = torch.cuda.Event()
                ev_m.record()
                self._m_store_slot_done.append(ev_m)
        self._staging_ready = True

    def _ensure_cond_events(self, cond: str) -> None:
        """为该 cond 分配 per-layer CUDA events（主线程，μs 级）。"""
        L = self._layer_num
        if cond not in self._s_fetch_events:
            self._s_fetch_events[cond] = [torch.cuda.Event() for _ in range(L)]
            self._m_fetch_events[cond] = [torch.cuda.Event() for _ in range(L)]
            self._s_store_events[cond] = [torch.cuda.Event() for _ in range(L)]
            self._m_store_events[cond] = [torch.cuda.Event() for _ in range(L)]

    def _ensure_cond_ready(self, cond: str) -> None:
        """等待该 cond 的后台分配线程完成，然后初始化 staging 和 events。"""
        if cond in self._pool_threads:
            with get_timer(f"offload.pool_wait[{cond}]"):
                self._pool_threads.pop(cond).join()
        # 线程已完成：pool 和形状已填充，现在初始化 staging/events
        self._ensure_staging()
        self._ensure_cond_events(cond)

    # =========================================================================
    # 后台预分配
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
        """
        在后台线程中为 cond 一次性分配两块 pinned memory pool：
            s_pool[cond]: Tensor[layer_num, *s_shape]
            m_pool[cond]: Tensor[layer_num, *m_shape]

        在 step_level==3 的 layer_idx==0 时调用，与 GPU 计算重叠。
        幂等：已分配过则直接返回。
        """
        if cond in self._s_pool:
            return   # 已分配

        self._ensure_streams()

        # 记录形状（主线程保存，供 _ensure_staging 使用）
        if self._s_shape is None:
            self._s_shape = tuple(s_shape)
            self._m_shape = tuple(m_shape)
            self._s_dtype = s_dtype
            self._m_dtype = m_dtype
            self._layer_num = layer_num

        def _alloc(c: str, L: int, ss: tuple, sd, ms: tuple, md):
            t0 = time.perf_counter()
            s_pool = torch.empty((L, *ss), dtype=sd, pin_memory=True)
            m_pool = torch.empty((L, *ms), dtype=md, pin_memory=True)
            elapsed = (time.perf_counter() - t0) * 1e3
            print(
                f"[OffloadManager] pool alloc cond={c}  "
                f"s_pool={s_pool.nbytes/1024**2:.1f}MB  "
                f"m_pool={m_pool.nbytes/1024**2:.1f}MB  "
                f"elapsed={elapsed:.2f}ms",
                flush=True,
            )
            self._s_pool[c] = s_pool
            self._m_pool[c] = m_pool

        t = threading.Thread(
            target=_alloc,
            args=(cond, layer_num, s_shape, s_dtype, m_shape, m_dtype),
            daemon=True,
        )
        self._pool_threads[cond] = t
        t.start()

    def is_ready(self, cond: Optional[str] = None) -> bool:
        """
        检查 pool 是否就绪。
        cond=None 时只要有任意 cond 已就绪（或分配中）即返回 True。
        """
        if cond is not None:
            return cond in self._s_pool or cond in self._pool_threads
        return bool(self._s_pool) or bool(self._pool_threads)

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
        异步将 GPU tensor 存入 pool[cond][layer_idx]。

        流程：
          1. tensor → store_staging[slot]（同步 GPU→GPU，隔离源 tensor）
          2. store_staging[slot] → pool[cond][layer_idx]（异步 D2H DMA）

        kv_type: 's'（static）或 'm'（medium）
        """
        self._ensure_cond_ready(cond)

        if kv_type == "s":
            pool        = self._s_pool[cond]
            staging_lst = self._s_store_staging
            slot_done   = self._s_store_slot_done
            store_ev    = self._s_store_events[cond][layer_idx]
            slot        = self._s_store_slot % self.k
            self._s_store_slot += 1
        else:
            pool        = self._m_pool[cond]
            staging_lst = self._m_store_staging
            slot_done   = self._m_store_slot_done
            store_ev    = self._m_store_events[cond][layer_idx]
            slot        = self._m_store_slot % self.k
            self._m_store_slot += 1

        # 确保此 staging 槽位上的上一条 DMA 已完成（通常已完成，~0μs）
        slot_done[slot].synchronize()

        staging = staging_lst[slot]

        # step1: 同步 GPU→GPU 拷贝到 staging，隔离源 tensor
        staging.copy_(tensor.detach(), non_blocking=False)

        # step2: staging → pinned pool[layer_idx] 异步 D2H DMA
        self.store_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.store_stream):
            pool[layer_idx].copy_(staging, non_blocking=True)
            slot_done[slot].record(self.store_stream)   # 记录此槽位 DMA 完成
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
        self._s_fetch_slot     = 0
        self._m_fetch_slot     = 0
        self._s_fetch_slot_map.clear()
        self._m_fetch_slot_map.clear()

        # 确保所有 store_stream D2H 写入完成后，fetch_stream 再发起 H2D 读取，
        # 防止 store D2H（写 CPU pool）与 fetch H2D（读 CPU pool）数据竞争。
        self.fetch_stream.wait_stream(self.store_stream)

        for _ in range(min(self.k, num_layers)):
            self._issue_prefetch_layer(self._fetch_next_layer)
            self._fetch_next_layer += 1

    def _issue_prefetch_layer(self, layer_idx: int) -> None:
        """在 fetch_stream 上为 layer_idx 发起异步 H2D DMA。"""
        if layer_idx < 0 or layer_idx >= self._fetch_num_layers:
            return
        cond = self._fetch_cond
        if cond not in self._s_pool:
            return

        # 等待当前计算流最新进度，防止 DMA 覆盖正在被读取的 staging slot
        self.fetch_stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self.fetch_stream):
            if self._fetch_step_level in (1, 2):
                slot = self._s_fetch_slot % self.k
                self._s_fetch_slot += 1
                self._s_fetch_staging[slot].copy_(
                    self._s_pool[cond][layer_idx], non_blocking=True
                )
                self._s_fetch_events[cond][layer_idx].record(self.fetch_stream)
                self._s_fetch_slot_map[layer_idx] = slot

            if self._fetch_step_level == 1:
                slot = self._m_fetch_slot % self.k
                self._m_fetch_slot += 1
                self._m_fetch_staging[slot].copy_(
                    self._m_pool[cond][layer_idx], non_blocking=True
                )
                self._m_fetch_events[cond][layer_idx].record(self.fetch_stream)
                self._m_fetch_slot_map[layer_idx] = slot

    def _issue_prefetch_fallback(self, layer_idx: int, kv_type: str) -> None:
        """Fallback：layer 未在预取窗口内时同步发起。"""
        print(
            f"[OffloadManager] Fallback prefetch: {kv_type}_kv_{layer_idx}",
            flush=True,
        )
        cond = self._fetch_cond
        self.fetch_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.fetch_stream):
            if kv_type == "s":
                slot = self._s_fetch_slot % self.k
                self._s_fetch_slot += 1
                self._s_fetch_staging[slot].copy_(
                    self._s_pool[cond][layer_idx], non_blocking=True
                )
                self._s_fetch_events[cond][layer_idx].record(self.fetch_stream)
                self._s_fetch_slot_map[layer_idx] = slot
            else:
                slot = self._m_fetch_slot % self.k
                self._m_fetch_slot += 1
                self._m_fetch_staging[slot].copy_(
                    self._m_pool[cond][layer_idx], non_blocking=True
                )
                self._m_fetch_events[cond][layer_idx].record(self.fetch_stream)
                self._m_fetch_slot_map[layer_idx] = slot

    def fetch(self, layer_idx: int, kv_type: str, cond: str) -> torch.Tensor:
        """
        等待 layer_idx 的预取完成，返回 GPU staging tensor。

        kv_type: 's'（static）或 'm'（medium）
        """
        self._ensure_cond_ready(cond)

        if kv_type == "s":
            if layer_idx not in self._s_fetch_slot_map:
                self._issue_prefetch_fallback(layer_idx, "s")
            torch.cuda.current_stream().wait_event(
                self._s_fetch_events[cond][layer_idx]
            )
            # ⚠ 必须先 clone，再触发下一层预取。
            # 若直接持有 staging 引用后再触发预取，_issue_prefetch_layer 内的
            # fetch_stream.wait_stream(current_stream) 只能看到此时 current_stream
            # 的快照（仅含 wait_event，尚未提交 attention 计算），fetch_stream
            # 因此提前启动 H2D DMA 并覆盖 staging[slot]，
            # 导致调用方从 staging 中读取到下一层的错误数据。
            result = self._s_fetch_staging[self._s_fetch_slot_map[layer_idx]].clone()
        else:
            if layer_idx not in self._m_fetch_slot_map:
                self._issue_prefetch_fallback(layer_idx, "m")
            torch.cuda.current_stream().wait_event(
                self._m_fetch_events[cond][layer_idx]
            )
            result = self._m_fetch_staging[self._m_fetch_slot_map[layer_idx]].clone()

        # 触发下一层预取（每层仅在 's' 时触发一次，避免重复）
        # clone() 之后 staging slot 与返回张量已解耦，可安全复用。
        if kv_type == "s":
            self._issue_prefetch_layer(self._fetch_next_layer)
            self._fetch_next_layer += 1

        return result

    # =========================================================================
    # 兼容旧接口（空操作）
    # =========================================================================

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
        cpu_bytes = sum(t.nbytes for t in self._s_pool.values()) + \
                    sum(t.nbytes for t in self._m_pool.values())
        gpu_fetch = sum(t.nbytes for t in self._s_fetch_staging) + \
                    sum(t.nbytes for t in self._m_fetch_staging)
        gpu_store = sum(t.nbytes for t in self._s_store_staging) + \
                    sum(t.nbytes for t in self._m_store_staging)
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

    def clear(self) -> None:
        """释放所有资源（推理结束后调用）。"""
        # 等待所有后台线程
        for t in self._pool_threads.values():
            t.join()
        self._pool_threads.clear()
        self._s_pool.clear()
        self._m_pool.clear()
        self._s_store_staging.clear()
        self._m_store_staging.clear()
        self._s_fetch_staging.clear()
        self._m_fetch_staging.clear()
        self._s_store_slot_done.clear()
        self._m_store_slot_done.clear()
        self._s_fetch_events.clear()
        self._m_fetch_events.clear()
        self._s_store_events.clear()
        self._m_store_events.clear()
        self._s_fetch_slot_map.clear()
        self._m_fetch_slot_map.clear()
        self._staging_ready   = False
        self._s_shape         = None
        self._m_shape         = None
        self._layer_num       = 0
        self._s_store_slot    = 0
        self._m_store_slot    = 0
        self.device           = None
        self.fetch_stream     = None
        self.store_stream     = None


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
