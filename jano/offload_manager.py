"""
Dual-Stream Prefetch Offload Manager for Jano

设计原则
--------
Warmup结束后模型层数、moderate/static的shape完全确定，因此：

CPU Pinned Memory（持久，预分配）
    key = f"{state_key}_{kv_type}"   e.g. "cond_0_5_s_kv", "cond_1_5_m_kv"
    每层每类型一个pinned buffer，warmup结束后一次性分配
    总量固定，DMA直接访问，无运行时内存分配开销

GPU Staging Buffer（循环复用，固定 k 个）
    每种独立shape各有 k 个 fetch staging（CPU→GPU 方向）
    每种独立shape各有 k 个 store staging（GPU→CPU 中转）
    运行时不新增/释放GPU内存

Prefetch策略（以k为步长的滑动窗口）
    begin_fetch_step(key_groups) 预取前k组
    每层第一个 fetch(key) 触发下一组的预取，与当前层计算重叠
    任意时刻 GPU staging buffer 数量 ≤ k × keys_per_layer

Store策略
    store_async(tensor, key): tensor.detach() → store staging → pinned (DMA)
    避免原始tensor被后续计算修改引发数据冒险
"""

from __future__ import annotations

import torch
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from utils.envs import GlobalEnv


def _fmt(n: int) -> str:
    return f"{n / 1024 ** 3:.2f} GB"


# (shape_tuple, dtype) → staging buffer pool的索引键
_ShapeKey = Tuple[tuple, torch.dtype]


class OffloadManager:
    """
    按需渐进分配 + 滑动窗口预取的 Offload Manager。

    使用流程
    --------
    1. 每个 step_level==3 步中，对每个 key 调用 register_shape(key, ...)：
       首次调用立即完成 CPU pinned buffer、GPU staging buffers、CUDA streams/events
       的分配；后续重复调用为幂等空操作。
    2. register_shape() 返回后即可调用 store_async() 将 KV 写入 CPU。
    3. step_level==1/2 时，begin_fetch_step() + fetch() 以滑动窗口方式回读。
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        prefetch_window: int = 1,
    ):
        # 设备延迟确定：__init__ 时 CUDA 可能未就绪，统一用 current_device()
        # build_pinned_buffers() 中再真正使用 self.device
        self._device_arg = device
        self.device: Optional[torch.device] = None  # 延迟到 build_pinned_buffers 赋值
        self.k = prefetch_window

        # ── CPU Pinned Memory ─────────────────────────────────────────────
        self._pinned: Dict[str, torch.Tensor] = {}
        self._warmup_done: bool = False
        # warmup 阶段收集的 shape：key → (shape, dtype)
        self._registered: Dict[str, Tuple[tuple, torch.dtype]] = {}

        # ── GPU Staging Buffers（固定 k 个，按 shape 分组）───────────────
        # shape_key → List[k 个 GPU tensor]
        self._fetch_staging: Dict[_ShapeKey, List[torch.Tensor]] = {}
        self._store_staging: Dict[_ShapeKey, List[torch.Tensor]] = {}
        # shape_key → 当前累计使用次数（% k 得到 slot）
        self._fetch_slot: Dict[_ShapeKey, int] = defaultdict(int)
        self._store_slot: Dict[_ShapeKey, int] = defaultdict(int)

        # ── CUDA Streams & Events（延迟到 build_pinned_buffers 创建）──────
        self.fetch_stream: Optional[torch.cuda.Stream] = None
        self.store_stream: Optional[torch.cuda.Stream] = None
        # key → CUDA event（每步 DMA 完成后 record）
        self._fetch_events: Dict[str, torch.cuda.Event] = {}
        self._store_events: Dict[str, torch.cuda.Event] = {}
        # key → (sk, slot) 记录本次预取用的 staging buffer 位置
        self._fetch_slot_map: Dict[str, Tuple[_ShapeKey, int]] = {}

        # ── 滑动窗口状态（每 step 重置）───────────────────────────────────
        self._key_groups: List[List[str]] = []
        self._next_prefetch_group: int = 0
        self._key_to_group_idx: Dict[str, int] = {}

    # =========================================================================
    # 资源分配
    # =========================================================================

    def _ensure_streams(self) -> None:
        """延迟初始化 CUDA device / streams（首次 register_shape 时调用）。"""
        if self.fetch_stream is not None:
            return
        self.device = self._device_arg or torch.device(
            f"cuda:{torch.cuda.current_device()}"
        )
        self.fetch_stream = torch.cuda.Stream(device=self.device)
        self.store_stream = torch.cuda.Stream(device=self.device)

    def register_shape(self, key: str, shape: tuple, dtype: torch.dtype) -> None:
        """
        注册 key 并立即分配 CPU pinned buffer、CUDA events 及 GPU staging buffers。
        幂等：重复注册同一 key 不会重复分配。
        调用后即可直接使用 store_async，无需等待 build_pinned_buffers()。
        """
        if key in self._registered:
            return
        self._ensure_streams()
        self._registered[key] = (tuple(shape), dtype)

        # 立即分配 CPU pinned buffer 和 CUDA events
        self._pinned[key] = torch.empty(shape, dtype=dtype, pin_memory=True, device="cpu")
        self._fetch_events[key] = torch.cuda.Event()
        self._store_events[key] = torch.cuda.Event()

        # 若该 shape 尚无 staging buffers，一并分配
        sk: _ShapeKey = (tuple(shape), dtype)
        if sk not in self._fetch_staging:
            self._fetch_staging[sk] = [
                torch.empty(shape, dtype=dtype, device=self.device)
                for _ in range(self.k)
            ]
            self._store_staging[sk] = [
                torch.empty(shape, dtype=dtype, device=self.device)
                for _ in range(self.k)
            ]
            self._fetch_slot[sk] = 0
            self._store_slot[sk] = 0

    def build_pinned_buffers(self) -> None:
        """[已废弃] 历史兼容接口。register_shape() 现在立即分配所有资源，本方法为空操作。"""
        pass

    # =========================================================================
    # Store（GPU → CPU，双流流水线）
    # =========================================================================

    def store_async(self, tensor: torch.Tensor, key: str) -> None:
        """
        异步将 GPU tensor 存入对应的 CPU pinned buffer。

        流程：
          1. tensor.detach() 在默认流做**同步**拷贝到 store staging（隔离原始 tensor）
          2. 在 store_stream 上从 staging async DMA 到 pinned memory
          3. record store_events[key]

        主调方无需等待，可继续计算。
        """
        assert key in self._pinned, f"未注册的 key: {key}，请先调用 register_shape()"

        sk: _ShapeKey = (tuple(self._pinned[key].shape), self._pinned[key].dtype)
        slot = self._store_slot[sk] % self.k
        self._store_slot[sk] += 1

        staging = self._store_staging[sk][slot]
        pinned = self._pinned[key]

        # step1: 同步拷贝到 staging（完全隔离原 tensor，避免数据冒险）
        staging.copy_(tensor.detach(), non_blocking=False)

        # step2: staging → pinned async DMA
        # 必须让 store_stream 等待默认流（或当前计算流）上的 GPU-to-GPU 拷贝完成
        self.store_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.store_stream):
            pinned.copy_(staging, non_blocking=True)
            self._store_events[key].record(self.store_stream)

    def wait_store(self, key: str) -> None:
        """阻塞等待指定 key 的 store DMA 完成。"""
        self._store_events[key].synchronize()

    # =========================================================================
    # Fetch（CPU → GPU，滑动窗口预取）
    # =========================================================================

    def begin_fetch_step(self, key_groups: List[List[str]]) -> None:
        """
        每个推理 step 的 fetch 阶段开始时调用。

        Parameters
        ----------
        key_groups : List[List[str]]
            按层排列的 key 分组，index i 对应第 i 层。
            例：[["cond0_0_s_kv","cond0_0_m_kv"], ["cond0_1_s_kv","cond0_1_m_kv"], ...]

        行为：重置窗口状态，预取前 k 组；其余组由 fetch() 触发。
        """
        self._key_groups = key_groups
        self._next_prefetch_group = 0
        self._fetch_slot_map.clear()

        # 构建 key → group_idx 映射，O(1) 查找
        self._key_to_group_idx = {
            key: g_idx
            for g_idx, group in enumerate(key_groups)
            for key in group
        }

        # 重置 fetch slot（每 step 从 0 开始，保证循环复用正确）
        for sk in self._fetch_slot:
            self._fetch_slot[sk] = 0

        # 预取前 k 组
        for _ in range(min(self.k, len(key_groups))):
            self._issue_prefetch_group(self._next_prefetch_group)
            self._next_prefetch_group += 1

    def _issue_prefetch_group(self, group_idx: int) -> None:
        """在 fetch_stream 上为第 group_idx 层的所有 key 发起异步 H2D DMA。"""
        if group_idx < 0 or group_idx >= len(self._key_groups):
            return
        
        # 必须让 fetch_stream 等待计算流到当前时刻，以防它跑得太快覆盖正在计算的 buffer
        self.fetch_stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self.fetch_stream):
            for key in self._key_groups[group_idx]:
                if key not in self._pinned:
                    continue
                sk: _ShapeKey = (
                    tuple(self._pinned[key].shape),
                    self._pinned[key].dtype,
                )
                slot = self._fetch_slot[sk] % self.k
                self._fetch_slot[sk] += 1
                staging = self._fetch_staging[sk][slot]
                staging.copy_(self._pinned[key], non_blocking=True)
                self._fetch_events[key].record(self.fetch_stream)
                # 记录 (sk, slot)，fetch() 时据此找到 staging buffer
                self._fetch_slot_map[key] = (sk, slot)

    def fetch(self, key: str) -> torch.Tensor:
        """
        等待 key 的预取 DMA 完成，返回 GPU staging buffer。

        顺序：
          1. Fallback（如未预取则先发起）
          2. wait_event：计算流等待 H2D DMA 完成
          3. 触发下一组预取（此时 staging 已被计算流"接管"，
             下一组可安全写入其他 slot）
        """
        assert key in self._pinned, f"未注册的 key: {key}"

        # Fallback：key 未被正常预取时，同步发起
        if key not in self._fetch_slot_map:
            self._issue_prefetch_fallback(key)

        # 先等待：计算流阻塞直到 DMA 完成，确保 staging 数据有效
        torch.cuda.current_stream().wait_event(self._fetch_events[key])

        # 再触发：staging 已就绪后，窗口滑动，下一组预取不与当前读冲突。
        # 条件：当前 key 是所在 group 的第一个 key，避免同一 group 内多次触发。
        # 注意 fetch_stream.wait_stream(current_stream) 在 _issue_prefetch_group 内部
        # 会等待 current_stream 上已入队的所有工作（包括上一层的 torch.cat），
        # 因此 k≥2 时不存在 DMA 覆盖正在读取的 staging buffer 的风险。
        # k=1 时 H2D PCIe DMA 调度延迟 >> GPU torch.cat，实践中同样安全。
        group_idx = self._find_group(key)
        if group_idx >= 0 and self._key_groups[group_idx][0] == key:
            self._issue_prefetch_group(self._next_prefetch_group)
            self._next_prefetch_group += 1

        sk, slot = self._fetch_slot_map[key]
        return self._fetch_staging[sk][slot]

    def _issue_prefetch_fallback(self, key: str) -> None:
        """在 fetch_stream 上为单个 key 发起预取（fallback，开销略高）。"""
        print(f"[OffloadManager] Fallback prefetch for key: {key}", flush=True)
        sk: _ShapeKey = (tuple(self._pinned[key].shape), self._pinned[key].dtype)
        slot = self._fetch_slot[sk] % self.k
        self._fetch_slot[sk] += 1
        staging = self._fetch_staging[sk][slot]
        
        # 同样需要等待计算流，避免覆盖正在被计算流使用的数据
        self.fetch_stream.wait_stream(torch.cuda.current_stream())
        
        with torch.cuda.stream(self.fetch_stream):
            staging.copy_(self._pinned[key], non_blocking=True)
            self._fetch_events[key].record(self.fetch_stream)
        self._fetch_slot_map[key] = (sk, slot)

    def _find_group(self, key: str) -> int:
        """O(1) 返回 key 所在 group 索引，找不到返回 -1。"""
        return self._key_to_group_idx.get(key, -1)

    def has_key(self, key: str) -> bool:
        return key in self._pinned

    # =========================================================================
    # Memory Stats
    # =========================================================================

    def memory_stats(self) -> dict:
        cpu_bytes = sum(t.nbytes for t in self._pinned.values())
        gpu_fetch = sum(
            t.nbytes for bufs in self._fetch_staging.values() for t in bufs
        )
        gpu_store = sum(
            t.nbytes for bufs in self._store_staging.values() for t in bufs
        )
        gpu_allocated = torch.cuda.memory_allocated(self.device)
        gpu_peak = torch.cuda.max_memory_allocated(self.device)
        return {
            "cpu_pinned_bytes": cpu_bytes,
            "gpu_fetch_staging_bytes": gpu_fetch,
            "gpu_store_staging_bytes": gpu_store,
            "gpu_allocated_bytes": gpu_allocated,
            "gpu_peak_bytes": gpu_peak,
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
        """释放所有 pinned memory（推理结束后调用）。"""
        self._pinned.clear()
        self._registered.clear()
        self._fetch_staging.clear()
        self._store_staging.clear()
        self._fetch_events.clear()
        self._store_events.clear()
        self._fetch_slot_map.clear()
        self._key_groups = []
        self._key_to_group_idx = {}
        self._next_prefetch_group = 0
        self._warmup_done = False
        self.device = None
        self.fetch_stream = None
        self.store_stream = None


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
