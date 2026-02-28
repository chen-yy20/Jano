"""
Dual-Stream Prefetch Offload Manager for Jano

This module provides a modular OffloadManager that any Jano model can use to
offload cached feature/KV tensors from GPU to CPU, and efficiently prefetch
them back using a dual-stream pipeline that overlaps data transfers with GPU
compute.

Core design
-----------
* CPU side  : page-locked (pinned) memory buffers per cache entry, enabling
              maximum-bandwidth DMA transfers.
* GPU side  : a staging buffer per cache entry that receives the prefetched
              data from the DMA engine.
* data_stream   : dedicated CUDA stream for all H2D / D2H copies.
* default stream: used for compute (the caller's stream, typically the
                  autocast/no_grad context).
* CUDA Events   : per-entry events recorded on the data_stream after each
                  prefetch; the compute stream calls wait_event before
                  consuming the staging buffer.

Pipeline (for a fetch step with N layers)
------------------------------------------
  begin_fetch_step(keys)           # issue all N prefetches on data_stream
                                   # ↑ DMA engine runs concurrently with GPU
  for i in range(N):
      compute layer i ...
      tensor_i = fetch(key_i)      # wait_event(events[key_i]) → returns
                                   #   gpu_staging[key_i] (already populated)

This overlaps CPU→GPU transfers for later layers with the GPU compute of
earlier layers, hiding the transfer latency almost entirely.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from utils.envs import GlobalEnv


def _format_bytes(n: int) -> str:
    return f"{n / 1024 ** 3:.2f} GB"


class OffloadManager:
    """
    Modular dual-stream prefetch offload manager.

    Any Jano mask-manager (or model) can create one instance of this class
    and use the three main operations:

    * ``store_async(tensor, key)``  – GPU → CPU pinned (on data_stream)
    * ``begin_fetch_step(keys)``    – issue all CPU → GPU staging prefetches
    * ``fetch(key)``                – wait for event, return GPU staging tensor

    Parameters
    ----------
    device : torch.device or str, optional
        CUDA device to allocate staging buffers on. Defaults to the current
        CUDA device.
    """

    def __init__(self, device: Optional[torch.device] = None):
        if not torch.cuda.is_available():
            raise RuntimeError("OffloadManager requires a CUDA-capable GPU.")

        self.device = device or torch.device(f"cuda:{torch.cuda.current_device()}")

        # Dedicated stream for all data transfers (D2H and H2D)
        self.data_stream: torch.cuda.Stream = torch.cuda.Stream(device=self.device)

        # CPU pinned-memory buffers: key → 1-D/N-D cpu tensor (pin_memory=True)
        self._pinned: Dict[str, torch.Tensor] = {}

        # GPU staging buffers: key → gpu tensor (same shape/dtype as pinned)
        self._staging: Dict[str, torch.Tensor] = {}

        # Per-key CUDA events (recorded on data_stream after each H2D copy)
        self._events: Dict[str, torch.cuda.Event] = {}

        # Set of keys whose H2D prefetch has been issued but not yet consumed
        self._prefetch_issued: set = set()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_pinned(self, tensor: torch.Tensor, key: str) -> None:
        """Allocate / reallocate a pinned CPU buffer for *key* if needed."""
        existing = self._pinned.get(key)
        if existing is None or existing.shape != tensor.shape or existing.dtype != tensor.dtype:
            self._pinned[key] = torch.empty(
                tensor.shape, dtype=tensor.dtype, pin_memory=True
            )

    def _ensure_staging(self, key: str) -> None:
        """Allocate / reallocate a GPU staging buffer matching the pinned buf."""
        pinned = self._pinned[key]
        existing = self._staging.get(key)
        if existing is None or existing.shape != pinned.shape or existing.dtype != pinned.dtype:
            self._staging[key] = torch.empty(
                pinned.shape, dtype=pinned.dtype, device=self.device
            )
        if key not in self._events:
            self._events[key] = torch.cuda.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store_async(self, tensor: torch.Tensor, key: str) -> None:
        """
        Asynchronously copy *tensor* (on GPU) to a CPU pinned-memory buffer.

        The copy is enqueued on ``self.data_stream`` and returns immediately;
        the caller must *not* modify or free *tensor* until the data_stream
        has advanced past this point (a stream synchronise or a later
        event-wait is sufficient).

        Parameters
        ----------
        tensor : torch.Tensor
            GPU tensor to store.
        key : str
            Unique identifier for this cache entry.
        """
        self._ensure_pinned(tensor, key)
        with torch.cuda.stream(self.data_stream):
            self._pinned[key].copy_(tensor, non_blocking=True)

    def begin_fetch_step(self, keys: List[str]) -> None:
        """
        Issue async H2D prefetches for all *keys* at the start of a fetch
        step.

        Enqueuing all copies at once lets the DMA engine saturate the PCIe
        bandwidth while GPU compute proceeds through the transformer layers.
        Each copy records a per-key CUDA event on the data_stream; the
        ``fetch`` method will wait for that event on the compute stream.

        Parameters
        ----------
        keys : list of str
            Cache keys to prefetch.  Keys not yet stored are silently skipped.
        """
        for key in keys:
            if key not in self._pinned:
                continue
            self._ensure_staging(key)
            with torch.cuda.stream(self.data_stream):
                self._staging[key].copy_(self._pinned[key], non_blocking=True)
                self._events[key].record()
            self._prefetch_issued.add(key)

    def fetch(self, key: str) -> Optional[torch.Tensor]:
        """
        Return the GPU staging buffer for *key*, blocking the current
        (compute) stream until the prefetch completes.

        If ``begin_fetch_step`` was not called for this key, a synchronous
        fallback prefetch is issued here (no overlap, but functionally
        correct).

        Parameters
        ----------
        key : str
            Cache key previously passed to ``store_async``.

        Returns
        -------
        torch.Tensor or None
            The GPU staging buffer (same shape/dtype as the original stored
            tensor), or *None* if the key was never stored.
        """
        if key not in self._pinned:
            return None

        # Fallback: issue prefetch now if not already in flight
        if key not in self._prefetch_issued:
            self._ensure_staging(key)
            with torch.cuda.stream(self.data_stream):
                self._staging[key].copy_(self._pinned[key], non_blocking=True)
                self._events[key].record()
            self._prefetch_issued.add(key)

        # Make the compute stream wait for the DMA to finish
        torch.cuda.current_stream().wait_event(self._events[key])
        self._prefetch_issued.discard(key)

        return self._staging[key]

    def has_key(self, key: str) -> bool:
        """Return True if *key* has been stored."""
        return key in self._pinned

    def clear(self) -> None:
        """
        Release all pinned-memory and GPU staging buffers.

        Should be called at the end of an inference session (e.g. after all
        denoising steps) to free resources.
        """
        self._pinned.clear()
        self._staging.clear()
        self._events.clear()
        self._prefetch_issued.clear()

    def print_memory_stats(self) -> None:
        """Print the total size of all pinned CPU and GPU staging buffers."""
        cpu_total = sum(
            t.element_size() * t.numel() for t in self._pinned.values()
        )
        gpu_total = sum(
            t.element_size() * t.numel() for t in self._staging.values()
        )
        print(
            f"[OffloadManager] pinned CPU: {_format_bytes(cpu_total)}, "
            f"GPU staging: {_format_bytes(gpu_total)}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Module-level singleton helpers (mirrors the pattern used by other managers)
# ---------------------------------------------------------------------------

_OFFLOAD_MANAGER: Optional[OffloadManager] = None


def init_offload_manager(num_layers: int = 0, device: Optional[torch.device] = None) -> OffloadManager:
    """
    Create and register the global OffloadManager instance.

    Parameters
    ----------
    num_layers : int, optional
        Kept for API compatibility; not used by OffloadManager internally.
    device : torch.device, optional
        CUDA device for staging buffers.

    Returns
    -------
    OffloadManager
    """
    global _OFFLOAD_MANAGER
    _OFFLOAD_MANAGER = OffloadManager(device=device)
    GlobalEnv.set_envs("offload_manager", _OFFLOAD_MANAGER)
    return _OFFLOAD_MANAGER


def get_offload_manager() -> Optional[OffloadManager]:
    """
    Return the global OffloadManager instance, or *None* if not initialised.
    """
    global _OFFLOAD_MANAGER
    return _OFFLOAD_MANAGER
