import os
from math import gcd
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Union

import torch
import torch.distributed

from utils.logger import init_logger
from .group_coordinate import GroupCoordinator

logger = init_logger(__name__)

_WORLD: Optional[GroupCoordinator] = None
_CFG: Optional[GroupCoordinator] = None


def get_world_group() -> GroupCoordinator:
  assert _WORLD is not None, ("world group is not initialized")
  return _WORLD

def get_cp_group() -> GroupCoordinator:
  assert _CFG is not None, ("classifier free guidance parallel group is not initialized")
  return _CFG

def get_cp_worldsize() -> int:
    if _CFG is not None:
        return _CFG.world_size
    else:
        return 1

def init_world_group(ranks: List[int], local_rank: int, backend: str) -> GroupCoordinator:
  return GroupCoordinator(
    group_ranks=[ranks],
    local_rank=local_rank,
    torch_distributed_backend=backend,
    group_name="world",
  )

def init_cp_group(
        group_ranks: List[List[int]],
        local_rank: int,
        backend: str,
    ) -> GroupCoordinator:
    global _CFG
    for group in group_ranks:
        assert len(group) <= 2, f'cfg_size can only be 1 or 2'
    _CFG =  GroupCoordinator(
        group_ranks=group_ranks,
        local_rank=local_rank,
        torch_distributed_backend=backend,
        group_name='cfg',
    )
    return _CFG

def init_distributed_environment(
  world_size: int = -1,
  rank: int = -1,
  distributed_init_method: str = "env://",
  local_rank: int = -1,
  backend: str = "nccl",
):
  logger.debug("world_size=%d rank=%d local_rank=%d distributed_init_method=%s backend=%s", world_size, rank, local_rank, distributed_init_method, backend)
  if not torch.distributed.is_initialized():
    assert distributed_init_method is not None, ("distributed_init_method must be provided when initializing "
                                                 "distributed environment")
    # this backend is used for WORLD
    torch.distributed.init_process_group(backend=backend, init_method=distributed_init_method, world_size=world_size, rank=rank)
  # set the local rank
  # local_rank is not available in torch ProcessGroup,
  # see https://github.com/pytorch/pytorch/issues/122816
  if local_rank == -1:
    # local rank not set, this usually happens in single-node
    # setting, where we can use rank as local rank
    if distributed_init_method == "env://":
      local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    else:
      local_rank = rank
  global _WORLD
  if _WORLD is None:
    ranks = list(range(torch.distributed.get_world_size()))
    _WORLD = init_world_group(ranks, local_rank, backend)
  else:
    assert _WORLD.world_size == torch.distributed.get_world_size(), ("world group already initialized with a different world size")


def destroy_distributed_environment():
  global _WORLD
  if _WORLD:
    _WORLD.destroy()
  _WORLD = None
  if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()