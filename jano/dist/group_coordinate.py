import pickle
import weakref
from collections import namedtuple
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from unittest.mock import patch

import torch
import torch.distributed
import torch.nn.functional as F
from torch.distributed import Backend, ProcessGroup

from utils.logger import init_logger

logger = init_logger(__name__)

_group_name_counter: Dict[str, int] = {}

# import toolbox


def _get_unique_name(name: str) -> str:
  """Get a unique name for the group.
    Example:
    _get_unique_name("tp") -> "tp:0"
    _get_unique_name("tp") -> "tp:1"
    """
  if name not in _group_name_counter:
    _group_name_counter[name] = 0
  newname = f"{name}:{_group_name_counter[name]}"
  _group_name_counter[name] += 1
  return newname


_groups: Dict[str, Callable[[], "GroupCoordinator"]] = {}


def _register_group(group: "GroupCoordinator") -> None:
  # looks like Python 3.8 does not understand `ReferenceType`
  _groups[group.unique_name] = weakref.ref(group)    # type: ignore

class GroupCoordinator:
  """
    PyTorch ProcessGroup wrapper for a group of processes.
    PyTorch ProcessGroup is bound to one specific communication backend,
        e.g. NCCL, Gloo, MPI, etc.
    GroupCoordinator takes charge of all the communication operations among
        the processes in the group. It can route the communication to
        a specific implementation (e.g. switch allreduce implementation
        based on the tensor size and cuda graph mode).
    """

  # available attributes:
  rank: int    # global rank
  ranks: List[int]    # global ranks in the group
  world_size: int    # size of the group
  # difference between `local_rank` and `rank_in_group`:
  # if we have a group of size 4 across two nodes:
  # Process | Node | Rank | Local Rank | Rank in Group
  #   0     |   0  |  0   |     0      |       0
  #   1     |   0  |  1   |     1      |       1
  #   2     |   1  |  2   |     0      |       2
  #   3     |   1  |  3   |     1      |       3
  local_rank: int    # local rank used to assign devices
  rank_in_group: int    # rank inside the group
  cpu_group: ProcessGroup    # group for CPU communication
  device_group: ProcessGroup    # group for device communication

  p2p_ops: List[torch.distributed.P2POp]

  def __init__(
    self,
    group_ranks: List[List[int]],
    local_rank: int,
    torch_distributed_backend: Union[str, Backend],
    group_name: Optional[str] = None,
  ):
    group_name = group_name or "anonymous"
    self.unique_name = _get_unique_name(group_name)
    _register_group(self)

    self.rank = torch.distributed.get_rank()
    self.local_rank = local_rank
    self.device_group = None
    self.cpu_group = None

    self.p2p_ops = []
    self.p2p_reqs = None

    for ranks in group_ranks:
      device_group = torch.distributed.new_group(ranks, backend=torch_distributed_backend)
      # a group with `gloo` backend, to allow direct coordination between
      # processes through the CPU.
      cpu_group = torch.distributed.new_group(ranks, backend="gloo")
      if self.rank in ranks:
        self.ranks = ranks
        self.world_size = len(ranks)
        self.rank_in_group = ranks.index(self.rank)
        self.device_group = device_group
        self.cpu_group = cpu_group

    assert self.cpu_group is not None
    assert self.device_group is not None

    if torch.cuda.is_available():
      self.device = torch.device(f"cuda:{local_rank}")
    else:
      self.device = torch.device("cpu")

    # Add for 2D-SP
    self.uneven_sizes = []
    self.parallel_type = None

  @property
  def first_rank(self):
    """Return the global rank of the first process in the group"""
    return self.ranks[0]

  @property
  def last_rank(self):
    """Return the global rank of the last process in the group"""
    return self.ranks[-1]

  @property
  def is_first_rank(self):
    """Return whether the caller is the first process in the group"""
    return self.rank == self.first_rank

  @property
  def is_last_rank(self):
    """Return whether the caller is the last process in the group"""
    return self.rank == self.last_rank

  @property
  def next_rank(self):
    """Return the global rank of the process that follows the caller"""
    rank_in_group = self.rank_in_group
    world_size = self.world_size
    return self.ranks[(rank_in_group + 1) % world_size]

  @property
  def prev_rank(self):
    """Return the global rank of the process that precedes the caller"""
    rank_in_group = self.rank_in_group
    world_size = self.world_size
    return self.ranks[(rank_in_group - 1) % world_size]

  def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
    # Bypass the function if we are using only 1 GPU.
    if self.world_size == 1:
      return input_
    else:
      torch.distributed.all_reduce(input_, group=self.device_group)
    return input_

  # @toolbox.timer.torch_function_decorator()
  def all_gather(self, input_: torch.Tensor, dim: int = -1, split: bool = False) -> torch.Tensor:
    world_size = self.world_size
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
      return input_
    assert -input_.dim() <= dim <= input_.dim(), (f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")

    if dim < 0:
      # Convert negative dim to positive.
      dim += input_.dim()
    input_size = input_.size()
    # Allocate output tensor.
    output_tensor = torch.empty((world_size,) + input_size, dtype=input_.dtype, device=input_.device)
    # All-gather.
    torch.distributed.all_gather_into_tensor(output_tensor, input_, group=self.device_group)
    # Reshape
    if dim != 0:
      output_tensor = output_tensor.movedim(0, dim)
    if split:
      tensor_list = [output_tensor.view(-1).narrow(0, input_.numel() * i, input_.numel()).view_as(input_) for i in range(world_size)]
      return tensor_list
    else:
      input_size = list(input_.size())
      input_size[dim] = input_size[dim] * world_size
      output_tensor = output_tensor.reshape(input_size)
      return output_tensor
  
  def async_all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    world_size = self.world_size
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
      return input_
    assert -input_.dim() <= dim <= input_.dim(), (f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")

    if dim < 0:
      # Convert negative dim to positive.
      dim += input_.dim()
    input_size = input_.size()
    # Allocate output tensor.
    output_tensor = torch.empty((world_size,) + input_size, dtype=input_.dtype, device=input_.device)
    # All-gather.
    work = torch.distributed.all_gather_into_tensor(output_tensor, input_, group=self.device_group, async_op=True)
    work.wait()
    # Reshape
    if dim != 0:
      output_tensor = output_tensor.movedim(0, dim)
    
    input_size = list(input_.size())
    input_size[dim] = input_size[dim] * world_size
    output_tensor = output_tensor.reshape(input_size)
    return output_tensor

  def gather_uneven_tensors(self, local_tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    world_size = self.world_size
    if world_size == 1:
        return local_tensor
    
    # 收集所有局部张量的形状
    local_shape = torch.tensor(local_tensor.shape, device=local_tensor.device)
    # 使用修改后的all_gather方法收集形状信息
    all_shapes = self.all_gather(local_shape, dim=0, split=True)
    
    # 计算每个rank的tensor在指定维度的大小
    dim_sizes = [shape[dim].item() for shape in all_shapes]
    max_size = max(dim_sizes)
    
    # 将local_tensor填充到最大尺寸
    pad_size = list(local_tensor.shape)
    pad_size[dim] = max_size - local_tensor.shape[dim]
    if pad_size[dim] > 0:
        # 创建padding
        pad_dims = [0] * (2 * len(local_tensor.shape))
        pad_dims[2 * (len(local_tensor.shape) - 1 - dim)] = pad_size[dim]
        local_tensor = F.pad(local_tensor, pad_dims)
    
    # 使用all_gather收集所有填充后的张量
    gathered_tensor = self.all_gather(local_tensor, dim=dim)
    
    # 移除padding，恢复到原始大小
    if dim < 0:
        dim += gathered_tensor.dim()
    
    slices = []
    start_idx = 0
    for size in dim_sizes:
        # 在指定维度上切片
        idx = [slice(None)] * gathered_tensor.dim()
        idx[dim] = slice(start_idx, start_idx + size)
        slices.append(gathered_tensor[idx])
        start_idx += size
    
    # 在指定维度上连接所有切片
    return torch.cat(slices, dim=dim)

  def gather(self, input_: torch.Tensor, dst: int = 0, dim: int = -1) -> Optional[torch.Tensor]:
    """
      NOTE: We assume that the input tensor is on the same device across
      all the ranks.
      NOTE: `dst` is the local rank of the destination rank.
    """
    world_size = self.world_size
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
      return input_
    assert -input_.dim() <= dim < input_.dim(), (f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
    if dim < 0:
      # Convert negative dim to positive.
      dim += input_.dim()
    # Allocate output tensor.
    if self.rank_in_group == dst:
      gather_list = [torch.empty_like(input_) for _ in range(world_size)]
    else:
      gather_list = None
    # Gather.
    torch.distributed.gather(input_, gather_list, dst=self.ranks[dst], group=self.device_group)
    if self.rank_in_group == dst:
      output_tensor = torch.cat(gather_list, dim=dim)
    else:
      output_tensor = None
    return output_tensor

  # @toolbox.timer.torch_function_decorator()
  def all_to_all(self, input_: torch.Tensor, scatter_dim: int = 2, gather_dim: int = 1) -> torch.Tensor:
    world_size = self.world_size
    if world_size == 1:
      return input_
    input_list = [t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    torch.distributed.all_to_all(output_list, input_list, group=self.device_group)
    return torch.cat(output_list, dim=gather_dim).contiguous()
  
  def get_uneven_seq_chunk_len(self, input_: torch.Tensor, dim: int = 1, seq_id: int = 0) -> list:
    if len(self.uneven_sizes) < seq_id + 1:
      local_splits = torch.tensor(input_.shape[dim], device=input_.device)
      all_splits = self.all_gather(local_splits, dim=0, split=True)
      self.uneven_sizes.append([split.item() for split in all_splits])
    return self.uneven_sizes[seq_id]
  def uneven_all_to_all(self, input_: torch.Tensor, scatter_dim: int = 2, gather_dim: int = 1, uneven_dim: int = None, seq_id: int = 0) -> torch.Tensor:
    world_size = self.world_size
    if world_size == 1:
        return input_
    if uneven_dim is None or (uneven_dim != scatter_dim and uneven_dim != gather_dim):
      return self.all_to_all(input_, scatter_dim, gather_dim)
    else:
      uneven_sizes = self.get_uneven_seq_chunk_len(input_, uneven_dim, seq_id)
      # print(f"{uneven_sizes=}", flush=True)
    
    if uneven_dim == gather_dim:
      max_gather_size = max(uneven_sizes)
      gather_pad_size = max_gather_size - input_.shape[gather_dim]
      if gather_pad_size > 0:
        pad_size = [0] * (2 * input_.dim())
        pad_size[2 * uneven_dim + 1] = gather_pad_size
        input_ = F.pad(input_, pad_size)
        input_ = input_.contiguous()
      
      padded_output = self.all_to_all(input_, scatter_dim, gather_dim)
      padded_output_list = torch.tensor_split(padded_output, world_size, dim=gather_dim)
      unpadded_output_list = []
      for id, output in enumerate(padded_output_list):
        idx = [slice(None)] * output.dim()
        idx[gather_dim] = slice(0, uneven_sizes[id])
        output = output[idx]
        unpadded_output_list.append(output)
      return torch.cat(unpadded_output_list, dim=gather_dim).contiguous()
      
    elif uneven_dim == scatter_dim:
      max_scatter_size = max(uneven_sizes)
      unpadded_input_list = []
      start_idx = 0
      # chunk
      for id in range(world_size):
        idx = [slice(None)] * input_.dim()
        idx[scatter_dim] = slice(start_idx, start_idx + uneven_sizes[id])
        start_idx += uneven_sizes[id]
        unpadded_input_list.append(input_[idx])
      # padding
      padded_input_list = []
      for id, input in enumerate(unpadded_input_list):
        pad_size = [0] * (2 * input.dim())
        pad_size[2 * uneven_dim + 1] = max_scatter_size - uneven_sizes[id]
        padded_input = F.pad(input, pad_size)
        padded_input_list.append(padded_input.contiguous())
      padded_input = torch.cat(padded_input_list, dim=scatter_dim)
      # a2a
      padded_output = self.all_to_all(padded_input, scatter_dim, gather_dim)
      # unpadding
      idx = [slice(None)] * padded_output.dim()
      idx[scatter_dim] = slice(0, uneven_sizes[self.rank_in_group])
      output = padded_output[idx].contiguous()
      return output
      

 # @toolbox.timer.torch_function_decorator("broadcast")
  def broadcast(self, input_: torch.Tensor, src: int = 0):
    """Broadcast the input tensor.
      NOTE: `src` is the local rank of the source rank.
    """
    assert src < self.world_size, f"Invalid src rank ({src})"

    # Bypass the function if we are using only 1 GPU.
    if self.world_size == 1:
      return input_
    # Broadcast.
    torch.distributed.broadcast(input_, src=self.ranks[src], group=self.device_group)
    return input_

  def broadcast_object(self, obj: Optional[Any] = None, src: int = 0):
    """Broadcast the input object.
      NOTE: `src` is the local rank of the source rank.
    """
    assert src < self.world_size, f"Invalid src rank ({src})"

    # Bypass the function if we are using only 1 GPU.
    if self.world_size == 1:
      return obj
    if self.mq_broadcaster is not None:
      assert src == 0, "Message queue broadcaster only supports src=0"
      return self.mq_broadcaster.broadcast_object(obj)
    if self.rank_in_group == src:
      torch.distributed.broadcast_object_list([obj], src=self.ranks[src], group=self.cpu_group)
      return obj
    else:
      recv = [None]
      torch.distributed.broadcast_object_list(recv, src=self.ranks[src], group=self.cpu_group)
      return recv[0]

  def broadcast_object_list(self, obj_list: List[Any], src: int = 0, group: Optional[ProcessGroup] = None):
    """Broadcast the input object list.
        NOTE: `src` is the local rank of the source rank.
        """
    assert src < self.world_size, f"Invalid src rank ({src})"

    # Bypass the function if we are using only 1 GPU.
    if self.world_size == 1:
      return obj_list
    # Broadcast.
    torch.distributed.broadcast_object_list(obj_list, src=self.ranks[src], group=self.device_group)
    return obj_list

  def send_object(self, obj: Any, dst: int) -> None:
    """Send the input object list to the destination rank."""
    """NOTE: `dst` is the local rank of the destination rank."""

    assert dst < self.world_size, f"Invalid dst rank ({dst})"

    assert dst != self.rank_in_group, ("Invalid destination rank. Destination rank is the same "
                                       "as the current rank.")

    # Serialize object to tensor and get the size as well
    object_tensor = torch.frombuffer(pickle.dumps(obj), dtype=torch.uint8)

    size_tensor = torch.tensor([object_tensor.numel()], dtype=torch.long, device="cpu")

    # Send object size
    torch.distributed.send(size_tensor, dst=self.ranks[dst], group=self.cpu_group)

    # Send object
    torch.distributed.send(object_tensor, dst=self.ranks[dst], group=self.cpu_group)
    return None

  def recv_object(self, src: int) -> Any:
    """Receive the input object list from the source rank."""
    """NOTE: `src` is the local rank of the source rank."""

    assert src < self.world_size, f"Invalid src rank ({src})"
    assert src != self.rank_in_group, ("Invalid source rank. Source rank is the same as the current rank.")

    size_tensor = torch.empty(1, dtype=torch.long, device="cpu")

    # Receive object size
    rank_size = torch.distributed.recv(size_tensor, src=self.ranks[src], group=self.cpu_group)

    # Tensor to receive serialized objects into.
    object_tensor = torch.empty(    # type: ignore[call-overload]
      size_tensor.item(),    # type: ignore[arg-type]
      dtype=torch.uint8,
      device="cpu")

    rank_object = torch.distributed.recv(object_tensor, src=self.ranks[src], group=self.cpu_group)
    assert rank_object == rank_size, ("Received object sender rank does not match the size sender rank.")
    obj = pickle.loads(object_tensor.numpy().tobytes())
    return obj

  def barrier(self):
    """Barrier synchronization among the group.
      NOTE: don't use `device_group` here! `barrier` in NCCL is
      terrible because it is internally a broadcast operation with
      secretly created GPU tensors. It is easy to mess up the current
      device. Use the CPU group instead.
    """
    torch.distributed.barrier(group=self.cpu_group)

  def send(self, tensor: torch.Tensor, dst: Optional[int] = None) -> None:
    """Sends a tensor to the destination rank in a non-blocking way"""
    """NOTE: `dst` is the local rank of the destination rank."""
    if dst is None:
      dst = (self.rank_in_group + 1) % self.world_size
    torch.distributed.send(tensor, self.ranks[dst], self.device_group)

  def recv(self, size: torch.Size, dtype: torch.dtype, src: Optional[int] = None) -> torch.Tensor:
    """Receives a tensor from the source rank."""
    """NOTE: `src` is the local rank of the source rank."""
    if src is None:
      src = (self.rank_in_group - 1) % self.world_size
    tensor = torch.empty(size, dtype=dtype, device=self.device)
    torch.distributed.recv(tensor, self.ranks[src], self.device_group)
    return tensor

  def destroy(self):
    if self.device_group is not None:
      torch.distributed.destroy_process_group(self.device_group)
      self.device_group = None
    if self.cpu_group is not None:
      torch.distributed.destroy_process_group(self.cpu_group)
      self.cpu_group = None

  # @toolbox.timer.torch_function_decorator("p2p_isend")
  def p2p_isend(self, tensor: torch.Tensor, dst: Optional[int] = None):
    if dst is None:
      dst = (self.rank_in_group + 1) % self.world_size
    send_op = torch.distributed.P2POp(torch.distributed.isend, tensor, self.ranks[dst], self.device_group)
    self.p2p_ops.append(send_op)

  # @toolbox.timer.torch_function_decorator("p2p_irecv")
  def p2p_irecv(self, size: torch.Size, dtype: torch.dtype, src: Optional[int] = None):
    if src is None:
      src = (self.rank_in_group - 1) % self.world_size
    tensor = torch.empty(size, dtype=dtype, device=self.device)
    recv_op = torch.distributed.P2POp(torch.distributed.irecv, tensor, self.ranks[src], self.device_group)
    self.p2p_ops.append(recv_op)
    return tensor

  # @toolbox.timer.torch_function_decorator("p2p_commit")
  def p2p_commit(self):
    assert self.p2p_reqs is None
    self.p2p_reqs = torch.distributed.batch_isend_irecv(self.p2p_ops)

  # @toolbox.timer.torch_function_decorator("p2p_wait")
  def p2p_wait(self):
    for req in self.p2p_reqs:
      req.wait()
    self.p2p_ops.clear()
    self.p2p_reqs = None