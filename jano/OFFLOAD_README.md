# Jano 双流预取 Offload 模块设计说明

## 背景

Jano 在 warmup 阶段完成了 cache/reuse 策略的制定，并在后续推理步骤中将静态 (static) 和中等动态 (medium) 区域的 KV / x 特征缓存起来，供后续步骤复用。

**目标**：将这些 cached tensors offload 到 CPU，不占用 GPU 显存，但性能非常接近全部保存在 GPU 内存中。

实现该目标的关键技术是**计算-访存 overlap**：
- 利用独立的 CUDA data_stream 驱动 DMA 引擎进行 CPU↔GPU 数据搬运
- 在 GPU compute stream 执行 Transformer 层计算的同时，data_stream 已在预取后续层所需的数据
- 通过 CUDA Event 实现两个流之间的同步

---

## 模块位置与结构

```
jano/
  offload_manager.py       ← 新增：独立的 OffloadManager 模块
  __init__.py              ← 修改：init_jano() 增加 offload 参数
  mask_manager/
    wan_mask_manager.py    ← 修改：集成 OffloadManager
```

---

## `OffloadManager` 设计

### 核心数据结构

| 属性 | 类型 | 说明 |
|------|------|------|
| `data_stream` | `torch.cuda.Stream` | 专用于所有 D2H / H2D 数据搬运的 CUDA 流 |
| `_pinned[key]` | `Dict[str, Tensor]` | CPU 锁页内存 buffer（每个 cache entry 一个） |
| `_staging[key]` | `Dict[str, Tensor]` | GPU 暂存 buffer（接收预取数据） |
| `_events[key]` | `Dict[str, cuda.Event]` | H2D 完成事件，用于 compute stream 等待 |
| `_prefetch_issued` | `set` | 已发起预取但尚未消费的 key 集合 |

### 公开 API

#### `store_async(tensor, key)`

在 `data_stream` 上异步将 GPU tensor 复制到 CPU 锁页内存。

```python
with torch.cuda.stream(self.data_stream):
    self._pinned[key].copy_(tensor, non_blocking=True)
```

- 调用后立即返回，DMA 引擎在后台执行搬运
- 适用于 step_level == 3（update step）存储 static/medium cache

#### `begin_fetch_step(keys)`

在一个 fetch step 开始时，**批量**发起所有 cache entry 的异步 H2D 预取。

```python
for key in keys:
    with torch.cuda.stream(self.data_stream):
        self._staging[key].copy_(self._pinned[key], non_blocking=True)
        self._events[key].record()
```

- 所有拷贝操作排入 data_stream 队列，DMA 引擎按序执行
- 由于 data_stream 与默认 compute stream 并行，GPU 计算早期层时，后续层的数据已在搬运途中

#### `fetch(key)`

等待指定 key 的预取完成，返回 GPU staging buffer。

```python
torch.cuda.current_stream().wait_event(self._events[key])
return self._staging[key]
```

- 让 compute stream 等待 data_stream 上对应 key 的 Event
- Event 通常在 `begin_fetch_step` 调用后很快被记录；到实际执行该层计算时，数据大概率已就绪，等待时间极短
- 如果 `begin_fetch_step` 未被提前调用（fallback 场景），在此处同步发起并等待

---

## 流水线时序

```
时间轴 →

data_stream:    [store l0] [store l1] ... [store lN]          [prefetch l0] [prefetch l1] ... [prefetch lN]
                ↑ step_level=3（update step）                  ↑ step_level=1/2（fetch step, begin_fetch_step()）

compute_stream:                                                [wait l0] [compute l0] [wait l1] [compute l1] ...
```

**关键 overlap**：当 compute_stream 计算第 i 层时，data_stream 已在搬运第 i+1, i+2, ... 层的数据。到 compute_stream 执行第 i+1 层的 `wait_event(events[l_{i+1}])` 时，数据通常已到达 GPU staging buffer，等待时间趋近于 0。

---

## 与 WAN MaskManager 的集成

### 初始化

```python
# jano/mask_manager/wan_mask_manager.py
def init_mask_manager(..., offload: bool = False) -> MaskManager:
    offload_manager = None
    if offload:
        offload_manager = init_offload_manager(layer_num)
    return MaskManager(..., offload_manager=offload_manager)
```

只需在调用 `init_jano()` 或 `init_mask_manager()` 时传入 `offload=True`，其余代码无需改动。

### Cache Key 命名约定

OffloadManager 使用字符串 key 区分不同的 cache entry：

| Key 格式 | 含义 |
|----------|------|
| `s_kv_{cond}_{layer_idx}` | cond 条件下第 layer_idx 层的 static KV cache |
| `m_kv_{cond}_{layer_idx}` | cond 条件下第 layer_idx 层的 medium KV cache |
| `s_x_{cond}_{layer_idx}` | cond 条件下第 layer_idx 层的 static x（hidden state）cache |
| `m_x_{cond}_{layer_idx}` | cond 条件下第 layer_idx 层的 medium x（hidden state）cache |

### step_level 3（update step）— 存储

```python
# process_kv_sequence, step_level == 3
if self.offload_manager is not None:
    self.offload_manager.store_async(static_data, f"s_kv_{state_key}")
    self.offload_manager.store_async(medium_data, f"m_kv_{state_key}")
```

### step_level 1/2（fetch step）— 预取 + 消费

`update_step_level()` 在每次 forward pass 开始时被调用，当检测到 fetch step 时，立即批量发起预取：

```python
# 在 update_step_level() 末尾
if self.offload_manager is not None and self.step_level in (1, 2):
    keys = self._get_prefetch_keys(cond)
    self.offload_manager.begin_fetch_step(keys)
```

在 `process_kv_sequence` / `process_x_sequence` 中消费：

```python
# step_level == 2，获取 static cache
static_kv = self.offload_manager.fetch(f"s_kv_{state_key}")
result = torch.cat([x, static_kv], dim=1)

# step_level == 1，获取 static + medium cache
medium_kv = self.offload_manager.fetch(f"m_kv_{state_key}")
static_kv  = self.offload_manager.fetch(f"s_kv_{state_key}")
result = torch.cat([x, medium_kv, static_kv], dim=1)
```

---

## 向后兼容性

- **默认行为不变**：`offload=False`（默认值）时，`offload_manager` 为 `None`，代码走原有路径
- **naive offload（`offload_kv=True`）**：当 `offload_manager is None` 但 `offload_kv=True` 时，使用原有的同步 `.cpu()` / `.cuda()` 方式（保留原有代码路径）
- **pipeline offload（`offload_manager` 不为 None）**：走新的异步路径

---

## 使用方式

### 在 `run_wan/` 脚本中启用 offload

```python
from jano import init_jano

init_jano(
    enable=True,
    ...,
    offload=True,   # ← 开启双流预取 offload
)
```

### 直接调用 `init_mask_manager`

```python
from jano.mask_manager.wan_mask_manager import init_mask_manager

mask_manager = init_mask_manager(
    patch_size=(1, 2, 2),
    seq_len=75600,
    num_inference_steps=50,
    layer_num=40,
    offload=True,   # ← 开启双流预取 offload
)
```

### 对其他模型（flux、cvx、sd3）的扩展

其他 mask manager 若要使用相同的 offload 机制，只需：

1. 在 `__init__` 中接受 `offload_manager: Optional[OffloadManager]` 参数
2. 在 store 路径调用 `offload_manager.store_async(tensor, key)`
3. 在 `update_step_level()` 末尾调用 `offload_manager.begin_fetch_step(keys)`
4. 在 fetch 路径调用 `offload_manager.fetch(key)`

`OffloadManager` 本身与模型架构无关，可直接复用。

---

## 性能分析

### 显存节省

以 WAN 14B 模型（40 层，cond + uncond，kv + x cache）为例：

- 每层 static KV cache（假设 static 区域占 60%）：大约几十 MB
- 全部 static + medium cache offload 后，GPU 显存占用可减少 **数 GB**

### 性能影响

| 场景 | 性能影响 |
|------|----------|
| offload=False（全 GPU）| baseline |
| offload=True，朴素实现（`.cpu()` / `.cuda()`）| 同步搬运，每层 block 前有明显等待 |
| offload=True，双流流水线（本实现）| 搬运与计算 overlap，等待时间趋近 0 |

理想情况下（PCIe 带宽充足，GPU 计算时间 > 搬运时间），双流流水线版本的性能与全 GPU 缓存版本接近。

---

## 注意事项

1. **staging buffer 内容不可修改**：`fetch()` 返回的是 GPU staging buffer 的引用，下一次预取会覆盖它。调用方若需保留数据，应在使用后立即 `cat` 或存入其他 tensor。
2. **data_stream 上的 store 操作**：`store_async` 返回后，source tensor 不应立即被 Python GC 或其他操作修改，直到 data_stream 完成该拷贝。实践中，由于 PyTorch 的 tensor 生命周期管理，这通常不是问题。
3. **多 GPU / 分布式场景**：`OffloadManager` 实例是 per-process 的，每个 GPU 进程维护自己的实例，无需额外修改。
4. **显存 vs 性能权衡**：GPU staging buffer 本身占用少量 GPU 显存（与 CPU pinned buffer 等量），但远小于不 offload 时的总 cache 大小。
