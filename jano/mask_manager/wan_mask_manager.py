import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from jano.block_manager import get_block_manager
from jano.stuff import get_timestep, visualize_mask
from jano.offload_manager import OffloadManager
from utils.envs import GlobalEnv
from utils.timer import get_timer
        
def create_random_latents_mask(x: torch.Tensor, ratio: float = 0.5, device=None):
    """
    创建空间随机mask，只对F、H、W三维进行随机采样
    
    Args:
        C: 通道数
        F: 帧数
        H: 高度
        W: 宽度
        ratio: 采样比例，表示被mask的比例
        device: torch设备，默认None表示使用CPU
    Returns:
        mask: bool张量 [C, F, H, W]
    """
    C, F, H, W = x.shape
    # 初始化全False的mask
    mask = torch.ones((C, F, H, W), dtype=torch.int8, device=device)
    
    # 计算F*H*W的总数
    total_points = F * H * W
    num_masked = int(total_points * ratio)
    
    # 对每个通道进行相同的随机mask
    indices = torch.randperm(total_points, device=device)[:num_masked]
    
    # 将一维索引转换为三维索引
    f_idx = (indices // (H * W)) % F
    h_idx = (indices // W) % H
    w_idx = indices % W
    
    # 对所有通道应用相同的mask
    for c in range(C):
        mask[c, f_idx, h_idx, w_idx] = 3
        
    return mask

def format_memory(bytes):
    """将字节数转换为可读格式（GB）"""
    return f"{bytes / 1024**3:.2f}GB"

def print_score_stats(tensor: torch.Tensor):
    """打印张量的统计信息
    Args:
        tensor: 一维张量
    """
    # 基础统计量
    mean = tensor.mean().item()
    std = tensor.std().item()
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    
    # 分位数
    quantiles = torch.quantile(tensor, torch.tensor([0.3, 0.5, 0.8]))
    
    print(f"========Score Statistics=========")
    print(f"Mean: {mean:.3f}")
    print(f"Std:  {std:.3f}")
    print(f"Min:  {min_val:.3f}")
    print(f"Max:  {max_val:.3f}")
    print(f"\nPercentiles:")
    print(f"30%: {quantiles[0]:.3f}")
    print(f"50%: {quantiles[1]:.3f}")
    print(f"80%: {quantiles[2]:.3f}")
    print(f"=================================")

class MemoryTracker:
    """精确的内存跟踪器，专门监控offload相关的内存使用"""
    def __init__(self):
        self.baseline_gpu_memory = torch.cuda.memory_allocated()
        self.peak_gpu_memory = self.baseline_gpu_memory
        
        # 分类内存统计
        self.cache_cpu_memory = 0  # CPU缓存内存 
        self.cache_gpu_memory = 0  # GPU staging buffer内存
        self.other_gpu_memory = self.baseline_gpu_memory  # 其他GPU内存
        
    def update_cache_memory(self, cpu_bytes: int, gpu_bytes: int):
        """更新缓存相关的内存统计"""
        self.cache_cpu_memory = cpu_bytes
        self.cache_gpu_memory = gpu_bytes
        
    def update_gpu_peak(self):
        """更新GPU峰值内存"""
        current = torch.cuda.memory_allocated()
        self.peak_gpu_memory = max(self.peak_gpu_memory, current)
        self.other_gpu_memory = current - self.cache_gpu_memory
        
    def get_memory_stats(self) -> dict:
        """获取详细的内存统计"""
        current_gpu = torch.cuda.memory_allocated()
        return {
            'current_gpu_gb': current_gpu / 1024**3,
            'peak_gpu_gb': self.peak_gpu_memory / 1024**3,
            'cache_cpu_gb': self.cache_cpu_memory / 1024**3, 
            'cache_gpu_gb': self.cache_gpu_memory / 1024**3,
            'other_gpu_gb': (current_gpu - self.cache_gpu_memory) / 1024**3,
            'gpu_saved_gb': (self.cache_cpu_memory - self.cache_gpu_memory) / 1024**3
        }
        
    def print_stats(self, step: int, offload_enabled: bool):
        """打印精简而关键的内存统计"""
        stats = self.get_memory_stats()
        
        if offload_enabled:
            print(f"Step {step} | GPU Peak: {stats['peak_gpu_gb']:.2f}GB, Current: {stats['current_gpu_gb']:.2f}GB")
            print(f"         | Cache - CPU: {stats['cache_cpu_gb']:.2f}GB, GPU: {stats['cache_gpu_gb']:.2f}GB, Saved: {stats['gpu_saved_gb']:.2f}GB")
        else:
            print(f"Step {step} | GPU Peak: {stats['peak_gpu_gb']:.2f}GB, Current: {stats['current_gpu_gb']:.2f}GB (no offload)")

class MaskManager:
    # 维护latent mask 和 sequence mask，提供apply mask和restore kv等接口
    def __init__(self, patch_size: tuple, seq_len: int, num_inference_steps: int, layer_nums: int,
                 offload_manager: Optional[OffloadManager] = None):
        self.patch_size = patch_size
        self.warmup_steps = GlobalEnv.get_envs("warmup_steps")
        self.cooldown_steps = GlobalEnv.get_envs("cooldown_steps")
        self.static_interval = GlobalEnv.get_envs("static_interval")
        self.medium_interval = GlobalEnv.get_envs("medium_interval")
        
        self.enable = GlobalEnv.get_envs("enable_stdit")
        
        self.num_inference_steps = num_inference_steps
        self.num_layers = layer_nums
        self.full_seq_len = seq_len
        self.medium_seqlen = seq_len
        self.active_seqlen = seq_len
        self.step_level = 0
        
        # mask
        self.active_bool_mask = None
        self.medium_bool_mask = None
        self.medium_bool_mask_in_l2 = None
        self.active_bool_mask_in_l2 = None
        self.static_bool_mask = None
        
        self.medium_cache = {}
        self.static_cache = {}
        
        # Per-cond restored_x buffers: keyed by the `name` argument passed to
        # process_masked_output (e.g. "output_c0", "output_c1").  Using a
        # single tensor was a bug: in 1-GPU mode the cond=0 and cond=1 forward
        # passes share the same MaskManager, so the L3 output of cond=1 would
        # overwrite the buffer and contaminate the cond=0 L1 result with the
        # wrong medium/static tokens.
        self.restored_x_dict: Dict[str, Optional[torch.Tensor]] = {}
        
        self.block_mask = None
        self.latent_mask = None
        self.sequence_mask = None
        
        # 记录最大内存使用
        self.max_memory = 0
        self.offload_kv = False
        self.memory_tracker = MemoryTracker()

        # Dual-stream pipeline offload manager (optional)
        # When set, cached KV/x tensors are stored in CPU pinned memory and
        # prefetched back to GPU just-in-time using a dedicated data_stream,
        # overlapping transfers with GPU compute.
        self.offload_manager: Optional[OffloadManager] = offload_manager
        if offload_manager is not None:
            self.offload_kv = True
            print(
                "[MaskManager] Dual-stream offload pipeline enabled "
                f"({layer_nums} layers).",
                flush=True,
            )
        

    def generate_mask(self, combined_score):
        """
        基于时空复杂度分析创建mask：
        1: 低动态区域 (< static_thresh)
        2: 中等动态区域 (static_thresh ~ medium_thresh)
        3: 高动态区域 (> medium_thresh)
        """
        static_thresh = GlobalEnv.get_envs("static_thresh")
        medium_thresh = GlobalEnv.get_envs("medium_thresh")
        
        bm = get_block_manager()
        C, T, H, W = bm.latent_shape
        
        # 创建块级别的标注mask (默认为1，表示低动态)
        self.block_mask = torch.ones_like(combined_score, dtype=torch.int8)
        
        # 中等动态区域: 任一维度超过static阈值但都不超过medium阈值
        medium_condition = (combined_score > static_thresh) & (combined_score <= medium_thresh)
        self.block_mask = torch.where(medium_condition, 2, self.block_mask)
        
        # 高动态区域: 任一维度超过medium阈值
        high_condition = combined_score > medium_thresh
        self.block_mask = torch.where(high_condition, 3, self.block_mask)
        
        # 将block mask转换为完整分辨率mask
        bt, bh, bw = bm.block_size
        nt, nh, nw = bm.padded_T // bt, bm.padded_H // bh, bm.padded_W // bw
        
        block_mask_3d = self.block_mask.reshape(nt, nh, nw)
        latent_mask = torch.zeros((T, H, W), dtype=torch.int64, device=torch.cuda.current_device())
        
        # 扩展block mask到完整分辨率
        for t in range(nt):
            for h in range(nh):
                for w in range(nw):
                    value = block_mask_3d[t, h, w]
                    latent_mask[t*bt:(t+1)*bt, 
                            h*bh:(h+1)*bh, 
                            w*bw:(w+1)*bw] = value
        
        latent_mask = latent_mask.unsqueeze(0).expand(C, -1, -1, -1)
        
        # 统计各个区域的比例
        total_pixels = C * T * H * W
        low_dynamic_ratio = (latent_mask == 1).sum().item() / total_pixels * 100
        medium_dynamic_ratio = (latent_mask == 2).sum().item() / total_pixels * 100
        high_dynamic_ratio = (latent_mask == 3).sum().item() / total_pixels * 100
        
        print(f"Created dynamics-based mask with:")
        print(f"Low dynamic regions (1): {low_dynamic_ratio:.2f}%")
        print(f"Medium dynamic regions (2): {medium_dynamic_ratio:.2f}%")
        print(f"High dynamic regions (3): {high_dynamic_ratio:.2f}%")
        
        # 可视化
        # latent_mask = create_random_latents_mask(latent_mask, GlobalEnv.get_envs("random")) # 启用了随机mask
        # latent_mask = self.generate_ratio_mask(combined_score)
        visualize_mask(latent_mask)
        self.latent_mask = latent_mask
        self.sequence_mask = self.transform_mask(latent_mask)
        
        self.static_bool_mask = (self.sequence_mask == 1).bool()
        self.medium_bool_mask = (self.sequence_mask == 2).bool()
        self.active_bool_mask = (self.sequence_mask == 3).bool()
        mask2 = self.sequence_mask[self.sequence_mask != 1]
        self.medium_bool_mask_in_l2 = (mask2 == 2).bool()
        self.active_bool_mask_in_l2 = (mask2 == 3).bool()
        
        return self.latent_mask
    
    def transform_mask(self, spatial_mask: torch.Tensor):
        """
        将空间形式的mask转换为序列形式，适配Conv3d patch embedding
        对于每个patch，取占比最高的等级作为该token的等级（向量化版本）
        
        Args:
            spatial_mask: [C, F, H, W] 形式的mask，值为1,2,3表示不同动态等级
        Returns:
            sequence_mask: [L] 形式的mask，一维数组，L = F'*H'*W'，值为1,2,3
        """
        C, F, H, W = spatial_mask.shape
        pF, pH, pW = self.patch_size
        
        # 计算下采样后的空间维度
        F_out = F // pF
        H_out = H // pH
        W_out = W // pW
        
        # 重塑spatial_mask以进行patch-wise操作
        spatial_mask = spatial_mask.reshape(
            C, F_out, pF, H_out, pH, W_out, pW
        )
        
        # 合并patch内的所有维度: [C, F_out, H_out, W_out, pF*pH*pW]
        patch_values = spatial_mask.permute(0, 1, 3, 5, 2, 4, 6).reshape(
            C, F_out, H_out, W_out, pF * pH * pW
        )
        
        # 进一步合并到 [F_out, H_out, W_out, total_pixels]
        patch_values = patch_values.permute(1, 2, 3, 0, 4).reshape(
            F_out, H_out, W_out, -1
        )
        
        # 向量化计算各等级的数量
        count_1 = (patch_values == 1).sum(dim=-1)  # [F_out, H_out, W_out]
        count_2 = (patch_values == 2).sum(dim=-1)  # [F_out, H_out, W_out]
        count_3 = (patch_values == 3).sum(dim=-1)  # [F_out, H_out, W_out]
        
        # 堆叠计数并找到最大值的索引
        counts = torch.stack([count_1, count_2, count_3], dim=-1)  # [F_out, H_out, W_out, 3]
        sequence_mask = torch.argmax(counts, dim=-1) + 1  # +1因为等级从1开始
        
        # 转换为int8类型
        sequence_mask = sequence_mask.to(torch.int8)
        
        # 展平到序列形式
        sequence_mask = sequence_mask.flatten()  # [F_out * H_out * W_out]
        
        # 计算各个等级的序列长度
        self.full_seq_len = sequence_mask.numel()  # 总序列长度（1+2+3）
        self.medium_mask = (sequence_mask >= 2).bool()
        self.medium_seqlen = (sequence_mask >= 2).sum().item()  # 中高动态区域长度（2+3）
        self.active_mask = (sequence_mask == 3).bool()
        self.active_seqlen = (sequence_mask == 3).sum().item()  # 高动态区域长度（3）
        
        print(f"Sequence lengths - Total: {self.full_seq_len}, Medium+High: {self.medium_seqlen}, High: {self.active_seqlen}")

        
        return sequence_mask
    
    def apply_sequence_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        单纯地应用mask到序列上，获取active部分
        
        Args:
            x: 输入序列 [B,S,*] (*表示任意维度)
        Returns:
            active_x: masked序列 [B,S',*]
        """
        
        if self.step_level == 0 or self.step_level == 3:
            return x 
        # 只在sequence维度展开mask
        if self.step_level == 1:
            sequence_mask = self.active_mask 
        elif self.step_level == 2:
            sequence_mask = self.medium_mask
        else:
            ValueError("Something went wrong with step level.")
        
        return x[:, sequence_mask, ...]  # ...会自动处理剩余维度

    def process_masked_output(self, x: torch.Tensor, name: str, layer_idx: int) -> torch.Tensor:
        """将本步计算出的局部 token 写回到 per-cond restored_x（全序列 GPU tensor）。

        每个 cond（name）独立维护一个 restored_x，避免 1-GPU 模式下 cond=0 和
        cond=1 共用同一缓冲区导致 medium/static token 相互污染。

        restored_x 在步间持久保留：
          step_level 3 → 全部 token 重新计算，直接替换
          step_level 2 → active+medium token 更新，static 区域维持上一次 level-3 的值
          step_level 1 → 仅 active token 更新，medium+static 维持上一步的值
        无需额外 cache，只做 scatter-write。
        """
        if self.step_level == 0:
            self.restored_x_dict[name] = x
            return x
        if self.get_seq_len() == 0:
            return self.restored_x_dict.get(name)

        if self.restored_x_dict.get(name) is None:
            B, _, D = x.shape
            self.restored_x_dict[name] = torch.zeros(
                B, self.full_seq_len, D, device=x.device, dtype=x.dtype
            )

        restored = self.restored_x_dict[name]
        if self.step_level == 3:
            self.restored_x_dict[name] = x
        elif self.step_level == 2:
            restored[:, ~self.static_bool_mask, :] = x
        elif self.step_level == 1:
            restored[:, self.active_mask, :] = x
        return self.restored_x_dict[name]
        
    def process_kv_sequence(self, kv: torch.Tensor, name: str, layer_idx: int) -> torch.Tensor:
        if self.step_level == 0:
            return kv

        name = str(name)   # GlobalEnv.get_envs("cond") 返回 int，统一转 str
        state_key = f"{name}_{layer_idx}"
        B, S, N, D = kv.shape
        x = kv.reshape(B, S, -1)
        
        if self.step_level == 3:           
            static_data = x[:, self.static_bool_mask, :]
            medium_data = x[:, self.medium_bool_mask, :]

            if layer_idx == 20:
                print(f"{get_timestep()} | Store to {state_key}, {static_data.shape=}, {static_data[0][2][:5]=}", flush=True)

            if self.offload_manager is not None:
                om = self.offload_manager
                # layer_idx==0 时启动后台线程预分配两块 pool（与当前层 GPU 计算重叠）
                if layer_idx == 0:
                    om.start_preallocate(
                        name,
                        self.num_layers,
                        tuple(static_data.shape),
                        static_data.dtype,
                        tuple(medium_data.shape),
                        medium_data.dtype,
                    )
                om.store_async(static_data, layer_idx, 's', name)
                om.store_async(medium_data, layer_idx, 'm', name)
                del static_data, medium_data
            elif self.offload_kv:
                self.static_cache[state_key] = static_data.cpu()
                self.medium_cache[state_key] = medium_data.cpu()
            else:
                self.static_cache[state_key] = static_data
                self.medium_cache[state_key] = medium_data

            result = x
        elif self.step_level == 2:
            # 存储 medium（当前帧 medium+active tokens → CPU）
            medium_data = x[:, self.medium_bool_mask_in_l2, :]
            if self.offload_manager is not None:
                om = self.offload_manager
                om.store_async(medium_data, layer_idx, 'm', name)
                del medium_data
            elif self.offload_kv:
                self.medium_cache[state_key] = medium_data.cpu()
            else:
                self.medium_cache[state_key] = medium_data
                
            # 恢复 static（从 CPU fetch 回 GPU）
            if self.offload_manager is not None:
                static_kv = om.fetch(layer_idx, 's', name)
                if layer_idx == 20:
                    print(f"{get_timestep()} | [L2] Fetch static from {state_key}, {static_kv.shape=}, {static_kv[0][2][:5]=}", flush=True)
            elif self.offload_kv:
                static_kv = self.static_cache[state_key].cuda()
            else:
                static_kv = self.static_cache[state_key]
                
            result = torch.cat([x, static_kv], dim=1)

        elif self.step_level == 1:
            # 恢复 medium + static
            if self.offload_manager is not None:
                om = self.offload_manager
                medium_kv = om.fetch(layer_idx, 'm', name)
                static_kv  = om.fetch(layer_idx, 's', name)
            elif self.offload_kv:
                medium_kv = self.medium_cache[state_key].cuda()
                static_kv = self.static_cache[state_key].cuda()
            else:
                medium_kv = self.medium_cache[state_key]
                static_kv = self.static_cache[state_key]
                
            if layer_idx == 20:
                print(f"{get_timestep()} | Fetch from {state_key}, {static_kv.shape=}, {static_kv[0][2][:5]=}", flush=True)

            result = torch.cat([x, medium_kv, static_kv], dim=1)
            
        return result.reshape(B, -1, N, D)
    
    def begin_cond_fetch(self, cond: str) -> None:
        """在每次 model forward 之前调用，为指定 cond 设置预取。

        必须在 GlobalEnv.set_envs("cond", ...) 已设置正确值之后、model forward 之前调用。
        这样可以保证 staging buffer 中预取的是当前 cond 的数据，而不是上一步遗留的其他 cond 数据。
        """
        if (
            self.offload_manager is not None
            and self.step_level in (1, 2)
            and self.offload_manager.is_ready(cond)
        ):
            self.offload_manager.begin_fetch_step(cond, self.num_layers, self.step_level)

    def clear_frozen_states(self):
        """清理frozen状态并重置内存统计"""
        self.static_cache.clear()
        self.medium_cache.clear()
        self.restored_x_dict.clear()
        if self.offload_manager is not None:
            self.offload_manager.clear()
            print("[MaskManager] Cleared all offload buffers and caches", flush=True)
        torch.cuda.reset_peak_memory_stats()
        self.max_memory = 0
        self.memory_tracker = MemoryTracker()  # 重置内存跟踪器
    
    def get_seq_len(self)->int:
        if self.step_level == 0 or self.step_level == 3:
            return self.full_seq_len
        elif self.step_level == 2:
            return self.medium_seqlen
        elif self.step_level == 1:
            return self.active_seqlen
        
    def update_step_level(self):
        timestep = get_timestep()

        if timestep is None or timestep <= self.warmup_steps \
            or timestep > self.num_inference_steps - self.cooldown_steps \
            or self.static_interval * self.medium_interval == 0:
            self.step_level = 0  # full compute wo update
        elif (timestep - self.warmup_steps - 1) % self.static_interval == 0:
            self.step_level = 3  # full compute w update
        elif (timestep - self.warmup_steps - 1) % self.medium_interval == 0:
            self.step_level = 2  # medium compute
        else:
            self.step_level = 1  # active compute

        # NOTE: begin_fetch_step 不在此处调用。
        # 它必须在每次 model forward 之前（cond 已正确设置后）通过 begin_cond_fetch() 触发，
        # 避免使用上一步遗留的错误 cond 值导致 staging buffer 数据错位。

        # 更新内存统计
        self._update_memory_stats()
    
    def _update_memory_stats(self):
        """每5步/rank0 打印一行关键运行状态。"""
        timestep = get_timestep() or 0
        if timestep % 5 != 0 and timestep > 3 and self.step_level != 3:
            return

        om = self.offload_manager
        if om is not None and om.is_ready():
            s = om.memory_stats()
            cpu_gb  = s["cpu_pinned_bytes"]        / 1024**3
            fgpu_gb = s["gpu_fetch_staging_bytes"] / 1024**3
            sgpu_gb = s["gpu_store_staging_bytes"] / 1024**3
            alloc   = s["gpu_allocated_bytes"]     / 1024**3
            peak    = s["gpu_peak_bytes"]          / 1024**3
            print(
                f"[Step {timestep:3d}|L{self.step_level}] "
                f"GPU alloc {alloc:.2f}GB  peak {peak:.2f}GB | "
                f"offload  CPU {cpu_gb:.2f}GB  "
                f"fetch-stg {fgpu_gb:.2f}GB  store-stg {sgpu_gb:.2f}GB  "
                f"saved {max(0.0, cpu_gb - fgpu_gb - sgpu_gb):.2f}GB",
                flush=True,
            )
        else:
            alloc = torch.cuda.memory_allocated() / 1024**3
            peak  = torch.cuda.max_memory_allocated() / 1024**3
            print(
                f"[Step {timestep:3d}|L{self.step_level}] "
                f"GPU alloc {alloc:.2f}GB  peak {peak:.2f}GB",
                flush=True,
            )
    
    def print_memory_stats(self):
        """保留原有接口兼容性，但使用新的统计方式"""
        self._update_memory_stats()

# ================================ APIs =================================
        
def init_mask_manager(patch_size, seq_len, num_inference_steps, layer_num,
                      offload: bool = False) -> MaskManager:
    """
    Create and register the global WAN MaskManager.

    Parameters
    ----------
    patch_size : tuple
        3-D patch size (pT, pH, pW).
    seq_len : int
        Full sequence length.
    num_inference_steps : int
        Total number of denoising steps.
    layer_num : int
        Number of transformer layers.
    offload : bool, optional
        When *True*, cache entries are stored in CPU pinned memory and
        prefetched back with a dual-stream pipeline.  Defaults to *False*.
    """
    offload_manager: Optional[OffloadManager] = None
    if offload and torch.cuda.is_available():
        from jano.offload_manager import init_offload_manager
        offload_manager = init_offload_manager()

    mask_manager = MaskManager(patch_size, seq_len, num_inference_steps, layer_num,
                               offload_manager=offload_manager)
    GlobalEnv.set_envs('MM', mask_manager)
    return mask_manager
    
def get_mask_manager() -> MaskManager:
    if GlobalEnv.get_envs("enable_stdit"):
        return GlobalEnv.get_envs('MM')
    else:
        return None
    
        