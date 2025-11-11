import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from jano.block_manager import get_block_manager
from jano.stuff import get_timestep, visualize_mask
from utils.envs import GlobalEnv
        
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
    mask = torch.ones((C, F, H, W), dtype=torch.bool, device=device)
    
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
        mask[c, f_idx, h_idx, w_idx] = False  # False 表示mask，不进行计算
        
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

class MaskManager:
    # 维护latent mask 和 sequence mask，提供apply mask和restore kv等接口
    def __init__(self, patch_size: tuple, seq_len: int, num_inference_steps: int, layer_nums: int):
        self.patch_size = patch_size
        self.warmup_steps = GlobalEnv.get_envs("warmup_steps")
        self.cooldown_steps = GlobalEnv.get_envs("cooldown_steps")
        self.static_interval = GlobalEnv.get_envs("static_interval")
        self.medium_interval = GlobalEnv.get_envs("medium_interval")
        
        # self.offload = GlobalEnv.get_envs("offload")
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
        
        self.restored_x = None
        
        self.block_mask = None
        self.latent_mask = None
        self.sequence_mask = None
        
        # 记录最大内存使用
        self.max_memory = 0
        
        # offload下的双流流水线实现【暂时停用】
        # if self.offload:
        #     print("Initializing MaskManager with dual-stream pipeline support.")
        #     self.frozen_states_cpu = {}
        #     # 用于记录第一次tensor的shape和dtype
        #     self.buffer_initialized = False
        #      # 1. 创建计算流和数据流
        #     self.compute_stream = torch.cuda.Stream()
        #     self.data_stream = torch.cuda.Stream()
            
        
        #     # Pinned Memory (CPU端) 和 GPU (设备端) 缓冲区
        #     self.pinned_buffers = {}
        #     self.gpu_staging_buffer = None
            
        #     # 同步事件
        #     self.prefetch_event = torch.cuda.Event()

    def generate_mask(self, combined_score):
        """
        基于时空复杂度分析创建mask：
        1: 低动态区域 (< static_thresh)
        2: 中等动态区域 (static_thresh ~ medium_thresh)
        3: 高动态区域 (> medium_thresh)
        """
        combined_score = torch.from_numpy(combined_score)
        print_score_stats(combined_score)
        # exit()
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
        if self.step_level == 0:
            self.restored_x = x
            return x
        if self.get_seq_len() == 0: # 特殊情况
            return self.restored_x
        
        state_key = f"{name}_{layer_idx}"
        device = x.device
        if self.restored_x is None:
            B = x.shape[0]
            D = x.shape[1]
            self.restored_x = torch.zeros(B, self.full_seq_len, D, device=device, dtype=x.dtype)
            
        # 在is_update_step时打印内存状态
        if self.step_level == 3: # 全部计算
            self.static_cache[state_key] = x[:, self.static_bool_mask, :]
            self.medium_cache[state_key] = x[:, self.medium_bool_mask, :]
            self.restored_x = x
        elif self.step_level == 2: # active + medium
            # 储存
            self.medium_cache[state_key] = x[:, self.medium_bool_mask_in_l2, :]
            # 恢复
            self.restored_x[:, ~self.static_bool_mask, :] = x
            self.restored_x[:, self.static_bool_mask, :] = self.static_cache[state_key]
        elif self.step_level == 1:
            self.restored_x[:, self.active_mask, :] = x
            self.restored_x[:, self.medium_bool_mask, :] = self.medium_cache[state_key]
            self.restored_x[:, self.static_bool_mask, :] = self.static_cache[state_key]
        return self.restored_x
        
    def process_kv_sequence(self, kv: torch.Tensor, name: str, layer_idx: int) -> torch.Tensor:
        if self.step_level == 0:
            return kv
        
        state_key = f"{name}_{layer_idx}"
        B, S, N, D = kv.shape
        x = kv.reshape(B, S, -1)
        
        if self.step_level == 3:            
            self.static_cache[state_key] = x[:, self.static_bool_mask, :]
            self.medium_cache[state_key] = x[:, self.medium_bool_mask, :]
            if get_timestep() == GlobalEnv.get_envs("warmup_steps") + 1:
                print(f"Stored {state_key}, {x.shape=}, "
                    f"tensor_MiB={x.element_size() * x.nelement() >> 20}, " # >>20，右移20位，Byte转换为MB
                    f"cuda_reserved_MiB={torch.cuda.memory_reserved() >> 20}, "
                    f"cuda_allocated_MiB={torch.cuda.memory_allocated() >> 20}", flush=True)
            result = x
        elif self.step_level == 2:
            # 储存
            self.medium_cache[state_key] = x[:, self.medium_bool_mask_in_l2, :]
            # 恢复
            static_kv = self.static_cache[state_key]
            if layer_idx == 20:
                print(f"{get_timestep()} | Fetch from {state_key}, {static_kv.shape=}", flush=True)
            result = torch.cat([x, static_kv], dim=1)
        elif self.step_level == 1:
            medium_kv = self.medium_cache[state_key]
            static_kv = self.static_cache[state_key]
            if layer_idx == 20:
                print(f"{get_timestep()} | Fetch from {state_key}, {static_kv.shape=} {medium_kv.shape=}", flush=True)
            result = torch.cat([x, medium_kv, static_kv], dim=1)
            
        return result.reshape(B, -1, N, D)

    def clear_frozen_states(self):
        """清理frozen状态并重置内存统计"""
        self.static_cache.clear()
        self.medium_cache.clear()
        torch.cuda.reset_peak_memory_stats()
        self.max_memory = 0
    
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
            self.step_level =  0 # full compute wo update
        elif (timestep-self.warmup_steps-1) % self.static_interval == 0:
            self.step_level = 3 # full compute w update
        elif (timestep-self.warmup_steps-1) % self.medium_interval == 0:
            self.step_level = 2 # medium compute
        else: 
            self.step_level = 1 # active compute
        
    
    def print_memory_stats(self):
        """打印当前GPU内存使用情况"""
        current_memory = torch.cuda.memory_allocated()
        max_memory = torch.cuda.max_memory_allocated()
        self.max_memory = max(self.max_memory, current_memory)
        
        print(f"step {get_timestep()} | GPU Memory Stats:")
        print(f"  Current Memory: {format_memory(current_memory)}")
        print(f"  Peak Memory: {format_memory(max_memory)}")
        print(f"  Session Peak Memory: {format_memory(self.max_memory)}", flush=True)
        
    #  ============================== Offload Codes ==============================
    # def _init_all_buffers(self, tensor: torch.Tensor):
    #     """初始化所有层的pinned buffers"""
    #     if not self.buffer_initialized and self.offload:
    #         print(f"Initializing pinned buffers for {self.num_layers} layers...", flush=True)
    #         shape = tensor.shape
    #         dtype = tensor.dtype
            
    #         self.gpu_staging_buffer = torch.empty(shape, dtype=dtype, device=torch.cuda.current_device())
    #         print(f"  - GPU Staging Buffer initialized. Size: {format_memory(self.gpu_staging_buffer.nelement() * self.gpu_staging_buffer.element_size())}")
            
    #         # 为每一层的x1和x2分配buffer
    #         for i in range(self.num_layers):
    #             x1_key = f"hidden_c0_{i}"
    #             x2_key = f"hidden_c1_{i}"
    #             self.pinned_buffers[x1_key] = torch.empty(shape, dtype=dtype, pin_memory=True, device='cpu')
    #             self.pinned_buffers[x2_key] = torch.empty(shape, dtype=dtype, pin_memory=True, device='cpu')
            
    #         total_size = sum(t.nelement() * t.element_size() for t in self.pinned_buffers.values())
    #         print(f"Initialized pinned memory buffers, total size: {format_memory(total_size)}", flush=True)
    #     self.buffer_initialized = True
            
    # def prefetch_to_staging_buffer(self, name: str, layer_idx: int):
    #     """
    #     在数据流上，将指定层的CPU数据预取到共享的GPU暂存区。
    #     """
    #     if not self.offload or self.is_warmup_or_cooldown_step() or self.is_update_step():
    #         return

    #     state_key = f"{name}_{layer_idx}"
    #     if state_key not in self.pinned_buffers: return

    #     with torch.cuda.stream(self.data_stream):
    #         # 异步从CPU拷贝到共享的GPU暂存区
    #         self.gpu_staging_buffer.copy_(self.pinned_buffers[state_key], non_blocking=True)
    #         # 记录事件，表示预取操作已入队
    #         self.prefetch_event.record()
            
    # def store_frozen_state_async(self, frozen_tensor: torch.Tensor, name: str, layer_idx: int):
    #     """
    #     在数据流上，将GPU上的frozen token异步写回到对应的CPU Pinned Buffer。
    #     """
    #     state_key = f"{name}_{layer_idx}"
        
    #     if not self.buffer_initialized:
    #         self._init_all_buffers(frozen_tensor)

    #     with torch.cuda.stream(self.data_stream):
    #         if self.offload:
    #             self.pinned_buffers[state_key].copy_(frozen_tensor, non_blocking=True)
    #             self.frozen_states_cpu[state_key] = self.pinned_buffers[state_key]
    #         else:
    #             self.frozen_states_gpu[state_key] = frozen_tensor
            
    # def restore_from_staging_buffer(self, active_tensor: torch.Tensor, name: str, layer_idx: int) -> torch.Tensor:
    #     """
    #     【流水线核心】在计算流上，等待预取完成，并从暂存区恢复完整序列。
    #     """
    #     # 1. 让计算流等待数据流上的预取完成
    #     if self.offload:
    #         self.compute_stream.wait_event(self.prefetch_event)
        
    #     # 2. 事件完成后，数据保证在 gpu_staging_buffer 中可用
    #     B, S_, D = active_tensor.shape
    #     S = self.full_seq_len
    #     device = active_tensor.device
    #     dtype = active_tensor.dtype

    #     restored_x = torch.zeros(B, S, D, device=device, dtype=dtype)
        
    #     # 合并 active 和 frozen (来自暂存区)
    #     if self.offload:
    #         frozen_tensor = self.gpu_staging_buffer
    #     else:
    #         frozen_tensor = self.frozen_states_gpu[f"{name}_{layer_idx}"]
            
    #     restored_x[:, self.sequence_mask[:S], :] = active_tensor
    #     restored_x[:, ~self.sequence_mask[:S], :] = frozen_tensor
        
    #     return restored_x
        
# ================================ APIs =================================
        
def init_mask_manager(patch_size, seq_len, num_inference_steps, layer_num) -> MaskManager:
    mask_manager = MaskManager(patch_size, seq_len, num_inference_steps, layer_num)
    GlobalEnv.set_envs('MM', mask_manager)
    return mask_manager
    
def get_mask_manager() -> MaskManager:
    if GlobalEnv.get_envs("enable_stdit"):
        return GlobalEnv.get_envs('MM')
    else:
        return None
    
        