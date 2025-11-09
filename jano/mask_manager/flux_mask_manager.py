import torch
import numpy as np

from jano.block_manager import get_block_manager
from jano.stuff import get_timestep, visualize_mask
from utils.envs import GlobalEnv
from utils.timer import get_timer

def create_random_latents_mask(x: torch.Tensor, ratio: float = 0.5, device=None):
    """
    创建空间随机mask，只对F、H、W三维进行随机采样
    
    Args:
        x: 输入张量 [C, F, H, W]
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

class MaskManager:
    def __init__(self, seq_len: int, num_inference_steps: int, layer_num: int):
        self.warmup_steps = GlobalEnv.get_envs("warmup_steps")
        self.cooldown_steps = GlobalEnv.get_envs("cooldown_steps")
        self.static_interval = GlobalEnv.get_envs("static_interval")
        self.medium_interval = GlobalEnv.get_envs("medium_interval")
        self.enable = GlobalEnv.get_envs("enable_stdit")
        
        self.num_inference_steps = num_inference_steps
        self.num_layers = layer_num
        self.full_seq_len = seq_len
        self.medium_seqlen = seq_len
        self.active_seqlen = seq_len
        
        self.step_level = 0
        
        self.static_bool_mask = None
        self.medium_bool_mask = None
        self.active_bool_mask = None
        
        self.medium_rotary = None
        self.active_rotary = None
        
        self.medium_cache = {}
        self.static_cache = {}
        
        self.block_mask = None
        self.latent_mask = None
        self.sequence_mask = None
        
        # 一些中间变量，存下来以加速
        self.restored_output = None
        self.txt_mask = None
        
        # 记录最大内存使用
        self.max_memory = 0
        
    def generate_mask(self, combined_score: torch.Tensor):
        
        if isinstance(combined_score, np.ndarray):
            combined_score = torch.from_numpy(combined_score)
        
        print(f"{combined_score.shape=}", flush=True)
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
        print(f"{self.latent_mask.shape=}")
        
        return self.latent_mask
    
    def transform_mask(self, latent_mask, height, width):
        pass # TODO
    
    def set_sequence_mask(self, sequence_mask: torch.Tensor):
        self.sequence_mask = sequence_mask

        self.medium_bool_mask = (sequence_mask == 2)
        self.medium_seqlen = self.medium_bool_mask.sum().item()

        self.active_bool_mask = (sequence_mask == 3)
        self.active_seqlen = self.active_bool_mask.sum().item()

        self.static_bool_mask = (sequence_mask == 1)
            
        mask2 = self.sequence_mask[~self.static_bool_mask]
        self.medium_bool_mask_in_l2 = (mask2 == 2).bool()

    def get_seq_len(self)->int:
        if self.step_level == 0 or self.step_level == 3:
            return self.full_seq_len
        elif self.step_level == 2:
            return self.medium_seqlen + self.active_seqlen
        elif self.step_level == 1:
            return self.active_seqlen
        
    def update_step_level(self):
        timestep = get_timestep()
        if timestep is None or timestep <= self.warmup_steps or timestep > self.num_inference_steps - self.cooldown_steps:
            self.step_level =  0 # full compute wo update
        elif (timestep-self.warmup_steps-1) % self.static_interval == 0:
            self.step_level = 3 # full compute w update
        elif (timestep-self.warmup_steps-1) % self.medium_interval == 0:
            self.step_level = 2 # medium compute
        else: 
            self.step_level = 1 # active compute
        
    def get_masked_rotary(self, rotary: tuple, txt_seq_len: int = 0) -> tuple:
        """
        对旋转位置编码应用mask
        
        Args:
            rotary: (cos, sin) 元组，形状为(S, D)
            txt_seq_len: 文本序列长度，0表示没有文本序列
        Returns:
            masked_rotary: (masked_cos, masked_sin) 元组
        """
        # if self.is_warmup_or_cooldown_step() or self.is_update_step():
        #     return rotary
        # elif self.masked_rotary is not None:
        #     return self.masked_rotary
        if self.step_level == 0 or self.step_level == 3:
            return rotary
        elif self.step_level == 2 and self.medium_rotary is not None:
            return self.medium_rotary
        elif self.step_level == 1 and self.active_rotary is not None: 
            return self.active_rotary
        
        cos, sin = rotary  # 形状如：(512 + 4096, 128)
        
        # 保留txt序列
        if txt_seq_len != 0:
            txt_cos, img_cos = torch.split(cos, [txt_seq_len, cos.shape[0] - txt_seq_len], dim=0)
            txt_sin, img_sin = torch.split(sin, [txt_seq_len, sin.shape[0] - txt_seq_len], dim=0)
        else:
            img_cos, img_sin = cos, sin
            txt_cos = txt_sin = None
        
        # 应用mask到图像序列部分
        if self.step_level == 2:
            sequence_mask = ~self.static_bool_mask[:img_cos.shape[0]]
        elif self.step_level == 1:
            sequence_mask = self.active_bool_mask[:img_cos.shape[0]]
            
        masked_cos = img_cos[sequence_mask, ...]
        masked_sin = img_sin[sequence_mask, ...]
        
        # 如果有文本序列，重新组合
        if txt_seq_len != 0:
            masked_cos = torch.cat([txt_cos, masked_cos], dim=0)
            masked_sin = torch.cat([txt_sin, masked_sin], dim=0)
        
        if self.step_level == 2:
            self.medium_rotary = (masked_cos, masked_sin)
        elif self.step_level == 1:
            self.active_rotary = (masked_cos, masked_sin)
        
        return (masked_cos, masked_sin)
    
    def apply_sequence_mask(self, x: torch.Tensor, txt_seq_len: int = 0) -> torch.Tensor:
        """
        单纯地应用mask到序列上，获取active部分
        
        Args:
            x: 输入序列 [B,S,*] (*表示任意维度)
        Returns:
            active_x: masked序列 [B,S',*]
        """
        if self.step_level == 0 or self.step_level == 3:
            return x
        
        if self.step_level == 1:
            sequence_mask = self.active_bool_mask 
        elif self.step_level == 2:
            sequence_mask = ~self.static_bool_mask
        else:
            ValueError("Something went wrong with step level.")
            
        if txt_seq_len > 0:
            # 在sequence_mask前面补充txt_seq_len个True
            if self.txt_mask is None:
                self.txt_mask = torch.ones(txt_seq_len, dtype=torch.bool, device=sequence_mask.device)
            sequence_mask = torch.cat([self.txt_mask, sequence_mask])
            
        return x[:, sequence_mask, ...]  # ...会自动处理剩余维度

    def process_masked_sequence(self, x: torch.Tensor, name: str, layer_idx: int) -> torch.Tensor:
        """处理masked序列"""
        if self.step_level == 0:
            return x
        
        state_key = f"{name}_{layer_idx}"
        device = x.device
        
        if self.restored_output is None:
            B, _, D = x.shape
            self.restored_output = torch.zeros(B, self.full_seq_len, D, device=device, dtype=x.dtype)
            
        if self.step_level == 3:
            # store
            self.medium_cache[state_key] = x[:, self.medium_bool_mask, :]
            self.static_cache[state_key] = x[:, self.static_bool_mask, :]
            return x
        
        elif self.step_level == 2:
            # store
            self.medium_cache[state_key] = x[:, self.medium_bool_mask_in_l2, :]
            # fetch
            self.restored_output[:, ~self.static_bool_mask, :] = x
            self.restored_output[:, self.static_bool_mask, :] = self.static_cache[state_key]
            return self.restored_output
        
        elif self.step_level == 1:
            # fetch
            self.restored_output[:, self.active_bool_mask, :] = x
            self.restored_output[:, self.medium_bool_mask, :] = self.medium_cache[state_key]
            self.restored_output[:, self.static_bool_mask, :] = self.static_cache[state_key]
            return self.restored_output
        
    def process_masked_kv_sequence(self, key: torch.Tensor, value: torch.Tensor, layer_idx: int, txt_seq_len: int = 0) -> tuple:
        """
        处理masked序列，适用于key和value张量
        
        Args:
            x: 输入张量 [B,H,S,D]
            name: 状态名称
            layer_idx: 层索引
            txt_seq_len: 文本序列长度，0表示没有文本序列
        Returns:
            处理后的张量 [B,H,S,D]
        """
        if self.step_level == 0:
            return key, value
        
        assert txt_seq_len > 0
        state_key = str(layer_idx)
        
        x = torch.cat([key, value], dim=0)
        
        # 分离文本和图像序列
        
        if self.step_level == 3:
            with get_timer("level3_kv"):
                # store
                img_seq = x[:, :, txt_seq_len:, :]  # [B,H,img_S,D]
                self.static_cache[state_key] = img_seq[:, :, self.static_bool_mask, :]
                self.medium_cache[state_key] = img_seq[:, :, self.medium_bool_mask, :]
                result = x
                
        elif self.step_level == 2:
            with get_timer("level2_kv"):
                # 分离文本和图像序列
                img_seq = x[:, :, txt_seq_len:, :]  # [B,H,img_S,D]
                self.medium_cache[state_key] = img_seq[:, :, self.medium_bool_mask_in_l2, :]
                # # 恢复图像序列部分
                result = torch.cat([x, self.static_cache[state_key]], dim=2) # 不重排，而是直接连接

        elif self.step_level == 1:
            with get_timer("level1_kv"):
                medium_seq = self.medium_cache[state_key]
                static_seq = self.static_cache[state_key]
                # print(f"{get_timestep()} | {medium_seq.shape=} {static_seq.shape=}", flush=True)
                result = torch.cat([x, medium_seq , static_seq], dim=2)
                if layer_idx == 20:
                    print(f"{get_timestep()} | Fetch from {state_key}, {static_seq.shape=} {medium_seq.shape=}", flush=True)

                
        return result.chunk(2, dim=0)
                
            
    def clear_frozen_states(self):
        """清理frozen状态并重置内存统计"""
        self.static_cache.clear()
        self.medium_cache.clear()
        torch.cuda.reset_peak_memory_stats()
        self.max_memory = 0
    
    def print_memory_stats(self):
        """打印当前GPU内存使用情况"""
        current_memory = torch.cuda.memory_allocated()
        max_memory = torch.cuda.max_memory_allocated()
        self.max_memory = max(self.max_memory, current_memory)
        
        print(f"step {get_timestep()} | GPU Memory Stats:")
        print(f"  Current Memory: {format_memory(current_memory)}")
        print(f"  Peak Memory: {format_memory(max_memory)}")
        print(f"  Session Peak Memory: {format_memory(self.max_memory)}", flush=True)

# ================================ APIs =================================
        
def init_mask_manager(seq_len, num_inference_steps, layer_num) -> MaskManager:
    """初始化MaskManager"""
    mask_manager = MaskManager(seq_len, num_inference_steps, layer_num)
    GlobalEnv.set_envs('MM', mask_manager)
    return mask_manager
    
def get_mask_manager() -> MaskManager:
    """获取MaskManager实例"""
    if GlobalEnv.get_envs("enable_stdit"):
        return GlobalEnv.get_envs('MM')
    else:
        return None