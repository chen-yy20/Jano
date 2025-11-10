import torch
import torch.nn.functional as F
import os 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap

from .block_manager import get_block_manager
from utils.envs import GlobalEnv

from datetime import datetime

def get_time_str():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def get_prompt_id(prompt):
    # 取关键词并过滤掉常用词
    skip_words = {'a', 'an', 'the', 'in', 'on', 'at', 'and', 'or', 'of', 'with', 'by'}
    words = [w for w in prompt.lower().split() if w not in skip_words]
    
    # 取前3-4个关键词
    key_words = words[:4]
    
    # 组合关键词
    id = '_'.join(key_words)
    
    return id

def store_feature(feature: torch.Tensor, timestep: int, layer: int, name: str, rank:int = 0):
    
    tag = GlobalEnv.get_envs("tag")
    dir_path = os.path.join("./exp3_conv_comp/wan-1.3B", tag)
    # name = f"{name}_t{timestep}l{layer}r{rank}.pt"
    name = f"{name}.pt"
    # 打印当前工作目录和目标目录
    print(f"Current working directory: {os.getcwd()}")
    print(f"Target directory: {dir_path}")
    
    # 检查目录是否存在
    if not os.path.exists(dir_path):
        print(f"Creating directory: {dir_path}")
        os.makedirs(dir_path, exist_ok=True)
    
    # 检查目录是否可写
    if not os.access(dir_path, os.W_OK):
        print(f"Directory {dir_path} is not writable!")
        return
        
    path = os.path.join(dir_path, name)
    print(f"Attempting to save to: {path}")
    
    try:
        torch.save(feature.detach().cpu(), path)  # 确保tensor在CPU上
        print(f"Successfully saved to {path}")
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        raise
    
    
def split_sequence_chunks(x, seq_lens, grid_sizes, temporal_chunks=2, height_chunks=2, width_chunks=2):
    """
    Split sequence into chunks while preserving spatial-temporal correspondence
    
    Args:
        x: tensor of shape [B, L, C] where L = F * (H/2) * (W/2)
        seq_lens: tensor of shape [B]
        grid_sizes: tensor of shape [B, 3] containing [F, H/2, W/2]
        temporal_chunks, height_chunks, width_chunks: number of chunks for each dimension
    """
    B, L, C = x.shape
    F, H, W = grid_sizes[0]  # 取第一个batch的grid size
    
    # 验证可分割性
    assert F % temporal_chunks == 0, f"Temporal dimension {F} not divisible by {temporal_chunks}"
    assert H % height_chunks == 0, f"Height {H} not divisible by {height_chunks}"
    assert W % width_chunks == 0, f"Width {W} not divisible by {width_chunks}"
    
    # 计算每个块的大小
    f_chunk_size = F // temporal_chunks
    h_chunk_size = H // height_chunks
    w_chunk_size = W // width_chunks
    
    # 重塑张量以便于分块
    x = x.reshape(B, F, H, W, C)
    
    total_chunks = temporal_chunks * height_chunks * width_chunks
    # 预分配输出张量
    chunks = torch.zeros(B * total_chunks, f_chunk_size * h_chunk_size * w_chunk_size, C, 
                        device=x.device, dtype=x.dtype)
    
    chunk_idx = 0
    for t in range(temporal_chunks):
        for h in range(height_chunks):
            for w in range(width_chunks):
                # 提取块
                t_start, t_end = t * f_chunk_size, (t + 1) * f_chunk_size
                h_start, h_end = h * h_chunk_size, (h + 1) * h_chunk_size
                w_start, w_end = w * w_chunk_size, (w + 1) * w_chunk_size
                
                chunk = x[:, t_start:t_end, h_start:h_end, w_start:w_end, :]
                # 重塑回序列形式
                chunk = chunk.reshape(B, -1, C)
                # 放入预分配的张量中
                chunks[chunk_idx*B:(chunk_idx+1)*B] = chunk
                chunk_idx += 1
    
    new_seq_lens = torch.full((B * total_chunks,), 
                            f_chunk_size * h_chunk_size * w_chunk_size, 
                            device=seq_lens.device)
    new_grid_sizes = torch.tensor([f_chunk_size, h_chunk_size, w_chunk_size], 
                                device=grid_sizes.device).repeat(B * total_chunks, 1)
    
    chunk_info = {
        'original_shape': (B, L, C),
        'original_grid': (F, H, W),
        'temporal_chunks': temporal_chunks,
        'height_chunks': height_chunks,
        'width_chunks': width_chunks,
        'f_chunk_size': f_chunk_size,
        'h_chunk_size': h_chunk_size,
        'w_chunk_size': w_chunk_size,
        'total_chunks': total_chunks
    }
    
    return chunks, new_seq_lens, new_grid_sizes, chunk_info

def merge_sequence_chunks(x, chunk_info):
    """
    Merge sequence chunks back to original tensor
    """
    B, L, C = chunk_info['original_shape']
    F, H, W = chunk_info['original_grid']
    temporal_chunks = chunk_info['temporal_chunks']
    height_chunks = chunk_info['height_chunks']
    width_chunks = chunk_info['width_chunks']
    f_chunk_size = chunk_info['f_chunk_size']
    h_chunk_size = chunk_info['h_chunk_size']
    w_chunk_size = chunk_info['w_chunk_size']
    total_chunks = chunk_info['total_chunks']
    
    # 初始化输出张量
    merged = torch.zeros((B, F, H, W, C), device=x.device)
    
    chunk_idx = 0
    for t in range(temporal_chunks):
        for h in range(height_chunks):
            for w in range(width_chunks):
                t_start = t * f_chunk_size
                t_end = (t + 1) * f_chunk_size
                h_start = h * h_chunk_size
                h_end = (h + 1) * h_chunk_size
                w_start = w * w_chunk_size
                w_end = (w + 1) * w_chunk_size
                
                # 从batch维度获取当前chunk
                current_chunk = x[chunk_idx*B:(chunk_idx+1)*B]
                current_chunk = current_chunk.reshape(B, f_chunk_size, h_chunk_size, w_chunk_size, C)
                merged[:, t_start:t_end, h_start:h_end, w_start:w_end, :] = current_chunk
                chunk_idx += 1
    
    # 重塑回原始序列形式
    merged = merged.reshape(B, L, C)
    
    return merged

# 使用示例
"""
x: [1, 32760, 1536]
seq_lens: [1]
grid_sizes: tensor([[21, 30, 52]])

# 分割
chunks, new_seq_lens, new_grid_sizes, chunk_info = split_sequence_chunks(
    x, seq_lens, grid_sizes,
    temporal_chunks=3, height_chunks=2, width_chunks=2
)
# chunks shape: [1*12, smaller_seq_len, 1536]

# 处理...

# 合并
merged = merge_sequence_chunks(chunks, chunk_info)
# merged shape: [1, 32760, 1536]
"""

TIMESTEP = -1

def update_timestep(timestep: int):
    global TIMESTEP
    TIMESTEP = timestep + 1
        
    
def get_timestep() -> int:
    global TIMESTEP
    return TIMESTEP
    
    
from utils.timer import get_timer
from utils.envs import GlobalEnv

def get_masked_timer(name):
    try:
        prefix = GlobalEnv.get_envs("timer_prefix")
    except KeyError:
        prefix = "full"
        
    return get_timer(f"{prefix}_{name}")


def compute_attention_scores(q, k):
    """
    逐个head计算注意力的softmax矩阵，减少内存使用
    Args:
        q: query tensor, shape (batch_size, seq_len, num_heads, head_dim)
        k: key tensor, shape (batch_size, seq_len, num_heads, head_dim)
    Returns:
        attention_scores: shape (batch_size, num_heads, seq_len, seq_len)
    """
    batch_size, seq_len, num_heads, head_dim = q.shape
    scale = head_dim ** -0.5
    
    # 预分配输出空间
    attention_scores = torch.zeros(batch_size, num_heads, seq_len, seq_len, 
                                 device=q.device, dtype=q.dtype)
    
    # 逐个head计算
    for h in range(num_heads):
        # 获取当前head的q和k
        q_h = q[:, :, h, :]  # (batch_size, seq_len, head_dim)
        k_h = k[:, :, h, :]  # (batch_size, seq_len, head_dim)
        
        # 计算注意力分数
        scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale  # (batch_size, seq_len, seq_len)
        scores = F.softmax(scores, dim=-1)
        
        # 存储结果
        attention_scores[:, h] = scores
        
        # 及时释放临时变量
        del scores
        
    print(f"Attention scores shape: {attention_scores.shape}", flush=True)
    return attention_scores


def visualize_attention_patterns(attention_scores, layer_id, cond, block_size=128):
    """
    将大型attention矩阵按block聚合并可视化
    Args:
        attention_scores: shape (batch_size, num_heads, seq_len, seq_len)
        block_size: 聚合的块大小
    """
    batch_size, num_heads, seq_len, _ = attention_scores.shape
    
    # 计算需要多少个block
    num_blocks = seq_len // block_size
    if seq_len % block_size != 0:
        num_blocks += 1
    
    # 创建聚合后的矩阵
    blocked_attention = torch.zeros(batch_size, num_heads, num_blocks, num_blocks)
    
    # 按block聚合
    for i in range(num_blocks):
        start_i = i * block_size
        end_i = min((i + 1) * block_size, seq_len)
        for j in range(num_blocks):
            start_j = j * block_size
            end_j = min((j + 1) * block_size, seq_len)
            # 计算每个block的平均attention score
            blocked_attention[:, :, i, j] = attention_scores[:, :, start_i:end_i, start_j:end_j].mean(dim=(2, 3))
    
    # 可视化
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.ravel()
    
    for h in range(num_heads):
        # 使用第一个batch的数据
        attention_map = blocked_attention[0, h].cpu().numpy()
        
        im = axes[h].imshow(attention_map, cmap='viridis')
        axes[h].set_title(f'Head {h}')
        axes[h].axis('off')
        plt.colorbar(im, ax=axes[h])
    
    plt.tight_layout()
    save_path = os.path.join(GlobalEnv.get_envs("save_dir"), f"softmax_{layer_id}_c{cond}.png")
    plt.savefig(save_path)
    
    # 返回聚合后的attention矩阵，方便进一步分析
    return blocked_attention

def print_gpu_memory(message: str = None):
    """
    打印当前显存使用的简单版本
    """
    print(f"{message} | GPU memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB "
          f"(Peak: {torch.cuda.max_memory_allocated()/1024**2:.2f}MB)", flush=True)


def visualize_dynamics(temporal_dynamics, spatial_dynamics, save_name):
        """可视化时空动态性（只显示有效区域）"""
        # 获取有效区域mask
        bm = get_block_manager()
        if bm.T == 1:
            # T=1的情况只显示空间动态性
            plt.figure(figsize=(10, 8))
            
            spat_map = spatial_dynamics.reshape(bm.nh, bm.nw).cpu().numpy()
            
            # 只显示有效区域
            valid_h = min(bm.nh, (bm.H + bm.block_size[1] - 1) // bm.block_size[1])
            valid_w = min(bm.nw, (bm.W + bm.block_size[2] - 1) // bm.block_size[2])
            
            sns.heatmap(spat_map[:valid_h, :valid_w], cmap='viridis', annot=False, fmt='.3f')
            plt.title(f'Spatial Dynamics (Block Size: {bm.block_size})')
            plt.xlabel('Width Blocks')
            plt.ylabel('Height Blocks')
            
        else:
            # T>1的情况显示时空动态性
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            temp_map = temporal_dynamics.reshape(bm.nt, bm.nh, bm.nw).cpu().numpy()
            spat_map = spatial_dynamics.reshape(bm.nt, bm.nh, bm.nw).cpu().numpy()
            # TODO: 把这个换到utils去，不要依赖self了！
            # 只显示有效区域
            valid_t = min(bm.nt, (bm.T + bm.block_size[0] - 1) // bm.block_size[0])
            valid_h = min(bm.nh, (bm.H + bm.block_size[1] - 1) // bm.block_size[1])
            valid_w = min(bm.nw, (bm.W + bm.block_size[2] - 1) // bm.block_size[2])
            
            temp_avg = temp_map[:valid_t, :valid_h, :valid_w].mean(axis=0)
            spat_avg = spat_map[:valid_t, :valid_h, :valid_w].mean(axis=0)
            
            sns.heatmap(temp_avg, ax=ax1, cmap='viridis', annot=False, fmt='.3f')
            ax1.set_title(f'Temporal Dynamics (Block Size: {bm.block_size})')
            ax1.set_xlabel('Width Blocks')
            ax1.set_ylabel('Height Blocks')
            
            sns.heatmap(spat_avg, ax=ax2, cmap='viridis', annot=False, fmt='.3f')
            ax2.set_title(f'Spatial Dynamics (Block Size: {bm.block_size})')
            ax2.set_xlabel('Width Blocks')
            ax2.set_ylabel('Height Blocks')
        
        plt.tight_layout()
        p_steps = GlobalEnv.get_envs("warmup_steps")
        save_path = os.path.join(GlobalEnv.get_envs("save_dir"), f"Dynamic_{save_name}_{p_steps}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Dynamic result saved to {save_path}.")
        
        
def visualize_mask(mask: torch.Tensor):
    """
    可视化mask的时空分布
    
    Args:
        mask: [C,F,H,W] 的int张量，值为1,2,3表示不同动态等级
    """
    # 使用第一个通道的mask进行可视化
    mask = mask[0].cpu().float()  # [F,H,W]
    
    bm = get_block_manager()
    
    # 定义颜色映射
    colors = ['#2E86C1', '#F4D03F', '#E74C3C']  # 蓝色(低)、黄色(中)、红色(高)
    cmap = ListedColormap(colors)
    
    if bm.T == 1:
        # T=1的情况只显示空间分布
        plt.figure(figsize=(10, 8))
        
        spatial_dist = mask[0].numpy()  # [H,W]
        im = plt.imshow(spatial_dist, cmap=cmap, vmin=1, vmax=3)
        plt.colorbar(im, ticks=[1, 2, 3], 
                    label='Dynamic Level',
                    boundaries=np.arange(0.5, 4.5, 1),
                    values=[1, 2, 3])
        plt.title('Spatial Distribution of Dynamic Regions')
        plt.xlabel('Width')
        plt.ylabel('Height')
        
    else:
        # T>1的情况显示时空分布
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 时间-空间平均图
        temporal_spatial = mask.mean(dim=0).numpy()  # [H,W]
        im = ax1.imshow(temporal_spatial, cmap='RdYlBu_r', vmin=1, vmax=3)
        plt.colorbar(im, ax=ax1, 
                    label='Average Dynamic Level',
                    ticks=[1, 2, 3])
        ax1.set_title('Spatial Distribution (Temporal Average)')
        ax1.set_xlabel('Width')
        ax1.set_ylabel('Height')
        
        # 2. 时间分布图 - 修正这里
        temporal_dist_low = (mask == 1).float().mean(dim=(1,2)).numpy()  # [F] -> numpy
        temporal_dist_med = (mask == 2).float().mean(dim=(1,2)).numpy()  # [F] -> numpy
        temporal_dist_high = (mask == 3).float().mean(dim=(1,2)).numpy()  # [F] -> numpy
        
        frames = np.arange(len(temporal_dist_low))
        
        # 修正stackplot的调用方式
        ax2.stackplot(frames, 
                     temporal_dist_low,  # 分别传递每个数组
                     temporal_dist_med, 
                     temporal_dist_high,
                     labels=['Low', 'Medium', 'High'],
                     colors=colors)
        
        ax2.set_title('Temporal Distribution of Dynamic Regions')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Region Ratio')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    tag = GlobalEnv.get_envs("tag")
    p_steps = GlobalEnv.get_envs("warmup_steps")
    save_path = os.path.join(GlobalEnv.get_envs("save_dir"), f"{tag}_{p_steps}_Mask.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Mask visualization saved to {save_path}")