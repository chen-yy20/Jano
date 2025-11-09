import torch
import os
import numpy as np
from pathlib import Path
from scipy import ndimage
from skimage import morphology
from skimage import measure
import matplotlib.pyplot as plt

from .block_manager import init_block_manager
from .stuff import get_timestep, store_feature
from utils.envs import GlobalEnv

def min_max_normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val != 0:
        normalized = (data - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(data)
    return normalized

class DynamicAnalyzer:
    def __init__(self, T, H, W, C):
        """
        初始化动态分析器
        T, H, W, C: 潜空间维度
        """
        self.block_manager = init_block_manager(T, H, W, C)
        self.analysis_steps = GlobalEnv.get_envs('warmup_steps')
        self.stored_features = []
        self.tag = GlobalEnv.get_envs('tag')
        
        # 参数：经过bayesian调优
        self.temporal_weight = GlobalEnv.get_envs("t_weight")
        self.spatial_weight = 1 - self.temporal_weight
        # 扩散强度的参数：warmup步数减少，就需要增强扩散强度。
        self.diffusion_strength = GlobalEnv.get_envs("d_strength")
        self.max_diffusion_distance = GlobalEnv.get_envs("d_distance")
        
        print("\nInitialized Dynamic Analyzer:")
        print("-" * 40)
        print(f"Tag: {self.tag}")
        print(f"Original dimensions: T={T}, H={H}, W={W}, C={C}")
        print(f"Block counts: nt={self.block_manager.nt}, nh={self.block_manager.nh}, nw={self.block_manager.nw}")
        print(f"Analysis steps: {self.analysis_steps}")
        print("-" * 40 + "\n")
        
        self.block_mask = None
        self.latent_mask = None
        
    def step(self, latent):
        """存储潜变量，供后续分析"""
        step = get_timestep()
        if step <= self.analysis_steps:
            self.stored_features.append(latent)
            print(f"Step {step} | Stored noise_pred (shape={latent.shape})", flush=True)
            if step == self.analysis_steps:
                enhanced_combined = self.analyze()
                self.stored_features = None
                return enhanced_combined
        return None
    
    def _compute_dynamics(self, blocked_latents):
        """计算时空动态性"""
        device = blocked_latents.device
        steps, block_num, t, s, c = blocked_latents.shape
        
        temporal_dynamics = torch.zeros(block_num, device=device)
        spatial_dynamics = torch.zeros(block_num, device=device)
        
        for b in range(block_num):
            # 时间动态性计算
            if self.block_manager.T > 1:
                for si in range(s):
                    features = blocked_latents[:, b, :, si, :]
                    feat_norm = features.norm(dim=-1).mean(dim=0)
                    valid_mask = feat_norm > 1e-6
                    if valid_mask.sum() == 0:
                        continue
                    
                    features_valid = features[:, valid_mask, :]
                    diff_matrix = features_valid.unsqueeze(1) - features_valid.unsqueeze(2)
                    diff_matrix_norm = diff_matrix.norm(dim=-1)
                    
                    interval = features_valid.shape[0] // 2
                    if interval > 0:
                        delta = diff_matrix_norm[interval:, :, :] - diff_matrix_norm[:-interval, :, :]
                        temporal_dynamics[b] += delta.abs().mean()
                temporal_dynamics[b] /= s
            
            # 空间动态性计算
            for ti in range(t):
                features = blocked_latents[:, b, ti, :, :]
                feat_norm = features.norm(dim=-1).mean(dim=0)
                valid_mask = feat_norm > 1e-6
                if valid_mask.sum() == 0:
                    continue
                
                features_valid = features[:, valid_mask, :]
                diff_matrix = features_valid.unsqueeze(1) - features_valid.unsqueeze(2)
                diff_matrix_norm = diff_matrix.norm(dim=-1)
                
                interval = features_valid.shape[0] // 2
                if interval > 0:
                    delta = diff_matrix_norm[interval:, :, :] - diff_matrix_norm[:-interval, :, :]
                    spatial_dynamics[b] += delta.abs().mean()
            spatial_dynamics[b] /= t
        
        if self.block_manager.T == 1:
            temporal_dynamics.fill_(0)
            
        return temporal_dynamics, spatial_dynamics
    
    def _enhance_combined_score(self, original_combined, block_dims):
        """使用向内扩散增强combined score"""
        nt, nh, nw = block_dims
        enhanced_combined = original_combined.copy()
        
        for t in range(nt):
            current_layer = original_combined.reshape(nt, nh, nw)[t, :, :]
            
            score_std = np.std(current_layer)
            score_mean = np.mean(current_layer)
            
            contour_threshold = score_mean + 0.3 * score_std
            contour_mask = current_layer > contour_threshold
            
            internal_threshold = score_mean
            potential_internal_mask = current_layer < internal_threshold
            
            if not np.any(contour_mask) or not np.any(potential_internal_mask):
                continue
            
            # 边界禁止区域
            forbidden_mask = np.zeros_like(current_layer, dtype=bool)
            forbidden_mask[0, :] = True
            forbidden_mask[-1, :] = True
            forbidden_mask[:, 0] = True
            forbidden_mask[:, -1] = True
            
            candidate_internal = potential_internal_mask & (~forbidden_mask)
            labeled_candidates = measure.label(candidate_internal)
            true_internal_mask = np.zeros_like(current_layer, dtype=bool)
            
            # 筛选有效的内部区域
            for region_id in range(1, labeled_candidates.max() + 1):
                region_mask = labeled_candidates == region_id
                region_boundary = morphology.binary_dilation(region_mask) & (~region_mask)
                
                if np.any(contour_mask):
                    distance_to_contour = ndimage.distance_transform_edt(~contour_mask)
                    boundary_distances = distance_to_contour[region_boundary]
                    close_to_contour_ratio = np.mean(boundary_distances <= 2.0)
                    
                    if close_to_contour_ratio >= 0.3 and np.sum(region_mask) >= 2:
                        true_internal_mask |= region_mask
            
            # 扩散增强
            if np.any(true_internal_mask) and np.any(contour_mask):
                distance_from_contour = ndimage.distance_transform_edt(~contour_mask)
                valid_diffusion_mask = true_internal_mask & (distance_from_contour <= self.max_diffusion_distance)
                
                if np.any(valid_diffusion_mask):
                    distances = distance_from_contour[valid_diffusion_mask]
                    diffusion_weights = np.exp(-distances / (self.max_diffusion_distance / 2))
                    avg_contour_score = np.mean(current_layer[contour_mask])
                    enhancement_values = diffusion_weights * avg_contour_score * self.diffusion_strength
                    enhanced_layer = current_layer.copy()
                    enhanced_layer[valid_diffusion_mask] += enhancement_values
                    
                    # 更新增强后的combined score
                    start_idx = t * nh * nw
                    end_idx = start_idx + nh * nw
                    enhanced_combined[start_idx:end_idx] = enhanced_layer.flatten()
        
        return enhanced_combined

    def analyze(self):
        """储存完profile steps后, 进行动态性分析并返回enhanced combined score"""
        if not self.stored_features:
            print("No features stored for analysis")
            return None
            
        print(f"\nAnalyzing {self.tag}...", flush=True)
        
        # 准备数据
        latents_tensor = torch.stack(self.stored_features, dim=0)
        latents_tensor = latents_tensor.cuda()
        blocked_latents = self.block_manager.block_3d(latents_tensor)
        
        if GlobalEnv.get_envs("cc_exp") and GlobalEnv.get_envs("static_interval") == 0:
            store_feature(latents_tensor, timestep=self.analysis_steps, layer=0, name="stack_latent")
            print(f"Stored stack_latent ({latents_tensor.shape}).", flush=True)
        
        # 计算动态性（在GPU上）
        temporal_dynamics, spatial_dynamics = self._compute_dynamics(blocked_latents)
        
        # 转换到CPU进行后续处理
        temporal_dynamics_cpu = temporal_dynamics.cpu().numpy()
        spatial_dynamics_cpu = spatial_dynamics.cpu().numpy()
        
        original_combined = self.temporal_weight * temporal_dynamics_cpu + self.spatial_weight * spatial_dynamics_cpu
        original_combined = min_max_normalize(original_combined)
        
        # 计算增强版本
        enhanced_combined = self._enhance_combined_score(
            original_combined,
            (self.block_manager.nt, self.block_manager.nh, self.block_manager.nw)
        )
        
        # 可视化 - 确保传入的都是numpy数组
        visualize_block_dynamics(
            temporal_dynamics_cpu, spatial_dynamics_cpu, original_combined, enhanced_combined,
            (self.block_manager.nt, self.block_manager.nh, self.block_manager.nw),
            save_name=self.tag
        )
        
        print(f"Analysis completed for {self.tag}\n", flush=True)
        
        return enhanced_combined
    
    
def visualize_block_dynamics(temporal_dynamics, spatial_dynamics, original_combined, enhanced_combined, 
                           block_dims, save_name="block_dynamics"):
    """
    独立的可视化函数，显示4个热力图
    
    Args:
        temporal_dynamics: 1D numpy array, 时间复杂度
        spatial_dynamics: 1D numpy array, 空间复杂度  
        original_combined: 1D numpy array, 原始加权复杂度
        enhanced_combined: 1D numpy array, 增强复杂度
        block_dims: tuple, (nt, nh, nw) 块维度
        save_name: str, 保存文件名
    """
    nt, nh, nw = block_dims
    
    # 确保所有输入都是numpy数组
    if isinstance(temporal_dynamics, torch.Tensor):
        temporal_dynamics = temporal_dynamics.cpu().numpy()
    if isinstance(spatial_dynamics, torch.Tensor):
        spatial_dynamics = spatial_dynamics.cpu().numpy()
    if isinstance(original_combined, torch.Tensor):
        original_combined = original_combined.cpu().numpy()
    if isinstance(enhanced_combined, torch.Tensor):
        enhanced_combined = enhanced_combined.cpu().numpy()
    
    # 将1D数据reshape为3D，然后计算时间平均
    temporal_3d = temporal_dynamics.reshape(nt, nh, nw)
    spatial_3d = spatial_dynamics.reshape(nt, nh, nw)
    original_combined_3d = original_combined.reshape(nt, nh, nw)
    enhanced_combined_3d = enhanced_combined.reshape(nt, nh, nw)
    
    # 计算时间平均的2D热力图
    temporal_2d = np.mean(temporal_3d, axis=0)
    spatial_2d = np.mean(spatial_3d, axis=0)
    original_combined_2d = np.mean(original_combined_3d, axis=0)
    enhanced_combined_2d = np.mean(enhanced_combined_3d, axis=0)
    
    # 创建2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 设置颜色映射
    cmap = 'viridis'
    
    # 1. 时间复杂度热力图
    im1 = axes[0, 0].imshow(temporal_2d, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    axes[0, 0].set_title(f'Temporal Dynamics\n(Mean: {temporal_2d.mean():.4f})', fontsize=12)
    axes[0, 0].set_xlabel('Width Blocks')
    axes[0, 0].set_ylabel('Height Blocks')
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
    
    # 2. 空间复杂度热力图
    im2 = axes[0, 1].imshow(spatial_2d, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Spatial Dynamics\n(Mean: {spatial_2d.mean():.4f})', fontsize=12)
    axes[0, 1].set_xlabel('Width Blocks')
    axes[0, 1].set_ylabel('Height Blocks')
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
    
    # 3. 原始加权复杂度热力图
    im3 = axes[1, 0].imshow(original_combined_2d, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    axes[1, 0].set_title(f'Combined Score\n(Mean: {original_combined_2d.mean():.4f})', fontsize=12)
    axes[1, 0].set_xlabel('Width Blocks')
    axes[1, 0].set_ylabel('Height Blocks')
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
    
    # 4. 增强复杂度热力图
    im4 = axes[1, 1].imshow(enhanced_combined_2d, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    axes[1, 1].set_title(f'Enhanced Score\n(Mean: {enhanced_combined_2d.mean():.4f})', fontsize=12)
    axes[1, 1].set_xlabel('Width Blocks')
    axes[1, 1].set_ylabel('Height Blocks')
    plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(GlobalEnv.get_envs("save_dir"), f"block_dynamics_{save_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Block dynamics visualization saved to: {save_path}")
    
    # 保存简单统计信息
    enhancement_ratio = enhanced_combined_2d.mean() / original_combined_2d.mean() if original_combined_2d.mean() > 0 else 1.0
    
    stats = {
        'temporal_mean': float(temporal_2d.mean()),
        'spatial_mean': float(spatial_2d.mean()),
        'combined_mean': float(original_combined_2d.mean()),
        'enhanced_mean': float(enhanced_combined_2d.mean()),
        'enhancement_ratio': float(enhancement_ratio)
    }
    
    stats_path = os.path.join(GlobalEnv.get_envs("save_dir"), f"block_stats_{save_name}.txt")
    
    with open(stats_path, 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value:.6f}\n")
    
    print(f"Block statistics saved to: {stats_path}")
    print(f"Enhancement ratio: {enhancement_ratio:.4f}")
    
    return save_path, stats