#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算两个文件夹中同名视频的 PSNR、SSIM 和 LPIPS 指标
"""

import os
import json
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image


# 尝试导入 LPIPS
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("警告: LPIPS 库未安装，将跳过 LPIPS 计算。")
    print("安装命令: pip install lpips")


def load_video_frames(video_path):
    """加载视频的所有帧"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 转换为 RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    return frames, fps, frame_count


def calculate_psnr(img1, img2):
    """计算 PSNR (Peak Signal-to-Noise Ratio)"""
    if img1.shape != img2.shape:
        # 如果尺寸不同，调整 img2 的尺寸
        h, w = img1.shape[:2]
        img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 转换为 float 类型
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # 计算 PSNR
    psnr = peak_signal_noise_ratio(img1, img2, data_range=255)
    return psnr


def calculate_ssim(img1, img2):
    """计算 SSIM (Structural Similarity Index)"""
    if img1.shape != img2.shape:
        # 如果尺寸不同，调整 img2 的尺寸
        h, w = img1.shape[:2]
        img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # SSIM 计算需要灰度图或单通道，但也可以处理多通道
    # 对于 RGB 图像，兼容不同版本的 scikit-image
    if len(img1.shape) == 3:
        try:
            # 新版本使用 channel_axis
            ssim = structural_similarity(img1, img2, channel_axis=2, data_range=255)
        except TypeError:
            # 旧版本使用 multichannel
            ssim = structural_similarity(img1, img2, multichannel=True, data_range=255)
    else:
        ssim = structural_similarity(img1, img2, data_range=255)
    
    return ssim


def calculate_lpips(img1, img2, lpips_model):
    """计算 LPIPS (Learned Perceptual Image Patch Similarity)"""
    if img1.shape != img2.shape:
        # 如果尺寸不同，调整 img2 的尺寸
        h, w = img1.shape[:2]
        img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 确保图像是 uint8 类型
    if img1.dtype != np.uint8:
        img1 = np.clip(img1, 0, 255).astype(np.uint8)
    if img2.dtype != np.uint8:
        img2 = np.clip(img2, 0, 255).astype(np.uint8)
    
    # 转换为 PIL Image 然后转换为 tensor
    img1_pil = Image.fromarray(img1)
    img2_pil = Image.fromarray(img2)
    
    # 转换为 tensor (范围 0-1)
    # LPIPS 期望输入在 [0, 1] 范围
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为 [0, 1]
    ])
    
    try:
        img1_tensor = transform(img1_pil).unsqueeze(0)
        img2_tensor = transform(img2_pil).unsqueeze(0)
        
        # 移动到 GPU 如果可用
        device = next(lpips_model.parameters()).device
        img1_tensor = img1_tensor.to(device)
        img2_tensor = img2_tensor.to(device)
        
        # 计算 LPIPS
        with torch.no_grad():
            lpips_value = lpips_model(img1_tensor, img2_tensor)
        
        # 提取数值，处理可能的 tensor
        if isinstance(lpips_value, torch.Tensor):
            lpips_value = lpips_value.cpu()
            # 如果是多维 tensor，取平均值或者第一个元素
            if lpips_value.numel() > 1:
                lpips_value = lpips_value.mean()
            lpips_value = lpips_value.item()
        
        # 检查是否为 nan 或 inf
        if np.isnan(lpips_value) or np.isinf(lpips_value):
            return None
        
        # 确保是有效数值
        if not isinstance(lpips_value, (int, float)):
            return None
        
        return float(lpips_value)
    except Exception as e:
        print(f"LPIPS 计算错误: {str(e)}")
        return None


def evaluate_videos(origin_dir, speed_up_dir, output_file='evaluation_results.json', use_lpips=True):
    """评估两个文件夹中的视频"""
    origin_dir = os.path.abspath(origin_dir)
    speed_up_dir = os.path.abspath(speed_up_dir)
    
    # 初始化 LPIPS 模型
    lpips_model = None
    if use_lpips and LPIPS_AVAILABLE:
        print("正在初始化 LPIPS 模型...")
        lpips_model = lpips.LPIPS(net='alex')  # 使用 alex 网络，也可以使用 'vgg' 或 'squeeze'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        lpips_model = lpips_model.to(device)
        lpips_model.eval()
        print(f"LPIPS 模型已加载到 {device}")
    
    # 获取两个文件夹中的所有视频文件
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    origin_files = {f for f in os.listdir(origin_dir) 
                   if any(f.lower().endswith(ext) for ext in video_extensions)}
    speed_up_files = {f for f in os.listdir(speed_up_dir) 
                     if any(f.lower().endswith(ext) for ext in video_extensions)}
    
    # 找到两个文件夹中都存在的文件
    common_files = sorted(origin_files & speed_up_files)
    
    if not common_files:
        print(f"警告: 在 {origin_dir} 和 {speed_up_dir} 中没有找到相同的视频文件")
        return
    
    print(f"找到 {len(common_files)} 对匹配的视频")
    
    results = []
    all_psnr_values = []
    all_ssim_values = []
    all_lpips_values = []
    
    # 逐个处理视频
    for filename in tqdm(common_files, desc="处理视频"):
        origin_path = os.path.join(origin_dir, filename)
        speed_up_path = os.path.join(speed_up_dir, filename)
        
        try:
            # 加载视频帧
            origin_frames, origin_fps, origin_frame_count = load_video_frames(origin_path)
            speed_up_frames, speed_up_fps, speed_up_frame_count = load_video_frames(speed_up_path)
            
            # 确保帧数相同（取较小值）
            min_frames = min(len(origin_frames), len(speed_up_frames))
            
            if min_frames == 0:
                print(f"警告: {filename} 没有有效帧")
                continue
            
            # 逐帧计算指标
            frame_psnr = []
            frame_ssim = []
            frame_lpips = []
            
            for i in range(min_frames):
                img_origin = origin_frames[i]
                img_speed = speed_up_frames[i]
                
                # 计算 PSNR
                psnr = calculate_psnr(img_origin, img_speed)
                frame_psnr.append(psnr)
                
                # 计算 SSIM
                ssim = calculate_ssim(img_origin, img_speed)
                frame_ssim.append(ssim)
                
                # 计算 LPIPS
                if lpips_model is not None:
                    lpips_value = calculate_lpips(img_origin, img_speed, lpips_model)
                    if lpips_value is not None:
                        frame_lpips.append(lpips_value)
            
            # 计算平均值
            avg_psnr = np.mean(frame_psnr)
            avg_ssim = np.mean(frame_ssim)
            avg_lpips = np.mean(frame_lpips) if frame_lpips and len(frame_lpips) > 0 else None
            
            results.append({
                'filename': filename,
                'frame_count': min_frames,
                'origin_fps': float(origin_fps),
                'speed_up_fps': float(speed_up_fps),
                'psnr_mean': float(avg_psnr),
                'psnr_std': float(np.std(frame_psnr)),
                'ssim_mean': float(avg_ssim),
                'ssim_std': float(np.std(frame_ssim)),
                'lpips_mean': float(avg_lpips) if avg_lpips is not None else None,
                'lpips_std': float(np.std(frame_lpips)) if frame_lpips else None,
            })
            
            all_psnr_values.extend(frame_psnr)
            all_ssim_values.extend(frame_ssim)
            if frame_lpips:
                all_lpips_values.extend(frame_lpips)
            
        except Exception as e:
            print(f"处理 {filename} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 计算总体统计信息
    if results:
        summary = {
            'total_videos': len(results),
            'total_frames': len(all_psnr_values),
            'psnr': {
                'mean': float(np.mean(all_psnr_values)),
                'std': float(np.std(all_psnr_values)),
                'min': float(np.min(all_psnr_values)),
                'max': float(np.max(all_psnr_values)),
            },
            'ssim': {
                'mean': float(np.mean(all_ssim_values)),
                'std': float(np.std(all_ssim_values)),
                'min': float(np.min(all_ssim_values)),
                'max': float(np.max(all_ssim_values)),
            },
        }
        
        # 过滤掉 None 和 nan 值
        valid_lpips_values = [v for v in all_lpips_values if v is not None and not np.isnan(v)]
        
        if valid_lpips_values:
            summary['lpips'] = {
                'mean': float(np.mean(valid_lpips_values)),
                'std': float(np.std(valid_lpips_values)),
                'min': float(np.min(valid_lpips_values)),
                'max': float(np.max(valid_lpips_values)),
            }
        
        summary['results'] = results
        
        # 保存结果到 JSON 文件
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 打印统计信息
        print("\n" + "="*70)
        print("评估结果统计:")
        print("="*70)
        print(f"总视频数: {len(results)}")
        print(f"总帧数: {len(all_psnr_values)}")
        print(f"\nPSNR:")
        print(f"  平均值: {summary['psnr']['mean']:.4f}")
        print(f"  标准差: {summary['psnr']['std']:.4f}")
        print(f"  最小值: {summary['psnr']['min']:.4f}")
        print(f"  最大值: {summary['psnr']['max']:.4f}")
        print(f"\nSSIM:")
        print(f"  平均值: {summary['ssim']['mean']:.4f}")
        print(f"  标准差: {summary['ssim']['std']:.4f}")
        print(f"  最小值: {summary['ssim']['min']:.4f}")
        print(f"  最大值: {summary['ssim']['max']:.4f}")
        if 'lpips' in summary:
            print(f"\nLPIPS:")
            print(f"  平均值: {summary['lpips']['mean']:.4f}")
            print(f"  标准差: {summary['lpips']['std']:.4f}")
            print(f"  最小值: {summary['lpips']['min']:.4f}")
            print(f"  最大值: {summary['lpips']['max']:.4f}")
            print(f"  有效帧数: {len(valid_lpips_values)} / {len(all_lpips_values)}")
        else:
            print(f"\nLPIPS: 无有效数据（所有帧计算失败或返回 nan）")
        print("="*70)
        print(f"\n详细结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='计算两个文件夹中视频的 PSNR、SSIM 和 LPIPS')
    parser.add_argument('--origin', type=str, 
                       default='/home/zlq/diffusion/vbench_result/wan2.1/origin',
                       help='原始视频文件夹路径')
    parser.add_argument('--speed_up', type=str,
                       default='/home/zlq/diffusion/vbench_result/wan2.1/STDit_new',
                       help='加速后视频文件夹路径')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='输出结果文件名')
    parser.add_argument('--no_lpips', action='store_true',
                       help='跳过 LPIPS 计算（加快速度）')
    
    args = parser.parse_args()
    
    evaluate_videos(args.origin, args.speed_up, args.output, use_lpips=not args.no_lpips)


if __name__ == '__main__':
    main()

