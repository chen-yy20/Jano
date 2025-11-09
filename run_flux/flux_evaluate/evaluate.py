#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算两个文件夹中同名图片的 PSNR 和 SSIM 指标
"""

import os
import json
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
import argparse


def load_image(image_path):
    """加载图片并转换为 numpy 数组"""
    img = Image.open(image_path).convert('RGB')
    return np.array(img)


def calculate_psnr(img1, img2):
    """计算 PSNR (Peak Signal-to-Noise Ratio)"""
    # PSNR 需要图片在相同尺寸
    if img1.shape != img2.shape:
        # 如果尺寸不同，将 img2 调整为 img1 的尺寸
        img2_resized = Image.fromarray(img2).resize((img1.shape[1], img1.shape[0]), Image.Resampling.LANCZOS)
        img2 = np.array(img2_resized)
    
    # 转换为 float 类型
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # 计算 PSNR
    psnr = peak_signal_noise_ratio(img1, img2, data_range=255)
    return psnr


def calculate_ssim(img1, img2):
    """计算 SSIM (Structural Similarity Index)"""
    # SSIM 需要图片在相同尺寸
    if img1.shape != img2.shape:
        # 如果尺寸不同，将 img2 调整为 img1 的尺寸
        img2_resized = Image.fromarray(img2).resize((img1.shape[1], img1.shape[0]), Image.Resampling.LANCZOS)
        img2 = np.array(img2_resized)
    
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


def evaluate_images(origin_dir, speed_up_dir, output_file='evaluation_results.json'):
    """评估两个文件夹中的图片"""
    origin_dir = os.path.abspath(origin_dir)
    speed_up_dir = os.path.abspath(speed_up_dir)
    
    # 获取两个文件夹中的所有 PNG 文件
    origin_files = {f for f in os.listdir(origin_dir) if f.lower().endswith('.png')}
    speed_up_files = {f for f in os.listdir(speed_up_dir) if f.lower().endswith('.png')}
    
    # 找到两个文件夹中都存在的文件
    common_files = sorted(origin_files & speed_up_files)
    
    if not common_files:
        print(f"警告: 在 {origin_dir} 和 {speed_up_dir} 中没有找到相同的图片文件")
        return
    
    print(f"找到 {len(common_files)} 对匹配的图片")
    
    results = []
    psnr_values = []
    ssim_values = []
    
    # 逐个处理图片
    for filename in tqdm(common_files, desc="处理图片"):
        origin_path = os.path.join(origin_dir, filename)
        speed_up_path = os.path.join(speed_up_dir, filename)
        
        try:
            # 加载图片
            img_origin = load_image(origin_path)
            img_speed = load_image(speed_up_path)
            
            # 计算指标
            psnr = calculate_psnr(img_origin, img_speed)
            ssim = calculate_ssim(img_origin, img_speed)
            
            results.append({
                'filename': filename,
                'psnr': float(psnr),
                'ssim': float(ssim)
            })
            
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            
        except Exception as e:
            print(f"处理 {filename} 时出错: {str(e)}")
            continue
    
    # 计算统计信息
    if results:
        psnr_mean = np.mean(psnr_values)
        psnr_std = np.std(psnr_values)
        ssim_mean = np.mean(ssim_values)
        ssim_std = np.std(ssim_values)
        
        summary = {
            'total_images': len(results),
            'psnr_mean': float(psnr_mean),
            'psnr_std': float(psnr_std),
            'ssim_mean': float(ssim_mean),
            'ssim_std': float(ssim_std),
            'results': results
        }
        
        # 保存结果到 JSON 文件
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 打印统计信息
        print("\n" + "="*60)
        print("评估结果统计:")
        print("="*60)
        print(f"总图片数: {len(results)}")
        print(f"\nPSNR:")
        print(f"  平均值: {psnr_mean:.4f}")
        print(f"  标准差: {psnr_std:.4f}")
        print(f"  最小值: {np.min(psnr_values):.4f}")
        print(f"  最大值: {np.max(psnr_values):.4f}")
        print(f"\nSSIM:")
        print(f"  平均值: {ssim_mean:.4f}")
        print(f"  标准差: {ssim_std:.4f}")
        print(f"  最小值: {np.min(ssim_values):.4f}")
        print(f"  最大值: {np.max(ssim_values):.4f}")
        print("="*60)
        print(f"\n详细结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='计算两个文件夹中图片的 PSNR 和 SSIM')
    parser.add_argument('--origin', type=str, 
                       default='/home/zlq/diffusion/vbench_result/flux/origin',
                       help='原始图片文件夹路径')
    parser.add_argument('--speed_up', type=str,
                       default='/home/zlq/diffusion/vbench_result/flux/pab',
                       help='加速后图片文件夹路径')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='输出结果文件名')
    
    args = parser.parse_args()
    
    evaluate_images(args.origin, args.speed_up, args.output)


if __name__ == '__main__':
    main()

