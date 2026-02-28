#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os


def load_image(image_path):
    from PIL import Image
    import numpy as np

    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def _resize_if_needed(img1, img2):
    if img1.shape != img2.shape:
        from PIL import Image
        import numpy as np

        img2 = np.array(
            Image.fromarray(img2).resize(
                (img1.shape[1], img1.shape[0]), Image.Resampling.LANCZOS
            )
        )
    return img1, img2


def calculate_psnr(img1, img2):
    from skimage.metrics import peak_signal_noise_ratio
    import numpy as np

    img1, img2 = _resize_if_needed(img1, img2)
    return peak_signal_noise_ratio(
        img1.astype(np.float64), img2.astype(np.float64), data_range=255
    )


def calculate_ssim(img1, img2):
    from skimage.metrics import structural_similarity

    img1, img2 = _resize_if_needed(img1, img2)
    if len(img1.shape) == 3:
        try:
            return structural_similarity(img1, img2, channel_axis=2, data_range=255)
        except TypeError:
            return structural_similarity(img1, img2, multichannel=True, data_range=255)
    return structural_similarity(img1, img2, data_range=255)


def evaluate_images(origin_dir, speed_up_dir, output_file="evaluation_results.json"):
    import numpy as np
    from tqdm import tqdm

    origin_dir = os.path.abspath(origin_dir)
    speed_up_dir = os.path.abspath(speed_up_dir)

    origin_files = {f for f in os.listdir(origin_dir) if f.lower().endswith(".png")}
    speed_up_files = {f for f in os.listdir(speed_up_dir) if f.lower().endswith(".png")}
    common_files = sorted(origin_files & speed_up_files)

    if not common_files:
        print(f"警告: 在 {origin_dir} 和 {speed_up_dir} 中没有找到相同的图片文件")
        return

    print(f"找到 {len(common_files)} 对匹配的图片")

    results, psnr_values, ssim_values = [], [], []
    for filename in tqdm(common_files, desc="处理图片"):
        origin_path = os.path.join(origin_dir, filename)
        speed_up_path = os.path.join(speed_up_dir, filename)
        try:
            img_origin = load_image(origin_path)
            img_speed = load_image(speed_up_path)
            psnr = calculate_psnr(img_origin, img_speed)
            ssim = calculate_ssim(img_origin, img_speed)
            results.append({"filename": filename, "psnr": float(psnr), "ssim": float(ssim)})
            psnr_values.append(psnr)
            ssim_values.append(ssim)
        except Exception as e:
            print(f"处理 {filename} 时出错: {str(e)}")

    if not results:
        return

    summary = {
        "total_images": len(results),
        "psnr_mean": float(np.mean(psnr_values)),
        "psnr_std": float(np.std(psnr_values)),
        "ssim_mean": float(np.mean(ssim_values)),
        "ssim_std": float(np.std(ssim_values)),
        "results": results,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("评估结果统计:")
    print("=" * 60)
    print(f"总图片数: {len(results)}")
    print(f"\nPSNR: 平均值={summary['psnr_mean']:.4f}, 标准差={summary['psnr_std']:.4f}")
    print(f"SSIM: 平均值={summary['ssim_mean']:.4f}, 标准差={summary['ssim_std']:.4f}")
    print("=" * 60)
    print(f"\n详细结果已保存到: {os.path.abspath(output_file)}")


def main():
    parser = argparse.ArgumentParser(description="计算两个文件夹中图片的 PSNR 和 SSIM")
    parser.add_argument("--origin", type=str, required=True, help="原始图片文件夹路径")
    parser.add_argument("--speed_up", type=str, required=True, help="加速后图片文件夹路径")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="输出结果文件名")
    args = parser.parse_args()
    evaluate_images(args.origin, args.speed_up, args.output)


if __name__ == "__main__":
    main()
