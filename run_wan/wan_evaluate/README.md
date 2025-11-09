# Wan 视频质量评估工具

这个工具用于计算原始视频和加速后视频之间的 PSNR (Peak Signal-to-Noise Ratio)、SSIM (Structural Similarity Index) 和 LPIPS (Learned Perceptual Image Patch Similarity) 指标。

## 安装依赖

```bash
pip install -r requirements.txt
```

**注意**: LPIPS 库需要额外的依赖，如果安装失败，可以跳过 LPIPS 计算。

## 使用方法

### 基本用法

```bash
python evaluate_videos.py
```

默认情况下，脚本会使用以下路径：
- 原始视频: `/home/zlq/diffusion/vbench_result/wan2.1/origin`
- 加速后视频: `/home/zlq/diffusion/vbench_result/wan2.1/STDit_new`

### 自定义路径

```bash
python evaluate_videos.py --origin /path/to/origin --speed_up /path/to/speed_up --output results.json
```

### 跳过 LPIPS 计算（加快速度）

```bash
python evaluate_videos.py --no_lpips
```

### 参数说明

- `--origin`: 原始视频文件夹路径
- `--speed_up`: 加速后视频文件夹路径
- `--output`: 输出结果文件名（默认为 `evaluation_results.json`）
- `--no_lpips`: 跳过 LPIPS 计算以加快评估速度

## 输出结果

脚本会生成一个 JSON 文件，包含：
- 每个视频的 PSNR、SSIM、LPIPS 平均值和标准差
- 所有视频和帧的总体统计信息（平均值、标准差、最小值、最大值）

同时在终端输出统计信息。

## 指标说明

- **PSNR (Peak Signal-to-Noise Ratio)**: 峰值信噪比，值越高表示视频质量越好（单位：dB）
- **SSIM (Structural Similarity Index)**: 结构相似性指数，范围在 0-1 之间，值越接近 1 表示视频越相似
- **LPIPS (Learned Perceptual Image Patch Similarity)**: 学习的感知图像块相似度，值越低表示感知质量越好（范围通常在 0-1 之间）

## 注意事项

1. 视频文件需要是常见的格式（mp4, avi, mov, mkv）
2. 如果两个视频的帧数不同，会使用较少的帧数进行计算
3. 如果两个视频的尺寸不同，会自动调整尺寸以匹配
4. LPIPS 计算需要 GPU 支持才能获得较好的性能，如果没有 GPU，计算会较慢

## 性能优化

- 如果只需要 PSNR 和 SSIM，可以使用 `--no_lpips` 参数跳过 LPIPS 计算
- LPIPS 计算较为耗时，建议在 GPU 环境下运行

