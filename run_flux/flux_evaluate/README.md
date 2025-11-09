# Flux 图片质量评估工具

这个工具用于计算原始图片和加速后图片之间的 PSNR (Peak Signal-to-Noise Ratio) 和 SSIM (Structural Similarity Index) 指标。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
python evaluate.py
```

默认情况下，脚本会使用以下路径：
- 原始图片: `/home/zlq/diffusion/vbench_result/flux/origin`
- 加速后图片: `/home/zlq/diffusion/vbench_result/flux/pab`

### 自定义路径

```bash
python evaluate.py --origin /path/to/origin --pab /path/to/pab --output results.json
```

### 参数说明

- `--origin`: 原始图片文件夹路径
- `--pab`: 加速后图片文件夹路径
- `--output`: 输出结果文件名（默认为 `evaluation_results.json`）

## 输出结果

脚本会生成一个 JSON 文件，包含：
- 每张图片的 PSNR 和 SSIM 值
- 所有图片的平均值、标准差、最小值、最大值

同时在终端输出统计信息。

## 指标说明

- **PSNR (Peak Signal-to-Noise Ratio)**: 峰值信噪比，值越高表示图片质量越好
- **SSIM (Structural Similarity Index)**: 结构相似性指数，范围在 0-1 之间，值越接近 1 表示图片越相似

