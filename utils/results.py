"""
统一结果保存工具
目录结构: ./results/{model}/{method}/{prompt_id}/
每个目录下只存两个文件:
  1. 生成结果 (图片/视频)
  2. {TAG}_params_metrics.json  (参数 + 计时统计 + 质量指标)
"""
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any


def save_params_and_metrics(
    output_dir: str,
    tag: str,
    params: Dict[str, Any],
    timing_stats: Optional[Dict] = None,
    quality_metrics: Optional[Dict] = None,
) -> str:
    """将运行参数、计时统计和质量指标统一写入一个 JSON 文件。

    Args:
        output_dir:      结果目录（已创建或将自动创建）
        tag:             本次运行的标签，用于文件命名
        params:          运行参数字典（模型、方法相关配置）
        timing_stats:    来自 get_time_statistics_dict() 的计时结果
        quality_metrics: 来自 evaluate_quality_with_origin() 的 metrics 字段

    Returns:
        保存的 JSON 文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    summary = {
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tag": tag,
        "parameters": params,
        "timing_stats": timing_stats or {},
        "quality_metrics": quality_metrics,
    }
    path = os.path.join(output_dir, f"{tag}_params_metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return path
