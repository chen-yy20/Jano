import time
import torch
from typing import Dict, Optional
from contextlib import ContextDecorator
import numpy as np

class _TimerCore:
    """Internal timer implementation"""
    def __init__(self, name: str):
        self.name = name
        self.reset()
        
    def reset(self):
        """Reset all recorded times"""
        self.times = []
        self._start_time = 0
        self._is_timing_valid = False  # 标记当前计时是否有效

    def start(self, global_enabled: bool):
        """Record start time"""
        if global_enabled:
            torch.cuda.synchronize()  # 确保之前的GPU操作完成
            self._start_time = time.perf_counter()
            self._is_timing_valid = True
        else:
            self._is_timing_valid = False

    def stop(self, global_enabled: bool):
        """Record end time and calculate duration"""
        # 只有在全局启用且当前计时有效时才记录
        if global_enabled and self._is_timing_valid:
            torch.cuda.synchronize()  # 确保GPU操作完成
            elapsed_time = (time.perf_counter() - self._start_time) * 1000  # 转换为毫秒
            self.times.append(elapsed_time)
        
        # 重置计时状态
        self._is_timing_valid = False

    @property
    def average(self) -> float:
        """Get average time in milliseconds"""
        return sum(self.times) / len(self.times) if self.times else 0.0

class TimerManager:
    """Global timer manager with control flags"""
    _instance = None
    _timers: Dict[str, _TimerCore] = {}
    _global_enabled = False

    def __new__(cls):
        """Singleton pattern implementation"""
        if cls._instance is None:
            cls._instance = super(TimerManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def init_timer(cls, enable: bool = True):
        """Initialize global timer settings
        
        Args:
            enable: Global enable/disable flag for all timers
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but required for timing")
        cls._global_enabled = enable
        # 重置所有现有的timer，清除无效的计时
        for timer in cls._timers.values():
            timer.reset()
        print(f"Timer initialized: enable={enable}")

    @classmethod
    def get_timer(cls, name: str) -> ContextDecorator:
        """Get or create a named timer
        
        Args:
            name: Unique identifier for the timer
        """
        if name not in cls._timers:
            cls._timers[name] = _TimerCore(name)
            
        class _TimerContext(ContextDecorator):
            """Context manager wrapper for timer"""
            def __enter__(self_ctx):
                cls._timers[name].start(cls._global_enabled)
                return self_ctx

            def __exit__(self_ctx, *args):
                cls._timers[name].stop(cls._global_enabled)

        return _TimerContext()

    @classmethod
    def print_time_statistics(cls, detect_outliers=False, threshold=1.5):
        """
        Print timing statistics (min/max/avg) for all timers
        Args:
            detect_outliers (bool): Whether to detect and report outliers
            threshold (float): IQR threshold for outlier detection
        """
        if not cls._global_enabled:
            print("Timing is disabled")
            return

        def detect_outliers_iqr(data, threshold):
            """使用IQR方法检测离群值"""
            if len(data) < 4:  # 数据太少不做离群值检测
                return [], data
            
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            normal = []
            outliers = []
            for x in data:
                if x < lower_bound or x > upper_bound:
                    outliers.append(x)
                else:
                    normal.append(x)
            return outliers, normal

        print("\n===== Timing Statistics =====")
        
        # 根据是否检测离群值选择不同的表头格式
        if detect_outliers:
            header_format = "{:<20} | {:>10} | {:>10} | {:>10} | {:>8} | {:>12} | {:>15}"
            print(header_format.format(
                "Timer Name", 
                "Min (ms)", 
                "Max (ms)", 
                "Avg (ms)", 
                "Samples",
                "Outliers #",
                "Outliers Avg"
            ))
            print("-" * 94)
            data_format = "{:<20} | {:>10.4f} | {:>10.4f} | {:>10.4f} | {:>8d} | {:>12d} | {:>15.4f}"
        else:
            header_format = "{:<20} | {:>10} | {:>10} | {:>10} | {:>8}"
            print(header_format.format(
                "Timer Name", 
                "Min (ms)", 
                "Max (ms)", 
                "Avg (ms)", 
                "Samples"
            ))
            print("-" * 66)
            data_format = "{:<20} | {:>10.4f} | {:>10.4f} | {:>10.4f} | {:>8d}"

        for name, timer in cls._timers.items():
            if timer.times:
                if detect_outliers:
                    outliers, normal_times = detect_outliers_iqr(timer.times, threshold)
                    
                    if normal_times:  # 有正常值
                        avg = sum(normal_times) / len(normal_times)
                        min_time = min(normal_times)
                        max_time = max(normal_times)
                    else:  # 全是离群值
                        avg = sum(timer.times) / len(timer.times)
                        min_time = min(timer.times)
                        max_time = max(timer.times)
                        
                    outliers_avg = sum(outliers) / len(outliers) if outliers else 0
                    
                    print(data_format.format(
                        name[:20], 
                        min_time,
                        max_time,
                        avg,
                        len(timer.times),
                        len(outliers),
                        outliers_avg
                    ))
                else:
                    # 不检测离群值的简单统计
                    avg = sum(timer.times) / len(timer.times)
                    min_time = min(timer.times)
                    max_time = max(timer.times)
                    
                    print(data_format.format(
                        name[:20], 
                        min_time,
                        max_time,
                        avg,
                        len(timer.times)
                    ))

        # 打印对应长度的分隔线
        print(("=" * 94 if detect_outliers else "=" * 66) + "\n")
        
    @classmethod
    def save_time_statistics_to_file(cls, filepath="timing_stats.txt"):
        """Save timing statistics (min/max/avg) to a text file and tracker"""
        if not cls._global_enabled:
            return

        # 获取当前时间戳
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 保存到文本文件
        with open(filepath, "a") as f:
            f.write("\n" + current_time + "\n")
            f.write("===== Timing Statistics =====\n")
            header_format = "{:<20} | {:>10} | {:>10} | {:>10} | {:>8}\n"
            f.write(header_format.format("Timer Name", "Min (ms)", "Max (ms)", "Avg (ms)", "Samples"))
            f.write("-" * 66 + "\n")
            
            data_format = "{:<20} | {:>10.4f} | {:>10.4f} | {:>10.4f} | {:>8d}\n"
            for name, timer in cls._timers.items():
                if timer.times:
                    avg = sum(timer.times) / len(timer.times)
                    min_time = min(timer.times)
                    max_time = max(timer.times)
                    f.write(data_format.format(
                        name[:20], 
                        min_time,
                        max_time,
                        avg,
                        len(timer.times)
                    ))
            f.write("=" * 66 + "\n")

        # 尝试记录到tracker
        # try:
        #     from yTrack.global_tracker import get_ytracker
        #     from ditango.core.config import get_config
        #     tracker = get_ytracker()
        #     config = get_config()
        #     # 记录每个计时器的统计信息
        #     for name, timer in cls._timers.items():
        #         if timer.times:
        #             avg = sum(timer.times) / len(timer.times)
        #             min_time = min(timer.times)
        #             max_time = max(timer.times)
                    
        #             tracker.log_inference_result(
        #                 num_gpus=config.world_size,  # 这里需要你提供实际的GPU数量
        #                 method=config.tag,  # 或者根据实际情况设置
        #                 latency=sum(timer.times),  # 使用总时间作为主要延迟指标
        #                 extra_info={
        #                     "timer_name": name,
        #                     "min_time": min_time,
        #                     "max_time": max_time,
        #                     "avg_time": avg,
        #                     "samples": len(timer.times),
        #                     "timestamp": current_time,
        #                     "statistics_type": "timing"
        #                 }
        #             )
        # except (ImportError, AttributeError, Exception) as e:
        #     # 如果tracker没有初始化或出现其他错误，静默处理
        #     print(f"yTracker error: {e}")
        
    @classmethod
    def enable_timing(cls):
        """Enable timing for all subsequent operations"""
        cls._global_enabled = True
        # 重置所有timer，清除在禁用期间的任何无效计时
        for timer in cls._timers.values():
            timer.reset()

    @classmethod
    def disable_timing(cls):
        """Disable timing for all subsequent operations"""
        cls._global_enabled = False
        # 重置所有timer的当前计时状态
        for timer in cls._timers.values():
            timer._is_timing_valid = False

    @classmethod
    def is_timing_enabled(cls) -> bool:
        """Check if timing is currently enabled"""
        return cls._global_enabled

# Public interface functions
def init_timer(enable: bool = True):
    TimerManager.init_timer(enable)

def get_timer(name: str) -> ContextDecorator:
    return TimerManager.get_timer(name)

def print_time_statistics():
    TimerManager.print_time_statistics()
    
def save_time_statistics_to_file(file_path="timing_stats.txt"):
    TimerManager.save_time_statistics_to_file(file_path)
    
def enable_timing():
    """Globally enable timing measurement"""
    TimerManager.enable_timing()

def disable_timing():
    """Globally disable timing measurement"""
    TimerManager.disable_timing()

def is_timing_enabled() -> bool:
    """Check global timing enable status"""
    return TimerManager.is_timing_enabled()