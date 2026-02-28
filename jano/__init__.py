import os
from utils.envs import GlobalEnv
from utils.timer import init_timer

def init_jano(
        enable,
        model,
        analyze_block_size,
        tag='no_tag',
        save_dir='./result',
        num_inference_steps=50,
        warmup_steps=6,
        cooldown_steps=4,
        # complexity
        t_weight=0.7,
        d_strength = 0.8,
        d_distance = 2,
        # convergence
        static_thresh = 0.2,
        static_interval = 10,
        medium_thresh = 0.3,
        medium_interval = 5,
    ):
    """
    初始化STDIT参数和全局环境
    Args:
        enable: bool, 是否启用STDIT
        model: torch.nn.Module, 待分析的模型
        analyze_block_size: int, 分析attention pattern的block大小
        tag: str, 实验标识
        save_dir: str, 结果保存路径
        num_inference_steps: int, 总推理步数
        warmup_steps: int, warmup stage采集信息的步数
        cooldown_steps: int, cooldown stage完整计算的步数
        update_interval: int, 更新弱token的间隔步数
        use_offload: bool, 是否将frozen states迁移到CPU
    """
    # 设置全局环境变量
    params = {
        'enable_stdit': enable,
        'model': model,
        'analyze_block_size': analyze_block_size,
        'tag': tag,
        'save_dir': save_dir,
        'num_inference_steps': num_inference_steps,
        'warmup_steps': warmup_steps,
        'cooldown_steps': cooldown_steps,
        't_weight': t_weight,
        'd_strength': d_strength,
        'd_distance': d_distance,
        'static_thresh' : static_thresh,
        'static_interval' : static_interval,
        'medium_thresh' : medium_thresh,
        'medium_interval' : medium_interval,
    }
    
    for key, value in params.items():
        GlobalEnv.set_envs(key, value)
        
    # 用于convergence analysis实验，会关闭优化，并保存一些实验需要的张量。
    do_cc_exp = False
    GlobalEnv.set_envs("cc_exp", do_cc_exp)
    
    if do_cc_exp:
        GlobalEnv.set_envs("static_thresh", 10) # 全是static
        GlobalEnv.set_envs("medium_thresh", 10)
        GlobalEnv.set_envs("medium_interval", 100)
        save_dir = os.path.join(f"./exp3_conv_comp/wan-1.3B/{tag}")
        GlobalEnv.set_envs("save_dir", save_dir)
    
    # 保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化计时器
    init_timer()
    
    # 打印所有参数
    print("\n" + "="*50)
    print("STDIT Initialization Parameters:")
    print("="*50)
    print(f"{'Parameter':<25} Value")
    print("-"*50)
    for key, value in params.items():
        print(f"{key:<25} {value}")
    print("="*50 + "\n")
    
    return params