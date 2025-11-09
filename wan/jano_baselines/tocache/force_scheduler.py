# tocache/force_scheduler.py
def force_scheduler(cache_dic, current):
    """
    强制全量刷新周期（步数）。Wan2.1 不区分时/空分支。
    """
    cache_dic['cal_threshold'] = {
        'attn'      : 3,
        'cross-attn': 6,
        'ffn'       : 3,
    }
