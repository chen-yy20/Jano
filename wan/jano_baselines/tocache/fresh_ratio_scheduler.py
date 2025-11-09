# tocache/fresh_ratio_scheduler.py
def fresh_ratio_scheduler(cache_dic, current):
    fr = float(cache_dic['fresh_ratio'])
    mode = cache_dic['fresh_ratio_schedule']
    step, num_steps = int(current['step']), max(1, int(current['num_steps']))
    weight = 0.9

    if mode == 'constant':
        r = fr
    elif mode == 'linear':
        r = fr * (1 + weight - 2 * weight * step / num_steps)
    elif mode == 'exp':
        r = fr * (weight ** (step / num_steps))
    elif mode == 'layerwise':
        r = fr * (1 + weight - 2 * weight * current['layer'] / 39)
    elif mode == 'ToCa':
        step_weight  = 0.0
        layer_weight = 0.0
        module_weight = 1.5
        module_time_weight = 0.33
        step_factor  = 1 + step_weight  - 2 * step_weight  * step / num_steps
        layer_factor = 1 + layer_weight - 2 * layer_weight * current['layer'] / 39
        module_factor = (1 - (1 - module_time_weight) * module_weight) if current['module'] == 'cross-attn' \
                        else (1 + module_time_weight * module_weight)
        r = fr * step_factor * layer_factor * module_factor
    else:
        raise ValueError(f"Unknown ratio scheduler: {mode}")

    return float(max(0.0, min(1.0, r)))
