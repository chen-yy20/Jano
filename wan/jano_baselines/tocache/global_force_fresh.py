# tocache/global_force_fresh.py
def global_force_fresh(cache_dic, current):
    step = int(current['step'])
    first_step = (step == 0)
    first_3    = (step <= 2)
    force_fresh = cache_dic['force_fresh']
    th = cache_dic.get('cal_threshold', {'attn':3,'cross-attn':6,'ffn':3})

    if force_fresh in ('local', 'none'):
        return {'attn': first_step, 'cross-attn': first_step, 'ffn': first_step}

    if force_fresh == 'global':
        return {
            'attn'      : first_3 or (step % th['attn']       == 0),
            'cross-attn': first_3 or (step % th['cross-attn'] == 0),
            'ffn'       : first_3 or (step % th['ffn']        == 0),
        }

    return {'attn': first_step, 'cross-attn': first_step, 'ffn': first_step}
