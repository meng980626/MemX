# objective.py

# ---- 三个固定权重 profile（§3.1.2，不在测试集上调参）----
WEIGHT_PROFILES = {
    'default':          (0.4, 0.2, 0.4),  # 设备差异大的异构集群：均衡与时间同等重要
    'safety_first':     (0.3, 0.5, 0.2),  # 最小显存余量 <10% 时：优先可行性与防 OOM
    'throughput_first': (0.7, 0.2, 0.1),  # 同构或弱异构集群：显存压力小，主打吞吐
}

EPSILON_RATIO = 0.01   # εV = EPSILON_RATIO * V_ref
SAFETY_MARGIN = 0.10   # 最小显存余量阈值，低于则切换 safety_first
HETEROGENEITY_THRESHOLD = 0.10  # 算力变异系数低于该值视为(弱)同构


def memory_pressure_index(min_available_mem, max_stage_mem_demand):
    """显存压力指数 = 最小可用显存 / 最大 stage 显存需求（§3.1.2）。"""
    return min_available_mem / max(max_stage_mem_demand, 1e-9)


def heterogeneity_index(compute_powers):
    """GPU 异构指数 = 设备算力的变异系数（标准差/均值，§3.1.2）。"""
    import numpy as np
    c = np.asarray(list(compute_powers), dtype=float)
    return float(c.std() / max(c.mean(), 1e-9))


def select_weight_profile(min_mem_margin, het_index, override=None):
    """按部署条件在三个固定 profile 中选择（§3.1.2）：
    - min_mem_margin < 10%       -> safety_first
    - 集群（弱）同构             -> throughput_first
    - 其余                       -> default
    """
    if override is not None:
        return WEIGHT_PROFILES[override]
    if min_mem_margin < SAFETY_MARGIN:
        return WEIGHT_PROFILES['safety_first']
    if het_index < HETEROGENEITY_THRESHOLD:
        return WEIGHT_PROFILES['throughput_first']
    return WEIGHT_PROFILES['default']


def compute_references(candidates):
    import numpy as np
    return {
        'T_ref': float(np.median([c['T'] for c in candidates])),
        'M_ref': float(np.median([c['M'] for c in candidates])),
        'V_ref': float(np.median([c['V'] for c in candidates])),
    }


def phi(T, M, V, refs, weights):
    wT, wM, wV = weights
    eps = EPSILON_RATIO * refs['V_ref']
    T_hat = T / refs['T_ref']
    M_hat = M / refs['M_ref']
    V_hat = (V + eps) / (refs['V_ref'] + eps)
    return (T_hat ** wT) * (M_hat ** wM) * (V_hat ** wV)


def select_schedule(min_available_mem, max_stage_mem_demand, het_index,
                    pressure_threshold=1.2):

    pressure = memory_pressure_index(min_available_mem, max_stage_mem_demand)
    if pressure < pressure_threshold or het_index > HETEROGENEITY_THRESHOLD:
        return '1F1B'
    return 'interleaved'
