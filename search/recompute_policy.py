# recompute_policy.py

LEVELS = ['null', 'selective', 'full-uniform-1']   # None / Selective / Full
NONE, SELECTIVE, FULL = LEVELS


def marginal_exchange_rate(t_from, t_to, m_from, m_to):
    """ρ_k(s1→s2) = ΔT/ΔM（Eq. (1)）：单位显存节省所付出的时间代价。"""
    dM = m_from - m_to
    if dM <= 0:          # 没有显存收益，交换率无意义
        return float('inf')
    return (t_to - t_from) / dM


def assign_recompute(config, stage_devices, capacities, predict_stage,
                     collapse_selective=True, safety_margin=0.0):
    levels = [NONE, FULL] if collapse_selective else LEVELS
    pp = config['PP_size']

    # ---- 第一级：硬可行性 ----
    mandatory = []     # 每个 stage 的可行策略集合
    for p, dev in enumerate(stage_devices):
        cap = capacities[dev] * (1.0 - safety_margin)
        feasible_lv = []
        for lv in levels:
            _, mem = predict_stage(p, config, dev, lv)
            if mem <= cap:
                feasible_lv.append(lv)
        if not feasible_lv:
            return False, None          # Full 也放不下 -> 整配置不可行
        mandatory.append(feasible_lv)

    # ---- 第二级：联合优化选择 ----
    # 基线：能 None 就 None（时间最优），否则 Full
    strategy = [NONE if NONE in lv else FULL for lv in mandatory]

    # 对被迫 Full 的 stage，检查是否存在占用介于两者之间的 Selective 可用
    # （仅当未坍缩中间层时；ρ 不连续时按论文保持二值选择）
    if not collapse_selective:
        for p, dev in enumerate(stage_devices):
            if strategy[p] != FULL or SELECTIVE not in mandatory[p]:
                continue
            t_full, m_full = predict_stage(p, config, dev, FULL)
            t_sel, m_sel = predict_stage(p, config, dev, SELECTIVE)
            rho = marginal_exchange_rate(t_sel, t_full, m_sel, m_full)
            # ρ 有限（确实有显存收益）且时间代价小于全量重计算的一半 -> 用 Selective
            if rho != float('inf') and t_sel < t_full * 1.5:
                strategy[p] = SELECTIVE

    return True, strategy
