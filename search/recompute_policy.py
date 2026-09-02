# recompute_policy.py
"""MemX 异构重计算策略选择（论文 §3.1.1），对应 Algorithm 1 第 15 行。

策略分两级：
  第一级（硬约束）：对每个 stage 比较 None 策略的显存占用 D̄^None 与设备容量
      M_k。若 Full 的占用都超过 M_k，配置不可行（剪枝）；若 None 就放得下，
      保留 None；否则强制 Full。
  第二级（软选择）：在可行域内按边际时间-显存交换率
      ρ_k(s1→s2) = ΔT_k(s1→s2) / ΔM_k(s1→s2)   (Eq. (1))
      做贪心精化——必须开 Full 的 stage 之外，对"放不下 None 但还有更小压力
      策略可选"的边界情形，优先把时间代价最小（ρ 最小）的 stage 放松回 None。

论文指出 Selective 在部分设备上收益不连续（接近零收益），此时把中间层
坍缩掉，有效选择在 None 与 Full 之间二值化（collapse_selective=True，
与 §3.1.1 末段一致）。
"""

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
    """两级重计算策略分配。

    predict_stage(stage_idx, config, device, recompute) -> (time, mem)
    capacities: dict, 设备类型 -> 物理显存容量（MB）
    safety_margin: 显存安全余量比例（如 0.05 表示容量打 95 折）

    返回 (feasible, per_stage_strategy list)。不可行时返回 (False, None)。
    """
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
