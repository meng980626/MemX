# pruning.py
"""MemX 启发式剪枝（论文 §3.1.2, Heuristic pruning），对应 Algorithm 1 第 2 行。

三条规则按开销从小到大排列，搜索时依次应用：
  规则 1（纯结构判断，零开销）-> 规则 2（一次显存下界预测）-> 规则 3（解析下界）。
"""


def rule1_tp_dp_within_stage(tp, pp, dp, stage_devices):
    """规则 1：跨设备通信限制。

    TP 和 DP 涉及频繁的延迟敏感同步，其通信组必须约束在单个流水 stage 内部，
    即每个 stage 内的 TP 个 rank 必须是同一设备类型（不允许 TP/DP 组横跨
    不同厂商的 stage）。

    stage_devices: 长度为 pp 的 list，stage_devices[p] = 第 p 个 stage 的设备类型。
    由于 MemX 的候选生成本身按 stage 分配设备，该函数同时作为防御性校验。
    """
    if len(stage_devices) != pp:
        return False
    # 每个 stage 只有一种设备类型，天然保证 TP/DP 组不跨设备
    return all(isinstance(dv, str) for dv in stage_devices)


def rule2_memory_overflow(config, stage_devices, capacities, predict_stage_mem):
    """规则 2：显存溢出剪枝。

    用显存预测模型估计每个 stage 在 Full recompute 下的显存占用——Full 保留
    最少激活，是显存占用的理论下界；若下界仍超过物理容量，则该配置无论换
    什么重计算策略都不可行，直接剪枝，避免在其他重计算配置上浪费预测调用。

    predict_stage_mem(stage_idx, config, device, recompute) -> 预测显存(MB)
    返回 (feasible, per_stage_lower_bound)
    """
    lower_bounds = []
    for p, dev in enumerate(stage_devices):
        cap = capacities[dev]
        lb = predict_stage_mem(p, config, dev, 'full-uniform-1')
        lower_bounds.append(lb)
        if lb > cap:
            return False, lower_bounds
    return True, lower_bounds


def rule3_time_lower_bound(config, stage_devices, peak_flops,
                           pp_bandwidth, dp_bandwidth, best_time):
    """规则 3：迭代时间下界剪枝。

    由设备峰值算力和理论通信量计算该配置的最小可达迭代时间；
    若下界已超过当前最优解，则该分支不可能产生更优配置，直接剪枝。

    下界 = 计算下界 + PP 通信下界 + DP 通信下界（全部忽略气泡与 kernel 效率，
    因此是真正的下界）。
    返回 (prune, T_lb)。
    """
    # ---- 计算下界：6 * P * tokens（fwd 2P + bwd 4P FLOPs/token）----
    params = (2 * config['vocab_size'] * config['hidden_state']
              + config['num_layers'] * (4 * config['hidden_state'] ** 2
              + 2 * config['hidden_state'] * config['ffn_hidden_state']))
    tokens = config['global_batch_size'] * config['sequence_length']
    total_flops = 6.0 * params * tokens
    total_peak = sum(peak_flops[dev] * config['TP_size'] * config['DP_size']
                     for dev in stage_devices)
    t_compute_lb = total_flops / total_peak

    # ---- PP 通信下界：每个 stage 边界一前向一反向两次激活/梯度传递 ----
    msg_bytes = (config['micro_batch_size'] * config['sequence_length']
                 * config['hidden_state'] * config['dtype_bytes'])
    t_pp_lb = sum(2.0 * msg_bytes / max(bw, 1e-9) for bw in pp_bandwidth)

    # ---- DP 通信下界：all-reduce 2(DP-1)/DP * M ----
    t_dp_lb = 0.0
    if config['DP_size'] > 1:
        t_dp_lb = (2.0 * (config['DP_size'] - 1) * params * config['dtype_bytes']
                   / config['DP_size'] / max(dp_bandwidth, 1e-9))

    T_lb = t_compute_lb + t_pp_lb + t_dp_lb
    return T_lb > best_time, T_lb
