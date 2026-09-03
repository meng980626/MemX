# pruning.py
def rule1_tp_dp_within_stage(tp, pp, dp, stage_devices):
    if len(stage_devices) != pp:
        return False
    # 每个 stage 只有一种设备类型，天然保证 TP/DP 组不跨设备
    return all(isinstance(dv, str) for dv in stage_devices)


def rule2_memory_overflow(config, stage_devices, capacities, predict_stage_mem):
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
