# search_configs.py
import argparse
import csv
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'modeling'))   # DualOutputPredictor 所在模块
from dual_predictor import DualOutputPredictor       # noqa: E402

import e2e_performance as e2e                         # noqa: E402
from objective import (compute_references, heterogeneity_index, phi,           # noqa: E402
                       select_schedule, select_weight_profile)
from pruning import (rule1_tp_dp_within_stage, rule2_memory_overflow,           # noqa: E402
                     rule3_time_lower_bound)
from recompute_policy import FULL, NONE, assign_recompute                       # noqa: E402

CLUSTER = {
    # 每种 PP 度数对应一个 stage->设备类型 的布局（规则 1：stage 内 TP/DP 组同构）
    'layouts': {
        3: ['v100', 'v100', 't4'],
        6: ['v100', 'v100', 'v100', 'v100', 't4', 't4'],
    },
    # 各设备类型的物理显存容量（MB）与峰值算力（FLOP/s）
    # TODO: vendor-b / vendor-c 的数值需按
    'capacity_MB': {'v100': 32768, 't4': 16384, 'a100': 81920,
                    'vendor-b': 32768, 'vendor-c': 32768},
    'normalized_compute': {'v100': 63, 't4': 28, 'a100': 128,
                   'vendor-b': 99, 'vendor-c': 46},
    # stage 间 PP 带宽（GB/s，用 tools/p2p_bw_test.py 实测）与 DP 带宽
    'pp_bandwidth_GBps': {3: [90, 10], 6: [90, 90, 90, 90, 10]},
    'dp_bandwidth_GBps': 70,
}

MODEL = {   # LLaMA2-7B
    'hidden_state': 4096, 'ffn_hidden_state': 11008, 'vocab_size': 32000,
    'num_layers': 32, 'dtype': 'fp16', 'dtype_bytes': 2,
}
GLOBAL_BATCH_SIZE = 32

# 候选空间（TP, PP, DP, MBS, 超参）
VALID_PARALLEL = [(2, 3, 1), (1, 6, 1)]
PARAM_GRID = {
    'SEQ_LENGTH': [1024, 2048, 4096],
    'MICRO_BATCH_SIZE': [2, 4, 8, 16],
    'MAX_POSITION_EMBEDDINGS': [4096],
    'NUM_ATTENTION_HEADS': [64],
    'NUM_QUERY_GROUPS': [4, 16],
}


def hash_signature(config, stage_devices):
    keys = ['TP_size', 'PP_size', 'DP_size', 'micro_batch_size',
            'sequence_length', 'num_layers', 'num_attention_heads',
            'num_query_groups', 'max_position_embedding', 'dtype']
    return (tuple(config[k] for k in keys), tuple(stage_devices))


def valid_config(tp, pp, dp, params):
    if tp * pp * dp > 8 * 8:            # 不超过集群规模
        return False
    if dp * params['MICRO_BATCH_SIZE'] > GLOBAL_BATCH_SIZE:
        return False
    if params['MAX_POSITION_EMBEDDINGS'] < params['SEQ_LENGTH']:
        return False
    if params['NUM_QUERY_GROUPS'] < tp:
        return False
    if params['NUM_ATTENTION_HEADS'] % tp != 0:
        return False
    if params['NUM_ATTENTION_HEADS'] % params['NUM_QUERY_GROUPS'] != 0:
        return False
    if MODEL['num_layers'] < pp:
        return False
    if pp not in CLUSTER['layouts']:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='../modeling/xgb_dual_aug.pkl',
                    help='双输出预测器 pkl 路径')
    ap.add_argument('--profile', default='auto',
                    choices=['auto', 'default', 'safety_first', 'throughput_first'],
                    help='权重 profile；auto = 按显存余量/异构指数自动选择')
    ap.add_argument('--out', default='memx_search_result.csv')
    ap.add_argument('--collapse-selective', action='store_true', default=True,
                    help='ρ 不连续时将重计算选择坍缩为 None/Full 二值（§3.1.1）')
    args = ap.parse_args()

    start_time = time.time()
    dual = DualOutputPredictor.load(args.model)

    cache = {}          # σ -> (T, M, V, strategies, schedule)
    candidates = []     # 可行候选的评估结果
    best_time = float('inf')
    n_all, n_pruned = 0, {'rule1': 0, 'rule2': 0, 'rule3': 0, 'cache': 0, 'invalid': 0}

    keys, values = zip(*PARAM_GRID.items())
    all_devs = [d for layout in CLUSTER['layouts'].values() for d in layout]
    het = heterogeneity_index([CLUSTER['peak_flops'][d] for d in all_devs])

    for tp, pp, dp in VALID_PARALLEL:
        stage_devices = CLUSTER['layouts'].get(pp)
        if stage_devices is None:
            continue
        for vals in itertools.product(*values):
            params = dict(zip(keys, vals))
            if not valid_config(tp, pp, dp, params):
                n_pruned['invalid'] += 1
                continue
            n_all += 1
            config = {
                'global_batch_size': GLOBAL_BATCH_SIZE,
                'micro_batch_size': params['MICRO_BATCH_SIZE'],
                'TP_size': tp, 'PP_size': pp, 'DP_size': dp,
                'sequence_length': params['SEQ_LENGTH'],
                'max_position_embedding': params['MAX_POSITION_EMBEDDINGS'],
                'num_attention_heads': params['NUM_ATTENTION_HEADS'],
                'num_query_groups': params['NUM_QUERY_GROUPS'],
                **MODEL,
            }

            if not rule1_tp_dp_within_stage(tp, pp, dp, stage_devices):
                n_pruned['rule1'] += 1
                continue

            sigma = hash_signature(config, stage_devices)
            if sigma in cache:
                n_pruned['cache'] += 1
                T, M, V, strategies, schedule, stage_mems = cache[sigma]
                candidates.append({'config': config, 'stage_devices': stage_devices,
                                   'T': T, 'M': M, 'V': V, 'stage_mems': stage_mems,
                                   'strategies': strategies, 'schedule': schedule})
                continue

            caps = CLUSTER['capacity_MB']

            feasible, _ = rule2_memory_overflow(
                config, stage_devices, caps,
                lambda p, c, dv, rg: e2e.stage_predict(p, c, dv, rg, dual)[2])
            if not feasible:
                n_pruned['rule2'] += 1
                continue

            prune3, _ = rule3_time_lower_bound(
                config, stage_devices, CLUSTER['peak_flops'],
                [b * 8e9 for b in CLUSTER['pp_bandwidth_GBps'][pp]],
                CLUSTER['dp_bandwidth_GBps'] * 8e9,
                best_time)
            if prune3:
                n_pruned['rule3'] += 1
                continue

            feasible, strategies = assign_recompute(
                config, stage_devices, caps,
                lambda p, c, dv, rg: (lambda r: (r[0], r[2]))(
                    e2e.stage_predict(p, c, dv, rg, dual)),
                collapse_selective=args.collapse_selective)
            if not feasible:
                n_pruned['rule2'] += 1
                continue

            mems = e2e.per_device_memory(config, stage_devices, strategies, dual)
            max_stage_mem = max(mems)
            min_avail = min(caps[d] for d in stage_devices) - max_stage_mem
            schedule = select_schedule(min_avail, max_stage_mem, het)

            vp_size = 1
            if schedule == 'interleaved':
                min_cap = min(caps[d] for d in stage_devices)
                vp_size = max(1, min(4, int(min_cap // max(max_stage_mem, 1))))
                if vp_size < 2:
                    schedule = '1F1B'   # 显存放不下更多虚拟 stage，回退 1F1B

            T = e2e.memx_e2e_time(config, stage_devices, strategies, schedule,
                                  CLUSTER['pp_bandwidth_GBps'][pp],
                                  CLUSTER['dp_bandwidth_GBps'], dual,
                                  vp_size=vp_size)
            M = max(mems)                          # 峰值显存 = 全集群最大
            V = float(np.var(mems))                # 显存方差 = 跨设备偏离均值的度量

            stage_mems = [mems[i * config['TP_size'] * config['DP_size']]
                          for i in range(config['PP_size'])]
            cache[sigma] = (T, M, V, strategies, schedule, stage_mems)
            best_time = min(best_time, T)
            candidates.append({'config': config, 'stage_devices': stage_devices,
                               'T': T, 'M': M, 'V': V, 'stage_mems': stage_mems,
                               'strategies': strategies, 'schedule': schedule})

    if not candidates:
        print('没有可行配置，请检查候选空间与集群描述。')
        return

    refs = compute_references(candidates)
    min_margin = min(
        (CLUSTER['capacity_MB'][dev] - sm) / CLUSTER['capacity_MB'][dev]
        for c in candidates
        for dev, sm in zip(c['stage_devices'], c['stage_mems']))
    weights = select_weight_profile(min_margin, het,
                                    override=None if args.profile == 'auto' else args.profile)
    print(f'weight profile: {weights} (min_mem_margin={min_margin:.2%}, het={het:.3f})')

    for c in candidates:
        c['Phi'] = phi(c['T'], c['M'], c['V'], refs, weights)
    candidates.sort(key=lambda c: c['Phi'])

    best = candidates[0]
    print(f"searched {n_all} configs "
          f"(pruned: {n_pruned}), cache_size={len(e2e.CACHE_DUAL)}, "
          f"search time: {time.time() - start_time:.2f}s")
    print(f"best: TP{best['config']['TP_size']} PP{best['config']['PP_size']} "
          f"DP{best['config']['DP_size']} MBS{best['config']['micro_batch_size']} "
          f"seq{best['config']['sequence_length']} strategies={best['strategies']} "
          f"schedule={best['schedule']} T={best['T']:.1f}ms M={best['M']:.0f}MB "
          f"V={best['V']:.3g} Phi={best['Phi']:.4f}")

    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['Phi', 'elapsed time per iteration', 'peak memory', 'memory variance',
                      'schedule', 'recompute strategies'] + list(best['config'].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in candidates:
            row = {'Phi': c['Phi'], 'elapsed time per iteration': c['T'],
                   'peak memory': c['M'], 'memory variance': c['V'],
                   'schedule': c['schedule'], 'recompute strategies': str(c['strategies'])}
            row.update(c['config'])
            writer.writerow(row)
    print(f'saved {len(candidates)} ranked candidates to {args.out}')


if __name__ == '__main__':
    main()
