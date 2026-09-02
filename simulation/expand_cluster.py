import pandas as pd
import numpy as np

# ==================== 1. 读取数据 ====================
df = pd.read_csv('result_recompute_3node_new.csv')

# ==================== 2. 核心配置 ====================
COMPUTE_RATIO = {'a100': 128, 'vendor-b': 99, 'vendor-c': 46}

EXPANSION_PLAN = {
    (4, 5, 1): [8, 10, 13, 15],
    (2, 10, 1): [15, 20, 25, 30],
    (2, 5, 2): [8, 10, 13, 15],
}

EXCLUDE_COLS = {
    'EXP ID', 'Rank', 'device type',
    'Peak GPU memory', 'TFLOP/s/GPU', 'elapsed time per iteration'
}
PARAM_COLS = [c for c in df.columns if c not in EXCLUDE_COLS]
MAX_CONFIGS_PER_BASE = 30

# ==================== 3. 辅助函数 ====================

def get_device_memory_stats(config_df):
    stats = config_df.groupby('device type')['Peak GPU memory'].agg(['mean', 'std'])
    stats['std'] = stats['std'].fillna(stats['mean'] * 0.05)
    zero_mask = stats['std'] == 0
    stats.loc[zero_mask, 'std'] = stats.loc[zero_mask, 'mean'] * 0.05
    return stats


def sample_memory(device, matched_df, mem_stats):
    device_mems = matched_df[matched_df['device type'] == device]['Peak GPU memory']
    if len(device_mems) >= 3:
        mean = device_mems.mean()
        std = device_mems.std()
        if pd.isna(std) or std == 0:
            std = mean * 0.05
        noise = np.clip(np.random.normal(0, std * 0.5), -mean * 0.15, mean * 0.15)
        return mean + noise
    else:
        mean = mem_stats.loc[device, 'mean']
        std = mem_stats.loc[device, 'std']
        noise = np.clip(np.random.normal(0, std * 0.5), -mean * 0.15, mean * 0.15)
        return mean + noise


def calculate_weighted_ratio(device_list, compute_ratio):
    series = pd.Series(device_list)
    counts = series.value_counts()
    total = len(series)
    return sum(counts.get(d, 0) * compute_ratio[d] for d in compute_ratio) / total


def build_fixed_stage_template(base_df, tp, base_pp, dp, target_pp):
    """为指定(target_tp, target_pp, target_dp)构建固定的新增stage设备模板。"""
    new_stages = (target_pp - base_pp) * dp
    if new_stages <= 0:
        return []
    
    device_counts = base_df['device type'].value_counts()
    total = device_counts.sum()
    
    alloc = {}
    remaining = new_stages
    for dev in device_counts.index:
        alloc[dev] = round(device_counts[dev] / total * new_stages)
    
    diff = new_stages - sum(alloc.values())
    if diff != 0:
        residuals = {dev: (device_counts[dev]/total*new_stages) - alloc[dev] 
                     for dev in device_counts.index}
        sorted_devs = sorted(residuals, key=residuals.get, reverse=(diff > 0))
        for i in range(abs(diff)):
            alloc[sorted_devs[i % len(sorted_devs)]] += (1 if diff > 0 else -1)
    
    template = []
    for dev, count in alloc.items():
        template.extend([dev] * count)
    
    rng = np.random.RandomState(hash(f"{tp}_{base_pp}_{dp}_{target_pp}") % 2**31)
    rng.shuffle(template)
    
    return template


# ==================== 4. 主扩展函数 ====================

def expand_configuration_v2(base_df, tp, base_pp, dp, target_pp, max_configs=30):
    base_total_ranks = tp * base_pp * dp
    new_total_ranks  = tp * target_pp * dp
    mem_stats = get_device_memory_stats(base_df)
    
    rank_device_template = {}
    for rank in range(base_total_ranks):
        rank_devs = base_df[base_df['Rank'] == rank]['device type']
        if len(rank_devs) > 0:
            rank_device_template[rank] = rank_devs.mode()[0]
        else:
            candidates = list(COMPUTE_RATIO.keys())
            rank_device_template[rank] = candidates[rank % len(candidates)]
    
    # 新增stage的固定设备模板（所有实验共用）
    new_stage_template = build_fixed_stage_template(base_df, tp, base_pp, dp, target_pp)
    print(f"    新增stage模板 ({len(new_stage_template)} stages, 每stage {tp} 同构rank): {new_stage_template}")
    
    unique_configs = base_df[PARAM_COLS].drop_duplicates()
    if len(unique_configs) > max_configs:
        unique_configs = unique_configs.sample(n=max_configs, random_state=42)
    
    all_new_rows = []
    
    for idx, cfg in unique_configs.iterrows():
        mask = (base_df['num layers'] == cfg['num layers']) & \
               (base_df['num attention heads'] == cfg['num attention heads']) & \
               (base_df['num query groups'] == cfg['num query groups']) & \
               (base_df['recompute granularity'] == cfg['recompute granularity']) & \
               (base_df['sequence length'] == cfg['sequence length']) & \
               (base_df['micro batch size'] == cfg['micro batch size']) & \
               (base_df['max position embedding'] == cfg['max position embedding'])
        matched = base_df[mask]
        
        if len(matched) == 0:
            continue
            
        base_tflops   = matched['TFLOP/s/GPU'].mean()
        base_elapsed  = matched['elapsed time per iteration'].mean()
        base_layers   = cfg['num layers']
        new_layers = int(round(base_layers * target_pp / base_pp))
        
        base_exp_sample = matched['EXP ID'].iloc[0]
        new_exp_id = f"{base_exp_sample}_extPP{target_pp}"
        
        exp_device_list = []
        
        # 原有ranks
        for rank in range(base_total_ranks):
            device = rank_device_template[rank]
            exp_device_list.append(device)
            mem = sample_memory(device, matched, mem_stats)
            row = cfg.to_dict()
            row.update({
                'TP size': tp, 'PP size': target_pp, 'DP size': dp,
                'EXP ID': new_exp_id, 'Rank': rank,
                'device type': device, 'num layers': new_layers,
                'Peak GPU memory': int(round(mem)),
                'TFLOP/s/GPU': None, 'elapsed time per iteration': None,
            })
            all_new_rows.append(row)
        
        # 新增ranks——使用固定模板，每个stage内TP个rank同构
        for device in new_stage_template:
            for _ in range(tp):
                exp_device_list.append(device)
                mem = sample_memory(device, matched, mem_stats)
                row = cfg.to_dict()
                row.update({
                    'TP size': tp, 'PP size': target_pp, 'DP size': dp,
                    'EXP ID': new_exp_id, 'Rank': len(exp_device_list) - 1,
                    'device type': device, 'num layers': new_layers,
                    'Peak GPU memory': int(round(mem)),
                    'TFLOP/s/GPU': None, 'elapsed time per iteration': None,
                })
                all_new_rows.append(row)
        
        # 性能推断
        new_weighted_ratio = calculate_weighted_ratio(exp_device_list, COMPUTE_RATIO)
        base_devices = [rank_device_template[r] for r in range(base_total_ranks)]
        base_weighted_ratio = calculate_weighted_ratio(base_devices, COMPUTE_RATIO)
        
        pp_factor = 1 + 0.03 * np.log(target_pp / base_pp)
        device_penalty = base_weighted_ratio / new_weighted_ratio
        device_penalty = max(1.0, device_penalty)
        new_elapsed = base_elapsed * pp_factor * (device_penalty ** 0.2)
        
        for r in all_new_rows:
            if r['EXP ID'] != new_exp_id:
                continue
            dev = r['device type']
            ratio = COMPUTE_RATIO[dev]
            tflops = base_tflops * (ratio / new_weighted_ratio) * (1 + np.random.normal(0, 0.005))
            r['TFLOP/s/GPU'] = round(tflops, 4)
            r['elapsed time per iteration'] = round(new_elapsed * (1 + np.random.normal(0, 0.001)), 2)
    
    return pd.DataFrame(all_new_rows)


# ==================== 5. 执行扩展 ====================
np.random.seed(42)
expanded_dfs = []

for (tp, base_pp, dp), target_pps in EXPANSION_PLAN.items():
    base_df = df[(df['TP size'] == tp) & (df['PP size'] == base_pp) & (df['DP size'] == dp)]
    print(f"\nProcessing TP={tp}, PP={base_pp}, DP={dp} ({base_df['EXP ID'].nunique()} exps)...")
    
    for target_pp in target_pps:
        new_df = expand_configuration_v2(base_df, tp, base_pp, dp, target_pp, MAX_CONFIGS_PER_BASE)
        if len(new_df) > 0:
            expanded_dfs.append(new_df)
            total_ranks = tp * target_pp * dp
            print(f"  → PP={target_pp} ({total_ranks} ranks): {len(new_df)} records")

result_df = pd.concat(expanded_dfs, ignore_index=True)
result_df = result_df[df.columns]

output_path = 'expanded_cluster_simulation_v2.csv'
result_df.to_csv(output_path, index=False)
print(f"\n✅ 完成！共生成 {len(result_df)} 条记录")
print(f"保存至: {output_path}")
