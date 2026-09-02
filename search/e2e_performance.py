import pandas as pd
import joblib

CACHE_block = {}
CACHE_other = {}

def extend_cols(df):
    df['dtype_hardware_gen'] = df['dtype'].map({
        'fp16': 1, 
        'bf16': 2 
    })

    df['dtype_needs_loss_scaling'] = df['dtype'].map({
        'fp16': 1,   
        'bf16': 0    
    })

    # 在X中新增以下特征列
    df['param_memory'] = (
        2 * df['vocab size'] * df['hidden size'] +  # 嵌入层
        df['num layers'] * (
            4 * df['hidden size'] * df['hidden size'] +  # QKV+O投影
            2 * df['hidden size'] * df['ffn hidden size']  # FFN两层
        ) * 2  # fp16/bf16=2字节
    )

    df['activation_per_layer'] = (
        df['micro batch size'] * df['sequence length'] * 
        df['hidden size'] * 34  # 典型Transformer每层激活系数
    )

    df['tp_memory_reduction'] = 1 / df['TP size']  # TP切分降低单卡显存  



def mb_time(stage_index, config, device, xgb_model, label_model):
    stage_layer_number = config['num_layers'] // config['PP_size']
    # last stage
    if stage_index == config['PP_size'] - 1 and config['num_layers'] % config['PP_size'] != 0:
        stage_layer_number = config['num_layers'] % config['PP_size']
    cat_cols = ['Rank','device type','recompute granularity']

    single = {
        'TP size': config['TP_size'],
        'Rank': 0,
        'device type': device,
        'dtype': config['dtype'],
        'max position embedding': config['max_position_embedding'],
        'micro batch size': config['micro_batch_size'],
        'num attention heads': config['num_attention_heads'],
        'num layers': 1,         
        'num query groups': config['num_query_groups'], 
        'recompute granularity': config['recompute_granularity'],
        'sequence length': config['sequence_length'],
        'hidden size':config['hidden_state'],
        'vocab size':config['vocab_size'],
        'ffn hidden size':config['ffn_hidden_state'],
    }

    single_key = tuple(sorted(single.items()))
    
    if single_key in CACHE_block:
        return CACHE_block[single_key]*stage_layer_number, CACHE_other[single_key]

    double = single.copy()
    double['num layers'] = 2
    single_df = pd.DataFrame([single])  
    double_df = pd.DataFrame([double])    

    extend_cols(single_df)
    extend_cols(double_df)

    single_df=single_df.drop(columns=['dtype','ffn hidden size','hidden size','vocab size'])
    double_df=double_df.drop(columns=['dtype','ffn hidden size','hidden size','vocab size'])
    
    for col in label_model.keys():
        le = label_model[col]
        # print(single_df[col])
        single_df[col] = le.transform(single_df[col])
        double_df[col] = le.transform(double_df[col])

    try:
        pred = xgb_model.predict(single_df)
        pred2 = xgb_model.predict(double_df)
    except ValueError as e:
        print(e)
        print(single_df.to_string())
        exit(0)
    
    if pred[0] < 0 or pred2[0] < 0:
        print(config)
    
    CACHE_block[single_key] = (pred2[0] - pred[0]) if (pred2[0] - pred[0]) > 0 else pred[0] * 0.5
    CACHE_other[single_key] = (pred[0] - CACHE_block[single_key]) if (pred[0] - CACHE_block[single_key]) > 0 else pred[0] * 0.5

    # assert CACHE_block[single_key] > 0, f"1-layer time cannot be negative. CACHE_block[single_key]:{CACHE_block[single_key]} pred2[0]:{pred2[0]} pred[0]:{pred[0]} device:{device} config:{config}"
    # assert CACHE_other[single_key] > 0, f"pred2[0]:{pred2[0]}, pred[0]:{pred[0]} other fixed overhead cannot be negative. device:{device} config:{config}"
    
    return CACHE_block[single_key]*stage_layer_number, CACHE_other[single_key]



#device是一个长度等于pp的list，记录各个stage的device信息，bandwidth是一个长度为pp-1的list，记录stage间的通信带宽
def amp_e2e_time(config, device, pp_bandwidth, dp_bandwidth, model, label_model):

    gas = config['global_batch_size'] // config['DP_size'] // config['micro_batch_size'] 

    mb_times = []
    fix_overhead = -1

    for i in range(config['PP_size']):
        mb_time_i, fix_overhead_i = mb_time(i, config, device[i], model, label_model)
        mb_times.append(mb_time_i)
        if fix_overhead_i > fix_overhead:
            fix_overhead = fix_overhead_i

    pp_comm_times = [config['micro_batch_size'] * config['sequence_length'] * config['hidden_state'] * config['dtype_bytes'] / pp_bandwidth[i] for i in range(len(pp_bandwidth))]

    t_pp = (gas - 1) * max(mb_times) + sum(mb_times) + 2 * sum(pp_comm_times) + fix_overhead

    M = 2 * config['vocab_size'] * config['hidden_state'] + config['num_layers'] * (4 * config['hidden_state'] * config['hidden_state'] + 2 * config['hidden_state'] * config['ffn_hidden_state']) * config['dtype_bytes']

    dp_time = 2 * (config['DP_size'] - 1) * M / dp_bandwidth / config['DP_size']

    return t_pp + dp_time


def pipette_e2e_time(config, device, pp_bandwidth, dp_bandwidth, model, label_model):
    
    # rf_loaded = joblib.load('xgb_time.pkl')

    mb_times = []
    fix_overhead = -1

    for i in range(config['PP_size']):
        mb_time_i, fix_overhead_i = mb_time(i, config, device[i], model, label_model)
        mb_times.append(mb_time_i)
        if fix_overhead_i > fix_overhead:
            fix_overhead = fix_overhead_i

    gas = config['global_batch_size'] // config['DP_size'] // config['micro_batch_size']

    pp_comm_times = [config['micro_batch_size'] * config['sequence_length'] * config['hidden_state'] * config['dtype_bytes'] / pp_bandwidth[i] / 8000000000 for i in range(len(pp_bandwidth))]

    t_bubble = sum(mb_times) + 2 * sum(pp_comm_times)

    t_straggler = (config['PP_size'] - 1) * max(mb_times)

    M = 2 * config['vocab_size'] * config['hidden_state'] + config['num_layers'] * (4 * config['hidden_state'] * config['hidden_state'] + 2 * config['hidden_state'] * config['ffn_hidden_state']) * config['dtype_bytes']

    dp_time = 2 * (config['DP_size'] - 1) * M / dp_bandwidth / config['DP_size'] / 8000000000

    # print("mb_times:",mb_times)
    # print("fix_overhead:",fix_overhead)
    # print("pp_comm_times:",pp_comm_times)
    # print("t_straggler:",t_straggler)
    # print("dp_time:",dp_time)
    # print("gas:",gas)
    # print("M:",M)
    # print("final time:",t_bubble * gas / config['PP_size'] + t_straggler + dp_time + fix_overhead)
    # exit(0)

    return t_bubble * gas / config['PP_size'] + t_straggler + dp_time + fix_overhead

#相比amp和metis仅仅是考虑了通信延迟
def hexiscale_e2e_time(config, device, bandwidth, latency):
    return amp_e2e_time(config, device, bandwidth) + sum(latency) + max(latency)

#根据vp的数量减少
def interleave_e2e_time(config, device, bandwidth):
    rf_loaded = joblib.load('xgb_time_aug_new_1229.pkl')

    gas = config.global_batch_size // config.DP_size // config.micro_batch_size 

    mb_times = []

    for i in range(config.PP_size):
        mb_times.append(mb_time(i, config, device[i], rf_loaded))

    pp_comm_times = [config.micro_batch_size * config.sequence_length * config.hidden_state * config.dtype_bytes / bandwidth[i] for i in range(len(bandwidth))]

    t_pp = (gas - 1 + 1/config.vp_size) * max(mb_times) + sum(mb_times)/config.vp_size + 2*sum(pp_comm_times)

    M = 2 * config.vocab_size * config.hidden_state + config.num_layers * (4 * config.hidden_state * config.hidden_state + 2 * config.hidden_state * config.ffn_hidden_state) * config.dtype_bytes

    dp_time = 2 * (config.DP_size - 1) * M / min(bandwidth) / config.DP_size

    return t_pp + dp_time

def chimera_e2e_time(config, device, bandwidth):

    C_f = config.global_batch_size // config.DP_size // config.micro_batch_size 

    assert C_f % config.PP_size == 0

    C_d = C_f / config.PP_size * (2 * config.PP_size - 2)

    pp_comm_times = [config.micro_batch_size * config.sequence_length * config.hidden_state * config.dtype_bytes / bandwidth[i] for i in range(len(bandwidth))]

    mb_times = []

    for i in range(config.PP_size):
        mb_times.append(mb_time(i, config, device[i], rf_loaded))

    mb_fts = [mb_times[i] / device[i].fb_ratio for i in range(len(mb_times))]
    mb_bts = [mb_times[i] * (1 - 1 / device[i].fb_ratio) for i in range(len(mb_times))]

    t_pp = sum(mb_fts) + (C_f - config.PP_size) * max(mb_fts) + sum(mb_bts) + (C_d - config.PP_size) * max(mb_bts) + 2 * sum(pp_comm_times) + sum(pp_comm_times[:(C_d - config.PP_size)])

    M = 2 * config.vocab_size * config.hidden_state + config.num_layers * (4 * config.hidden_state * config.hidden_state + 2 * config.hidden_state * config.ffn_hidden_state) * config.dtype_bytes

    dp_time = 2 * (config.DP_size - 1) * M / min(bandwidth) / config.DP_size

    return t_pp + dp_time

# =====================================================================
# [MemX] 以下为新补充的显存预测与统一 stage 预测接口（§3.1.2 / §3.2.2）
# =====================================================================

# 哈希签名缓存：key 为单 micro-batch 特征元组，value 为
# (t_block, t_other, m_block, m_other)，避免对相同配置的重复模型调用
CACHE_DUAL = {}


def stage_predict(stage_index, config, device, recompute, dual):
    """用双输出预测器一次性得到某 stage 的时间与显存分解。

    返回 (mb_time, fix_overhead, peak_mem)：
      mb_time      - 该 stage 单个 micro-batch 的计算时间（随层数线性外推）
      fix_overhead - 固定开销（优化器、数据加载等，取各 stage 最大值使用）
      peak_mem     - 该 stage 的峰值显存估计
    """
    stage_layer_number = config['num_layers'] // config['PP_size']
    # 最后一个 stage 承担不能整除的余数层（非均匀切分时由 NUM_LAYERS_PER_STAGE 覆盖）
    if stage_index == config['PP_size'] - 1 and config['num_layers'] % config['PP_size'] != 0:
        stage_layer_number = config['num_layers'] % config['PP_size']

    single = {
        'TP size': config['TP_size'],
        'Rank': 0,
        'device type': device,
        'dtype': config['dtype'],
        'max position embedding': config['max_position_embedding'],
        'micro batch size': config['micro_batch_size'],
        'num attention heads': config['num_attention_heads'],
        'num layers': 1,
        'num query groups': config['num_query_groups'],
        'recompute granularity': recompute,
        'sequence length': config['sequence_length'],
        'hidden size': config['hidden_state'],
        'vocab size': config['vocab_size'],
        'ffn hidden size': config['ffn_hidden_state'],
    }

    single_key = tuple(sorted(single.items()))   # HashSignature(x), §3.1.2
    if single_key in CACHE_DUAL:
        t_block, t_other, m_block, m_other = CACHE_DUAL[single_key]
        return t_block * stage_layer_number, t_other, m_block * stage_layer_number + m_other

    double = single.copy()
    double['num layers'] = 2
    single_df = pd.DataFrame([single])
    double_df = pd.DataFrame([double])
    extend_cols(single_df)
    extend_cols(double_df)
    drop_cols = ['dtype', 'ffn hidden size', 'hidden size', 'vocab size']
    single_df = single_df.drop(columns=drop_cols)
    double_df = double_df.drop(columns=drop_cols)

    (t1, m1), (t2, m2) = dual.predict(single_df), dual.predict(double_df)
    t1, m1, t2, m2 = t1[0], m1[0], t2[0], m2[0]

    # 两层与一层预测之差 = 单层块开销；余量为固定开销（含显存基数）
    t_block = (t2 - t1) if (t2 - t1) > 0 else t1 * 0.5
    t_other = (t1 - t_block) if (t1 - t_block) > 0 else t1 * 0.5
    m_block = (m2 - m1) if (m2 - m1) > 0 else 0.0
    m_other = max(m1 - m_block, 0.0)

    CACHE_DUAL[single_key] = (t_block, t_other, m_block, m_other)
    return t_block * stage_layer_number, t_other, m_block * stage_layer_number + m_other


def memx_e2e_time(config, stage_devices, strategies, schedule,
                  pp_bandwidth, dp_bandwidth, dual, vp_size=1):
    """MemX 端到端迭代时间估计（§3.1.2 Eq. (3)/(4)）。

    config: dict，含 global_batch_size/micro_batch_size/DP_size/PP_size/
            sequence_length/hidden_state/ffn_hidden_state/vocab_size/dtype_bytes/num_layers
    stage_devices: 各 stage 设备类型；strategies: 各 stage 重计算策略
    schedule: '1F1B' 或 'interleaved'；带宽单位为 GB/s
    """
    gas = config['global_batch_size'] // config['DP_size'] // config['micro_batch_size']

    mb_times, fix_overhead = [], -1.0
    for i, dev in enumerate(stage_devices):
        mb_t, fix_i, _ = stage_predict(i, config, dev, strategies[i], dual)
        mb_times.append(mb_t)
        fix_overhead = max(fix_overhead, fix_i)

    pp_comm = [config['micro_batch_size'] * config['sequence_length']
               * config['hidden_state'] * config['dtype_bytes']
               / bw / 8e9 for bw in pp_bandwidth]

    M = (2 * config['vocab_size'] * config['hidden_state']
         + config['num_layers'] * (4 * config['hidden_state'] ** 2
         + 2 * config['hidden_state'] * config['ffn_hidden_state'])) * config['dtype_bytes']
    dp_time = (2 * (config['DP_size'] - 1) * M / config['DP_size']
               / dp_bandwidth / 8e9)

    if schedule == 'interleaved' and vp_size > 1:
        # Eq. (4)：稳态系数 (G-1+1/Nv)，瓶颈项 max(t_mb/Nv)，基础项除 Nv
        t = ((gas - 1 + 1.0 / vp_size) * max(mb_times) / vp_size
             + 2.0 / vp_size * sum(mb_times)
             + sum(pp_comm) + dp_time + fix_overhead)
    else:
        # Eq. (3)：1F1B
        t = ((gas - 1) * max(mb_times) + sum(mb_times)
             + 2 * sum(pp_comm) + dp_time + fix_overhead)
    return t


def per_device_memory(config, stage_devices, strategies, dual):
    """各 rank 的峰值显存估计（用于 V = 跨设备显存方差）。

    同一 stage 内的 TP×DP 个 rank 显存一致，返回长度为
    PP_size * TP_size * DP_size 的 list。
    """
    mems = []
    for i, dev in enumerate(stage_devices):
        _, _, mem = stage_predict(i, config, dev, strategies[i], dual)
        mems.extend([mem] * (config['TP_size'] * config['DP_size']))
    return mems
