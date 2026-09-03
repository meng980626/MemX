import itertools
import subprocess
import os, signal, threading, time, psutil
import json
from datetime import datetime
from collections import Counter
import math
import pandas as pd
import glob
import argparse
from itertools import cycle, islice

import fcntl
SYNC_DIR = os.environ.get("MEMX_SYNC_DIR", "/path/to/shared/sync")  # 所有节点共享的目录，如 NFS
os.makedirs(SYNC_DIR, exist_ok=True)

import random

RANDOM_SEED = 24
random.seed(RANDOM_SEED)




def barrier(config_idx, total_nodes=3, timeout=5000):
    """改进的屏障同步 - 两阶段释放"""
    ready_file = os.path.join(SYNC_DIR, f"ready_{args.noderank}_{config_idx}")
    done_file = os.path.join(SYNC_DIR, f"done_{config_idx}")
    
    # 阶段1：标记本节点就绪
    with open(ready_file, 'w') as f:
        f.write(datetime.now().isoformat())
    
    # 阶段2：等待所有节点就绪（由rank 0创建done标记）
    start = time.time()
    while True:
        # 检查是否所有节点已就绪
        ready_count = len([f for f in os.listdir(SYNC_DIR) 
                          if f.startswith(f"ready_") and f"_{config_idx}" in f])
        
        # rank 0负责创建完成标记
        if args.noderank == 0 and ready_count >= total_nodes:
            with open(done_file, 'w') as f:
                f.write('all_ready')
        
        # 所有节点检查done_file是否存在
        if os.path.exists(done_file):
            break
            
        if time.time() - start > timeout:
            print(f"[WARNING] 配置{config_idx}同步超时，就绪节点: {ready_count}/{total_nodes}")
            kill_bash_on_port_6000()
            if args.noderank == 0:
                # 清理所有旧的同步文件
                patterns = ['ready_*', 'done_*', 'lock_*', 'barrier_*']
                for pattern in patterns:
                    for f in glob.glob(os.path.join(SYNC_DIR, pattern)):
                        try:
                            os.remove(f)
                        except:
                            pass

                print(f"[INIT] Rank 0 已清理同步目录")
            return False
        time.sleep(0.2)
    
    # 阶段3：所有节点确认后，rank 0最后清理
    # 给一点时间让其他节点看到done_file
    time.sleep(0.5)
    
    # 非rank节点先删除自己的ready文件
    if args.noderank != 0:
        try:
            os.remove(ready_file)
        except:
            pass
    else:
        # rank 0等其他人清理完再清理done_file
        time.sleep(1)
        for f in os.listdir(SYNC_DIR):
            if f"_{config_idx}" in f:
                try:
                    os.remove(os.path.join(SYNC_DIR, f))
                except:
                    pass
    
    return True

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="v100")
parser.add_argument("--noderank", type=int, default=0)
args = parser.parse_args()

if args.noderank == 0:
    # 清理所有旧的同步文件
    patterns = ['ready_*', 'done_*', 'lock_*', 'barrier_*']
    for pattern in patterns:
        for f in glob.glob(os.path.join(SYNC_DIR, pattern)):
            try:
                os.remove(f)
            except:
                pass

    print(f"[INIT] Rank 0 已清理同步目录")

# def suicide(sec=60):
#     """1 分钟后自动强退，防止 NCCL 死锁"""
#     def _kill():
#         os.kill(os.getpid(), signal.SIGKILL)
#     threading.Timer(sec, _kill).start()

def kill_bash_on_port_6000():
    """
    检查 6000 端口是否被占用，若占用则 kill 对应的 bash 进程
    仅依赖 ps + lsof / ss / netstat（镜像环境）
    """
    # 1. 判断 6000 端口是否有进程监听
    try:
        # 方案 A：lsof（优先）
        out = subprocess.check_output(
            ['lsof', '-ti', ':6000'], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        # 方案 B：ss
        try:
            out = subprocess.check_output(
                ['ss', '-lptn', 'sport = :6000'], stderr=subprocess.DEVNULL, text=True
            )
            # 提取 PID（ss 输出最后一列格式类似 "users:(("bash",pid=1234,fd=3))"）
            import re
            m = re.search(r'pid=(\d+)', out)
            out = m.group(1) if m else ''
        except (FileNotFoundError, subprocess.CalledProcessError):
            # 方案 C：netstat（最兼容）
            try:
                out = subprocess.check_output(
                    ['netstat', '-tunlp'], stderr=subprocess.DEVNULL, text=True
                )
                for line in out.splitlines():
                    if ':6000' in line:
                        parts = line.split()
                        pid_slash = parts[-1].split('/')[0]
                        out = pid_slash if pid_slash.isdigit() else ''
                        break
                else:
                    out = ''
            except (FileNotFoundError, subprocess.CalledProcessError):
                out = ''

    if not out:
        # 端口空闲
        return

    pids = [int(p) for p in out.strip().split('\n') if p]

    # 2. 用 ps aux 验证进程名是否包含 bash
    try:
        for pid in pids:
            ps_line = subprocess.check_output(
                ['ps', '-o', 'pid,comm,args', '-p', str(pid)],
                text=True
            ).splitlines()[1]  # 跳过表头
            comm = ps_line.split()[1]
            if comm == 'bash' or 'bash' in ps_line:
                print(f"[PORT-CLEAN] kill bash({pid}) on port 6000")
                os.kill(pid, signal.SIGKILL)
    except (IndexError, subprocess.CalledProcessError):
        pass

def execute_experiment(env, experiment_id, args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/ex{experiment_id}_{args.noderank}.log"
    config_file = f"{log_dir}/ex{experiment_id}_{args.noderank}.json"

    with open(config_file, "w") as f:
        json.dump(env, f, indent=2)
    
    with open(log_file, "w") as log:
        try:
            time.sleep(30)
            kill_bash_on_port_6000()
            result = subprocess.run(
                [base_script],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=5000         
            )
            print(f"Finished with exit code {result.returncode}")
        except subprocess.TimeoutExpired:
            # 超时日志
            log.write("\n[TIMEOUT] 子进程运行超过20分钟，已被强制终止\n")
            log.flush()
            print(f"[TIMEOUT] 子进程运行超过20分钟，已被强制终止")

TOTAL_GPU = 4 
if args.device == "t4":
    TOTAL_GPU = 2
elif args.device == "a100":
    TOTAL_GPU = 8
elif args.device == "3090":
    TOTAL_GPU = 8
elif args.device == "v100":
    TOTAL_GPU = 4
elif args.device == "vendor-b":
    TOTAL_GPU = 8
elif args.device == "2node":
    TOTAL_GPU = 8
elif args.device == "3node":
    TOTAL_GPU = 20
 
VALID_PARALLEL = []

base_script = "../scripts/train_llama2_7b_multi_node.sh"

if args.device == "vendor-b":
    base_script = "../scripts/train_llama2_7b_single_node.sh"
elif args.device == "2node":
    base_script = "../scripts/train_llama2_7b_multi_node.sh"
elif args.device == "3node":
    base_script = "../scripts/train_llama2_7b_multi_node.sh"


log_dir = "sweep_logs"
os.makedirs(log_dir, exist_ok=True)



VALID_PARALLEL.append((4, 5, 1))
VALID_PARALLEL.append((2, 10, 1))
VALID_PARALLEL.append((2, 5, 2))

print(VALID_PARALLEL)

# exit(0)

GLOBAL_BATCH_SIZE = 32
FFN_HIDDEN_SIZE = 11008
HIDDEN_SIZE = 4096
CP_SIZE = 1
DTYPE = ['fp16']

if args.device == "a100":
    DTYPE = ['fp16','bf16']
elif args.device == "vendor-b" or args.device == "2node" or args.device == "3node":
    DTYPE = ['bf16']

recompute_method = []

NUM_ATTENTION_HEADS_RANGE=[1,2,4,8,16,32,64,128]
if args.device == "vendor-b" or args.device == "2node" or args.device == "3node":
    NUM_ATTENTION_HEADS_RANGE = [64,128]

param_grid = {
    "NUM_LAYERS": [12],
    "SEQ_LENGTH": [1024,2048,4096],
    "MICRO_BATCH_SIZE": [x for x in range(2, GLOBAL_BATCH_SIZE+1) if (x & (x - 1)) == 0],
    "MAX_POSITION_EMBEDDINGS": [2048,4096,8192],
    "NUM_ATTENTION_HEADS": NUM_ATTENTION_HEADS_RANGE,
    "NUM_QUERY_GROUPS": [1,2,4,8,16,32],
    "DTYPE":DTYPE
}

keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]


num_experiments = 0

min_experiment_dict = {}
num_layers_experiment_count = Counter()
seq_length_experiment_count = Counter()
micro_bs_experiment_count = Counter()

for tp, pp, dp in VALID_PARALLEL:
    for i, params in enumerate(combinations):
        # print(f"TP{tp} PP{pp} DP{dp} [{i+1}/{len(combinations)}] Running with params: {params}")
        if params["MAX_POSITION_EMBEDDINGS"] < params["SEQ_LENGTH"]:
            #print(f"MAX_POSITION_EMBEDDINGS {params['MAX_POSITION_EMBEDDINGS']} cannot be less than SEQ_LENGTH {params['SEQ_LENGTH']}!")
            continue
        
        if params["NUM_QUERY_GROUPS"] < tp:
            #print(f"NUM_QUERY_GROUPS {params['NUM_QUERY_GROUPS']} must be a multiple of tensor_parallel_size ({str(tp)})")
            continue
        
        if params["NUM_LAYERS"] < pp:
            #print(f"NUM_LAYERS {params['NUM_LAYERS']} cannot be less than pipeline_parallel_size ({str(pp)})")
            continue
        
        if params["NUM_ATTENTION_HEADS"] % tp != 0:
            #print(f"NUM_QUERY_GROUPS {params['NUM_QUERY_GROUPS']} must be a multiple of tensor_parallel_size ({str(tp)})")
            continue
        
        if params["NUM_ATTENTION_HEADS"] % params["NUM_QUERY_GROUPS"] != 0:
            #print(f"AssertionError: The number of attention heads {params["NUM_ATTENTION_HEADS"]} must be divisible by the number of GQA groups {params["NUM_QUERY_GROUPS"]}! when instantiating TEDotProductAttention")
            continue

        if pp == 3:
            NUM_LAYERS_PER_STAGE="4,4,4"
        elif pp == 6:
            NUM_LAYERS_PER_STAGE="2,2,2,2,2,2"


        key = (tp, pp, dp, params["MAX_POSITION_EMBEDDINGS"], params["NUM_QUERY_GROUPS"], params["NUM_ATTENTION_HEADS"], params["NUM_LAYERS"], params["MICRO_BATCH_SIZE"], params["SEQ_LENGTH"],params["DTYPE"], NUM_LAYERS_PER_STAGE) 

        min_experiment_dict[key] = None

print(len(min_experiment_dict))
recompute_num=0

RECOMPUTE_GRANULARITY = ['full','selective','null']
RECOMPUTE_METHOD = ['uniform','block']

RESULT_RECOMPUTE_CSV="result_recompute_3node.csv"
csv_path = "./"+RESULT_RECOMPUTE_CSV
executed_configs = set()

if os.path.exists(csv_path):
    df_existing = pd.read_csv(csv_path, keep_default_na=False)
    # 构建唯一标识：把关键参数组合成 tuple
    for _, row in df_existing.iterrows():
        config_key = (
            int(row['TP size']),
            int(row['PP size']),
            int(row['DP size']),
            int(row['Rank']),
            int(row['max position embedding']),
            int(row['num query groups']),
            int(row['num attention heads']),
            int(row['num layers']),
            int(row['micro batch size']),
            int(row['sequence length']),
            str(row['dtype']),
            str(row['recompute granularity']),
        )
        print(config_key)
        executed_configs.add(config_key)
    print(f"已加载 {len(executed_configs)} 条历史记录")
else:
    print("未找到历史记录，从头开始")

NODES = [0, 1, 2]
RECOMPUTE_STRATEGIES = ['null', 'selective', 'full-block-1', 'full-uniform-1']

# 预生成所有64种配置
ALL_CONFIGS = []
for combo in itertools.product(RECOMPUTE_STRATEGIES, repeat=len(NODES)):
    # cfg = {node: strategy for node, strategy in zip(NODES, combo)}
    # if cfg[1] == 'full-block-1' or cfg[2] == 'full-block-1':#br上使用full-block-1有bug
    #     continue
    # # if cfg[1] == 'null' or cfg[2] == 'null':#br上使用recompute
    # #     continue
    # if cfg[2] != 'null':#ts上固定用null
    #     continue
    # if cfg[0] == 'null':#nv上需要用recompute
    #     continue
    ALL_CONFIGS.append(cfg)

print(f"总共有 {len(ALL_CONFIGS)} 种不同的多节点配置组合")

kill_bash_on_port_6000()
i=0
# for tp, pp, dp, max_position_embedding, num_query_groups, num_attention_heads, num_layers, micro_batch_size, sequence_length, dtype in min_experiment_dict:

len(min_experiment_dict)
random_configs = cycle(
    random.choices(ALL_CONFIGS, k=len(min_experiment_dict))
)

# skip_iters=[94,114,125,126,127,128,129,135,136,137,138,139,144,145,146,147,148,149,150,151,157,158,160,161,162,167,171,172,173]
skip_iters=list(range(96, 460))
for i, key in enumerate(min_experiment_dict.keys()):
    tp, pp, dp, max_position_embedding, num_query_groups, num_attention_heads, num_layers, micro_batch_size, sequence_length, dtype, num_layers_per_stage = key
    print(f"TP{tp} PP{pp} DP{dp} MAX_POSITION_EMBEDDINGS{max_position_embedding} NUM_QUERY_GROUPS{num_query_groups} NUM_ATTENTION_HEADS{num_attention_heads} NUM_LAYERS{num_layers} MICRO_BATCH_SIZE{micro_batch_size} SEQ_LENGTH{sequence_length} DTYPE{dtype} [{i+1}/{len(min_experiment_dict)}]")
    if i in skip_iters:
        recompute_config = next(random_configs)
        recompute_num+=1
        print(f"Iter {i}: config = {recompute_config}")
        continue

    env = os.environ.copy()
    GPUS_PER_NODE = 4 if args.noderank == 0 else 8
    if args.noderank == 0:
        env.update({"CUDA_VISIBLE_DEVICES":"4,5,6,7"})
    env.update({
        "RESULT_RECOMPUTE_CSV":RESULT_RECOMPUTE_CSV,
        "TP_SIZE": str(tp),
        "PP_SIZE": str(pp),
        "DP_SIZE": str(dp),
        "MAX_POSITION_EMBEDDINGS": str(max_position_embedding),
        "NUM_QUERY_GROUPS": str(num_query_groups),
        "NUM_ATTENTION_HEADS": str(num_attention_heads),
        "NUM_LAYERS": str(num_layers),
        "MICRO_BATCH_SIZE": str(micro_batch_size),
        "SEQ_LENGTH": str(sequence_length),
        "DTYPE":dtype,
        "GLOBAL_BATCH_SIZE": str(GLOBAL_BATCH_SIZE),
        "NUM_LAYERS_PER_STAGE":str(num_layers_per_stage),
        "NODE_RANK":str(args.noderank),
        "GPUS_PER_NODE":str(GPUS_PER_NODE)
    })
    if args.noderank==2:
        env.update({"SIMULATE_VENDOR_C_ON_B":"true"})

    # for config_idx, config in enumerate(ALL_CONFIGS):
    #     print(f"Config {config_idx+1}/{len(ALL_CONFIGS)}: {config}")

    recompute_config = next(random_configs)
    print(f"Iter {i}: config = {recompute_config}")    
        # 获取当前node应该使用的策略
    recompute_num+=1
    current_key1 = (int(tp), int(pp), int(dp), 0, int(max_position_embedding), int(num_query_groups), int(num_attention_heads), int(num_layers), int(micro_batch_size), int(sequence_length), str(dtype), recompute_config[0])
    current_key2 = (int(tp), int(pp), int(dp), 4, int(max_position_embedding), int(num_query_groups), int(num_attention_heads), int(num_layers), int(micro_batch_size), int(sequence_length), str(dtype), recompute_config[1])
    current_key3 = (int(tp), int(pp), int(dp), 12, int(max_position_embedding), int(num_query_groups), int(num_attention_heads), int(num_layers), int(micro_batch_size), int(sequence_length), str(dtype), recompute_config[2])
    if current_key1 in executed_configs and current_key2 in executed_configs and current_key3 in executed_configs :
        print(f"[SKIP] 已存在")
        continue

    kill_bash_on_port_6000()
    if not barrier(recompute_num):
        continue  # 超时则跳过

    my_strategy = recompute_config[args.noderank]

    if my_strategy == 'null':
        env.update({"RECOMPUTE_GRANULARITY":"null"})
    elif my_strategy == 'selective':
        env.update({"RECOMPUTE_GRANULARITY":"selective"})
    elif my_strategy == 'full-block-1':
        env.update({
                "RECOMPUTE_GRANULARITY":"full",
                "RECOMPUTE_METHOD":"block",
                "RECOMPUTE_NUM_LAYERS":'1',
            })
    elif my_strategy == 'full-uniform-1':
        env.update({
                "RECOMPUTE_GRANULARITY":"full",
                "RECOMPUTE_METHOD":"uniform",
                "RECOMPUTE_NUM_LAYERS":'1',
            })
    
    execute_experiment(env, recompute_num, args)
    
    # for granu in RECOMPUTE_GRANULARITY:
    #     if granu == 'null':
    #         env.update({"RECOMPUTE_GRANULARITY":"null"})
    #         current_key = (int(tp), int(pp), int(dp), int(max_position_embedding), int(num_query_groups), 
    #                       int(num_attention_heads), int(num_layers), int(micro_batch_size), 
    #                       int(sequence_length), str(dtype), 'null')
    #         print(current_key)
    #         if current_key in executed_configs:
    #             print(f"[SKIP] RECOMPUTE={granu} 已存在")
    #             continue
    #         recompute_num+=1
    #         execute_experiment(env, recompute_num)
    #     elif granu == 'selective':
    #         env.update({"RECOMPUTE_GRANULARITY":"selective"})
    #         current_key = (int(tp), int(pp), int(dp), int(max_position_embedding), int(num_query_groups), 
    #                       int(num_attention_heads), int(num_layers), int(micro_batch_size), 
    #                       int(sequence_length), str(dtype), 'selective')
    #         print(current_key)
    #         if current_key in executed_configs:
    #             print(f"[SKIP] RECOMPUTE={granu} 已存在")
    #             continue
    #         recompute_num+=1
    #         execute_experiment(env, recompute_num)
    #     else: # granu == 'full'
    #         #5:27切分
    #         # RECOMPUTE_NUM_LAYERS = [
    #         #     n for n in range(1, 5)
    #         # ]  
    #         for method in RECOMPUTE_METHOD:
    #             # for re_nl in RECOMPUTE_NUM_LAYERS:
                
    #             env.update({
    #                 "RECOMPUTE_GRANULARITY":"full",
    #                 "RECOMPUTE_METHOD":method,
    #                 "RECOMPUTE_NUM_LAYERS":'1',
    #             })
                
    #             current_key = (int(tp), int(pp), int(dp), int(max_position_embedding), int(num_query_groups), 
    #                       int(num_attention_heads), int(num_layers), int(micro_batch_size), 
    #                       int(sequence_length), str(dtype), 'full-'+method+'-1')
    #             print(current_key)
    #             if current_key in executed_configs:
    #                 print(f"[SKIP] RECOMPUTE={granu} 已存在")
    #                 continue
    #             recompute_num+=1
    #             execute_experiment(env, recompute_num)
    
    # i+=1


