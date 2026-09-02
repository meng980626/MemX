import itertools
import subprocess
import os, signal, threading, time, psutil
import json
from datetime import datetime
from collections import Counter

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="v100")
args = parser.parse_args()


def suicide(sec=60):
    """1 分钟后自动强退，防止 NCCL 死锁"""
    def _kill():
        os.kill(os.getpid(), signal.SIGKILL)
    threading.Timer(sec, _kill).start()

def kill_previous_gpu_processes(exclude_cmd="Xorg"):
    try:
        out = subprocess.check_output([
            'nvidia-smi',
            '--query-compute-apps=pid,used_memory',
            '--format=csv,noheader,nounits'
        ], text=True)
    except subprocess.CalledProcessError:
        return

    for line in out.strip().splitlines():
        if not line:
            continue
        pid_str, mem_str = line.split(',')
        pid, mem = int(pid_str.strip()), int(mem_str.strip())
        if mem <= 0:
            continue
        try:
            proc = psutil.Process(pid)
            cmd = " ".join(proc.cmdline())
            if exclude_cmd in cmd:
                print(f"[GPU-CLEAN] Skip {exclude_cmd}: {pid}")
                continue
            os.kill(pid, signal.SIGTERM)
            print(f"[GPU-CLEAN] SIGTERM -> {pid} ({cmd})")
        except (psutil.NoSuchProcess, ProcessLookupError):
            pass

    time.sleep(2)
    for line in out.strip().splitlines():
        if not line:
            continue
        pid_str, mem_str = line.split(',')
        pid, mem = int(pid_str.strip()), int(mem_str.strip())
        if mem <= 0:
            continue
        try:
            proc = psutil.Process(pid)
            cmd = " ".join(proc.cmdline())
            if exclude_cmd in cmd:
                continue
            os.kill(pid, signal.SIGKILL)
            print(f"[GPU-CLEAN] SIGKILL -> {pid} ({cmd})")
        except (psutil.NoSuchProcess, ProcessLookupError):
            pass

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

    pid = int(out.strip())

    # 2. 用 ps aux 验证进程名是否包含 bash
    try:
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

def execute_experiment(env, experiment_id):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/ex{experiment_id}_{timestamp}.log"
    config_file = f"{log_dir}/ex{experiment_id}_{timestamp}.json"

    with open(config_file, "w") as f:
        json.dump(env, f, indent=2)
    
    with open(log_file, "w") as log:
        try:
            kill_bash_on_port_6000()
            result = subprocess.run(
                [base_script],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=300          # 5分钟后强制超时
            )
            print(f"Finished with exit code {result.returncode}")
        except subprocess.TimeoutExpired:
            # 超时日志
            log.write("\n[TIMEOUT] 子进程运行超过5分钟，已被强制终止\n")
            log.flush()
            print(f"[TIMEOUT] 子进程运行超过5分钟，已被强制终止")

TOTAL_GPU = 4 
if args.device == "t4":
    TOTAL_GPU = 2
elif args.device == "a100":
    TOTAL_GPU = 8
 
VALID_PARALLEL = []

base_script = "../scripts/train_llama2_7b_single_node.sh"
if args.device == "t4":
    base_script = "../scripts/train_llama2_7b_single_node.sh"  # 通过环境变量传入设备参数
log_dir = "sweep_logs"
os.makedirs(log_dir, exist_ok=True)

for tp in range(1, TOTAL_GPU + 1):
    if (tp & (tp - 1)) != 0:
        continue
    VALID_PARALLEL.append((tp, 1, 1))

GLOBAL_BATCH_SIZE = 32
FFN_HIDDEN_SIZE = 11008
HIDDEN_SIZE = 4096
CP_SIZE = 1
DTYPE = ['fp16']

if args.device == "a100":
    DTYPE = ['bf16']

recompute_method = []

param_grid = {
    "NUM_LAYERS": [1,2,4,8,16,32],
    "SEQ_LENGTH": [1024,2048],
    "MICRO_BATCH_SIZE": [x for x in range(2, GLOBAL_BATCH_SIZE+1) if (x & (x - 1)) == 0],
    "MAX_POSITION_EMBEDDINGS": [2048,4096,8192],
    "NUM_ATTENTION_HEADS": [1,2,4,8,16,32],
    "NUM_QUERY_GROUPS": [1,2,4,8,16,32],
}

keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
num_experiments = 0

min_experiment_dict = set()
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
        
        key_num_layers = (tp, pp, dp, params["MAX_POSITION_EMBEDDINGS"], params["NUM_QUERY_GROUPS"], params["NUM_ATTENTION_HEADS"], params["MICRO_BATCH_SIZE"], params["SEQ_LENGTH"])

        num_layers_experiment_count[key_num_layers]+=1

        key_seq_length = (tp, pp, dp, params["MAX_POSITION_EMBEDDINGS"], params["NUM_QUERY_GROUPS"], params["NUM_ATTENTION_HEADS"], params["NUM_LAYERS"], params["MICRO_BATCH_SIZE"])

        seq_length_experiment_count[key_seq_length]+=1

        key_micro_bs = (tp, pp, dp, params["MAX_POSITION_EMBEDDINGS"], params["NUM_QUERY_GROUPS"], params["NUM_ATTENTION_HEADS"], params["NUM_LAYERS"], params["SEQ_LENGTH"])

        micro_bs_experiment_count[key_micro_bs]+=1
        
        if num_layers_experiment_count[key_num_layers]<=2 and seq_length_experiment_count[key_seq_length]<=2 and micro_bs_experiment_count[key_micro_bs] <=2:
            key = (tp, pp, dp, params["MAX_POSITION_EMBEDDINGS"], params["NUM_QUERY_GROUPS"], params["NUM_ATTENTION_HEADS"], params["NUM_LAYERS"], params["MICRO_BATCH_SIZE"], params["SEQ_LENGTH"]) 
            min_experiment_dict.add(key)

print(len(min_experiment_dict))
recompute_num=0

RECOMPUTE_GRANULARITY = ['full','selective','null']
RECOMPUTE_METHOD = ['uniform','block']

i=0
for tp, pp, dp, max_position_embedding, num_query_groups, num_attention_heads, num_layers, micro_batch_size, sequence_length in min_experiment_dict:
    print(f"TP{tp} PP{pp} DP{dp} MAX_POSITION_EMBEDDINGS{max_position_embedding} NUM_QUERY_GROUPS{num_query_groups} NUM_ATTENTION_HEADS{num_attention_heads} NUM_LAYERS{num_layers} MICRO_BATCH_SIZE{micro_batch_size} SEQ_LENGTH{sequence_length} [{i+1}/{len(min_experiment_dict)}]")
    env = os.environ.copy()
    env.update({
        "TP_SIZE": str(tp),
        "PP_SIZE": str(pp),
        "DP_SIZE": str(dp),
        "MAX_POSITION_EMBEDDINGS": str(max_position_embedding),
        "NUM_QUERY_GROUPS": str(num_query_groups),
        "NUM_ATTENTION_HEADS": str(num_attention_heads),
        "NUM_LAYERS": str(num_layers),
        "MICRO_BATCH_SIZE": str(micro_batch_size),
        "SEQ_LENGTH": str(sequence_length),
    })
    for granu in RECOMPUTE_GRANULARITY:
        if granu == 'null':
            env.update({"RECOMPUTE_GRANULARITY":"null"})
            recompute_num+=1
            execute_experiment(env, recompute_num)
        elif granu == 'selective':
            env.update({"RECOMPUTE_GRANULARITY":"selective"})
            recompute_num+=1
            execute_experiment(env, recompute_num)
        else: # granu == 'full'
            #暂时假定NUM_LAYERS能被pp整除，实际上有可能会不均匀切分
            RECOMPUTE_NUM_LAYERS = [
                n for n in range(1, num_layers // pp + 1)
            ]  
            for method in RECOMPUTE_METHOD:
                for re_nl in RECOMPUTE_NUM_LAYERS:
                    recompute_num+=1
                    env.update({
                        "RECOMPUTE_GRANULARITY":"full",
                        "RECOMPUTE_METHOD":method,
                        "RECOMPUTE_NUM_LAYERS":str(re_nl),
                    })
                    execute_experiment(env, recompute_num)
    
    i+=1