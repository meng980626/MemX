# csv_writer.py
import os, csv, io, subprocess, re
import torch
import torch.distributed as dist
import time
import json
from pathlib import Path

import pickle



CSV_FILE = os.getenv('RESULT_RECOMPUTE_CSV', 'result_recompute.csv')
_row_dict = {}

_gloo_pg = None

def get_gloo_pg():
    """获取或创建 Gloo 进程组"""
    global _gloo_pg
    
    if _gloo_pg is not None:
        return _gloo_pg
    
    world_size = dist.get_world_size()
    _gloo_pg = dist.new_group(
        ranks=list(range(world_size)),
        backend=dist.Backend.GLOO,
    )
    
    return _gloo_pg

def write_csv_column(name, value):
    """写某一列数据"""
    global _row_dict
    try:
        v = float(value)
        _row_dict[name] = f"{v:g}"
    except (TypeError, ValueError):
        _row_dict[name] = str(value)

def all_gather_object_bytes(obj, group=None):
    """用字节传输，不关心内部数据类型"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # 序列化为字节
    buffer = io.BytesIO()
    pickle.dump(obj, buffer)
    obj_bytes = buffer.getvalue()
    
    # 关键：用 uint8 tensor 传输字节，所有 rank 完全一致
    obj_tensor = torch.tensor(list(obj_bytes), dtype=torch.uint8)
    
    # 收集大小（long）
    local_size = torch.tensor([obj_tensor.numel()], dtype=torch.long)
    size_list = [torch.zeros(1, dtype=torch.long) for _ in range(world_size)]
    dist.all_gather(size_list, local_size, group=group)
    
    # 收集数据（uint8，所有 rank 一致）
    max_size = max(s.item() for s in size_list)
    padded = torch.zeros(max_size, dtype=torch.uint8)
    padded[:obj_tensor.numel()] = obj_tensor
    
    gathered_tensors = [torch.zeros(max_size, dtype=torch.uint8) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, padded, group=group)
    
    # 反序列化
    result = []
    for i, tensor in enumerate(gathered_tensors):
        size = size_list[i].item()
        bytes_data = bytes(tensor[:size].tolist())
        result.append(pickle.loads(bytes_data))
    
    return result

@torch.no_grad()
def write_csv_newline():
    """
    使用 all_gather_object 把各 rank 的 dict 一次性收齐，
    由 rank0 统一写 CSV。
    """
    world_size = dist.get_world_size()
    rank       = dist.get_rank()

    gloo_pg = get_gloo_pg()

    # gathered_list = [None] * world_size
    gathered_list = all_gather_object_bytes(_row_dict, group=gloo_pg)
    # dist.all_gather_object(gathered_list, _row_dict, group=gloo_pg)

    if rank == 0:
        print(gathered_list)
        all_keys = {k for d in gathered_list for k in d}

        first_cols = ['TP size','PP size','DP size']
        last_cols = ['Peak GPU memory', 'TFLOP/s/GPU', 'elapsed time per iteration']

        rest_cols = sorted(all_keys - set(first_cols)- set(last_cols))

        final_header = first_cols + rest_cols + last_cols

        first_write = not os.path.exists(CSV_FILE)
        with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=final_header)
            if first_write:
                writer.writeheader()
            for row in gathered_list:   # 每个 rank 一行
                writer.writerow(row)

    dist.barrier(group=gloo_pg)
    _row_dict.clear()

def gpu_memory_used(gpu_id=0):

    if os.getenv('USE_CUDA') == 'true':
        gpu_id+=4
        cmd = ["nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-i", str(gpu_id)]
        out = subprocess.check_output(cmd, text=True).strip()
        return int(out)
    elif os.getenv('USE_VENDOR_B') == 'true':
        txt = subprocess.check_output(["vendor-smi"], text=True).replace(",", "")
        block = re.search(
            rf"\|\s*{gpu_id}\b.*?\n(.*?)\n\+-",
            txt, re.S
        )
        if not block:
            return -100
        # 在块里找 数字MiB / 数字MiB
        m = re.search(r"\|\s+(\d+)MiB\s+/\s+(\d+)MiB", block.group(1))
        if m:
            return int(m.group(1))          # 已用 MiB
        return -100