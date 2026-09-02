#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨节点 GPU P2P 带宽测试 v2 (PyTorch + NCCL, all_reduce 版)
节点0: 2 x T4   -> global rank 0, 1
节点1: 4 x V100 -> global rank 2, 3, 4, 5
"""

import argparse
import os
import socket

import torch
import torch.distributed as dist


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--size-mb", type=float, default=256.0, help="消息大小 (MB)")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    assert world == 6, f"world_size 应为 6，当前 {world}"

    numel = int(args.size_mb * 1024 * 1024 / 4)  # float32
    buf = torch.randn(numel, device=device)
    msg_bytes = numel * 4

    print(f"[rank {rank}] host={socket.gethostname()} "
          f"gpu={torch.cuda.get_device_name(local_rank)}", flush=True)

    # 所有 rank 按相同顺序为每个 pair 建子通信组（new_group 是集合调用，不能漏）
    pairs = [(i, j) for i in range(world) for j in range(i + 1, world)]
    groups = {p: dist.new_group(ranks=list(p)) for p in pairs}
    dist.barrier()

    matrix = torch.zeros(world, world, device=device)

    for (i, j) in pairs:
        pg = groups[(i, j)]
        if rank in (i, j):
            for _ in range(args.warmup):
                dist.all_reduce(buf, group=pg)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iters):
                dist.all_reduce(buf, group=pg)
            end.record()
            torch.cuda.synchronize()
            t = start.elapsed_time(end) / 1e3
            # 2-rank all_reduce: 每 rank 发送 msg_bytes、接收 msg_bytes
            bw_gbs = msg_bytes * args.iters / t / 1e9
            matrix[rank, j if rank == i else i] = bw_gbs
        dist.barrier()

    dist.all_reduce(matrix, op=dist.ReduceOp.SUM)
    dist.barrier()

    if rank == 0:
        m = matrix.cpu()
        print("\n" + "=" * 60)
        print(f"P2P 带宽矩阵 (单向等效, {args.size_mb:.0f} MB x {args.iters} iters)")
        print("rank 0,1 = 节点0 (T4) | rank 2,3,4,5 = 节点1 (V100)")
        print("=" * 60)
        print("        " + "".join(f"rank{j:<5d}" for j in range(world)))
        for i in range(world):
            row = f"rank{i:<4d}"
            for j in range(world):
                if i == j:
                    row += "    --    "
                else:
                    v = min(m[i, j].item(), m[j, i].item())
                    if v == 0:
                        v = max(m[i, j].item(), m[j, i].item())
                    row += f"{v:8.2f}  "
            print(row)
        print("单位: GB/s")

    for pg in groups.values():
        dist.destroy_process_group(pg)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()