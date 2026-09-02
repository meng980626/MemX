#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import torch

SIZE_MB = 256
ITERS, WARMUP = 20, 5
numel = SIZE_MB * 1024 * 1024 // 4
n = torch.cuda.device_count()

print(f"检测到 {n} 张 GPU: {[torch.cuda.get_device_name(i) for i in range(n)]}")
print(f"消息大小: {SIZE_MB} MB x {ITERS} iters\n")

for i in range(n):
    for j in range(n):
        if i == j:
            continue
        torch.cuda.set_device(i)          # 关键修复：切到源设备
        p2p = torch.cuda.can_device_access_peer(i, j)
        a = torch.randn(numel, device=f"cuda:{i}")
        b = torch.empty(numel, device=f"cuda:{j}")
        for _ in range(WARMUP):
            b.copy_(a, non_blocking=True)
        torch.cuda.synchronize(i)
        torch.cuda.synchronize(j)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            b.copy_(a, non_blocking=True)
        torch.cuda.synchronize(i)         # 关键修复：两个设备都等
        torch.cuda.synchronize(j)
        t = time.perf_counter() - t0
        bw = numel * 4 * ITERS / t / 1e9
        print(f"GPU{i} -> GPU{j}: {bw:7.2f} GB/s   "
              f"({'P2P 直连' if p2p else '无 P2P，经主机内存'})")
        del a, b
        torch.cuda.empty_cache()