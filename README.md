# MemX: Memory-Aware Automatic Parallelism for Cross-Vendor Heterogeneous GPU Training

MemX 是面向跨厂商异构 GPU 集群的显存感知自动并行系统。它在配置搜索中同时优化
训练吞吐、峰值显存与设备间显存方差，支持逐 stage 的异构重计算策略与非均匀流水
层切分；配套的轻量化性能建模模块通过最小集合采样 + 线性插值数据增强训练
XGBoost 预测器，实现毫秒级的配置性能预测。

本仓库包含论文实验所用的全部外围代码：profiling 采样、性能建模、配置搜索、
集群模拟扩展与带宽测试工具。

## 目录结构

```
MemX/
├── profiling/                    # 最小集合 Profiler（论文 §3.2.1）
│   ├── run_sweep_single_node.py  # 单机配置采样（关键参数识别 + 边界-中点采样）
│   ├── run_sweep_multi_node.py   # 多机异构集群采样（含节点间屏障同步、断点续跑）
│   ├── csv_writer.py             # 训练进程内的分布式 CSV 数据收集（all-gather 汇总）
│   └── megatron_patch/
│       └── training.py           # 植入埋点的 Megatron-LM 训练主循环（补丁形式）
├── modeling/                     # 性能预测模型（论文 §3.2.2）
│   ├── train_predictor.py        # 线性插值数据增强 + XGBoost 时间/显存模型训练
│   ├── dual_predictor.py         # 双输出预测器封装（一次推理同时输出时间和显存）
│   └── legacy_random_forest.py   # 早期随机森林版本（仅存档，不建议使用）
├── search/                       # 配置搜索（论文 §3.1.1/§3.1.2，Algorithm 1）
│   ├── search_configs.py         # 主程序：枚举 -> 剪枝 -> 重计算策略 -> 调度选择 -> Φ 排序
│   ├── objective.py              # Φ=T̂^wT·M̂^wM·V̂^wV 评分、三个权重 profile、调度选择
│   ├── pruning.py                # 三条启发式剪枝规则
│   ├── recompute_policy.py       # 两级异构重计算策略分配（硬可行性 + ρ 边际交换率）
│   └── e2e_performance.py        # 端到端时间/显存估计（1F1B/interleaved + 哈希缓存）
├── simulation/
│   └── expand_cluster.py         # 20/40/60 卡模拟集群数据扩展（论文 §4.5）
├── scripts/
│   ├── env.sh                    # [重建模板] 集群网络环境变量
│   ├── args.sh                   # [重建模板] Megatron 参数组装
│   ├── train_llama2_7b_single_node.sh
│   └── train_llama2_7b_multi_node.sh
└── tools/
    ├── p2p_bw_test.py            # 跨节点 GPU P2P 带宽矩阵测试（NCCL）
    ├── intra_p2p.py              # 节点内 GPU P2P 带宽测试
    └── test_nccl.py              # NCCL 连通性冒烟测试
```

## 运行流程

1. **带宽测量**：`tools/` 下的脚本测量节点内/节点间 P2P 带宽，作为搜索时的
   通信参数输入。
2. **最小集合采样**：`profiling/run_sweep_*.py` 枚举关键参数子集，逐配置调用
   `scripts/train_llama2_7b_*.sh` 启动真实训练，每个配置只跑若干步；
   训练循环（`megatron_patch/training.py`）把每 rank 的配置、TFLOP/s、
   峰值显存写入 CSV。
3. **模型训练**：`modeling/train_predictor.py` 读入采样 CSV，做线性插值数据
   增强后训练 XGBoost 时间/显存预测器，导出 `.pkl`。
4. **配置搜索**：`search/search_configs.py` 加载双输出预测器，枚举候选配置，
   经三条启发式规则剪枝后逐配置分配异构重计算策略、选择流水线调度方案，
   按 Φ=T̂^wT·M̂^wM·V̂^wV 加权乘积评分排序，输出到 CSV：

   ```bash
   cd search
   python search_configs.py --model ../modeling/xgb_dual_aug.pkl --profile auto
   ```

   `--profile` 可选 `default`(0.4,0.2,0.4) / `safety_first`(0.3,0.5,0.2) /
   `throughput_first`(0.7,0.2,0.1)；`auto` 按显存余量与集群异构指数自动选择
   （规则见 §3.1.2）。集群拓扑与各设备显存容量/算力在文件顶部 `CLUSTER`
   字典中配置。

## 环境变量

| 变量 | 含义 |
|---|---|
| `MEMX_WORKSPACE` | 工作目录（代码、数据、日志的根路径） |
| `MEMX_SYNC_DIR` | 多节点采样时的共享同步目录（NFS 等） |
| `DATA_PARENT_PATH` | RedPajama 数据集路径 |
| `MASTER_ADDR` / `MASTER_PORT` | 分布式训练主节点地址 |
| `RESULT_RECOMPUTE_CSV` | profiling 结果 CSV 输出路径 |

## 说明

- 设备类型字符串（`v100` / `t4` / `a100` / `vendor-b` / `vendor-c`）仅作为
  分类特征参与建模，`vendor-b` / `vendor-c` 为匿名化的非 NVIDIA 厂商。
- `profiling/megatron_patch/training.py` 以补丁形式提供，基于 Megatron-LM
  （Apache-2.0）修改，其中 `megatron_vendorb.*`、`torch_vendorb.*` 等导入来自
  厂商内部分支，公开环境中无法直接运行，仅供参考埋点位置（搜索 `[MemX]` 标记）。
- `scripts/env.sh` 与 `scripts/args.sh` 为重建模板（原文件未归档），使用前请核对。
- 预测模型权重（`xgb_*.pkl`）未包含在仓库中，可用 `modeling/train_predictor.py`
  基于采样数据自行训练，产物 `xgb_dual_aug.pkl` 为时间+显存双输出预测器。

## License

TBD（注意：`profiling/megatron_patch/training.py` 衍生自 Apache-2.0 的
Megatron-LM，发布时需保留其版权声明）
