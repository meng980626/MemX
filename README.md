# MemX: Memory-Aware Automatic Parallelism for Cross-Vendor Heterogeneous GPU Training

MemX is a memory-aware automatic parallelism system for cross-vendor heterogeneous
GPU clusters. It jointly optimizes training throughput, peak memory usage, and the
cross-device memory variance during configuration search, and supports per-stage
heterogeneous recomputation strategies and non-uniform pipeline layer partitioning.
Its lightweight performance modeling module samples a minimal set of configurations,
augments them through validated linear interpolation, and trains an XGBoost
predictor that provides millisecond-level performance prediction.

This repository contains all the peripheral code used in the paper's experiments:
profiling/sampling, performance modeling, configuration search, cluster-scale
simulation, and bandwidth measurement tools.

## Repository Layout

```
MemX/
├── profiling/                    # Minimal-set profiler (Section 3.2.1)
│   ├── run_sweep_single_node.py  # Single-node config sampling (key parameter
│   │                             #   identification + boundary-midpoint sampling)
│   ├── run_sweep_multi_node.py   # Multi-node heterogeneous cluster sampling
│   │                             #   (cross-node barrier sync, resume support)
│   └── megatron_patch/
│       ├── csv_writer.py         # Distributed CSV collection inside the training
│       │                         #   process (all-gather aggregation)
│       └── training.py           # Instrumented Megatron-LM training loop (patch)
├── modeling/                     # Performance prediction model (Section 3.2.2)
│   ├── train_predictor.py        # Linear-interpolation data augmentation +
│   │                             #   XGBoost time/memory model training
│   ├── dual_predictor.py         # Dual-output predictor wrapper (one inference
│   │                             #   call returns both time and memory)
│   └── legacy_random_forest.py   # Early random forest version (archived; do not use)
├── search/                       # Configuration search (Sections 3.1.1/3.1.2, Algorithm 1)
│   ├── search_configs.py         # Main entry: enumerate -> prune -> recompute policy
│   │                             #   -> schedule selection -> Phi ranking
│   ├── objective.py              # Phi = T^wT * M^wM * V^wV scoring, three weight
│   │                             #   profiles, schedule selection
│   ├── pruning.py                # The three heuristic pruning rules
│   ├── recompute_policy.py       # Two-level heterogeneous recomputation assignment
│   │                             #   (hard feasibility + marginal exchange rate rho)
│   └── e2e_performance.py        # End-to-end time/memory estimation
│                                 #   (1F1B / interleaved + hash-signature cache)
├── simulation/
│   └── expand_cluster.py         # 20/40/60-GPU simulated cluster expansion (Section 4.5)
├── scripts/
│   ├── env.sh                    # [RECONSTRUCTED TEMPLATE] cluster network env vars
│   ├── args.sh                   # [RECONSTRUCTED TEMPLATE] Megatron argument assembly
│   ├── train_llama2_7b_single_node.sh
│   └── train_llama2_7b_multi_node.sh
├── tools/
│   ├── p2p_bw_test.py            # Cross-node GPU P2P bandwidth matrix test (NCCL)
│   ├── intra_p2p.py              # Intra-node GPU P2P bandwidth test
│   └── test_nccl.py              # NCCL connectivity smoke test
└── data/                         # Three anonymized profiling CSVs (sample data)
```

## Workflow

1. **Bandwidth measurement**: the scripts under `tools/` measure intra-node and
   cross-node P2P bandwidth, which the search uses as communication parameters.
2. **Minimal-set profiling**: `profiling/run_sweep_*.py` enumerates the key
   parameter subset and launches real training runs via
   `scripts/train_llama2_7b_*.sh`, running only a few iterations per
   configuration; the training loop (`megatron_patch/training.py`) writes each
   rank's configuration, TFLOP/s, and peak memory to CSV.
3. **Model training**: `modeling/train_predictor.py` loads the profiling CSV,
   performs linear-interpolation data augmentation, and trains the XGBoost
   time/memory predictors, exporting a dual-output predictor `xgb_dual_aug.pkl`.
4. **Configuration search**: `search/search_configs.py` loads the dual-output
   predictor, enumerates candidate configurations, prunes them with the three
   heuristic rules, assigns per-stage heterogeneous recomputation strategies,
   selects the pipeline schedule, and ranks the survivors by the weighted-product
   objective Phi = T^wT * M^wM * V^wV:

## Environment Variables

| Variable | Meaning |
|---|---|
| `MEMX_WORKSPACE` | Working directory (root for code, data, and logs) |
| `MEMX_SYNC_DIR` | Shared synchronization directory for multi-node profiling (e.g., NFS) |
| `DATA_PARENT_PATH` | Path to the RedPajama dataset |
| `MASTER_ADDR` / `MASTER_PORT` | Rendezvous address of the distributed training master |
| `RESULT_RECOMPUTE_CSV` | Output path of the profiling result CSV |

## Notes

- Device-type strings (`v100` / `t4` / `a100` / `vendor-b` / `vendor-c`) are used
  only as categorical features for modeling; `vendor-b` and `vendor-c` are
  anonymized non-NVIDIA vendors.
- `profiling/megatron_patch/training.py` is provided as a patch based on
  Megatron-LM (Apache-2.0). Imports such as `megatron_vendorb.*` and
  `torch_vendorb.*` come from a vendor-internal fork and cannot run in a public
  environment; the file is intended as a reference for the instrumentation
  points (search for the `[MemX]` markers).
- `scripts/env.sh` and `scripts/args.sh` are reconstructed templates (the
  original files were not archived); please verify them before use.
- Predictor weights (`xgb_*.pkl`) are not included in the repository; you can
  retrain them from the profiling data with `modeling/train_predictor.py`,
  which produces `xgb_dual_aug.pkl` (the time + memory dual-output predictor).

