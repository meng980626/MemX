#!/bin/bash
_THIS_DIR=$(dirname "$0")

select_idle_gpus(){
    local need=$1
    local free_list="" found=0

    # 1. 按行解析 vendor-smi 输出
    while read -r idx; do
        free_list="$free_list,$idx"
        ((found++))
        [[ $found -eq $need ]] && break
    done < <(
        vendor-smi | awk '
            /^\|[[:space:]]+[0-9]+/{
              idx=$2;  split($6,a,"/");
              used=a[1]; total=a[2];
              if (used*100/total <= 10) print idx
            }'
    )

    free_list=${free_list#,}
    [[ $found -lt $need ]] && { echo "[ERROR] 仅找到 $found 张空闲卡（需要 $need 张）" >&2; exit 1; }

    export VENDOR_B_VISIBLE_DEVICES=$free_list
    echo "[INFO] 已设置 VENDOR_B_VISIBLE_DEVICES=$free_list"
}

# Distributed Config
NUM_NODES=3
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=6000

# Model Config
MODEL_SIZE=7
DROP_OUT=0.0
STEP=10

# Parallel Strategy Config
TP=${TP_SIZE:-4}
PP=${PP_SIZE:-5}
DP=${DP_SIZE:-1}

NUM_HEAD=${NUM_ATTENTION_HEADS:-64}
NUM_QUERY_GROUP=${NUM_QUERY_GROUPS:-2}
NUM_LAYERS=${NUM_LAYERS:-32}
SEQ_LENGTH=${SEQ_LENGTH:-1024}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-4096}

DTYPE=${DTYPE:-bf16}
RECOMPUTE_GRANULARITY=${RECOMPUTE_GRANULARITY:-'selective'}
RECOMPUTE_METHOD=${RECOMPUTE_METHOD:-'block'}
RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-1}


# export NUM_LAYERS_PER_STAGE=3,5,22 #在run_sweep脚本中设置
export NUM_LAYERS_PER_STAGE=${NUM_LAYERS_PER_STAGE:-"21,2,2,3,4"}
# select_idle_gpus $GPUS_PER_NODE

# WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2} 
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}

# Enviroment
source $_THIS_DIR/env.sh
# Code Path

WORKSPACE=${MEMX_WORKSPACE:-/path/to/workspace}
MEGATRON_HOME=$WORKSPACE/megatron-lm
MEGATRON_VENDORB_HOME=$WORKSPACE/megatron-lm-vendorb
if [ "$USE_VENDOR_B" = "true" ]; then
    LOCAL_MEGATRON=$MEGATRON_VENDORB_HOME
    # VENDORB_TORCH_HOME=${VENDORB_TORCH_HOME:-/path/to/vendor-b-torch/packages}
    SCRIPT_FILE=${SCRIPT_FILE:-$MEGATRON_VENDORB_HOME/examples/llama2/model/pretrain_llama.py}
    export PYTHONPATH=$MEGATRON_VENDORB_HOME:$VENDORB_TORCH_HOME:$PYTHONPATH
elif [ "$USE_MX" = "true" ]; then
    LOCAL_MEGATRON=$WORKSPACE/megatron-lm-mx
    MEGATRON_HOME=$WORKSPACE/megatron-lm-mx
    SCRIPT_FILE=${SCRIPT_FILE:-$LOCAL_MEGATRON/pretrain_gpt.py}
else
    LOCAL_MEGATRON=$MEGATRON_HOME
    SCRIPT_FILE=${SCRIPT_FILE:-$MEGATRON_HOME/pretrain_gpt.py}
fi
export PYTHONPATH=$MEGATRON_HOME:$PYTHONPATH


# Data & Vocab Path
DATA_PARENT_PATH=${DATA_PARENT_PATH:-/path/to/RedPajama-Data-1T-Sample}
DATA_PATH="$DATA_PARENT_PATH/redpajama-llama2_text_document"
VOCAB_FILE=${VOCAB_FILE:-$MEGATRON_HOME/examples/llama2/tokenizer}
TOKENIZER_MODEL=$VOCAB_FILE/tokenizer.model

# Log & Tensorboard Path
# DP=$((WORLD_SIZE / TP / PP))
ALL_LOGS_PATH=logs/llama2-7B
mkdir -p ${ALL_LOGS_PATH}
LOGS_PATH=mb${MICRO_BATCH_SIZE}_gb${GLOBAL_BATCH_SIZE}_l${NUM_LAYERS}_tp${TP}_pp${PP}_dp${DP}_n${NUM_NODES}
mkdir -p $ALL_LOGS_PATH/${LOGS_PATH}
timestamp=$(date +%s)
TENSORBOARD_LOGS_PATH=$ALL_LOGS_PATH/${LOGS_PATH}/${timestamp}

# Arguments

## Special Argument for specific device
if [ "$USE_CUDA" = "true" ]; then
    export HGCT_LOCAL_IB_HCAS=mlx5_0,mlx5_1,mlx5_2,mlx5_3
    export CUDA_VISIBLE_DEVICES=4,5,6,7
    NODE_RANK=${NODE_RANK:-0}
    export XCCL_IB_HCA=mlx5_bond_0
    GPUS_PER_NODE=${GPUS_PER_NODE:-4}
    export USE_CUDA=true
    HGCT_HOME=$WORKSPACE/hgct_nv
    SYSSTAT_HOME=$WORKSPACE/hgct_nv/sysstat
    export PYTHONPATH=$HGCT_HOME:$SYSSTAT_HOME:$PYTHONPATH
    CUDA_ARGS="
        --use-flash-attn
        --transformer-impl transformer_engine
        --normalization RMSNorm
        "
    DEVICE_ARGS=$CUDA_ARGS
elif [ "$USE_VENDOR_C" = "true" ]; then
    export HGCT_LOCAL_IB_HCAS=mlx5_2,mlx5_3,mlx5_4,mlx5_5
    TX_ARGS="
        --use-flash-attn
        --transformer-impl local
        --normalization LayerNorm
        --recompute-num-layers 1
        --recompute-granularity=full 
        --recompute-method=uniform 
    "
    DEVICE_ARGS=$TX_ARGS
elif [ "$USE_MX" = "true" ]; then
    export HGCT_LOCAL_IB_HCAS=mlx5_0,mlx5_1,mlx5_3,mlx5_4
    MX_ARGS="
        --use-flash-attn
        --transformer-impl local
        --normalization LayerNorm
        --pipline-num-layers-list 16 16
        --no-masked-softmax-fusion
        --accumulate-bf16
    "
    DEVICE_ARGS=$MX_ARGS
else
    export HGCT_LOCAL_IB_HCAS=mlx5_2,mlx5_4,mlx5_8,mlx5_10
    NODE_RANK=${NODE_RANK:-1}
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
    export USE_VENDOR_B=true
    export SIMULATE_VENDOR_C_ON_B=${SIMULATE_VENDOR_C_ON_B:-'false'}
    export XCCL_IB_HCA=mlx5_2
    HGCT_HOME=$WORKSPACE/hgct_vendorb
    SYSSTAT_HOME=$WORKSPACE/hgct_vendorb/sysstat
    export PYTHONPATH=$HGCT_HOME:$SYSSTAT_HOME:$PYTHONPATH
    CONFIG_PATH=$_THIS_DIR/async_offload_config.yaml
    SUPA_ARGS="
        --normalization RMSNorm
        --use-llama-mlp
        --vendorb-fuse-attention-transform
        --vendorb-fuse-split-qkv
        --vendorb-fuse-rope
        --fused-mlp
        --cross-entropy-loss-fusion
        --vendor-dnn-attention
        --bind-numa-node
        --use-tensor-pool
        --inplace-comm
        --llama2mlp-with-tensor-pool
        --vendorb-fuse-embedding
        --vendorb-fuse-rmsnorm
        --bf16-optimizer-use-flat-buffers
        --sccl-enable-input-buffer-writing
        --pp-comm-directly
        --tp-comm-overlap-rs-row-bpw
        --async-grad-allgather
        --use-commsplit=false
        --transformer-impl local
        "
    # if use recompute
    # SUPA_ARGS="$SUPA_ARGS 
    #             --recompute-num-layers 8
    #             --recompute-granularity=full 
    #             --recompute-method=uniform"
    DEVICE_ARGS=$SUPA_ARGS
fi

source $_THIS_DIR/args.sh

# HGCT Config
export HGCT_DEBUG=INFO
export XCCL_NET=Socket
export HGCT_PERMUTE_SHAPE=0

export XCCL_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME
# hgct local test




echo "[CMD] torchrun ${DISTRIBUTED_ARGS[@]} \"${SCRIPT_FILE}\" \\"
printf "      %s \\\n" "${MODEL_ARGS[@]}" "${TRAINING_ARGS[@]}" "${RECOMPUTE_ARGS[@]}" \
        "${MODEL_PARALLEL_ARGS[@]}" "${DATA_ARGS[@]}" "${EVAL_AND_LOGGING_ARGS[@]}" "${DEVICE_ARGS[@]}"

torchrun ${DISTRIBUTED_ARGS[@]} ${SCRIPT_FILE} \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${RECOMPUTE_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${DEVICE_ARGS[@]} \
    2>&1 | tee -a $ALL_LOGS_PATH/${LOGS_PATH}/${timestamp}.log

set -ex
