#!/usr/bin/env bash

set -eo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate orca
set -u

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

seed=1
lr="2e-4"
global_batch_size="${GLOBAL_BATCH_SIZE:-32}"
epoch=5
max_steps="${MAX_STEPS:--1}"
input_steps=1
predict_steps=1
max_t=6
save_eval_steps="${SAVE_EVAL_STEPS:-2000}"
use_pretrained="${USE_PRETRAINED:-0}"
train_micro_batch_size="${TRAIN_MICRO_BATCH_SIZE:-4}"
eval_per_device_batch_size="${EVAL_PER_DEVICE_BATCH_SIZE:-$train_micro_batch_size}"
use_bf16="${USE_BF16:-True}"
use_fp16="${USE_FP16:-False}"
use_tf32="${USE_TF32:-True}"
gradient_checkpointing="${GRADIENT_CHECKPOINTING:-True}"
pytorch_cuda_alloc_conf="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
ddp_find_unused_parameters="${DDP_FIND_UNUSED_PARAMETERS:-True}"

input_var_list=(so thetao tos uo vo zos hfds mlotst rsntds sob sos tob wfo wo)
atmo_var_list=(tauu tauv)

data_dir=/mnt/data/zhu.yishun/ORCA-DL-main/data/1840-2014
soda_dir=/mnt/data/zhu.yishun/ORCA-DL-main/data/valid_test_data/SODA2
oras5_dir=/mnt/data/zhu.yishun/ORCA-DL-main/data/valid_test_data/ORAS5
model_config_path="$repo_root/model_config.json"
pretrained_model_path="$repo_root/ori-model"

timestamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${OUTPUT_DIR:-$repo_root/output-pretrain-e${epoch}-bs${global_batch_size}-${timestamp}}"
logging_dir="$output_dir/log"

if [[ ! -d "$data_dir" ]]; then
    echo "训练数据目录不存在: $data_dir" >&2
    exit 1
fi

for valid_dir in "$soda_dir" "$oras5_dir"; do
    if [[ ! -d "$valid_dir" ]]; then
        echo "验证数据目录不存在: $valid_dir" >&2
        exit 1
    fi
done

if [[ ! -f "$model_config_path" ]]; then
    echo "模型配置不存在: $model_config_path" >&2
    exit 1
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
    nproc_per_node="${#visible_gpus[@]}"
else
    nproc_per_node="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
fi

if [[ "$nproc_per_node" -lt 1 ]]; then
    echo "未检测到可用 GPU。" >&2
    exit 1
fi

if (( train_micro_batch_size < 1 )); then
    echo "TRAIN_MICRO_BATCH_SIZE 必须 >= 1，当前值: ${train_micro_batch_size}" >&2
    exit 1
fi

if (( global_batch_size % (train_micro_batch_size * nproc_per_node) != 0 )); then
    echo "global_batch_size=${global_batch_size} 不能被 train_micro_batch_size=${train_micro_batch_size} * GPU 数 ${nproc_per_node} 整除。" >&2
    exit 1
fi

per_device_batch_size="$train_micro_batch_size"
gradient_accumulation_steps=$((global_batch_size / (train_micro_batch_size * nproc_per_node)))
dist_port=$((12345 + RANDOM % 20000))
dataloader_num_workers="${DATALOADER_NUM_WORKERS:-4}"
omp_num_threads="${OMP_NUM_THREADS:-4}"
mkl_num_threads="${MKL_NUM_THREADS:-$omp_num_threads}"
torchrun_log_dir="${TORCHRUN_LOG_DIR:-$repo_root/torchrun-logs/${timestamp}}"
torchrun_tee="${TORCHRUN_TEE:-0}"

mkdir -p "$torchrun_log_dir"

cmd=(
    env
    OMP_NUM_THREADS="$omp_num_threads"
    MKL_NUM_THREADS="$mkl_num_threads"
    PYTHONUNBUFFERED=1
    PYTHONFAULTHANDLER=1
    PYTORCH_CUDA_ALLOC_CONF="$pytorch_cuda_alloc_conf"
    torchrun
    --standalone
    --log-dir "$torchrun_log_dir"
    --tee "$torchrun_tee"
    --nproc_per_node="$nproc_per_node"
    train.py
    --model_config_path "$model_config_path"
    --max_t "$max_t"
    --atmo_var_list "${atmo_var_list[@]}"
    --atmo_dims 2
    --do_train
    --do_eval
    --dist_port "$dist_port"
    --data_dir "$data_dir"
    --valid_data_dir "$soda_dir" "$oras5_dir"
    --input_var_list "${input_var_list[@]}"
    --input_steps "$input_steps"
    --predict_steps "$predict_steps"
    --output_dir "$output_dir"
    --seed "$seed"
    --report_to none
    --log_level info
    --logging_dir "$logging_dir"
    --logging_steps 100
    --log_on_each_node False
    --save_strategy steps
    --save_steps "$save_eval_steps"
    --save_total_limit 3
    --ddp_find_unused_parameters "$ddp_find_unused_parameters"
    --num_train_epochs "$epoch"
    --max_steps "$max_steps"
    --per_device_train_batch_size "$per_device_batch_size"
    --per_device_eval_batch_size "$eval_per_device_batch_size"
    --gradient_accumulation_steps "$gradient_accumulation_steps"
    --dataloader_num_workers "$dataloader_num_workers"
    --gradient_checkpointing "$gradient_checkpointing"
    --bf16 "$use_bf16"
    --fp16 "$use_fp16"
    --tf32 "$use_tf32"
    --fsdp ""
    --learning_rate "$lr"
    --weight_decay 0.1
    --max_grad_norm 0.0
    --adam_beta1 0.9
    --adam_beta2 0.95
    --adam_epsilon 1e-6
    --lr_scheduler_type cosine
    --warmup_ratio 0.1
    --evaluation_strategy steps
    --eval_steps "$save_eval_steps"
    --load_best_model_at_end True
    --end_year 2010
)

if [[ "$use_pretrained" == "1" && -d "$pretrained_model_path" && -f "$pretrained_model_path/pytorch_model.bin" ]]; then
    cmd+=(--model_path "$pretrained_model_path")
fi

echo "repo_root=$repo_root"
echo "output_dir=$output_dir"
echo "nproc_per_node=$nproc_per_node"
echo "global_batch_size=$global_batch_size"
echo "max_steps=$max_steps"
echo "train_micro_batch_size=$train_micro_batch_size"
echo "per_device_train_batch_size=$per_device_batch_size"
echo "per_device_eval_batch_size=$eval_per_device_batch_size"
echo "gradient_accumulation_steps=$gradient_accumulation_steps"
echo "learning_rate=$lr"
echo "use_pretrained=$use_pretrained"
echo "dataloader_num_workers=$dataloader_num_workers"
echo "OMP_NUM_THREADS=$omp_num_threads"
echo "MKL_NUM_THREADS=$mkl_num_threads"
echo "PYTORCH_CUDA_ALLOC_CONF=$pytorch_cuda_alloc_conf"
echo "bf16=$use_bf16"
echo "fp16=$use_fp16"
echo "tf32=$use_tf32"
echo "gradient_checkpointing=$gradient_checkpointing"
echo "ddp_find_unused_parameters=$ddp_find_unused_parameters"
echo "torchrun_log_dir=$torchrun_log_dir"
echo "torchrun_tee=$torchrun_tee"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'DRY_RUN command:\n'
    printf '%q ' "${cmd[@]}"
    printf '\n'
    exit 0
fi

"${cmd[@]}"
