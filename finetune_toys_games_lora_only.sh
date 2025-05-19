#!/usr/bin/env bash
n_proc=8
#model_name="meta-llama/Llama-2-7b-hf"
model_name="meta-llama/Llama-3.2-3B"
#tokenizer_path="./tokenizers/Llama-2-7b-hf/toys/tokenizer"
tokenizer_path="./tokenizers/Llama-3.2-3B/sports/tokenizer"
dataset_path="./dataset/amazon/raw/toys"
num_train_epochs=3
batch_size=2
gradient_accumulation_steps=2
lr=1e-4
seed=0
train_embed_only="false"
#target_modules_string="q_proj v_proj"
target_modules_string="q_proj v_proj k_proj o_proj down_proj gate_proj up_proj"
model_max_length=1024
filter_long_context_examples_arg=False
num_hashes=4
item_nums=11924

if [ ! -z "$1" ]; then
  batch_size=$1
  lr=$2
  num_train_epochs=$3
  adapter_dim=$4
  compression_rate=$5
  num_hashes=$6
fi

effective_batch_size=$((n_proc*gradient_accumulation_steps*batch_size))

dataset_name="llama3.23B-lora-only-noqloraall-ours-toys-${compression_rate}-lora${adapter_dim}-batch${effective_batch_size}-num_hashes${num_hashes}-lr${lr}"
dst_path="./llama3.23B-lora-only-noqloraall-ours-toys-${compression_rate}-lora${adapter_dim}-batch${effective_batch_size}-num_hashes${num_hashes}-lr${lr}"


# convert to array
if [ -n "$target_modules_string" ]; then
    IFS=' ' read -ra target_modules <<< "$target_modules_string"
    target_modules_arg="--target_modules ${target_modules[*]}"
else
    target_modules_arg=""
fi


torchrun --standalone --nproc_per_node ${n_proc} finetune-main-lora-only.py \
    --model_name $model_name \
    --dataset_name $dataset_name \
    --train_embed_only $train_embed_only \
    --dataset_path $dataset_path \
    --tokenizer_path $tokenizer_path \
    --compression_rate $compression_rate \
    --num_hashes $num_hashes \
    --item_nums $item_nums \
    --bf16 True \
    --output_dir $dst_path \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --do_train True \
    --learning_rate $lr \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --lora_dim $adapter_dim \
    --lora_alpha 16 \
    --adapter_type lora \
    --seed $seed \
    --model_max_length $model_max_length \
    --lora_init true \
    $filter_long_context_examples_arg \
    $target_modules_arg