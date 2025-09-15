#!/bin/bash

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_ASYNC_ERROR_HANDLING=1

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=${master_addr:-"127.0.0.1"}
export CURRENT_RANK=${SLURM_PROCID:-"0"}
worker_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')
n_node=${SLURM_JOB_NUM_NODES:-1}

echo "MASTER_ADDR="$MASTER_ADDR
echo "JobID: $SLURM_JOB_ID | Full list: $worker_list"

global_bs=${BATCH_SIZE:-64}
acc_step=${ACC_STEP:-1}
bs=$((global_bs / n_node / acc_step))

echo "PER_DEVICE_TRAIN_BATCH_SIZE="$bs

torchrun --nnodes=$n_node --nproc_per_node=2 --master_port=25001 \
    --master_addr $MASTER_ADDR --node_rank=$SLURM_PROCID \
    vila_u/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --version v1 \
    --data_mixture debias_gender \
    --chunk_sampler True \
    --mm_projector mlp2x_gelu \
    --tune_mm_projector False \
    --tune_language_model True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --mm_use_vi_start_end True \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir ./checkpoints/vila-u-lora \
    --num_train_epochs 3 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $acc_step \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --vflan_no_system_prompt True \
    --report_to wandb \
    --use_peft True

CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/occupation.json \
    --num_imgs_per_prompt 160 \
    --save_dir /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images/test_prompts_occupation_understanding/ \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora
