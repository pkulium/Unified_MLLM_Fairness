# finetune

CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/occupation_w_style_and_context.json \
    --save_dir=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images-ablation/test_prompts_occupation_w_style_and_context_sft/ \
    --num_imgs_per_prompt=160 \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora-generation-v6

CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/personal_descriptor.json \
    --save_dir=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images-ablation/test_prompts_personal_descriptor_sft/ \
    --num_imgs_per_prompt=160 \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora-generation-v6

CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/sports.json \
    --save_dir=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images-ablation/test_prompts_sports_sft/ \
    --num_imgs_per_prompt=160 \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora-generation-v6

CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/LAION-aesthetics-V2-occupation-related.json \
    --save_dir=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images-ablation/test_prompts_format_sft/ \
    --num_imgs_per_prompt=160 \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora-generation-v6

CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/occupation_w_style_and_context.json \
    --save_dir=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images-ablation/test_prompts_occupation_w_style_and_context_sft_race/ \
    --num_imgs_per_prompt=160 \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora-generation-race-v6

CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/personal_descriptor.json \
    --save_dir=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images-ablation/test_prompts_personal_descriptor_sft_race/ \
    --num_imgs_per_prompt=160 \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora-generation-race-v6

CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/sports.json \
    --save_dir=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images-ablation/test_prompts_sports_sft_race/ \
    --num_imgs_per_prompt=160 \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora-generation-race-v6


CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/LAION-aesthetics-V2-occupation-related.json \
    --save_dir=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images-ablation/test_prompts_format_sft_race/ \
    --num_imgs_per_prompt=160 \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora-generation-race-v6

CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/occupation_w_style_and_context.json \
    --save_dir=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images-ablation/test_prompts_occupation_w_style_and_context_sft_mix/ \
    --num_imgs_per_prompt=160 \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora-generation-mix-v6

CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/personal_descriptor.json \
    --save_dir=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images-ablation/test_prompts_personal_descriptor_sft_mix/ \
    --num_imgs_per_prompt=160 \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora-generation-mix-v6

CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/sports.json \
    --save_dir=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images-ablation/test_prompts_sports_sft_mix/ \
    --num_imgs_per_prompt=160 \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora-generation-mix-v6


CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/LAION-aesthetics-V2-occupation-related.json \
    --save_dir=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images-ablation/test_prompts_format_sft_mix/ \
    --num_imgs_per_prompt=160 \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora-generation-mix-v6

bpo-gender
CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/occupation_w_style_and_context.json \
    --save_dir=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images-ablation/test_prompts_occupation_w_style_and_context_bpo/ \
    --num_imgs_per_prompt=160 \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora-generation-v4.20 

CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/personal_descriptor.json \
    --save_dir=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images-ablation/test_prompts_personal_descriptor_bpo/ \
    --num_imgs_per_prompt=160 \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora-generation-v4.20 

CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/sports.json \
    --save_dir=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images-ablation/test_prompts_sports_bpo/ \
    --num_imgs_per_prompt=160 \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora-generation-v4.20 


CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/LAION-aesthetics-V2-occupation-related.json \
    --save_dir=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images-ablation/test_prompts_format_bpo/ \
    --num_imgs_per_prompt=160 \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora-generation-v4.20 

# bpo-mix

CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/occupation_w_style_and_context.json \
    --save_dir=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images-ablation/test_prompts_occupation_w_style_and_context_bpo_mix/ \
    --num_imgs_per_prompt=160 \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora-generation-mix-v4.20 

CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/personal_descriptor.json \
    --save_dir=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images-ablation/test_prompts_personal_descriptor_bpo_mix/ \
    --num_imgs_per_prompt=160 \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora-generation-mix-v4.20 

CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/sports.json \
    --save_dir=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images-ablation/test_prompts_sports_bpo_mix/ \
    --num_imgs_per_prompt=160 \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora-generation-mix-v4.20 

CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/LAION-aesthetics-V2-occupation-related.json \
    --save_dir=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images-ablation/test_prompts_format_bpo_mix/ \
    --num_imgs_per_prompt=160 \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora-generation-mix-v4.20

# bpo-race   

CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/occupation_w_style_and_context.json \
    --save_dir=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images-ablation/test_prompts_occupation_w_style_and_context_bpo_race/ \
    --num_imgs_per_prompt=160 \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora-generation-race-v4.20 

CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/personal_descriptor.json \
    --save_dir=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images-ablation/test_prompts_personal_descriptor_bpo_race/ \
    --num_imgs_per_prompt=160 \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora-generation-race-v4.20 

CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/sports.json \
    --save_dir=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images-ablation/test_prompts_sports_bpo_race/ \
    --num_imgs_per_prompt=160 \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora-generation-race-v4.20 



CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/LAION-aesthetics-V2-occupation-related.json \
    --save_dir=/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images-ablation/test_prompts_format_bpo_race/ \
    --num_imgs_per_prompt=160 \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora-generation-race-v4.20