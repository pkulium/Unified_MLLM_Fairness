# Baseline Model
CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/occupation.json \
    --num_imgs_per_prompt 160 \
    --save_dir /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/original-vila-generated-images/test_prompts_occupation/ \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt

=================================================================================================
# Naive finetuned Model on understanding dataset
CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/occupation.json \
    --num_imgs_per_prompt 160 \
    --save_dir /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images/test_prompts_occupation_understanding/ \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora

# Naive finetuned Model on generation dataset
CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/occupation.json \
    --num_imgs_per_prompt 160 \
    --save_dir /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images/test_prompts_occupation/ \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora-generation

# Finetuned Model on generation dataset with BPO
CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/occupation.json \
    --num_imgs_per_prompt 160 \
    --save_dir /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images/test_prompts_occupation_bpo/ \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora-generation-bpo


# Baseline Model with suffix prompt
CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/occupation.json \
    --num_imgs_per_prompt 160 \
    --save_dir /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images/test_prompts_occupation_suffix/ \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --debias_suffix ", any gender" 



=================================================================================================
# Naive finetuned Model on understanding dataset
CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/occupation.json \
    --num_imgs_per_prompt 160 \
    --save_dir /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images/test_prompts_occupation_understanding_race/ \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora-race

# Naive finetuned Model on generation dataset
CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/occupation.json \
    --num_imgs_per_prompt 160 \
    --save_dir /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images/test_prompts_occupation_race/ \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora-generation-race

# Finetuned Model on generation dataset with BPO
CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/occupation.json \
    --num_imgs_per_prompt 160 \
    --save_dir /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images/test_prompts_occupation_bpo_race/ \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora-generation-bpo-race

# Baseline Model with suffix prompt
CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/occupation.json \
    --num_imgs_per_prompt 160 \
    --save_dir /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images/test_prompts_occupation_suffix_race/ \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --debias_suffix ", any gender" 






=================================================================================================





# Naive finetuned Model on generation dataset more epochs
CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/occupation.json \
    --num_imgs_per_prompt 160 \
    --save_dir /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images/test_prompts_occupation_20epochs/ \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora-generation-epochs20


# Finetuned Model on generation dataset with BPO beta 0
CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/occupation.json \
    --num_imgs_per_prompt 160 \
    --save_dir /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images/test_prompts_occupation_bpo_beta0/ \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora-generation-bpo-beta0

# Finetuned Model on generation dataset with BPO beta 0
CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/occupation.json \
    --num_imgs_per_prompt 160 \
    --save_dir /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images/test_prompts_occupation_bpo_beta025/ \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --adapter_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora-generation-bpo-race-beta025


CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/LAION-aesthetics-V2-occupation-related.json \
    --num_imgs_per_prompt 160 \
    --save_dir /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/original-vila-generated-images/test_prompts_format/ \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt 

# Initialize conda for the shell
eval "$(conda shell.bash hook)"

# Deactivate the current environment
conda deactivate

conda activate /scratch/bcey/hchen10/pkulium/envs/transformer

cd /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion
python eval-generated-images.py \
    --generated_imgs_dir "/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/original-vila-generated-images/test_prompts_format" \
    --save_dir "/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/original-vila-generated-images/test_prompts_format_results"

sh run_post_quality_abalation.sh
