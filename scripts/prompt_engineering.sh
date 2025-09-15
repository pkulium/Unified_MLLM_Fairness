# Baseline Model with suffix prompt
CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/occupation.json \
    --num_imgs_per_prompt 160 \
    --save_dir /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images/test_prompts_occupation_suffix/ \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --debias_suffix ", any gender" 


# Baseline Model with suffix prompt
CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/occupation.json \
    --num_imgs_per_prompt 160 \
    --save_dir /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images/test_prompts_occupation_suffix_race/ \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --debias_suffix ", any race" 


# Baseline Model with suffix prompt
CUDA_VISIBLE_DEVICES=0 python gen-images.py \
    --prompts_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/occupation.json \
    --num_imgs_per_prompt 160 \
    --save_dir /scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/finetuned-vila-generated-images/test_prompts_occupation_suffix_mix/ \
    --model_path /scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt \
    --debias_suffix "any gender, any race" 