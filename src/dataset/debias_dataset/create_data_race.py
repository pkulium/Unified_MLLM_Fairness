import json
prompts_path = "/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/occupation.json"
import os
with open(prompts_path, 'r') as f:
    experiment_data = json.load(f)

train_prompts = experiment_data.get("train_prompts", [])

images_prefix = '/scratch/bcey/hchen10/pkulium/code/VLM_Bias/SD/train_prompts_occupation'
images_number = 10

data_list = []
variants = ['asian', 'black', 'indian', 'white']
data_id = 0
for prompt_id, prompt in enumerate(train_prompts):
    for variant in variants:
        for image_id in range(images_number): 
            image_path = f"{images_prefix}/{variant}/prompt_{prompt_id}/img_{image_id}.jpg" 
            if not os.path.exists(image_path):
                continue
            data_item =  {
                "id": str(data_id),
                "image": image_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": "Give a discription of the image.\n<image>",
                    },
                    {
                        "from": "gpt",
                        "value": prompt,
                    }
                ]
            }
            data_id += 1
            data_list.append(data_item)

with open('debias_race.json', 'w') as json_file:
    json.dump(data_list, json_file, indent=4)
