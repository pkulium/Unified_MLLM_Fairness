import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_prompt(args) -> List[Dict]:
    """Process a single prompt and generate data items for all its images."""
    prompt_id, prompt, train_prompts_augumented, images_prefix, images_number, variants = args
    data_items = []
    
    for image_id in range(images_number):
        image_paths = []
        
        for variant in variants:
            image_path = f"{images_prefix}/{variant}/prompt_{prompt_id}/img_{image_id}.jpg"
            
            if not os.path.exists(image_path):
                continue
            image_paths.append(image_path)
        
        if len(image_paths) == 1:
            continue
            
        for i in range(len(image_paths)):
            data_item = {
                "__key__": None,  # Will be assigned later
                ".jpg": [image_paths[i], image_paths[i + 1]] if i != len(image_paths) - 1 else [image_paths[i - 1], image_paths[0]],
                ".json": {
                    "sharegpt4v": train_prompts_augumented[prompt_id][image_id]
                }
            }
            data_items.append(data_item)
        

        image_paths = []
        
        extra_variants = ['male', 'female']
        for variant in extra_variants:
            image_path = f"{images_prefix}/{variant}/prompt_{prompt_id}/img_{image_id}.jpg"
            
            if not os.path.exists(image_path):
                continue
            image_paths.append(image_path)
        
        if len(image_paths) == 1:
            continue
            
        for i in range(len(image_paths) - 1):
            data_item = {
                "__key__": None,  # Will be assigned later
                ".jpg": [image_paths[i], image_paths[i + 1]],
                ".json": {
                    "sharegpt4v": train_prompts_augumented[prompt_id][image_id]
                }
            }
            data_items.append(data_item)
    return data_items

def main():
    # Load data
    prompts_path = "/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/occupation.json"
    with open(prompts_path, 'r') as f:
        experiment_data = json.load(f)

    train_prompts = experiment_data.get("train_prompts", [])
    train_prompts_augumented = experiment_data.get("train_prompts_augumented", [])

    images_prefix = '/scratch/bcey/hchen10/pkulium/code/VLM_Bias/SD/train_prompts_occupation'
    images_number = 5
    variants = ['asian', 'black', 'indian', 'white']

    # Prepare arguments for parallel processing
    process_args = [
        (prompt_id, prompt, train_prompts_augumented, images_prefix, images_number, variants)
        for prompt_id, prompt in enumerate(train_prompts)
    ]

    # Process prompts in parallel
    data_list = []
    with ThreadPoolExecutor(max_workers = 16) as executor:
        logging.info(f"Starting parallel processing of {len(process_args)} prompts...")
        futures = executor.map(process_prompt, process_args)
        
        # Collect results and assign data IDs
        data_id = 0
        for result in futures:
            for data_item in result:
                data_item["__key__"] = data_id
                data_id += 1
                data_list.append(data_item)

    logging.info(f"Processing complete. Generated {len(data_list)} data items.")

    # Save results
    output_path = 'debias_mix_generation_bpo.json'
    with open(output_path, 'w') as json_file:
        json.dump(data_list, json_file, indent=4)
    logging.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()