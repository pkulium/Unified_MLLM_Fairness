import json
import os
from openai import OpenAI
import openai
import time

# Set your OpenAI API key securely using environment variables

import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_prompt(prompt_id, prompt, images_prefix, images_number, variants, train_prompts_augumented):
    """
    Processes a single prompt: paraphrases it and prepares data items.

    Args:
        prompt_id (int): The ID of the prompt.
        prompt (str): The original prompt text.
        images_prefix (str): The prefix path for images.
        images_number (int): Number of paraphrased prompts per original prompt.
        variants (list): List of variants (e.g., ["male", "female"]).

    Returns:
        list: A list of data items for this prompt.
    """
    print(f"Paraphrasing prompt {prompt_id + 1}...")
    paraphrased_prompts = train_prompts_augumented[prompt_id]
    
    data_items = []
    for variant in variants:
        if variant == 'male' or variant == 'female':
            images_number = 10
        else:
            images_number = 5
        for image_id in range(images_number):
            # Ensure we don't exceed the number of paraphrased prompts
            if image_id >= len(paraphrased_prompts):
                current_prompt = paraphrased_prompts[-1]
            else:
                current_prompt = paraphrased_prompts[image_id]
            
            image_path = f"{images_prefix}/{variant}/prompt_{prompt_id}/img_{image_id}.jpg" 
            if not os.path.exists(image_path):
                continue  # Skip if the image does not exist
            
            data_item = {
                "__key__": f"{prompt_id}_{variant}_{image_id}",
                ".jpg": image_path,
                ".json": {
                    "sharegpt4v": current_prompt
                }
            }
            data_items.append(data_item)
    
    return data_items

def main():
    prompts_path = "/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/occupation.json"

    with open(prompts_path, 'r') as f:
        experiment_data = json.load(f)

    train_prompts = experiment_data.get("train_prompts", [])
    train_prompts_augumented = experiment_data.get("train_prompts_augumented", []) 

    images_prefix = '/scratch/bcey/hchen10/pkulium/code/VLM_Bias/SD/train_prompts_occupation'
    images_number = 10  # Number of paraphrased prompts per original prompt

    variants = ['male', 'female', 'asian', 'black', 'indian', 'white']

    data_list = []

    max_workers = 16  # Adjust based on your system's capabilities and API rate limits

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all prompts to the executor
        future_to_prompt = {
            executor.submit(process_prompt, prompt_id, prompt, images_prefix, images_number, variants, train_prompts_augumented): prompt_id
            for prompt_id, prompt in enumerate(train_prompts)
        }

        for future in as_completed(future_to_prompt):
            prompt_id = future_to_prompt[future]
            try:
                data_items = future.result()
                data_list.extend(data_items)
                print(f"Completed processing prompt {prompt_id + 1}/{len(train_prompts)}")
            except Exception as e:
                print(f"Error processing prompt {prompt_id + 1}: {e}")

    # Optionally, sort data_list or perform additional processing here

    output_path = 'debias_mix_generation.json'
    with open(output_path, 'w') as json_file:
        json.dump(data_list, json_file, indent=4)

    print("Paraphrasing and data preparation complete.")

if __name__ == "__main__":
    main()
