import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# Paths and constants
prompts_path = "/scratch/bcey/hchen10/pkulium/code/VLM_Bias/finetune-fair-diffusion/data/1-prompts/occupation.json"
images_prefix = '/scratch/bcey/hchen10/pkulium/code/VLM_Bias/SD/train_prompts_occupation'
# List the variants once so they can be used in the worker function
variants = ['male', 'female', 'asian', 'black', 'indian', 'white']

# Load experiment data
with open(prompts_path, 'r') as f:
    experiment_data = json.load(f)

train_prompts = experiment_data.get("train_prompts", [])

def process_prompt(item):
    """
    Process one prompt (identified by prompt_id and its text) into a list of data items.
    For each prompt the function iterates over the variants and available images,
    checks if the file exists, and returns a list of valid data items.
    """
    prompt_id, prompt = item
    items = []
    # Process each variant
    for variant in variants:
        # Use different image counts based on the variant
        if variant in ['male', 'female']:
            images_number = 10
        else:
            images_number = 5
        for image_id in range(images_number):
            image_path = f"{images_prefix}/{variant}/prompt_{prompt_id}/img_{image_id}.jpg"
            if not os.path.exists(image_path):
                continue
            # Create a temporary data_item. We leave the "id" as None (or you can generate one based on the input)
            data_item = {
                "id": None,  # Will be assigned later after merging results.
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
            items.append(data_item)
    return items

def main():
    # Collect all data items in parallel
    all_data = []
    # Prepare items to process: a list of tuples (prompt_id, prompt)
    items_to_process = list(enumerate(train_prompts))
    
    # Use ProcessPoolExecutor to execute in parallel
    with ProcessPoolExecutor() as executor:
        # Map our process_prompt function to each item.
        # This will return an iterator of lists.
        futures = executor.map(process_prompt, items_to_process)
        
        # Merge the results (each future returns a list of data items).
        for result in futures:
            all_data.extend(result)
    
    # Now assign unique incremental IDs to each data item.
    for idx, data_item in enumerate(all_data):
        data_item["id"] = str(idx)
    
    # Save the merged data to a JSON file.
    with open('debias_mix.json', 'w') as json_file:
        json.dump(all_data, json_file, indent=4)

if __name__ == "__main__":
    main()
