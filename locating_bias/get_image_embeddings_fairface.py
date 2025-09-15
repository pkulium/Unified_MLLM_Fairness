import argparse
import os
import vila_u
import torch
import logging
from tqdm import tqdm
from datasets import load_dataset

import argparse
import os
import vila_u
import torch
import logging
from tqdm import tqdm
from datasets import load_dataset

def process_and_save_embeddings_batch(model, dataset, save_dir, batch_size=100):
    male_embeddings = {'image_paths': [], 'embeddings': []}
    female_embeddings = {'image_paths': [], 'embeddings': []}
    total_batches = (len(dataset) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(dataset), batch_size), total=total_batches, desc="Processing batches"):
        batch = dataset[i:i+batch_size]
        
        with torch.no_grad():
            for j in range(len(batch['image'])):
                try:
                    # Access the correct fields from the batch dictionary
                    # image = vila_u.Image(batch['image'][j])
                    image = batch['image'][j]
                    image_feature, _ = model.get_image_embeddings([image, ""])
                    embedding_tensor = image_feature.cpu().float().detach()
                    
                    # Gender is encoded as 0 for Male, 1 for Female
                    if batch['gender'][j] == 0:  # Male
                        male_embeddings['image_paths'].append(f"image_{i+j}")
                        male_embeddings['embeddings'].append(embedding_tensor)
                    else:  # Female
                        female_embeddings['image_paths'].append(f"image_{i+j}")
                        female_embeddings['embeddings'].append(embedding_tensor)
                except Exception as e:
                    logging.error(f"Error processing image {i+j}: {e}")

        # Save intermediate results
        if male_embeddings['embeddings']:
            male_tensor = torch.stack(male_embeddings['embeddings'])
            torch.save({
                'image_paths': male_embeddings['image_paths'],
                'embeddings': male_tensor
            }, os.path.join(save_dir, 'male_embeddings.pt'))
        
        if female_embeddings['embeddings']:
            female_tensor = torch.stack(female_embeddings['embeddings'])
            torch.save({
                'image_paths': female_embeddings['image_paths'],
                'embeddings': female_tensor
            }, os.path.join(save_dir, 'female_embeddings.pt'))

        logging.info(f"Saved batch {i//batch_size + 1}/{total_batches}")

    logging.info(f"Completed saving all embeddings to {save_dir}")

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Generate and save image embeddings for FairFace dataset.")
    parser.add_argument("--model_path", type=str, default="/scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/ckpt")
    parser.add_argument("--save_path", type=str, default="/scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/fairface_embeddings")
    parser.add_argument("--batch_size", type=int, default=100, help="Number of images to process in each batch")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    logging.info(f"Save directory: {args.save_path}")

    try:
        model = vila_u.load(args.model_path)
        logging.info(f"Loaded model from {args.model_path}")
    except Exception as e:
        raise ValueError(f"Failed to load model from {args.model_path}: {e}")

    # Load FairFace dataset
    dataset = load_dataset("HuggingFaceM4/FairFace", "1.25", split="train")
    logging.info("Loaded FairFace dataset")

    process_and_save_embeddings_batch(model, dataset, args.save_path, args.batch_size)

if __name__ == "__main__":
    main()
