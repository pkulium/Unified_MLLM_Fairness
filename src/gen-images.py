import argparse
import os
import torch
import numpy as np
from tqdm.auto import tqdm
import json
import cv2
import math
import vila_u  # Ensure this module is installed and accessible
DEBUG = False
if DEBUG:
    print('=' * 64)
    print('DEBUG')


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Script to generate images using the vila_u model.")

    # Model and generation arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pretrained vila_u model.",
    )
    parser.add_argument(
        "--prompts_path",
        type=str,
        required=True,
        help="Path to the JSON file containing prompts.",
    )
    parser.add_argument(
        "--num_imgs_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate per prompt.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save the generated images.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=1997,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="The value of temperature for text generation.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.6,
        help="The value of top-p for text generation.",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=3.0,
        help="The value of the classifier free guidance for image generation.",
    )
    parser.add_argument(
        "--video_generation",
        action='store_true',
        help="Flag to indicate video generation instead of image generation.",
    )
    parser.add_argument(
        "--generation_nums",
        type=int,
        default=8,
        help="Number of images/videos to generate per batch per prompt.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=16,
        help="Patch size used in the model.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=384,
        help="Size of the generated images.",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="",
        help="Size of the generated images.",
    )
    parser.add_argument(
        "--debias_suffix",
        type=str,
        default="",
        help="prompt used to debias generation.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def save_image(images, save_dir_prompt_i, img_idx):
    os.makedirs(save_dir_prompt_i, exist_ok=True)
    for i in range(images.shape[0]):
        image = images[i].permute(1, 2, 0)
        image = image.cpu().numpy().astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_save_path = os.path.join(save_dir_prompt_i, f"img_{img_idx + i}.jpg")
        cv2.imwrite(img_save_path, image)


def save_video(response, save_dir_prompt_i, img_idx):
    os.makedirs(save_dir_prompt_i, exist_ok=True)
    for i in range(response.shape[0]):
        video = response[i].permute(0, 2, 3, 1)
        video = video.cpu().numpy().astype(np.uint8)
        video = np.concatenate(video, axis=1)  # Concatenate frames horizontally
        video = cv2.cvtColor(video, cv2.COLOR_RGB2BGR)
        img_save_path = os.path.join(save_dir_prompt_i, f"img_{img_idx + i}.jpg")
        cv2.imwrite(img_save_path, video)


def main(args):
    # Load the base model first (as you already have)
    if args.model_path is not None:
        model = vila_u.load(args.model_path)
    else:
        raise ValueError("No model path provided!")

    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    if args.adapter_path:
        # Load and merge the adapter
        from peft import PeftModel
        
        # Load the base model with the adapter
        model = PeftModel.from_pretrained(
            model,
            args.adapter_path,
            is_trainable=False  # Set to True if you want to continue training
        )

        # Merge adapter weights with the base model
        model = model.merge_and_unload()

        # Set to evaluation mode
        model.eval()
        model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Read prompts from the JSON file
    with open(args.prompts_path, 'r') as f:
        experiment_data = json.load(f)

    test_prompts = experiment_data.get("test_prompts", [])
    if DEBUG:
        test_prompts = experiment_data.get("train_prompts", [])
    
    if not test_prompts:
        raise ValueError("No prompts found in the JSON file under the key 'test_prompts'.")

    if args.debias_suffix:
        # Append the suffix to each prompt
        test_prompts = [prompt + args.debias_suffix for prompt in test_prompts]

    # Set random seed for reproducibility
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Variants for replacements
    replacements = ["person"]
    if DEBUG:
        replacements = ["male", "female", "person", "white", "asian", "black", "indian"]

    # Iterate over each prompt with a progress bar
    for idx, prompt_text in tqdm(
        enumerate(test_prompts),
        total=len(test_prompts),
        desc="Prompts"
    ): 
        # For each prompt, create multiple variants by replacing 'person' with each replacement
        for variant in replacements:
            variant_prompt = prompt_text.replace("person", variant)
            
            # Changed the directory structure as requested
            if DEBUG:
                variant_save_dir = os.path.join(args.save_dir, variant, f"prompt_{idx}")
            else:
                variant_save_dir = os.path.join(args.save_dir, f"prompt_{idx}")
            os.makedirs(variant_save_dir, exist_ok=True)
            
            total_images = args.num_imgs_per_prompt
            num_batches = math.ceil(total_images / args.generation_nums)
            
            all_responses = []
            all_image_ids = []
            all_image_embeds = []

            # Progress bar for batch generation within the current prompt and variant
            for batch_num in tqdm(
                range(num_batches),
                desc=f"Generating Batches for Prompt {idx}, Variant '{variant}'",
                leave=False
            ):
                # Determine the number of images in the current batch
                tmp_generation_nums = min(args.generation_nums, total_images - batch_num * args.generation_nums)
                
                # Set a unique seed per batch for reproducibility
                seed = args.random_seed + idx * 1000 + batch_num
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                # Generate images for the current batch
                response, image_ids, image_embeds = model.generate_image_content(
                    variant_prompt,
                    args.cfg,
                    tmp_generation_nums
                )
                
                all_responses.append(response)
                if DEBUG:
                    all_image_ids.extend(image_ids)
                    all_image_embeds.append(image_embeds)

            # Concatenate all batches into a single tensor
            if len(all_responses) > 0:
                all_responses = torch.cat(all_responses, dim=0)  # Shape: (total_images, C, H, W)
            if len(all_image_embeds) > 0:
                all_image_embeds = torch.cat(all_image_embeds, dim=0)  # Shape: (total_images, embed_dim)
            
            # Save all images for the current variant
            save_image(all_responses, variant_save_dir, img_idx=0)

            # Save all_responses, all_image_ids, and all_image_embeds into a single torch file
            if DEBUG:
                data_dict = {
                    "image_ids": all_image_ids,    # List of IDs for each image
                    "image_embeds": all_image_embeds  # Tensor of shape (N, embed_dim)
                }

                torch_save_path = os.path.join(variant_save_dir, "data.pt")
                torch.save(data_dict, torch_save_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
