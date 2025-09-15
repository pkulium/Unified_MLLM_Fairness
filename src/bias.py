import argparse
import cv2
import numpy as np
import os
import vila_u

def read_prompts_from_file(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def create_folder(parent_folder, prompt):
    folder_name = prompt.replace(" ", "_")[:50]  # Use first 50 characters of prompt as folder name
    full_path = os.path.join(parent_folder, folder_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path

def save_image(response, path):
    os.makedirs(path, exist_ok=True)
    for i in range(response.shape[0]):
        image = response[i].permute(1, 2, 0)
        image = image.cpu().numpy().astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(path, f"image_{i}.png"), image)

def save_video(response, path):
    os.makedirs(path, exist_ok=True)
    for i in range(response.shape[0]):
        video = response[i].permute(0, 2, 3, 1)
        video = video.cpu().numpy().astype(np.uint8)
        video = np.concatenate(video, axis=1)
        video = cv2.cvtColor(video, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(path, f"video_{i}.png"), video)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompts_file", type=str, required=True)
    parser.add_argument("--video_generation", action="store_true")
    parser.add_argument("--cfg", type=float, default=3.0, help="The value of the classifier free guidance for image/video generation.")
    parser.add_argument("--save_path", type=str, default="vila_generated_content/")
    parser.add_argument("--generation_nums", type=int, default=1)
    args = parser.parse_args()

    if args.model_path is not None:
        model = vila_u.load(args.model_path)
    else:
        raise ValueError("No model path provided!")

    # Create parent folder
    os.makedirs(args.save_path, exist_ok=True)

    # Read prompts from file
    prompts = read_prompts_from_file(args.prompts_file)

    # Process each prompt
    for prompt in prompts:
        folder_path = create_folder(args.save_path, prompt)
        
        if args.video_generation:
            response = model.generate_video_content(prompt, args.cfg, args.generation_nums)
            save_video(response, folder_path)
            print(f"Video for prompt '{prompt}' saved in {folder_path}")
        else:
            response = model.generate_image_content(prompt, args.cfg, args.generation_nums)
            # save_image(response, folder_path)
            print(f"Image for prompt '{prompt}' saved in {folder_path}")