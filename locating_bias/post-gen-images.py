# js_divergence_analysis.py

import os
import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate JS-Divergence between variants and 'person' for each prompt.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/train_prompts_occupation",
        help="Directory where the generated data.pt files are saved.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/train_prompts_occupation_results/js_divergence_results.json",
        help="Path to save the JS-Divergence results.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs='+',
        default=["person", "male", "female", "white", "asian", "black", "indian"],
        help="List of variants to consider.",
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=16384,
        help="Total number of possible token IDs.",
    )
    return parser.parse_args()

def load_data_pt(data_pt_path):
    if not os.path.exists(data_pt_path):
        raise FileNotFoundError(f"data.pt not found at {data_pt_path}")
    data = torch.load(data_pt_path, map_location='cpu')
    image_ids = data.get("image_ids", [])
    return image_ids

def count_token_ids(image_ids, num_tokens):
    counts = np.zeros(num_tokens, dtype=np.int64)
    for token_id in image_ids:
        if isinstance(token_id, torch.Tensor):
            token_id = token_id.cpu().numpy()
        valid_tokens = (token_id >= 0) & (token_id < num_tokens)
        unique, counts_per_id = np.unique(token_id[valid_tokens], return_counts=True)
        counts[unique] += counts_per_id
        if not np.all(valid_tokens):
            print(f"Warning: Some token IDs out of range [0, {num_tokens-1}].")
    return counts


def compute_js_divergence(p, q):
    # scipy's jensenshannon returns the square root of JS divergence
    return jensenshannon(p, q, base=2.0) ** 2  # To get the actual JS divergence

def main():
    args = parse_args()
    save_dir = args.save_dir
    variants = args.variants
    num_tokens = args.num_tokens
    output_file = args.output_file

    # Check if 'person' variant is included
    if "person" not in variants:
        raise ValueError("'person' variant must be included in the variants list.")

    # Collect all prompt directories (assuming all variants have the same prompts)
    # Find prompts by listing one variant's subdirectories
    sample_variant_dir = os.path.join(save_dir, variants[0])
    if not os.path.exists(sample_variant_dir):
        raise FileNotFoundError(f"Variant directory not found: {sample_variant_dir}")

    prompt_dirs = [d for d in os.listdir(sample_variant_dir) if os.path.isdir(os.path.join(sample_variant_dir, d))]
    prompt_dirs_sorted = sorted(prompt_dirs, key=lambda x: int(x.split('_')[-1]))  # Assuming 'prompt_{idx}' format
    num_prompts = len(prompt_dirs_sorted)
    print(f"Found {num_prompts} prompts.")

    results = {}

    for prompt_dir in tqdm(prompt_dirs_sorted, desc="Processing Prompts"):
        prompt_idx = prompt_dir.split('_')[-1]
        results[prompt_idx] = {}
        token_counts_per_variant = {}

        # Load and count token ids for each variant
        for variant in variants:
            data_pt_path = os.path.join(save_dir, variant, prompt_dir, "data.pt")
            try:
                image_ids = load_data_pt(data_pt_path)
                image_ids = image_ids[:len(image_ids)//2]
                counts = count_token_ids(image_ids, num_tokens)
                token_counts_per_variant[variant] = counts
            except FileNotFoundError as e:
                print(e)
                token_counts_per_variant[variant] = np.zeros(num_tokens, dtype=np.int64)

        # Create probability distributions
        distributions = {}
        for variant, counts in token_counts_per_variant.items():
            total = counts.sum()
            if total > 0:
                distributions[variant] = counts / total
            else:
                print(f"Warning: No tokens found for variant '{variant}' in prompt '{prompt_dir}'.")
                distributions[variant] = np.zeros(num_tokens, dtype=np.float64)

        # Reference distribution: 'person'
        p_distribution = distributions["person"]

        # Calculate JS-divergence for other variants compared to 'person'
        js_divergences = {}
        for variant, q_distribution in distributions.items():
            if variant == "person":
                continue
            # Handle cases where both p and q are zero vectors
            if np.all(p_distribution == 0) and np.all(q_distribution == 0):
                js_div = 0.0
            elif np.all(p_distribution == 0) or np.all(q_distribution == 0):
                js_div = 1.0  # Maximum divergence
            else:
                js_div = compute_js_divergence(p_distribution, q_distribution)
            js_divergences[variant] = js_div

        results[prompt_idx] = js_divergences

    # Save the results to a JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"JS-Divergence results saved to {output_file}")

if __name__ == "__main__":
    main()
