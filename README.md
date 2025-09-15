# On Fairness of Unified Multimodal Large Language Model for Image Generation

[![Paper](https://img.shields.io/badge/arXiv-2502.03429-b31b1b.svg)](https://arxiv.org/abs/2502.03429)

This repository contains the implementation code for the paper "On Fairness of Unified Multimodal Large Language Model for Image Generation".


## Overview

This work examines demographic biases (gender and race) in unified multimodal large language models (U-MLLMs) for image generation. The key contributions include:

- **Bias Detection**: Benchmarking latest U-MLLMs to identify demographic biases in image generation
- **Locate-then-Fix Strategy**: Identifying that bias primarily originates from the language model component
- **Balanced Preference Optimization (BPO)**: A novel approach to balance demographic distribution using synthetic data while maintaining semantic fidelity

## Key Findings

- Most U-MLLMs exhibit significant demographic biases in image generation
- A "partial alignment" phenomenon exists where understanding bias is minimal but generation bias remains substantial
- The proposed balanced preference model effectively reduces demographic bias while preserving image quality

## Repository Structure

```
.
├── src/                           # Main source code
│   ├── vila_u/                   # VILA-U model implementation
│   │   ├── train/                # Training scripts and utilities
│   │   ├── model/                # Model architecture components
│   │   ├── data/                 # Data loading and processing
│   │   └── utils/                # Utility functions
│   ├── dataset/                  # Dataset preparation
│   │   └── debias_dataset/       # Debiasing dataset creation scripts
│   ├── app.py                    # Gradio demo application
│   ├── gen-images.py             # Image generation script
│   └── bias.py                   # Bias evaluation utilities
├── scripts/                       # Training and inference scripts
│   ├── lora_generation_*.sh      # LoRA fine-tuning scripts for generation
│   ├── lora_understanding_*.sh   # LoRA fine-tuning scripts for understanding
│   ├── inference.sh              # Inference script
│   └── zero2.json                # DeepSpeed configuration
└── locating_bias/                # Bias localization experiments
    ├── detect_gender.py          # Gender bias detection
    └── post-gen-images.py        # Post-generation image analysis
```

## Installation

### Prerequisites
- CUDA-enabled GPU
- Python 3.8+
- Conda (recommended)

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/pkulium/Unified_MLLM_Fairness.git
cd Unified_MLLM_Fairness

# Create conda environment
conda create -n vila-u python=3.9
conda activate vila-u

# Install dependencies
cd src
bash environment_setup.sh
```

## Usage

### 1. Image Generation with baseline model

Generate images from a list of prompts:

```bash
python src/gen-images.py \
    --model_path /path/to/vila-u/model \
    --prompts_path /path/to/prompts.json \
    --save_dir ./generated_images \
    --num_imgs_per_prompt 10 \
    --cfg 3.0
```

### 2. Image Generation with our model


```bash
python src/gen-images.py \
    --model_path /path/to/vila-u/model \
    --adapter_path /path/to/bpo/adapter \
    --prompts_path /path/to/prompts.json \
    --save_dir ./debiased_images
```

### 3. Training with BPO

Train with BPO for debiasing:

```bash
# For gender debiasing
bash scripts/lora_generation_bpo_gender_v4.20.sh

# For race debiasing
bash scripts/lora_generation_bpo_race_v4.20.sh

# For mixed debiasing
bash scripts/lora_generation_bpo_mix_v4.20.sh
```

## Configuration

### Training Parameters
- **Batch Size**: Configurable via `BATCH_SIZE` environment variable
- **Learning Rate**: Set in training scripts (default: 2e-5)
- **LoRA Rank**: 128 (configurable in scripts)
- **Training Epochs**: 2-3 (adjustable based on dataset size)

## Bias Locating and Evaluation

We use the bias evaluation benchmark from [Finetuning Text-to-Image Diffusion Models for Fairness](https://github.com/sail-sg/finetune-fair-diffusion) for measuring demographic bias in generated images.

For detailed bias metrics and evaluation protocols, please refer to the [finetune-fair-diffusion](https://github.com/sail-sg/finetune-fair-diffusion) repository.

To locating bias in the model:

```bash
python locating_bias/detect_gender.py --model_path /path/to/model
```

## Datasets

### Training Dataset
The debiasing training dataset is built using Flux and is available on Hugging Face:
- **Dataset**: [pkulium/bpo](https://huggingface.co/datasets/pkulium/bpo)
- **Description**: Synthetic paired images with balanced demographic representations for preference optimization

## Citation

If you use this code in your research, please cite:

```bibtex
@article{liu2025fairness,
  title={On Fairness of Unified Multimodal Large Language Model for Image Generation},
  author={Liu, Ming and Chen, Hao and Wang, Jindong and Wang, Liwen and Ramakrishnan, Bhiksha Raj and Zhang, Wensheng},
  journal={arXiv preprint arXiv:2502.03429},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This work builds upon the VILA-U model architecture. We thank the original authors for their contributions.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors through the paper correspondence.
