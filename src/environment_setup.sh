#!/usr/bin/env bash
set -e

# conda install -c nvidia cuda-toolkit -y
# This is required to enable PEP 660 support
pip install --upgrade pip setuptools

# Install FlashAttention2 based on your machine
pip install flash-attn==2.5.8

# Install VILA
pip install -e ".[train,eval]"

# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
# pip install git+https://github.com/huggingface/transformers@v4.36.2

# Replace transformers and deepspeed files
site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -rv ./vila_u/train/transformers_replace/* $site_pkg_path/transformers/
# Avoid confused warning
rm -rf $site_pkg_path/lmms_eval/models/mplug_owl_video/modeling_mplug_owl.py