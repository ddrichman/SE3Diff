#!/bin/bash

# Enable ColabFold for WSL2
export TF_FORCE_UNIFIED_MEMORY="1"
export XLA_PYTHON_CLIENT_MEM_FRACTION="4.0"
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"
export TF_FORCE_GPU_ALLOW_GROWTH="true"

# Change venv directory to the current conda environment
export BIOEMU_COLABFOLD_DIR=$CONDA_PREFIX

python -m bioemu.sample \
    --sequence ANQASVVANQLIPINVALTLVMMRSEVVTPVGIPAEDIPRLVSMQVNRAVPLGTTLMPDMVKGYAA \
    --num_samples 100 \
    --output_dir ./test/1msj \
    --batch_size_100 100 \
    --model_name bioemu-v1.0 \
    --ckpt_path None \
    --model_config_path None \
    --denoiser_type dpm \
    --denoiser_config_path None \
    --cache_embeds_dir ~/.cache/bioemu/embeds \
    --cache_so3_dir ~/.cache/bioemu/so3 \
    --msa_host_url None \
    --filter_samples True
