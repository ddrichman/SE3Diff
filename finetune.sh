#!/bin/bash

# Enable ColabFold for WSL2
export TF_FORCE_UNIFIED_MEMORY="1"
export XLA_PYTHON_CLIENT_MEM_FRACTION="4.0"
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"
export TF_FORCE_GPU_ALLOW_GROWTH="true"

# Change venv directory to the current conda environment
export BIOEMU_COLABFOLD_DIR=$CONDA_PREFIX

python -m bioemu.finetune \
    --csv_path ./test/megascale/train_sample.csv \
    --csv_path_val ./test/megascale/val_sample.csv \
    --sequence_col aa_seq \
    --h_stars_cols p_folded \
    --output_dir None \
    --finetune_config_path None \
    --model_name bioemu-v1.0 \
    --ckpt_path None \
    --finetune_ckpt_path None \
    --model_config_path None \
    --denoiser_type euler_maruyama_finetune \
    --denoiser_config_path None \
    --h_func_type folding_stability \
    --h_func_config_path None \
    --cache_embeds_dir ~/.cache/bioemu/embeds \
    --cache_so3_dir ~/.cache/bioemu/so3 \
    --msa_file None \
    --msa_host_url None \
