#!/bin/bash
#SBATCH --job-name=vq_vaeolmo_1b_base_gsm8k     # create a short name for your job
#SBATCH --output=logs/vq_vae_olmo_1b_base_gsm8k.log
#SBATCH --partition=pli-c
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G         # memory per cpu-core (4G is default)
#SBATCH --gpus=1                 # number of gpus per node
#SBATCH --time=21:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=xw2259@princeton.edu

module purge
module load anaconda3/2025.6
conda activate latent

python train_vq_vae.py \
    --pretrained_model_name ../models/OLMo-2-0425-1B-Base \
    --train_data_path data/GSM8K/train.jsonl \
    --val_data_path data/GSM8K/test.jsonl \
    --num_latent_tokens 100 \
    --num_heads 4 \
    --k 3 \
    --max_length 1024 \
    --train_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --val_batch_size 64 \
    --num_steps 10000 \
    --learning_rate 1e-3 \
    --output_dir ./outputs/vq_vae/olmo_1b_base_gsm8k_debug \
    --save_steps 100 \
    --log_steps 10 \
    --test_steps 200 \
    --seed 42 \
    --use_bf16 \
    --use_wandb \
    --wandb_project vq-vae-text-reconstructor \
    --wandb_run_name vq_vae_olmo_1b_base_gsm8k_debug \
    --wandb_offline \
    # --resume_from gpt2_outputs/checkpoints/step_100 \
    # --use_gradient_checkpointing \  # Uncomment to enable gradient checkpointing (saves memory, slightly slower)