python train_vq_vae.py \
    --pretrained_model_name gpt2 \
    --num_latent_tokens 16 \
    --num_heads 2 \
    --k 2 \
    --batch_size 4 \
    --num_epochs 5 \
    --learning_rate 1e-3 \
    --output_dir ./gpt2_outputs \
    --save_steps 100 \
    --log_steps 100 \
    --seed 42 \
    --use_wandb \
    --wandb_project vq-vae-text-reconstructor \
    --wandb_run_name gpt2_test \
    --wandb_offline \
    --resume_from gpt2_outputs/checkpoints/step_100 \
    # --use_gradient_checkpointing \  # Uncomment to enable gradient checkpointing (saves memory, slightly slower)