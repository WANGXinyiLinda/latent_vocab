#!/usr/bin/env python3
"""
Training script for VQVAETextReconstructor

This script trains a VQ-VAE model for text reconstruction using pretrained language models.
It includes data loading, training loops, validation, logging, and checkpointing.
"""

import os
import sys
import argparse
import logging
from typing import Tuple, Dict, Optional
from dataloader import TextDataset, load_text_data, create_sample_data
from utils import setup_logging, set_seed, get_random_state, set_random_state, get_deterministic_seed, DeterministicSampler
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import tqdm
import wandb
import os
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vq_vae import VQVAETextReconstructor

def train_loop(
    model: VQVAETextReconstructor,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    device: torch.device,
    logger: logging.Logger,
    total_steps: int,
    test_steps: int = None,
    log_steps: int = None,
    save_steps: int = None,
    warmup_steps: int = 1000,
    start_step: int = 0,
    batch_size: int = 32,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_fsdp: bool = False,
    gradient_clip: float = 1.0,
    use_wandb: bool = False,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    best_val_loss: float = float('inf'),
    output_dir: str = "./outputs",
    base_seed: int = None,
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: VQVAETextReconstructor model
        dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        logger: Logger instance
        total_steps: Total number of steps to train
        start_step: Starting step
        batch_size: Batch size
        scaler: Scaler for mixed precision training
        use_fsdp: Whether to use FSDP
        gradient_clip: Gradient clipping norm
        
    Returns:
        Dictionary with training metrics
    """
    
    total_loss = 0.0
    total_vq_loss = 0.0
    total_recon_loss = 0.0
    total_perplexity = 0.0
    num_batches = 0
    num_data = len(train_dataloader.dataset)
    
    # Calculate epochs more intuitively
    # Note: Epochs are 1-indexed for user display (Epoch 1, 2, 3...)
    # But internally we use 0-indexed for calculations
    if num_data == 0:
        logger.warning("No training data available")
        return {}
    
    if batch_size <= 0:
        logger.error(f"Invalid batch size: {batch_size}")
        return {}
    
    # Calculate steps per epoch and total epochs
    # steps_per_epoch: how many training steps make up one epoch
    # current_epoch: which epoch we're currently in (0-indexed)
    # total_epochs: total number of epochs we'll train for
    steps_per_epoch = max(1, num_data // batch_size)
    current_epoch = start_step // steps_per_epoch
    total_epochs = (total_steps + steps_per_epoch - 1) // steps_per_epoch  # Ceiling division
    
    if start_step >= total_steps:
        return {}
    
    progress_bar = tqdm.tqdm(total=total_steps, initial=start_step, desc=f"Epoch {current_epoch + 1}/{total_epochs}")
    
    # Track current step and epoch
    current_step = 0
    
    # Main training loop
    while current_step < total_steps:
        current_epoch = current_step // steps_per_epoch
        progress_bar.set_description(f"Epoch {current_epoch + 1}/{total_epochs}")
        logger.info(f"Epoch {current_epoch + 1}/{total_epochs}")
        
        # Update sampler epoch for deterministic shuffling
        if hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(current_epoch)
        
        # Iterate through the dataloader for this epoch
        for batch in train_dataloader:
            # Skip steps if we're resuming from a checkpoint
            if current_step < start_step:
                current_step += 1
                continue
                
            model.train()
            
            # Handle batch format - batch could be a single item or a list
            if not isinstance(batch, list):
                batch = [batch]  
            # Ensure batch is in the correct format
            batch = [(q, a) for q, a in batch]
            
            # Forward pass
            optimizer.zero_grad()
            
            # Use mixed precision if enabled
            if scaler is not None:
                with torch.amp.autocast():
                    vq_loss, perplexity, recon_loss, total_batch_loss = model(batch)
                
                # Backward pass with scaler
                scaler.scale(total_batch_loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                
                # Optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()
            else:
                vq_loss, perplexity, recon_loss, total_batch_loss = model(batch)
                
                # Backward pass
                total_batch_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                
                optimizer.step()
                
            # Step the scheduler after training
            if scheduler:
                scheduler.step()
                    
            # Update metrics
            total_loss += total_batch_loss.item()
            total_vq_loss += vq_loss.item()
            total_recon_loss += recon_loss.item()
            total_perplexity += perplexity.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_batch_loss.item():.4f}',
                'VQ_Loss': f'{vq_loss.item():.4f}',
                'Recon_Loss': f'{recon_loss.item():.4f}',
                'Perplexity': f'{perplexity.item():.2f}'
            })
        
            # Calculate moving averages
            metrics = {
                'train_loss': total_loss / (current_step + 1),
                'train_vq_loss': total_vq_loss / (current_step + 1),
                'train_recon_loss': total_recon_loss / (current_step + 1),
                'train_perplexity': total_perplexity / (current_step + 1)
            }
            
            if test_steps and current_step % test_steps == 0:
                test_metrics = validate_loop(model, val_dataloader, logger)
                metrics = {**metrics, **test_metrics}
                # Update best validation loss if we have validation metrics
                if test_metrics['val_loss'] < best_val_loss:
                    # Save best model
                    best_model_path = os.path.join(output_dir, "best_model.pt")
                    model.save_checkpoint(best_model_path)
                    best_val_loss = test_metrics['val_loss']  # Update best_val_loss
                    logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
                    
                    # Log best model info to WandB
                    if use_wandb:
                        wandb.log({
                            "best_val_loss": best_val_loss,
                            "best_model_saved": True,
                            "best_model_path": best_model_path
                        }, step=current_step)
            
            # Log metrics
            if log_steps and current_step % log_steps == 0:
                # Log step summary
                logger.info(f"Step {current_step} completed")
                for key, value in metrics.items():
                    logger.info(f"{key}: {value:.4f}")
                    # Log to WandB if enabled
                    if use_wandb:
                        wandb.log({key: value}, step=current_step)
            
            # Save checkpoint
            if save_steps and current_step % save_steps == 0:
                checkpoint_path = save_checkpoint(
                    model, optimizer, scheduler, 
                    current_epoch + 1, warmup_steps, current_step, total_steps, 
                    learning_rate, weight_decay, metrics,
                    os.path.join(output_dir, "checkpoints"), logger,
                    base_seed=base_seed,
                    current_random_state=get_random_state(),
                    use_fsdp=use_fsdp
                )
                
                # Log checkpoint info to WandB
                if use_wandb:
                    wandb.log({
                        "checkpoint_saved": True,
                        "checkpoint_path": checkpoint_path,
                        "checkpoint_epoch": current_epoch + 1
                    }, step=current_step)
            
            # Increment step counter
            current_step += 1
            
            # Check if we've reached total_steps
            if current_step >= total_steps:
                break
    
    return metrics


def validate_loop(
    model: VQVAETextReconstructor,
    dataloader: DataLoader,
    logger: logging.Logger,
) -> Dict[str, float]:
    """
    Validate for one epoch.
    
    Args:
        model: VQVAETextReconstructor model
        dataloader: Validation data loader
        device: Device to validate on
        logger: Logger instance
        epoch: Current epoch number
        
    Returns:
        Dictionary with validation metrics
    """
    if len(dataloader) == 0:
        logger.warning("No validation data found")
        return {}
    
    model.eval()
    total_loss = 0.0
    total_vq_loss = 0.0
    total_recon_loss = 0.0
    total_perplexity = 0.0
    num_batches = 0
    
    progress_bar = tqdm.tqdm(dataloader, desc=f"Validation")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            # Handle batch format - batch could be a single item or a list
            if not isinstance(batch, list):
                batch = [batch]
            # Ensure batch is in the correct format
            batch = [(q, a) for q, a in batch]
            
            # Forward pass
            vq_loss, perplexity, recon_loss, total_batch_loss = model(batch)
            
            # Update metrics
            total_loss += total_batch_loss.item()
            total_vq_loss += vq_loss.item()
            total_recon_loss += recon_loss.item()
            total_perplexity += perplexity.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_batch_loss.item():.4f}',
                'VQ_Loss': f'{vq_loss.item():.4f}',
                'Recon_Loss': f'{recon_loss.item():.4f}',
                'Perplexity': f'{perplexity.item():.2f}'
            })
    
    # Calculate averages
    if num_batches == 0:
        metrics = {
            'val_loss': float('inf'),
            'val_vq_loss': float('inf'),
            'val_recon_loss': float('inf'),
            'val_perplexity': 0.0
        }
    else:
        metrics = {
            'val_loss': total_loss / num_batches,
            'val_vq_loss': total_vq_loss / num_batches,
            'val_recon_loss': total_recon_loss / num_batches,
            'val_perplexity': total_perplexity / num_batches
        }
    
    return metrics


def save_checkpoint(
    model: VQVAETextReconstructor,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    epoch: int,
    warmup_steps: int,
    trained_steps: int,
    total_training_steps: int,
    learning_rate: float,
    weight_decay: float,
    metrics: Dict[str, float],
    checkpoint_dir: str,
    logger: logging.Logger,
    base_seed: int = None,
    current_random_state: dict = None,
    use_fsdp: bool = False
) -> str:
    """
    Save training checkpoint.
    
    Args:
        model: VQVAETextReconstructor model
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        trained_steps: Trained steps
        total_training_steps: Total training steps
        learning_rate: Learning rate
        weight_decay: Weight decay
        metrics: Current metrics
        checkpoint_dir: Directory to save checkpoint
        logger: Logger instance
        base_seed: Base random seed for deterministic shuffling
        current_random_state: Current random state for reproducibility
        use_fsdp: Whether the model is using FSDP
        
    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir = os.path.join(checkpoint_dir, f"epoch_{epoch}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model checkpoint
    model_checkpoint_path = os.path.join(checkpoint_dir, f"vq_vae_model.pt")
    training_checkpoint_path = os.path.join(checkpoint_dir, f"training_state.pt")
    
    if use_fsdp:
        # FSDP-specific checkpointing
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import StateDictType
            
            # Check if model is wrapped with FSDP
            if isinstance(model, FSDP):
                logger.info("Saving FSDP model checkpoint...")
                
                # Gather model state dict with full state dict type
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                    model_state = model.state_dict()
                
                # Gather optimizer state dict
                optim_state = FSDP.optim_state_dict(model, optimizer)
                
                # Save on rank 0 only to avoid conflicts
                if not dist.is_initialized() or dist.get_rank() == 0:
                    torch.save(model_state, model_checkpoint_path)
                    torch.save({
                        'optimizer_state_dict': optim_state,
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'epoch': epoch,
                        'learning_rate': learning_rate,
                        'weight_decay': weight_decay,
                        'trained_steps': trained_steps,
                        'total_training_steps': total_training_steps,
                        'warmup_steps': warmup_steps,
                        'metrics': metrics,
                        'base_seed': base_seed,
                        'random_state': current_random_state,
                        'fsdp_wrapped': True
                    }, training_checkpoint_path)
                    
                    logger.info(f"Saved FSDP training state checkpoint to {training_checkpoint_path}")
                    logger.info(f"Saved FSDP model checkpoint to {model_checkpoint_path}")
                
                # Synchronize all processes
                if dist.is_initialized():
                    dist.barrier()
                    
            else:
                # Model is not FSDP-wrapped, use regular saving
                logger.warning("Model is not FSDP-wrapped but use_fsdp=True. Using regular saving.")
                model.save_checkpoint(model_checkpoint_path)
                
        except ImportError:
            logger.error("FSDP not available. Falling back to regular model saving.")
            model.save_checkpoint(model_checkpoint_path)
        except Exception as e:
            logger.error(f"FSDP checkpointing failed: {e}. Falling back to regular saving.")
            model.save_checkpoint(model_checkpoint_path)
    else:
        # Regular model checkpointing
        model.save_checkpoint(model_checkpoint_path)
        
    
    # Save training state on rank 0 only for FSDP
    if not use_fsdp or not dist.is_initialized() or dist.get_rank() == 0:
        # Save training state (non-model components)
        training_state = {
            'epoch': epoch,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'trained_steps': trained_steps,
            'total_training_steps': total_training_steps,
            'warmup_steps': warmup_steps,  
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'base_seed': base_seed,
            'random_state': current_random_state,
            'fsdp_wrapped': use_fsdp
        }
        torch.save(training_state, training_checkpoint_path)
        logger.info(f"Saved training state to {training_checkpoint_path}")
    
    # Synchronize all processes for FSDP
    if use_fsdp and dist.is_initialized():
        dist.barrier()
    
    logger.info(f"Saved checkpoint for epoch {epoch}")
    return training_checkpoint_path


def load_checkpoint(
    checkpoint_dir: str,
    logger: logging.Logger,
    device: str = "auto",
    use_fsdp: bool = False
) -> Tuple[VQVAETextReconstructor, optim.Optimizer, Optional[optim.lr_scheduler._LRScheduler], int, int, Dict[str, float], int]:
    """
    Load training checkpoint.
    
    Args:
        checkpoint_dir: Path to checkpoints
        logger: Logger instance
        device: Device to load checkpoint on
        use_fsdp: Whether the model is using FSDP
        
    Returns:
        Tuple of (model, optimizer, scheduler, start_epoch, start_step, metrics, base_seed)
    """
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    device = torch.device(device) if device != "auto" \
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_checkpoint_path = os.path.join(checkpoint_dir, f"vq_vae_model.pt")
    training_checkpoint_path = os.path.join(checkpoint_dir, f"training_state.pt")
    
    # Load training state first
    training_checkpoint = torch.load(training_checkpoint_path, map_location=device)
    
    # Check if this was an FSDP checkpoint
    was_fsdp_checkpoint = training_checkpoint.get('fsdp_wrapped', False)
    
    if use_fsdp and was_fsdp_checkpoint:
        # FSDP checkpoint loading
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import StateDictType
            
            # Load the FSDP checkpoint
            fsdp_checkpoint = torch.load(model_checkpoint_path, map_location=device)
            
            # Create a fresh model (not wrapped with FSDP yet)
            # We'll need to create the base model first, then wrap it
            logger.info("Loading FSDP checkpoint - model will be created and wrapped after loading")
            
            # For now, we'll need to create a temporary model to get the class
            # This is a limitation - we need the model class to recreate it
            # In practice, you might want to save the model class info in the checkpoint
            
            # Load training state
            trained_steps = training_checkpoint['trained_steps']
            epoch = training_checkpoint['epoch']
            metrics = training_checkpoint['metrics']
            base_seed = training_checkpoint.get('base_seed', None)
            
            # Note: For FSDP, we typically need to recreate the model and wrap it
            # before loading the state dict. This is a limitation of the current approach.
            logger.warning("FSDP checkpoint loading requires model recreation. Please ensure the model is properly wrapped with FSDP before calling this function.")
            
            # Return placeholder values - the calling code should handle model creation
            return None, None, None, epoch, trained_steps, metrics, base_seed
            
        except ImportError:
            logger.error("FSDP not available. Cannot load FSDP checkpoint.")
            raise
        except Exception as e:
            logger.error(f"FSDP checkpoint loading failed: {e}")
            raise
    else:
        # Regular checkpoint loading
        model = VQVAETextReconstructor.load_from_checkpoint(model_checkpoint_path, device=device)
        
        # Initialize optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=training_checkpoint['learning_rate'],
            weight_decay=training_checkpoint['weight_decay']
        )
        
        # Initialize scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_checkpoint['warmup_steps'],
            num_training_steps=training_checkpoint['total_training_steps']
        )

        # Load optimizer state if available
        if 'optimizer_state_dict' in training_checkpoint:
            optimizer.load_state_dict(training_checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if training_checkpoint.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(training_checkpoint['scheduler_state_dict'])
        
        trained_steps = training_checkpoint['trained_steps']
        epoch = training_checkpoint['epoch']
        metrics = training_checkpoint['metrics']
        base_seed = training_checkpoint.get('base_seed', None)
        
        # Restore random state if available
        if 'random_state' in training_checkpoint and training_checkpoint['random_state']:
            set_random_state(training_checkpoint['random_state'])
            logger.info("Random state restored from checkpoint")
        
        logger.info(f"Loaded checkpoint from epoch {epoch}")
        return model, optimizer, scheduler, epoch, trained_steps, metrics, base_seed


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train VQVAETextReconstructor")
    
    # Model parameters
    parser.add_argument("--pretrained_model_name", type=str, required=True,
                       help="Pretrained model name or path")
    parser.add_argument("--num_latent_tokens", type=int, default=64,
                       help="Number of latent tokens")
    parser.add_argument("--num_heads", type=int, default=8,
                       help="Number of attention heads")
    parser.add_argument("--k", type=int, default=8,
                       help="Number of quantized vectors")
    parser.add_argument("--commitment_cost", type=float, default=0.25,
                       help="Commitment cost for vector quantization")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--train_decoder", action="store_true",
                       help="Whether to train the decoder")
    parser.add_argument("--previous_segments_mode", type=str, default="text",
                       choices=["text", "latent", "none"],
                       help="Mode for how to handle previous segments as context")
    
    # Training parameters
    parser.add_argument("--train_data_path", type=str, default=None,
                       help="Path to training data")
    parser.add_argument("--val_data_path", type=str, default=None,
                       help="Path to validation data")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--num_steps", type=int, default=None,
                       help="Number of training steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                       help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                       help="Number of warmup steps")
    parser.add_argument("--gradient_clip", type=float, default=1.0,
                       help="Gradient clipping norm")
    
    # Data parameters
    parser.add_argument("--val_split", type=float, default=0.1,
                       help="Validation split ratio")
    parser.add_argument("--compress_cot_only", action="store_true",
                       help="Compress only chain-of-thought reasoning")
    parser.add_argument("--explain_token", type=str, default="<EXPLAIN>",
                       help="Explain token for reconstruction")
    parser.add_argument("--think_token", type=str, default="<THINK>",
                       help="Think token for reconstruction")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Optional prompt prefix")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Output directory for checkpoints and logs")
    parser.add_argument("--save_steps", type=int, default=100,
                       help="Save checkpoint every N steps")
    parser.add_argument("--log_steps", type=int, default=100,
                       help="Log metrics every N steps")
    parser.add_argument("--test_steps", type=int, default=100,
                       help="Test every N steps")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Resume training
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Resume training from checkpoint")
    
    # Distributed training and precision parameters
    parser.add_argument("--use_fsdp", action="store_true",
                       help="Use FSDP for distributed training")
    parser.add_argument("--use_bf16", action="store_true",
                       help="Use bfloat16 precision for training")
    parser.add_argument("--use_fp16", action="store_true",
                       help="Use float16 precision for training")
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="Local rank for distributed training")
    parser.add_argument("--world_size", type=int, default=1,
                       help="World size for distributed training")
    parser.add_argument("--fsdp_transformer_layer_cls_to_wrap", type=str, default=None,
                       help="Transformer layer class to wrap for FSDP (e.g., 'LlamaDecoderLayer')")
    parser.add_argument("--fsdp_min_num_params", type=int, default=1e8,
                       help="Minimum number of parameters for FSDP auto wrapping")
    parser.add_argument("--fsdp_cpu_offload", action="store_true",
                       help="Enable CPU offloading for FSDP")
    parser.add_argument("--fsdp_backward_prefetch", type=str, default="BACKWARD_PRE",
                       choices=["BACKWARD_PRE", "BACKWARD_POST", "NONE"],
                       help="FSDP backward prefetch strategy")
    parser.add_argument("--fsdp_sharding_strategy", type=str, default="FULL_SHARD",
                       choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"],
                       help="FSDP sharding strategy")
    
    # Weights & Biases parameters
    parser.add_argument("--use_wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="vq-vae-text-reconstructor",
                       help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="WandB run name")
    parser.add_argument("--wandb_offline", action="store_true",
                       help="Run WandB in offline mode")
    parser.add_argument("--wandb_dir", type=str, default=None,
                       help="Directory to store WandB offline files")
    
    args = parser.parse_args()
    
    # Setup distributed training if enabled
    if args.use_fsdp:
        # When using torchrun, local_rank should be set by the environment
        if args.local_rank != -1:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(args.local_rank)
            args.device = torch.device("cuda", args.local_rank)
        else:
            # Check if we're in a distributed environment
            if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
                rank = int(os.environ["RANK"])
                world_size = int(os.environ["WORLD_SIZE"])
                local_rank = int(os.environ["LOCAL_RANK"])
                dist.init_process_group(backend="nccl")
                torch.cuda.set_device(local_rank)
                args.device = torch.device("cuda", local_rank)
                args.local_rank = local_rank
                args.world_size = world_size
            else:
                args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set random seeds
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    log_dir = os.path.join(args.output_dir, "logs")
    logger = setup_logging(log_dir)
    
    logger.info("Starting VQVAETextReconstructor training")
    logger.info(f"Arguments: {vars(args)}")
    
    if not args.train_data_path: # Create sample data if not provided
        sample_data_path = os.path.join(args.output_dir, "sample_data.json")
        create_sample_data(sample_data_path, num_samples=100)
        args.train_data_path = sample_data_path
        logger.info(f"Created sample data at {sample_data_path}")

    try:
        train_data = load_text_data(args.train_data_path)
        logger.info(f"Loaded {len(train_data)} training examples")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    if args.val_data_path:
        try:
            val_data = load_text_data(args.val_data_path)
            logger.info(f"Loaded {len(val_data)} validation examples")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return
    else:
        # Split data
        val_size = int(len(train_data) * args.val_split)
        train_size = len(train_data) - val_size
        val_data = train_data[train_size:]
        train_data = train_data[:train_size]
    
    logger.info(f"Training set size: {len(train_data)}")
    logger.info(f"Validation set size: {len(val_data)}")
    
    # Create datasets and dataloaders
    train_dataset = TextDataset(train_data, max_length=args.max_length)
    val_dataset = TextDataset(val_data, max_length=args.max_length)
    
    # Setup distributed sampling if using FSDP
    if args.use_fsdp and args.local_rank != -1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=args.local_rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=args.world_size,
            rank=args.local_rank,
            shuffle=False
        )
        shuffle = False  # Sampler handles shuffling
    else:
        # Use deterministic sampler for consistent data ordering across checkpoints
        if args.seed is not None:
            train_sampler = DeterministicSampler(train_dataset, args.seed, shuffle=True)
            val_sampler = DeterministicSampler(val_dataset, args.seed, shuffle=False)
            shuffle = False  # Sampler handles shuffling
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=0,
        drop_last=False
    )
    
    # Setup device
    device = args.device
    logger.info(f"Using device: {device}")
    
    # Setup mixed precision
    if args.use_bf16:
        scaler = None
        dtype = torch.bfloat16
        logger.info("Using bfloat16 precision")
    elif args.use_fp16:
        scaler = torch.amp.GradScaler()
        dtype = torch.float16
        logger.info("Using float16 precision with GradScaler")
    else:
        scaler = None
        dtype = torch.float32
        logger.info("Using float32 precision")
    
    # Create model
    model = VQVAETextReconstructor(
        num_latent_tokens=args.num_latent_tokens,
        num_heads=args.num_heads,
        k=args.k,
        pretrained_model_name=args.pretrained_model_name,
        commitment_cost=args.commitment_cost,
        explain_token=args.explain_token,
        think_token=args.think_token,
        prompt=args.prompt,
        compress_cot_only=args.compress_cot_only,
        max_length=args.max_length,
        train_decoder=args.train_decoder,
        previous_segments_mode=args.previous_segments_mode,
        device=device
    )
    logger.info("Model created successfully")
    
    # Print model information
    model.print_parameter_summary()
    
    # Move model to device
    model = model.to(device)
    
    # Setup FSDP if enabled
    if args.use_fsdp:
        logger.info("Setting up FSDP...")
        
        # Configure FSDP parameters
        fsdp_config = {
            "mixed_precision": MixedPrecision(
                param_dtype=dtype,
                reduce_dtype=dtype,
                buffer_dtype=dtype,
            ),
            "sharding_strategy": getattr(ShardingStrategy, args.fsdp_sharding_strategy),
            "backward_prefetch": getattr(BackwardPrefetch, args.fsdp_backward_prefetch),
            "cpu_offload": CPUOffload(offload_params=args.fsdp_cpu_offload),
        }
        
        # Auto-wrap policy
        if args.fsdp_transformer_layer_cls_to_wrap:
            # Use transformer auto-wrap policy
            auto_wrap_policy = transformer_auto_wrap_policy(
                model_class=eval(args.fsdp_transformer_layer_cls_to_wrap),
                transformer_layer_cls_to_wrap=eval(args.fsdp_transformer_layer_cls_to_wrap),
            )
        else:
            # Use size-based auto-wrap policy
            auto_wrap_policy = size_based_auto_wrap_policy(
                min_num_params=args.fsdp_min_num_params
            )
        
        # Wrap model with FSDP
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            **fsdp_config
        )
        
        logger.info("FSDP setup completed")
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    if args.num_steps:
        total_steps = args.num_steps
    else:
        total_steps = len(train_loader) * args.num_epochs
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Setup Weights & Biases
    if args.use_wandb:
        # Set offline mode if requested
        if args.wandb_offline:
            os.environ["WANDB_MODE"] = "offline"
            if args.wandb_dir:
                os.environ["WANDB_DIR"] = args.wandb_dir
            logger.info("WandB running in offline mode")
        
        # Initialize wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            dir=args.wandb_dir if args.wandb_offline else None,
            mode="offline" if args.wandb_offline else "online"
        )
        logger.info(f"WandB initialized successfully. Project: {args.wandb_project}")
        
        # Log model architecture info
        if hasattr(model, 'get_model_size_info'):
            model_info = model.get_model_size_info()
            wandb.config.update({
                "model_total_params": model_info['total_parameters'],
                "model_trainable_params": model_info['trainable_parameters'],
                "model_size_mb": model_info['model_size_mb']
            })
                
    else:
        logger.info("WandB logging disabled")
    
    # Resume training if specified
    start_epoch = 0
    start_step = 0
    best_val_loss = float('inf')
    
    if args.resume_from:
        model, optimizer, scheduler, start_epoch, start_step, metrics, base_seed = load_checkpoint(
            args.resume_from, logger, device, args.use_fsdp
        )
        if 'val_loss' in metrics:
            best_val_loss = metrics['val_loss']
        logger.info(f"Resuming training from epoch {start_epoch}")
    
    # Training loop
    logger.info("Starting training loop")
    
    # Train
    metrics = train_loop(
        model, train_loader, val_loader, optimizer, scheduler, device, logger, total_steps, 
        args.test_steps, args.log_steps, args.save_steps, args.warmup_steps,
        start_step, args.batch_size, scaler, args.use_fsdp, args.gradient_clip, args.use_wandb,
        args.learning_rate, args.weight_decay, best_val_loss, args.output_dir, base_seed
    )
    
    # Save final model
    final_checkpoint_path = save_checkpoint(
        model, optimizer, scheduler, 
        args.num_epochs, args.warmup_steps, total_steps, total_steps, 
        args.learning_rate, args.weight_decay, metrics,
        os.path.join(args.output_dir, "checkpoints"), logger,
        base_seed=args.seed,
        current_random_state=get_random_state(),
        use_fsdp=args.use_fsdp
    )
    
    # Cleanup distributed training
    if args.use_fsdp and args.local_rank != -1:
        dist.destroy_process_group()
        logger.info("Distributed training process group destroyed")
    
    # Log final training results to WandB
    if args.use_wandb:
        wandb.log({
            "training_completed": True,
            "final_best_val_loss": best_val_loss,
            "final_checkpoint_path": final_checkpoint_path,
            "best_model_path": os.path.join(args.output_dir, 'best_model.pt'),
            "total_epochs": args.num_epochs
        })
        
        # Finish wandb run
        wandb.finish()
        logger.info("WandB run finished")
    
    logger.info("Training completed successfully!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Final model saved at: {final_checkpoint_path}")
    logger.info(f"Best model saved at: {os.path.join(args.output_dir, 'best_model.pt')}")


if __name__ == "__main__":
    main()
