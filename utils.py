import logging
import os
import torch
import random
import numpy as np
from torch.utils.data import Sampler

def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_random_state():
    """
    Get the current random state for all random generators.
    
    Returns:
        Dictionary containing random states
    """
    return {
        'python_random': random.getstate(),
        'numpy_random': np.random.get_state(),
        'torch_random': torch.get_rng_state(),
        'torch_cuda_random': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    }

def set_random_state(random_state):
    """
    Set the random state for all random generators.
    
    Args:
        random_state: Dictionary containing random states
    """
    if 'python_random' in random_state:
        random.setstate(random_state['python_random'])
    if 'numpy_random' in random_state:
        np.random.set_state(random_state['numpy_random'])
    if 'torch_random' in random_state:
        torch.set_rng_state(random_state['torch_random'])
    if 'torch_cuda_random' in random_state and random_state['torch_cuda_random'] is not None:
        torch.cuda.set_rng_state(random_state['torch_cuda_random'])

def get_deterministic_seed(base_seed: int, epoch: int):
    """
    Generate a deterministic seed for a specific epoch.
    This ensures consistent shuffling order when resuming from checkpoints.
    
    Args:
        base_seed: Base random seed
        epoch: Current epoch number
        
    Returns:
        Deterministic seed for the epoch
    """
    # Use a hash-like function to generate epoch-specific seeds
    # This ensures the same epoch always gets the same seed
    return hash((base_seed, epoch)) % (2**32)

class DeterministicSampler(Sampler):
    """
    A deterministic sampler that ensures consistent data ordering across checkpoints.
    This is useful for resuming training from checkpoints while maintaining the same data order.
    """
    
    def __init__(self, data_source, base_seed: int, shuffle: bool = True):
        """
        Initialize the deterministic sampler.
        
        Args:
            data_source: The dataset to sample from
            base_seed: Base random seed for deterministic shuffling
            shuffle: Whether to shuffle the data
        """
        self.data_source = data_source
        self.base_seed = base_seed
        self.shuffle = shuffle
        self.epoch = 0
        
    def set_epoch(self, epoch: int):
        """
        Set the epoch for deterministic shuffling.
        
        Args:
            epoch: Current epoch number
        """
        self.epoch = epoch
        
    def __iter__(self):
        """
        Generate indices for the current epoch.
        
        Returns:
            Iterator over indices
        """
        if self.shuffle:
            # Generate deterministic indices for this epoch
            epoch_seed = get_deterministic_seed(self.base_seed, self.epoch)
            g = torch.Generator()
            g.manual_seed(epoch_seed)
            indices = torch.randperm(len(self.data_source), generator=g).tolist()
        else:
            indices = list(range(len(self.data_source)))
            
        return iter(indices)
        
    def __len__(self):
        return len(self.data_source)

def setup_logging(log_dir: str, log_level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level
        
    Returns:
        Configured logger
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("VQVAETrainer")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create handlers
    log_file = os.path.join(log_dir, "training.log")
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Create formatters and add it to handlers
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def get_checkpoint_info(checkpoint_path: str) -> dict:
    """
    Get information about a saved VQ-VAE checkpoint without loading the full model.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Dictionary containing checkpoint information
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    info = {
        'checkpoint_path': checkpoint_path,
        'file_size_mb': os.path.getsize(checkpoint_path) / (1024 * 1024),
        'checkpoint_type': 'full_model' if 'model_state_dict' in checkpoint else 'trained_components_only'
    }
    
    if 'config' in checkpoint:
        info['config'] = checkpoint['config']
    
    if 'trained_components' in checkpoint:
        info['trained_components'] = list(checkpoint['trained_components'].keys())
    
    return info