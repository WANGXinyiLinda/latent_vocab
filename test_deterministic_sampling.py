#!/usr/bin/env python3
"""
Test script to demonstrate deterministic sampling for checkpoint consistency.
This script shows how the DeterministicSampler ensures the same data order
when resuming from checkpoints.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from utils import DeterministicSampler, get_deterministic_seed, set_seed
import numpy as np

class SimpleDataset(Dataset):
    """Simple dataset for testing deterministic sampling."""
    
    def __init__(self, size=100):
        self.data = [f"sample_{i}" for i in range(size)]
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

def test_deterministic_sampling():
    """Test that deterministic sampling produces consistent results."""
    
    print("=== Testing Deterministic Sampling ===")
    
    # Create dataset
    dataset = SimpleDataset(20)
    base_seed = 42
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Base seed: {base_seed}")
    
    # Test 1: Same epoch should produce same order
    print("\n1. Testing same epoch consistency:")
    sampler1 = DeterministicSampler(dataset, base_seed, shuffle=True)
    sampler1.set_epoch(0)
    
    sampler2 = DeterministicSampler(dataset, base_seed, shuffle=True)
    sampler2.set_epoch(0)
    
    indices1 = list(sampler1)
    indices2 = list(sampler2)
    
    print(f"Epoch 0, Run 1: {indices1[:10]}...")
    print(f"Epoch 0, Run 2: {indices2[:10]}...")
    print(f"Same order: {indices1 == indices2}")
    
    # Test 2: Different epochs should produce different orders
    print("\n2. Testing different epoch orders:")
    sampler1.set_epoch(1)
    sampler2.set_epoch(2)
    
    indices_epoch1 = list(sampler1)
    indices_epoch2 = list(sampler2)
    
    print(f"Epoch 1: {indices_epoch1[:10]}...")
    print(f"Epoch 2: {indices_epoch2[:10]}...")
    print(f"Different orders: {indices_epoch1 != indices_epoch2}")
    
    # Test 3: Same epoch should always produce same order after reset
    print("\n3. Testing epoch reset consistency:")
    sampler1.set_epoch(0)
    indices_reset = list(sampler1)
    print(f"Epoch 0 after reset: {indices_reset[:10]}...")
    print(f"Consistent with original: {indices_reset == indices1}")
    
    # Test 4: Deterministic seed generation
    print("\n4. Testing deterministic seed generation:")
    seed_epoch0 = get_deterministic_seed(base_seed, 0)
    seed_epoch1 = get_deterministic_seed(base_seed, 1)
    seed_epoch0_again = get_deterministic_seed(base_seed, 0)
    
    print(f"Seed for epoch 0: {seed_epoch0}")
    print(f"Seed for epoch 1: {seed_epoch1}")
    print(f"Seed for epoch 0 (again): {seed_epoch0_again}")
    print(f"Epoch 0 seeds consistent: {seed_epoch0 == seed_epoch0_again}")
    print(f"Different epochs have different seeds: {seed_epoch0 != seed_epoch1}")
    
    # Test 5: DataLoader integration
    print("\n5. Testing DataLoader integration:")
    sampler = DeterministicSampler(dataset, base_seed, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=4, sampler=sampler)
    
    print("First epoch:")
    sampler.set_epoch(0)
    batch1 = next(iter(dataloader))
    print(f"First batch: {batch1}")
    
    print("Second epoch:")
    sampler.set_epoch(1)
    batch2 = next(iter(dataloader))
    print(f"First batch: {batch2}")
    
    print("Back to first epoch:")
    sampler.set_epoch(0)
    batch1_again = next(iter(dataloader))
    print(f"First batch: {batch1_again}")
    print(f"Consistent across epochs: {batch1 == batch1_again}")
    
    print("\n=== Deterministic Sampling Test Completed Successfully! ===")

if __name__ == "__main__":
    test_deterministic_sampling()
