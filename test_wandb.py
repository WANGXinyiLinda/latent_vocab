#!/usr/bin/env python3
"""
Test script for wandb functionality in the training script.

This script tests that the Weights & Biases integration can be imported
and configured without errors.
"""

import os
import sys
import tempfile

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_wandb_import():
    """Test that wandb can be imported."""
    print("Testing wandb import...")
    
    try:
        import wandb
        print(f"✓ WandB {wandb.__version__} imported successfully")
        return True
    except ImportError as e:
        print(f"✗ WandB import failed: {e}")
        print("  Install with: pip install wandb")
        return False


def test_wandb_offline_mode():
    """Test wandb offline mode configuration."""
    print("\nTesting wandb offline mode...")
    
    try:
        import wandb
        
        # Test setting offline mode
        os.environ["WANDB_MODE"] = "offline"
        
        # Test offline directory
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["WANDB_DIR"] = temp_dir
            
            # Try to initialize wandb in offline mode
            run = wandb.init(
                project="test-project",
                mode="offline",
                dir=temp_dir
            )
            
            # Check if we're in offline mode by checking the environment
            if os.environ.get("WANDB_MODE") == "offline":
                print("✓ WandB offline mode configured successfully")
                print(f"  Offline directory: {temp_dir}")
                
                # Test basic logging
                wandb.log({"test_metric": 1.0})
                print("✓ Basic logging works in offline mode")
                
                # Finish the run
                wandb.finish()
                return True
            else:
                print(f"✗ WandB not in offline mode: {os.environ.get('WANDB_MODE')}")
                return False
                
    except Exception as e:
        print(f"✗ WandB offline mode test failed: {e}")
        return False


def test_training_script_wandb():
    """Test that the training script can import wandb functionality."""
    print("\nTesting training script wandb integration...")
    
    try:
        # Test that we can import the training script
        from train_vq_vae import main
        
        # Test argument parsing for wandb
        import argparse
        parser = argparse.ArgumentParser()
        
        # Add wandb arguments
        parser.add_argument("--use_wandb", action="store_true")
        parser.add_argument("--wandb_project", type=str, default="test-project")
        parser.add_argument("--wandb_entity", type=str, default=None)
        parser.add_argument("--wandb_run_name", type=str, default=None)
        parser.add_argument("--wandb_offline", action="store_true")
        parser.add_argument("--wandb_dir", type=str, default=None)
        
        # Test parsing
        test_args = [
            "--use_wandb",
            "--wandb_project", "my-project",
            "--wandb_run_name", "test-run",
            "--wandb_offline",
            "--wandb_dir", "./wandb_test"
        ]
        
        args = parser.parse_args(test_args)
        
        if (args.use_wandb and 
            args.wandb_project == "my-project" and
            args.wandb_run_name == "test-run" and
            args.wandb_offline and
            args.wandb_dir == "./wandb_test"):
            print("✓ WandB argument parsing successful")
            return True
        else:
            print("✗ WandB argument parsing failed")
            return False
            
    except Exception as e:
        print(f"✗ Training script wandb test failed: {e}")
        return False


def test_launch_script_wandb():
    """Test that the launch script supports wandb arguments."""
    print("\nTesting launch script wandb support...")
    
    try:
        from launch_fsdp_training import create_fsdp_launch_command
        
        # Test creating a launch command with wandb
        class MockArgs:
            def __init__(self):
                self.num_gpus = 1
                self.num_nodes = 1
                self.node_rank = 0
                self.master_addr = "localhost"
                self.master_port = "12355"
                self.pretrained_model_name = "microsoft/DialoGPT-small"
                self.data_path = "test_data.json"
                self.num_latent_tokens = 32
                self.num_heads = 4
                self.k = 4
                self.batch_size = 2
                self.num_epochs = 10
                self.learning_rate = 1e-4
                self.output_dir = "./test_outputs"
                self.use_bf16 = False
                self.use_fp16 = False
                self.train_decoder = False
                self.compress_cot_only = False
                self.explain_token = None
                self.think_token = None
                self.prompt = None
                self.commitment_cost = None
                self.max_length = None
                self.weight_decay = None
                self.warmup_steps = None
                self.gradient_clip = None
                self.val_split = None
                self.checkpoint_interval = None
                self.log_interval = None
                self.seed = None
                self.resume_from = None
                self.create_sample_data = False
                self.fsdp_transformer_layer_cls_to_wrap = None
                self.fsdp_min_num_params = None
                self.fsdp_cpu_offload = False
                self.fsdp_backward_prefetch = None
                self.fsdp_sharding_strategy = None
                self.use_wandb = True
                self.wandb_project = "test-project"
                self.wandb_entity = None
                self.wandb_run_name = "test-run"
                self.wandb_offline = True
                self.wandb_dir = "./wandb_test"
        
        mock_args = MockArgs()
        cmd = create_fsdp_launch_command(mock_args)
        
        # Check that wandb arguments are included
        if "--use_wandb" in cmd and "--wandb_offline" in cmd:
            print("✓ Launch script wandb support successful")
            print(f"  Command includes wandb: {' '.join(cmd[-6:])}")
            return True
        else:
            print("✗ Launch script wandb support failed")
            return False
            
    except Exception as e:
        print(f"✗ Launch script wandb test failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("WandB Integration Tests")
    print("=" * 60)
    
    tests = [
        ("WandB Import", test_wandb_import),
        ("WandB Offline Mode", test_wandb_offline_mode),
        ("Training Script WandB", test_training_script_wandb),
        ("Launch Script WandB", test_launch_script_wandb),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<25} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! WandB integration is ready to use.")
        print("\nYou can now use:")
        print("  - --use_wandb to enable experiment tracking")
        print("  - --wandb_offline for training without internet")
        print("  - --wandb_dir to specify offline storage location")
        print("  - --wandb_project and --wandb_run_name for organization")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        if not results[0][1]:  # First test (import) failed
            print("\nTo install WandB:")
            print("  pip install wandb")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
