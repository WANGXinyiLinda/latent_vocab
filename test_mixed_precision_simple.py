#!/usr/bin/env python3
"""
Simple manual test script for mixed precision training functionality.

This script allows you to manually test different precision settings
without running the full automated test suite.
"""

import os
import sys
import subprocess
import json
import torch

def create_minimal_test_data():
    """Create minimal test data for training."""
    test_data = [
        {
            "text": "This is a test sentence for VQ-VAE training.",
            "explanation": "A simple test case to verify mixed precision training."
        },
        {
            "text": "Another test sentence with different content.",
            "explanation": "Second test case to ensure data loading works."
        }
    ]
    
    test_data_path = "./test_data.json"
    with open(test_data_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"✅ Created test data at: {test_data_path}")
    return test_data_path

def check_hardware_support():
    """Check hardware support for different precision types."""
    print("\n🔍 Hardware Support Check:")
    print("=" * 40)
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        print(f"✅ CUDA version: {torch.version.cuda}")
        
        if torch.cuda.is_bf16_supported():
            print("✅ bfloat16 supported on this CUDA device")
        else:
            print("⚠️  bfloat16 not supported on this CUDA device")
            
        print("✅ float16 supported on all CUDA devices")
    else:
        print("⚠️  CUDA not available - some tests may fail")
    
    print(f"✅ PyTorch version: {torch.__version__}")

def run_precision_test(test_name, precision_args, expected_dtype):
    """Run a single precision test."""
    print(f"\n🧪 Testing: {test_name}")
    print(f"Expected dtype: {expected_dtype}")
    print(f"Args: {precision_args}")
    print("-" * 50)
    
    # Create output directory
    test_output_dir = f"./test_outputs/{test_name}"
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable, "train_vq_vae.py",
        "--pretrained_model_name", "microsoft/DialoGPT-small",  # Small model for testing
        "--num_latent_tokens", "16",
        "--num_heads", "4",
        "--k", "4",
        "--max_length", "64",
        "--batch_size", "2",
        "--num_steps", "10",  # Very short training for testing
        "--learning_rate", "1e-4",
        "--warmup_steps", "2",
        "--save_steps", "5",
        "--log_steps", "2",
        "--test_steps", "5",
        "--train_data_path", "./test_data.json",
        "--output_dir", test_output_dir,
        "--seed", "42",
    ] + precision_args
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the training script
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            print("✅ Training completed successfully!")
            
            # Check for expected precision in logs
            if expected_dtype == torch.bfloat16:
                if "Using bfloat16 precision" in result.stdout:
                    print("✅ bfloat16 precision confirmed in logs")
                else:
                    print("⚠️  bfloat16 precision not found in logs")
                    
            elif expected_dtype == torch.float16:
                if "Using float16 precision with GradScaler" in result.stdout:
                    print("✅ float16 precision with GradScaler confirmed in logs")
                else:
                    print("⚠️  float16 precision not found in logs")
                    
            elif expected_dtype == torch.float32:
                if "Using float32 precision" in result.stdout:
                    print("✅ float32 precision confirmed in logs")
                else:
                    print("⚠️  float32 precision not found in logs")
            
            # Check for other important messages
            if "Model created successfully" in result.stdout:
                print("✅ Model creation confirmed")
            if "Training completed successfully!" in result.stdout:
                print("✅ Training completion confirmed")
                
            return True
            
        else:
            print(f"❌ Training failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Training timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Training failed with exception: {e}")
        return False

def main():
    """Main test function."""
    print("Mixed Precision Training Test")
    print("=" * 40)
    
    # Check hardware support
    check_hardware_support()
    
    # Create test data
    test_data_path = create_minimal_test_data()
    
    # Test different precision settings
    tests = [
        ("Default FP32", [], torch.float32),
        ("BF16 Precision", ["--use_bf16"], torch.bfloat16),
        ("FP16 Precision", ["--use_fp16"], torch.float16),
        ("BF16 + FSDP", ["--use_bf16", "--use_fsdp"], torch.bfloat16),
        ("FP16 + FSDP", ["--use_fp16", "--use_fsdp"], torch.float16),
    ]
    
    results = []
    
    for test_name, args, expected_dtype in tests:
        success = run_precision_test(test_name, args, expected_dtype)
        results.append((test_name, success))
        print(f"\n{'='*60}\n")
    
    # Summary
    print("📊 TEST SUMMARY")
    print("=" * 40)
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("💥 Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    main()
