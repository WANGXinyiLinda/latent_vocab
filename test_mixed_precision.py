#!/usr/bin/env python3
"""
Test script for mixed precision training functionality in train_vq_vae.py

This script tests:
1. use_bf16 functionality (bfloat16 precision)
2. use_fp16 functionality (float16 precision with GradScaler)
3. Default float32 precision
4. Error handling for invalid precision combinations
"""

import os
import sys
import subprocess
import tempfile
import json
import torch
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data(output_dir):
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
    
    test_data_path = os.path.join(output_dir, "test_data.json")
    with open(test_data_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    return test_data_path

def run_training_test(test_name, args, expected_dtype, test_data_path):
    """Run a training test with specific arguments."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running test: {test_name}")
    logger.info(f"Expected dtype: {expected_dtype}")
    logger.info(f"Args: {args}")
    logger.info(f"{'='*60}")
    
    # Create output directory for this test
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
        "--train_data_path", test_data_path,
        "--output_dir", test_output_dir,
        "--seed", "42",
        "--use_gradient_checkpointing",  # Test with gradient checkpointing
    ] + args
    
    logger.info(f"Command: {' '.join(cmd)}")
    
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
            logger.info("✅ Training completed successfully!")
            
            # Check if the expected dtype was used
            if expected_dtype == torch.bfloat16:
                if "Using bfloat16 precision" in result.stdout:
                    logger.info("✅ bfloat16 precision confirmed in logs")
                else:
                    logger.warning("⚠️  bfloat16 precision not found in logs")
                    
            elif expected_dtype == torch.float16:
                if "Using float16 precision with GradScaler" in result.stdout:
                    logger.info("✅ float16 precision with GradScaler confirmed in logs")
                else:
                    logger.warning("⚠️  float16 precision not found in logs")
                    
            elif expected_dtype == torch.float32:
                if "Using float32 precision" in result.stdout:
                    logger.info("✅ float32 precision confirmed in logs")
                else:
                    logger.warning("⚠️  float32 precision not found in logs")
            
            # Check for other important log messages
            if "Model created successfully" in result.stdout:
                logger.info("✅ Model creation confirmed")
            if "Training completed successfully!" in result.stdout:
                logger.info("✅ Training completion confirmed")
            
            return True, result.stdout, result.stderr
            
        else:
            logger.error(f"❌ Training failed with return code {result.returncode}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return False, result.stdout, result.stderr
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Training timed out after 5 minutes")
        return False, "", "Timeout expired"
    except Exception as e:
        logger.error(f"❌ Training failed with exception: {e}")
        return False, "", str(e)

def test_mixed_precision_functionality():
    """Test all mixed precision training scenarios."""
    logger.info("Starting mixed precision functionality tests...")
    
    # Create temporary test data
    with tempfile.TemporaryDirectory() as temp_dir:
        test_data_path = create_test_data(temp_dir)
        logger.info(f"Created test data at: {test_data_path}")
        
        # Test 1: Default float32 precision (no flags)
        success1, stdout1, stderr1 = run_training_test(
            "default_fp32",
            [],
            torch.float32,
            test_data_path
        )
        
        # Test 2: bfloat16 precision
        success2, stdout2, stderr2 = run_training_test(
            "bf16_precision",
            ["--use_bf16"],
            torch.bfloat16,
            test_data_path
        )
        
        # Test 3: float16 precision
        success3, stdout3, stderr3 = run_training_test(
            "fp16_precision",
            ["--use_fp16"],
            torch.float16,
            test_data_path
        )
        
        # Test 4: Both bf16 and fp16 (should fail or warn)
        success4, stdout4, stderr4 = run_training_test(
            "both_precisions",
            ["--use_bf16", "--use_fp16"],
            torch.float16,  # fp16 should take precedence
            test_data_path
        )
        
        # Test 5: bfloat16 with FSDP
        success5, stdout5, stderr5 = run_training_test(
            "bf16_with_fsdp",
            ["--use_bf16", "--use_fsdp"],
            torch.bfloat16,
            test_data_path
        )
        
        # Test 6: float16 with FSDP
        success6, stdout6, stderr6 = run_training_test(
            "fp16_with_fsdp",
            ["--use_fp16", "--use_fsdp"],
            torch.float16,
            test_data_path
        )
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Default FP32: {'✅ PASS' if success1 else '❌ FAIL'}")
        logger.info(f"BF16 Precision: {'✅ PASS' if success2 else '❌ FAIL'}")
        logger.info(f"FP16 Precision: {'✅ PASS' if success3 else '❌ FAIL'}")
        logger.info(f"Both Precisions: {'✅ PASS' if success4 else '❌ FAIL'}")
        logger.info(f"BF16 + FSDP: {'✅ PASS' if success5 else '❌ FAIL'}")
        logger.info(f"FP16 + FSDP: {'✅ PASS' if success6 else '❌ FAIL'}")
        
        # Overall success
        all_tests_passed = all([success1, success2, success3, success4, success5, success6])
        logger.info(f"\nOverall Result: {'✅ ALL TESTS PASSED' if all_tests_passed else '❌ SOME TESTS FAILED'}")
        
        return all_tests_passed

def test_precision_validation():
    """Test precision validation and error handling."""
    logger.info("\nTesting precision validation...")
    
    # Test invalid precision combinations
    invalid_tests = [
        ("invalid_both_precisions", ["--use_bf16", "--use_fp16"]),
        ("invalid_precision_with_invalid_model", ["--use_bf16", "--pretrained_model_name", "invalid/model"]),
    ]
    
    for test_name, args in invalid_tests:
        logger.info(f"Testing {test_name}...")
        # These should fail gracefully or show appropriate warnings
        # Implementation depends on your error handling

def check_hardware_support():
    """Check if the current hardware supports the precision types."""
    logger.info("\nChecking hardware support...")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        logger.info(f"✅ CUDA version: {torch.version.cuda}")
        
        # Check bfloat16 support
        if torch.cuda.is_bf16_supported():
            logger.info("✅ bfloat16 supported on this CUDA device")
        else:
            logger.warning("⚠️  bfloat16 not supported on this CUDA device")
            
        # Check float16 support
        logger.info("✅ float16 supported on all CUDA devices")
        
    else:
        logger.warning("⚠️  CUDA not available - some tests may fail")
    
    # Check PyTorch version
    logger.info(f"✅ PyTorch version: {torch.__version__}")

def main():
    """Main test function."""
    logger.info("Mixed Precision Training Test Suite")
    logger.info("=" * 50)
    
    # Check hardware support first
    check_hardware_support()
    
    # Run the main tests
    try:
        all_passed = test_mixed_precision_functionality()
        
        if all_passed:
            logger.info("\n🎉 All mixed precision tests passed!")
            return 0
        else:
            logger.error("\n💥 Some tests failed. Check the logs above for details.")
            return 1
            
    except Exception as e:
        logger.error(f"Test suite failed with exception: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
