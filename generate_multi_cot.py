#!/usr/bin/env python3
"""
Improved script to generate multiple Chain of Thought (CoT) solutions for GSM8K questions.
Only keeps CoTs that arrive at the correct answer.

This version uses better models and includes debugging capabilities.

Usage:
    python generate_multi_cot_improved.py --data_path ./data/GSM8K/train.jsonl --output_path ./multi_cot_data.json --k 5
"""

import os
import sys
import json
import argparse
import re
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import random

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import set_seed


def extract_answer_from_cot(cot_text: str) -> Optional[str]:
    """
    Extract the final answer from a CoT solution.
    GSM8K answers are typically in the format "#### 123" at the end.
    
    Args:
        cot_text: The chain of thought text
        
    Returns:
        The extracted answer as a string, or None if not found
    """
    # Look for the pattern "#### number" at the end
    pattern = r'####\s*([+-]?\d+(?:\.\d+)?)'
    matches = re.findall(pattern, cot_text)
    
    if matches:
        return matches[-1]  # Return the last match (final answer)
    
    # If no #### pattern, try to find numbers at the end
    lines = cot_text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and re.match(r'^[+-]?\d+(?:\.\d+)?$', line):
            return line
    
    return None


def extract_ground_truth_answer(gsm8k_item: Dict) -> str:
    """
    Extract the ground truth answer from a GSM8K item.
    
    Args:
        gsm8k_item: Dictionary containing GSM8K question and answer
        
    Returns:
        The ground truth answer as a string
    """
    answer_text = gsm8k_item['answer']
    # Extract the final answer from the GSM8K format
    pattern = r'####\s*([+-]?\d+(?:\.\d+)?)'
    matches = re.findall(pattern, answer_text)
    
    if matches:
        return matches[-1]
    
    return answer_text


def generate_cot_with_pipeline(question: str, generator, k: int = 1, max_length: int = 512) -> str:
    """
    Generate a Chain of Thought solution using a text generation pipeline.
    
    Args:
        question: The math problem question
        generator: The text generation pipeline
        k: Number of CoT solutions to generate
        max_length: Maximum generation length
        
    Returns:
        Generated CoT solution
    """
    # Create prompt for CoT generation
    prompt = f"""Solve this math problem step by step. Show your reasoning clearly and end with the final answer in the format "#### [number]".

Problem: {question}

Solution:"""
    
    # Generate response
    result = generator(
        prompt,
        max_length=max_length,
        num_return_sequences=k,
        temperature=0.7,
        do_sample=True,
        pad_token_id=generator.tokenizer.eos_token_id
    )
    
    # Extract the generated text
    generated_text = [r['generated_text'].split("Solution:")[-1].strip() for r in result]
    
    return generated_text


def generate_cot_with_manual_prompt(question: str, model, tokenizer, device: str, 
                                  temperature: float = 0.7, max_length: int = 512, k: int = 1) -> str:
    """
    Generate a Chain of Thought solution using manual prompting.
    
    Args:
        question: The math problem question
        model: The language model
        tokenizer: The tokenizer
        device: Device to run on
        temperature: Sampling temperature
        max_length: Maximum generation length
        
    Returns:
        Generated CoT solution
    """
    # Create a more specific prompt for mathematical reasoning
    prompt = f"""Let's solve this math problem step by step.

Problem: {question}

Let me think through this step by step:

1. First, I need to understand what the problem is asking.
2. Then, I'll identify the key information and what I need to calculate.
3. I'll work through the calculation step by step.
4. Finally, I'll provide the answer in the format "#### [number]".

Let me start:

Step 1: Understanding the problem
The problem is asking me to: {question.split('?')[0]}?

Step 2: Identifying key information
Looking at the problem, I can see the following information:
"""
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=k
        )
    
    # Decode response
    generated_text = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    
    # Extract only the generated part (after the prompt)
    generated_text = [g.split("Let me start:")[-1].strip() for g in generated_text]
    
    return generated_text


def load_gsm8k_data(data_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """
    Load GSM8K data from JSONL file.
    
    Args:
        data_path: Path to the GSM8K data file
        max_samples: Maximum number of samples to load (None for all)
        
    Returns:
        List of GSM8K items
    """
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                data.append(item)
                if max_samples and len(data) >= max_samples:
                    break
    
    print(f"Loaded {len(data)} GSM8K examples from {data_path}")
    return data


def generate_multi_cot_data(data_path: str, output_path: str, k: int = 5, 
                           model_name: str = "microsoft/DialoGPT-medium",
                           max_samples: Optional[int] = None,
                           temperature: float = 0.7,
                           max_length: int = 512,
                           seed: int = 42,
                           debug: bool = False) -> None:
    """
    Generate multiple CoT solutions for GSM8K questions.
    
    Args:
        data_path: Path to GSM8K training data
        output_path: Path to save the multi-CoT data
        k: Number of CoT solutions to generate per question
        model_name: Name of the language model to use
        max_samples: Maximum number of questions to process
        temperature: Sampling temperature for generation
        max_length: Maximum generation length
        seed: Random seed for reproducibility
        debug: Whether to print debug information
    """
    # Set random seed
    set_seed(seed)
    
    # Load GSM8K data
    gsm8k_data = load_gsm8k_data(data_path, max_samples)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    try:
        # Try to use pipeline for better generation
        generator = pipeline(
            "text-generation",
            model=model_name,
            device=0 if device == "cuda" else -1,
            return_full_text=False
        )
        use_pipeline = True
    except Exception as e:
        print(f"Pipeline failed, falling back to manual generation: {e}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)
        model.eval()
        use_pipeline = False
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Generate multi-CoT data
    multi_cot_data = []
    correct_cot_count = 0
    total_cot_count = 0
    
    print(f"Generating {k} CoT solutions per question...")
    
    for item_idx, item in enumerate(tqdm(gsm8k_data, desc="Processing questions")):
        question = item['question']
        ground_truth = extract_ground_truth_answer(item)
        
        if debug:
            print(f"\n--- Question {item_idx + 1} ---")
            print(f"Question: {question}")
            print(f"Ground truth: {ground_truth}")
        
        # Generate k CoT solutions
        correct_cots = []
        try:
            if use_pipeline:
                cots = generate_cot_with_pipeline(
                    question, generator, k, max_length
                )
            else:
                cots = generate_cot_with_manual_prompt(
                    question, model, tokenizer, device,
                    temperature=current_temperature,
                    max_length=max_length,
                    k=k
                )
            
            for i, cot in enumerate(cots):
                # Extract answer from CoT
                predicted_answer = extract_answer_from_cot(cot)
                total_cot_count += 1
                
                if debug:
                    print(f"\nCoT {i+1}:")
                    print(f"Generated: {cot[:200]}...")
                    print(f"Predicted answer: {predicted_answer}")
                    print(f"Correct: {predicted_answer == ground_truth}")
                
                # Only keep CoTs with correct answers
                if predicted_answer and predicted_answer == ground_truth:
                    correct_cots.append(cot)
                    correct_cot_count += 1
                
        except Exception as e:
            print(f"Error generating CoT {i+1} for question: {e}")
            continue
        
        # Only include questions that have at least one correct CoT
        if correct_cots:
            multi_cot_item = {
                'question': question,
                'ground_truth': ground_truth,
                'num_correct_cots': len(correct_cots)
            }
            
            # Add CoT solutions
            for i, cot in enumerate(cots):
                multi_cot_item[f'cot{i+1}'] = cot
            
            multi_cot_data.append(multi_cot_item)
            
            if debug:
                print(f"✓ Question {item_idx + 1} has {len(cots)} correct CoTs")
    
    # Save results
    print(f"\nSaving results to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(multi_cot_data, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    print(f"\n=== Generation Statistics ===")
    print(f"Total questions processed: {len(gsm8k_data)}")
    print(f"Questions with at least one correct CoT: {len(multi_cot_data)}")
    print(f"Total CoTs generated: {total_cot_count}")
    print(f"Correct CoTs: {correct_cot_count}")
    if total_cot_count > 0:
        print(f"Accuracy: {correct_cot_count/total_cot_count*100:.2f}%")
    print(f"Average correct CoTs per question: {correct_cot_count/len(gsm8k_data):.2f}")
    
    if multi_cot_data:
        print(f"Average CoTs per successful question: {sum(item['num_correct_cots'] for item in multi_cot_data)/len(multi_cot_data):.2f}")


def main():
    parser = argparse.ArgumentParser(description="Generate multiple CoT solutions for GSM8K questions")
    parser.add_argument("--data_path", type=str, default="./data/GSM8K/train.jsonl",
                       help="Path to GSM8K training data")
    parser.add_argument("--output_path", type=str, default="./multi_cot_data.json",
                       help="Path to save the multi-CoT data")
    parser.add_argument("--k", type=int, default=5,
                       help="Number of CoT solutions to generate per question")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B",
                       help="Name of the language model to use")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of questions to process (None for all)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature for generation")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum generation length")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--debug", action="store_true",
                       help="Print debug information")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found: {args.data_path}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    print("=== Multi-CoT Generation Script (Improved) ===")
    print(f"Data path: {args.data_path}")
    print(f"Output path: {args.output_path}")
    print(f"Number of CoTs per question: {args.k}")
    print(f"Model: {args.model_name}")
    print(f"Max samples: {args.max_samples or 'All'}")
    print(f"Temperature: {args.temperature}")
    print(f"Max length: {args.max_length}")
    print(f"Seed: {args.seed}")
    print(f"Debug: {args.debug}")
    print()
    
    # Generate multi-CoT data
    generate_multi_cot_data(
        data_path=args.data_path,
        output_path=args.output_path,
        k=args.k,
        model_name=args.model_name,
        max_samples=args.max_samples,
        temperature=args.temperature,
        max_length=args.max_length,
        seed=args.seed,
        debug=args.debug
    )
    
    print(f"\n✅ Multi-CoT generation completed!")
    print(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()

