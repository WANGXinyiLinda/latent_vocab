python generate_multi_cot.py \
    --data_path ./data/GSM8K/train.jsonl \
    --output_path ./data/GSM8K/multi_cot_data.json \
    --k 5 \
    --model_name Qwen/Qwen3-8B \
    --max_samples 100 \
    --temperature 1 \
    --max_length 1024 \
    --seed 42 \
    --debug