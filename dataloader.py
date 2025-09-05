from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from random import shuffle
import json

class TextDataset(Dataset):
    """
    Dataset for text data in the format expected by VQVAETextReconstructor.
    Expects data in the format: List[Tuple[str, str]] where each tuple is (question, CoT)
    or List[Tuple[str, List[str]]] where each tuple is (question, [CoT1, CoT2, ..., CoTn])
    """
    
    def __init__(self, data: List[Tuple[str, str]], max_length: int = 512, num_cot_traces: int = 1):
        self.data = data
        self.max_length = max_length
        self.num_cot_traces = num_cot_traces
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def load_text_data(data_path: str) -> List[Tuple[str, str]]:
    """
    Load text data from various formats.
    
    Args:
        data_path: Path to data file (supports .json, .txt, .jsonl)
        
    Returns:
        List of (question, answer) tuples
    """
    data = []
    
    if data_path.endswith('.json'):
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        # Handle different JSON formats
        if isinstance(raw_data, list):
            for item in raw_data:
                if isinstance(item, dict):
                    # Try different key combinations
                    if 'question' in item and 'answer' in item:
                        data.append((item['question'], item['answer']))
                    elif 'question' in item and 'explanation' in item:
                        data.append((item['question'], item['explanation']))
                    elif 'input' in item and 'output' in item:
                        data.append((item['input'], item['output']))
                    elif 'prompt' in item and 'completion' in item:
                        data.append((item['prompt'], item['completion']))
                    elif 'text' in item and 'summary' in item:
                        data.append((item['text'], item['summary']))
                    elif len(item) == 2:
                        # Assume first is question, second is answer
                        keys = list(item.keys())
                        data.append((str(item[keys[0]]), str(item[keys[1]])))
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    data.append((str(item[0]), str(item[1])))
        elif isinstance(raw_data, dict):
            # Single example
            if 'question' in raw_data and 'answer' in raw_data:
                data.append((raw_data['question'], raw_data['answer']))
                
    elif data_path.endswith('.jsonl'):
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    if isinstance(item, dict):
                        if 'question' in item and 'answer' in item:
                            data.append((item['question'], item['answer']))
                        elif 'question' in item and 'explanation' in item:
                            data.append((item['question'], item['explanation']))
                        elif 'input' in item and 'output' in item:
                            data.append((item['input'], item['output']))
                        elif 'prompt' in item and 'completion' in item:
                            data.append((item['prompt'], item['completion']))
                        elif 'text' in item and 'summary' in item:
                            data.append((item['text'], item['summary']))
                        elif len(item) == 2:
                            keys = list(item.keys())
                            data.append((str(item[keys[0]]), str(item[keys[1]])))
                    elif isinstance(item, (list, tuple)) and len(item) == 2:
                        data.append((str(item[0]), str(item[1])))
                        
    elif data_path.endswith('.txt'):
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                if i + 1 < len(lines):
                    question = lines[i].strip()
                    answer = lines[i + 1].strip()
                    if question and answer:
                        data.append((question, answer))
    
    if not data:
        raise ValueError(f"No valid data found in {data_path}")
    
    print(f"Loaded {len(data)} question-answer pairs from {data_path}")
    return data


def load_multi_cot_data(data_path: str, num_cot_traces: int = 1) -> List[Tuple[str, List[str]]]:
    """
    Load text data with multiple CoT traces per question.
    
    Args:
        data_path: Path to data file (supports .json, .jsonl)
        num_cot_traces: Number of CoT traces expected per question
        
    Returns:
        List of (question, [CoT1, CoT2, ..., CoTn]) tuples
    """
    data = []
    
    if data_path.endswith('.json'):
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        # Handle different JSON formats
        if isinstance(raw_data, list):
            for item in raw_data:
                if isinstance(item, dict):
                    question = None
                    cots = []
                    
                    # Extract question
                    if 'question' in item:
                        question = item['question']
                    elif 'input' in item:
                        question = item['input']
                    elif 'prompt' in item:
                        question = item['prompt']
                    
                    # Extract CoT traces
                    if 'answers' in item and isinstance(item['answers'], list):
                        cots = item['answers']
                    elif 'answer' in item:
                        cots = [item['answer']]
                    elif 'output' in item:
                        cots = [item['output']]
                    elif 'completion' in item:
                        cots = [item['completion']]
                    elif 'explanation' in item:
                        cots = [item['explanation']]
                    
                    if question and cots:
                        # Ensure we have the right number of CoT traces
                        if len(cots) < num_cot_traces:
                            # Repeat the last CoT to fill up to num_cot_traces
                            while len(cots) < num_cot_traces:
                                cots.append(cots[-1])
                        elif len(cots) > num_cot_traces:
                            # Truncate to num_cot_traces
                            cots = cots[:num_cot_traces]
                        
                        data.append((question, cots))
                        
    elif data_path.endswith('.jsonl'):
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    if isinstance(item, dict):
                        question = None
                        cots = []
                        
                        # Extract question
                        if 'question' in item:
                            question = item['question']
                        elif 'input' in item:
                            question = item['input']
                        elif 'prompt' in item:
                            question = item['prompt']
                        
                        # Extract CoT traces
                        if 'answers' in item and isinstance(item['answers'], list):
                            cots = item['answers']
                        elif 'answer' in item:
                            cots = [item['answer']]
                        elif 'output' in item:
                            cots = [item['output']]
                        elif 'completion' in item:
                            cots = [item['completion']]
                        elif 'explanation' in item:
                            cots = [item['explanation']]
                        
                        if question and cots:
                            # Ensure we have the right number of CoT traces
                            if len(cots) < num_cot_traces:
                                # Repeat the last CoT to fill up to num_cot_traces
                                while len(cots) < num_cot_traces:
                                    cots.append(cots[-1])
                            elif len(cots) > num_cot_traces:
                                # Truncate to num_cot_traces
                                cots = cots[:num_cot_traces]
                            
                            data.append((question, cots))
    
    if not data:
        raise ValueError(f"No valid multi-CoT data found in {data_path}")
    
    print(f"Loaded {len(data)} question-multi-CoT pairs from {data_path}")
    print(f"Each question has {num_cot_traces} CoT traces")
    return data


def create_sample_data(output_path: str = "sample_data.json", num_samples: int = 100, num_cot_traces: int = 1):
    """
    Create sample data for testing the training script.
    
    Args:
        output_path: Path to save sample data
        num_samples: Number of sample question-answer pairs to create
        num_cot_traces: Number of CoT traces per question (for multi-CoT training)
    """
    sample_data = []
    
    questions = [
        "What is the capital of France?",
        "How do you solve a quadratic equation?",
        "What is the difference between a list and a tuple in Python?",
        "Explain the concept of machine learning.",
        "What are the benefits of exercise?",
        "How does photosynthesis work?",
        "What is the meaning of life?",
        "How do you bake a chocolate cake?",
        "What is the theory of relativity?",
        "How do you write a good essay?"
    ]
    
    # Multiple CoT traces for each question to encourage diverse reasoning
    multi_cot_answers = {
        "What is the capital of France?": [
            "The capital of France is Paris. Paris is located in the northern part of France and is known for its rich history, culture, and landmarks like the Eiffel Tower.",
            "France's capital city is Paris. This city has been the political and cultural center of France for centuries, home to famous monuments and museums.",
            "Paris serves as the capital of France. It's a major European city known for art, fashion, and cuisine, situated along the Seine River."
        ],
        "How do you solve a quadratic equation?": [
            "To solve a quadratic equation ax² + bx + c = 0, you can use the quadratic formula: x = (-b ± √(b² - 4ac)) / (2a). First, identify the coefficients a, b, and c, then substitute them into the formula.",
            "There are several methods to solve quadratic equations. The quadratic formula is most general: x = (-b ± √(b² - 4ac)) / (2a). You can also factor if possible or complete the square.",
            "For ax² + bx + c = 0, the quadratic formula gives x = (-b ± √(b² - 4ac)) / (2a). Calculate the discriminant (b² - 4ac) first to determine the nature of solutions."
        ],
        "What is the difference between a list and a tuple in Python?": [
            "In Python, a list is mutable (can be changed after creation) and uses square brackets []. A tuple is immutable (cannot be changed after creation) and uses parentheses (). Lists are typically used for collections that might change, while tuples are used for fixed collections.",
            "Lists and tuples differ in mutability. Lists use [] and can be modified, while tuples use () and are immutable. Lists are for dynamic data, tuples for fixed data.",
            "The key difference is mutability: lists are changeable with [] syntax, tuples are unchangeable with () syntax. Lists are for variable collections, tuples for constant data."
        ],
        "Explain the concept of machine learning.": [
            "Machine learning is a subset of artificial intelligence where computers learn patterns from data without being explicitly programmed. It involves algorithms that can identify patterns and make predictions or decisions based on input data.",
            "ML is AI that learns from examples. Instead of programming rules, you provide data and let algorithms find patterns to make predictions or classifications.",
            "Machine learning enables computers to learn and improve from experience. It uses statistical techniques to build models that can make predictions on new, unseen data."
        ],
        "What are the benefits of exercise?": [
            "Exercise provides numerous benefits including improved cardiovascular health, stronger muscles and bones, better mental health, weight management, increased energy levels, and better sleep quality.",
            "Regular exercise improves physical fitness, reduces disease risk, enhances mood, and increases longevity. It strengthens the heart, muscles, and immune system.",
            "Exercise benefits include better heart health, stronger muscles, improved mental well-being, weight control, higher energy, and enhanced sleep patterns."
        ]
    }
    
    # Fallback single answers for questions not in multi_cot_answers
    fallback_answers = [
        "This is a complex topic that requires careful consideration and analysis.",
        "The answer involves multiple factors that need to be examined systematically.",
        "Understanding this concept requires breaking it down into its fundamental components.",
        "This question can be approached from several different perspectives and methodologies.",
        "The solution involves applying established principles and logical reasoning."
    ]
    
    for i in range(num_samples):
        q_idx = i % len(questions)
        question = questions[q_idx]
        
        if question in multi_cot_answers and num_cot_traces > 1:
            # Use multiple CoT traces for this question
            available_cots = multi_cot_answers[question]
            # Select up to num_cot_traces, cycling through available ones
            selected_cots = []
            for j in range(num_cot_traces):
                selected_cots.append(available_cots[j % len(available_cots)])
            
            if num_cot_traces == 1:
                sample_data.append({
                    "question": question,
                    "answer": selected_cots[0]
                })
            else:
                sample_data.append({
                    "question": question,
                    "answers": selected_cots
                })
        else:
            # Use single answer (backward compatibility)
            a_idx = i % len(fallback_answers)
            sample_data.append({
                "question": question,
                "answer": fallback_answers[a_idx]
            })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created sample data with {num_samples} examples at {output_path}")
    if num_cot_traces > 1:
        print(f"Using {num_cot_traces} CoT traces per question for multi-CoT training")