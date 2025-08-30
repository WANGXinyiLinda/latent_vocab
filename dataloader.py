from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from random import shuffle
import json

class TextDataset(Dataset):
    """
    Dataset for text data in the format expected by VQVAETextReconstructor.
    Expects data in the format: List[Tuple[str, str]] where each tuple is (question, CoT)
    """
    
    def __init__(self, data: List[Tuple[str, str]], max_length: int = 512):
        self.data = data
        shuffle(self.data)
        self.max_length = max_length
        
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


def create_sample_data(output_path: str = "sample_data.json", num_samples: int = 100):
    """
    Create sample data for testing the training script.
    
    Args:
        output_path: Path to save sample data
        num_samples: Number of sample question-answer pairs to create
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
    
    answers = [
        "The capital of France is Paris. Paris is located in the northern part of France and is known for its rich history, culture, and landmarks like the Eiffel Tower.",
        "To solve a quadratic equation ax² + bx + c = 0, you can use the quadratic formula: x = (-b ± √(b² - 4ac)) / (2a). First, identify the coefficients a, b, and c, then substitute them into the formula.",
        "In Python, a list is mutable (can be changed after creation) and uses square brackets []. A tuple is immutable (cannot be changed after creation) and uses parentheses (). Lists are typically used for collections that might change, while tuples are used for fixed collections.",
        "Machine learning is a subset of artificial intelligence where computers learn patterns from data without being explicitly programmed. It involves algorithms that can identify patterns and make predictions or decisions based on input data.",
        "Exercise provides numerous benefits including improved cardiovascular health, stronger muscles and bones, better mental health, weight management, increased energy levels, and better sleep quality.",
        "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen. This process occurs in the chloroplasts of plant cells and is essential for life on Earth.",
        "The meaning of life is a philosophical question that has been debated for centuries. Different cultures, religions, and individuals have various perspectives on this profound question.",
        "To bake a chocolate cake, you'll need flour, sugar, cocoa powder, eggs, milk, and butter. Mix dry ingredients, beat wet ingredients, combine them, pour into a greased pan, and bake at 350°F for 25-30 minutes.",
        "Einstein's theory of relativity consists of two parts: special relativity and general relativity. Special relativity deals with space and time being relative to the observer's motion, while general relativity explains gravity as a curvature of spacetime.",
        "To write a good essay, start with a clear thesis statement, organize your ideas logically, provide evidence and examples, use clear and concise language, and conclude with a strong summary of your main points."
    ]
    
    for i in range(num_samples):
        q_idx = i % len(questions)
        a_idx = i % len(answers)
        sample_data.append({
            "question": questions[q_idx],
            "answer": answers[a_idx]
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created sample data with {num_samples} examples at {output_path}")