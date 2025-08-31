import torch
from transformers import AutoTokenizer, AutoModel
from typing import Union, List, Tuple, Sequence
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionQuantizer(nn.Module):
    """
    Multi-head attention based quantizer that maps representations to k vectors.
    Uses k separate attention mechanisms, each with its own learnable query vector.
    """
    
    def __init__(self, hidden_size: int, num_heads: int, k: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.k = k
        
        # k separate learnable query vectors, each for a different attention mechanism
        self.learnable_queries = nn.Parameter(torch.randn(k, hidden_size))
        
        # k separate multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(k)
        ])
        
        # Output projection for each attention output
        self.output_projections = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(k)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, representations: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            representations: Input representations of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask of shape (batch_size, seq_len)
        Returns:
            Quantized vectors of shape (batch_size, k, hidden_size)
        """
        batch_size = representations.shape[0]
        outputs = []
        
        # Process each attention mechanism separately
        for i in range(self.k):
            # Get the i-th learnable query and expand to batch size
            # query: (batch_size, 1, hidden_size)
            query = self.learnable_queries[i:i+1].unsqueeze(0).expand(batch_size, -1, -1)
            
            # Apply the i-th attention mechanism
            # query as query, representations as key and value
            attended_output, _ = self.attention_layers[i](
                query=query,
                key=representations,
                value=representations,
                attn_mask=attention_mask
            )
            
            # Apply output projection
            output = self.output_projections[i](attended_output)
            outputs.append(output)
        
        # Concatenate all outputs along the k dimension
        # outputs: list of (batch_size, 1, hidden_size) -> (batch_size, k, hidden_size)
        concatenated_output = torch.cat(outputs, dim=1)
        
        # Apply layer norm
        final_output = self.layer_norm(concatenated_output)
        
        return final_output


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer that maps continuous vectors to discrete indices.
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Embedding table
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        # Initialize embeddings
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: Input vectors of shape (batch_size, k, embedding_dim)
            
        Returns:
            Tuple of (quantized, loss, perplexity, encoding_indices)
        """
        # Reshape inputs for easier processing
        batch_size, k, embedding_dim = inputs.shape
        flat_inputs = inputs.view(-1, embedding_dim)
        
        # Calculate distances to all embeddings
        distances = torch.sum(flat_inputs**2, dim=1, keepdim=True) + \
                   torch.sum(self.embedding.weight**2, dim=1) - \
                   2 * torch.matmul(flat_inputs, self.embedding.weight.t())
        
        # Find closest embeddings
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized.view(batch_size, k, embedding_dim)
        
        # Calculate loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Calculate perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, loss, perplexity, encoding_indices.view(batch_size, k)


def clean_tokens_for_display(tokens: List[str]) -> List[str]:
    """
    Clean BPE tokens for human-readable display by removing special characters.
    
    Args:
        tokens: List of tokenized tokens
        
    Returns:
        List of cleaned tokens
    """
    cleaned = []
    for token in tokens:
        # Remove common BPE artifacts
        clean_token = token.replace('Ġ', ' ').replace('Ċ', '\n').replace('▁', ' ')
        # Remove other common artifacts
        clean_token = clean_token.replace('Ċ', '\n').replace('ċ', '\n')
        clean_token = clean_token.replace('Ġ', ' ').replace('ġ', ' ')
        cleaned.append(clean_token)
    return cleaned


def get_last_layer_representations(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    inputs: Union[str, Sequence],
    device: str = "cpu",
    max_length: int = 512,
    return_token_text: bool = False,
    split_delimiters: Sequence[str] = ["\n", ".", "!", "?"],
    return_cot_only: bool = False
) -> Union[Tuple[List[List[torch.Tensor]], List[List[List[str]]], List[str]], 
           Tuple[List[List[torch.Tensor]], List[List[List[str]]]],
           Tuple[List[List[torch.Tensor]], List[List[List[int]]], List[str]], 
           Tuple[List[List[torch.Tensor]], List[List[List[int]]]],
           Tuple[List[torch.Tensor], List[List[str]], List[str]], 
           Tuple[List[torch.Tensor], List[List[str]]],
           Tuple[List[torch.Tensor], List[List[int]], List[str]], 
           Tuple[List[torch.Tensor], List[List[int]]]
           ]:
    """
    Extract the last layer representations from a Hugging Face pretrained language model.
    
    Args:
        model (AutoModel): Hugging Face pretrained model
        tokenizer (AutoTokenizer): Hugging Face pretrained tokenizer
        inputs (Union[str, Sequence]): 
            - Single string input
            - Question sequence, CoT sequence, and optional return_cot_only sequence
        device (str): Device to run the model on ("cpu", "cuda", or specific device)
        max_length (int): Maximum sequence length for tokenization
        return_token_text (bool): Whether to return token text (True) or token indices (False)
        split_delimiters (Sequence[str]): Sequence of text delimiters to split on (e.g., ["\n", ".", "Answer:"])
        return_cot_only (bool): For question+cot pairs, if True, only return representations and tokens from the CoT part
        
    Returns:
        For text input:
            - If split_delimiters=None and return_token_text=False: tuple of (tensor, token_indices_list)
            - If split_delimiters=None and return_token_text=True: tuple of (tensor, token_text_list)
            - If split_delimiters is provided and return_token_text=False: tuple of (list_of_tensors, list_of_token_indices_lists)
            - If split_delimiters is provided and return_token_text=True: tuple of (list_of_tensors, list_of_token_text_lists)
            
        For question+cot pairs input:
            - If split_delimiters=None and return_token_text=False: tuple of (list_of_tensors, list_of_token_indices_lists, list_of_questions)
            - If split_delimiters=None and return_token_text=True: tuple of (list_of_tensors, list_of_token_text_lists, list_of_questions)
            - If split_delimiters is provided and return_token_text=False: tuple of (list_of_lists_of_tensors, list_of_lists_of_token_indices_lists, list_of_questions)
            - If split_delimiters is provided and return_token_text=True: tuple of (list_of_lists_of_tensors, list_of_lists_of_token_text_lists, list_of_questions)
    """
    
    # Handle batch processing
    if isinstance(inputs, str):
        # Single regular text
        is_qc_pairs = False
        questions = [None]
        cots = [None]
        cot_only_flags = [False]
        concatenated_texts = [inputs]
    elif isinstance(inputs, Sequence):
        if len(inputs) == 2:
            # Question+cot pair tuple
            is_qc_pairs = True
            questions = inputs[0]
            cots = inputs[1]
            cot_only_flags = [return_cot_only] * len(inputs[0])
            concatenated_texts = [f"{q} {c}" for q, c in zip(questions, cots)]
        elif len(inputs) == 3:
            # Question+cot pair tuple with return_cot_only flag
            is_qc_pairs = True
            questions = inputs[0]
            cots = inputs[1]
            cot_only_flags = inputs[2]
            concatenated_texts = [f"{q} {c}" for q, c in zip(questions, cots)]
        else:
            raise ValueError(f"Invalid tuple length {len(inputs)}. Expected 2 or 3 elements.")
    else:
        raise ValueError(f"Invalid text type: {type(inputs)}")
    
    # Tokenize the input texts
    inputs = tokenizer(
        concatenated_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True
    )
    
    # Move inputs to device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Get the offset mappings for all texts
    offset_mappings = [offset.cpu().numpy() for offset in inputs["offset_mapping"]]
    
    # Run the model
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
    
    # Extract the last layer hidden states
    # hidden_states is a tuple of (num_layers + 1) tensors
    # The last element is the final layer output
    last_hidden_states = outputs.hidden_states[-1]  # Shape: (batch_size, seq_len, hidden_size)
    
    # Convert to CPU if using CUDA
    if device != "cpu":
        last_hidden_states = last_hidden_states.cpu()
    
    # Process each text in the batch
    batch_results = []
    batch_tokens = []
    batch_questions = []
    
    for batch_idx in range(len(concatenated_texts)):
        # Get the hidden states for this text
        text_hidden_states = last_hidden_states[batch_idx]  # Shape: (seq_len, hidden_size)
        
        if return_token_text:
            text_tokens = tokenizer.convert_ids_to_tokens(input_ids[batch_idx])
        else:
            text_tokens = input_ids[batch_idx].cpu().tolist()
        
        # Get the offset mapping for this text
        text_offset_mapping = offset_mappings[batch_idx]
        
        # Filter out special tokens and get meaningful representations
        meaningful_tokens = []
        meaningful_representations = []
        meaningful_offsets = []
        
        for i, (token, offset) in enumerate(zip(text_tokens, text_offset_mapping)):
            # Skip special tokens and padding
            if return_token_text:
                if token in tokenizer.all_special_tokens or offset[0] == offset[1]:
                    continue
            else:
                if token in tokenizer.all_special_ids or offset[0] == offset[1]:
                    continue
            
            meaningful_tokens.append(token)
            meaningful_representations.append(text_hidden_states[i])
            meaningful_offsets.append(offset)
        
        # Handle CoT-only extraction for question+cot pairs
        if is_qc_pairs and cot_only_flags[batch_idx]:
            # Find where the CoT part starts in the concatenated text
            cot_start_pos = len(questions[batch_idx])
            
            # Filter tokens to only include those from the CoT part
            cot_tokens = []
            cot_representations = []
            
            for token, repr_tensor, offset in zip(meaningful_tokens, meaningful_representations, meaningful_offsets):
                # Check if this token belongs to the CoT part
                if offset[0] >= cot_start_pos:
                    cot_tokens.append(token)
                    cot_representations.append(repr_tensor)
            
            # Use CoT-only tokens and representations
            meaningful_tokens = cot_tokens
            meaningful_representations = cot_representations
            meaningful_offsets = [offset for offset in meaningful_offsets if offset[0] >= cot_start_pos]
        
        # Handle splitting for this text
        if split_delimiters:
            # Split representations and tokens based on text-level delimiters
            split_representations = []
            split_token_lists = []
            
            # First, find all split positions in the original text
            split_positions = []
            for delimiter in split_delimiters:
                if is_qc_pairs and cot_only_flags[batch_idx]:
                    start = cot_start_pos
                else:
                    start = 0
                while True:
                    pos = concatenated_texts[batch_idx].find(delimiter, start)
                    if pos == -1:
                        break
                    split_positions.append(pos + len(delimiter))  # Split after the delimiter
                    start = pos + 1
            
            split_positions = sorted(split_positions)
            
            # Group tokens by their text positions
            current_reprs = []
            current_tokens = []
            
            for token, repr_tensor, offset in zip(meaningful_tokens, meaningful_representations, meaningful_offsets):
                current_reprs.append(repr_tensor)
                current_tokens.append(token)
                
                # Check if we've passed a split position
                while split_positions and offset[1] >= split_positions[0]:
                    # Finalize current segment
                    if current_reprs:
                        split_representations.append(torch.stack(current_reprs))
                        split_token_lists.append(current_tokens)
                        current_reprs = []
                        current_tokens = []
                    
                    split_positions.pop(0)  # Remove the processed split position
            
            # Add final segment if there are remaining tokens
            if current_reprs:
                split_representations.append(torch.stack(current_reprs))
                split_token_lists.append(current_tokens)
            
            # Handle empty case
            if not split_representations:
                empty_tensor = torch.empty(0, last_hidden_states.shape[-1])
                split_representations = [empty_tensor]
                split_token_lists = [[]]
            
            batch_results.append(split_representations)
            batch_tokens.append(split_token_lists)
            if is_qc_pairs:
                batch_questions.append(questions[batch_idx])
            else:
                batch_questions.append(None)
        else:
            # No splitting - stack the representations
            if meaningful_representations:
                meaningful_representations = torch.stack(meaningful_representations)
            else:
                # If no meaningful tokens, return empty tensor
                meaningful_representations = torch.empty(0, last_hidden_states.shape[-1])
            
            batch_results.append(meaningful_representations)
            batch_tokens.append(meaningful_tokens)
            if is_qc_pairs:
                batch_questions.append(questions[batch_idx])
            else:
                batch_questions.append(None)
    # Return results based on input type and parameters
    if is_qc_pairs:
        return batch_results, batch_tokens, batch_questions
    else:
        return batch_results, batch_tokens


# Example usage
if __name__ == "__main__":
    # Run the main demo
    print("Running VQ-VAE main demo...")
    
    # Example 1: Basic usage
    text = "Hello world! This is a test sentence."
    model_name = "../models/OLMo-2-0425-1B-Base"
    
    # Load tokenizer and model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Example 3: Split by text delimiters (handles tokenization mismatches)
    print("\n=== Split by Tokens ===")
    text_with_splits = "First sentence.\nSecond sentence.\nThird sentence."
    split_reprs, split_tokens = get_last_layer_representations(
        model,
        tokenizer,
        text_with_splits, 
        return_token_text=True,
        split_delimiters=["\n", "."]
    )
    print(f"Number of segments: {len(split_reprs[0])}")
    for i, (reprs, tokens) in enumerate(zip(split_reprs[0], split_tokens[0])):
        print(f"Segment {i}: {tokens} -> Shape: {reprs.shape}")
        
    # Example 3: Split by text delimiters (handles tokenization mismatches)
    print("\n=== Token indices version===")
    text_with_splits = "First sentence.\nSecond sentence.\nThird sentence."
    split_reprs, split_tokens = get_last_layer_representations(
        model,
        tokenizer,
        text_with_splits, 
        split_delimiters=["\n", "."]
    )
    print(f"Number of segments: {len(split_reprs[0])}")
    for i, (reprs, tokens) in enumerate(zip(split_reprs[0], split_tokens[0])):
        print(f"Segment {i}: {tokens} -> Shape: {reprs.shape}")
    
    # Example 4: Split by words that might be tokenized into multiple tokens
    print("\n=== Split by Multiple Tokens ===")
    text_multi = "Question: What is AI? Answer: AI is intelligence. Explanation: It's complex."
    split_reprs, split_tokens = get_last_layer_representations(
        model,
        tokenizer,
        text_multi, 
        return_token_text=True,
        split_delimiters=["Question:", "Answer:", "Explanation:"]
    )
    print(f"Number of segments: {len(split_reprs[0])}")
    for i, (reprs, tokens) in enumerate(zip(split_reprs[0], split_tokens[0])):
        print(f"Segment {i}: {tokens} -> Shape: {reprs.shape}")
        
    print("\n=== Token indices version===")
    split_reprs, split_tokens = get_last_layer_representations(
        model,
        tokenizer,
        text_multi, 
        split_delimiters=["Question:", "Answer:", "Explanation:"]
    )
    print(f"Number of segments: {len(split_reprs[0])}")
    for i, (reprs, tokens) in enumerate(zip(split_reprs[0], split_tokens[0])):
        print(f"Segment {i}: {tokens} -> Shape: {reprs.shape}")
    
    # Example 5: Demonstrate handling of tokenization mismatches
    print("\n=== Tokenization Mismatch Handling ===")
    text_demo = "Hello world! How are you?\nI'm doing great."
    print(f"Original text: {repr(text_demo)}")
    # Split by '!' which might be tokenized with the word before it
    split_reprs, split_tokens = get_last_layer_representations(
        model,
        tokenizer,
        text_demo, 
        return_token_text=True,
        split_delimiters=["!", "?", "\n"]
    )
    print(f"Number of segments: {len(split_reprs[0])}")
    for i, (reprs, tokens) in enumerate(zip(split_reprs[0], split_tokens[0])):
        print(f"Segment {i}: {tokens} -> Shape: {reprs.shape}")
    
    print("\n=== Token indices version===")
    split_reprs, split_tokens = get_last_layer_representations(
        model,
        tokenizer,
        text_demo, 
        split_delimiters=["!", "?", "\n"]
    )
    print(f"Number of segments: {len(split_reprs[0])}")
    for i, (reprs, tokens) in enumerate(zip(split_reprs[0], split_tokens[0])):
        print(f"Segment {i}: {tokens} -> Shape: {reprs.shape}")

    # Example 6: Batch processing
    print("\n=== Batch Processing ===")
    batch_texts = [
        "First text. Second sentence.",
        "Another text with multiple sentences. Here is more content.",
        "Short text."
    ]
    
    # Process batch without splitting
    batch_reprs, batch_tokens = get_last_layer_representations(
        model,
        tokenizer,
        batch_texts, 
        return_token_text=True,
        split_delimiters=None
    )
    print(f"Batch size: {len(batch_reprs)}")
    for i, (reprs, tokens) in enumerate(zip(batch_reprs, batch_tokens)):
        print(f"Text {i}: {clean_tokens_for_display(tokens)} -> Shape: {reprs.shape}")
    
    # Process batch with splitting
    batch_split_reprs, batch_split_tokens = get_last_layer_representations(
        model,
        tokenizer,
        batch_texts, 
        return_token_text=True,
        split_delimiters=["."]
    )
    print(f"\nBatch with splitting:")
    for i, (text_reprs, text_tokens) in enumerate(zip(batch_split_reprs, batch_split_tokens)):
        print(f"Text {i}: {len(text_reprs)} segments")
        for j, (reprs, tokens) in enumerate(zip(text_reprs, text_tokens)):
            print(f"  Segment {j}: {clean_tokens_for_display(tokens)} -> Shape: {reprs.shape}")
            
    print("\n=== Token indices version===")
    batch_split_reprs, batch_split_tokens = get_last_layer_representations(
        model,
        tokenizer,
        batch_texts, 
        split_delimiters=["."]
    )
    print(f"Batch size: {len(batch_split_reprs)}")
    for i, (reprs, tokens) in enumerate(zip(batch_split_reprs, batch_split_tokens)):
        print(f"Text {i}: {len(reprs)} segments")
        for j, (reprs, tokens) in enumerate(zip(reprs, tokens)):
            print(f"  Segment {j}: {tokens} -> Shape: {reprs.shape}")

    # Example 7: Demonstrate BPE token cleaning
    print("\n=== BPE Token Cleaning Demo ===")
    demo_text = "Hello world!\nHow are you?"
    print(f"Original text: {repr(demo_text)}")
    
    demo_reprs, demo_tokens = get_last_layer_representations(
        model, 
        tokenizer, 
        demo_text, 
        return_token_text=True,
        split_delimiters=None
    )
    
    print(f"Raw BPE tokens: {demo_tokens[0]}")
    print(f"Clean tokens: {clean_tokens_for_display(demo_tokens[0])}")
    
    # Show what each special character means
    print(f"\nBPE Character Meanings:")
    print(f"Ġ -> space character")
    print(f"Ċ -> newline character") 
    print(f"▁ -> space character (SentencePiece)")
    
    # Alternative: Use tokenizer's built-in method
    clean_text = tokenizer.convert_tokens_to_string(demo_tokens[0])
    print(f"Tokenizer clean text: {repr(clean_text)}")
    
    # Example 8: Question + CoT pairs with CoT-only extraction
    print("\n=== Question + CoT Pairs Demo ===")
    question_cot_pairs = [
        ("What is 2+2?", "Let me think about this step by step. 2+2 equals 4."),
        ("Explain photosynthesis", "Photosynthesis is the process where plants convert sunlight into energy. This happens in the chloroplasts."),
        ("How do computers work?", "Computers work by processing binary data through logic gates and memory units.")
    ]
    
    # Process with CoT-only extraction
    cot_only_reprs, cot_only_tokens, questions = get_last_layer_representations(
        model,
        tokenizer,
        question_cot_pairs,
        return_token_text=True,
        return_cot_only=True
    )
    
    print(f"CoT-only processing with questions:")
    for i, (reprs, tokens, question) in enumerate(zip(cot_only_reprs, cot_only_tokens, questions)):
        print(f"  Pair {i}: Question: {question}")
        for j, (token, repr_tensor) in enumerate(zip(tokens, reprs)):
            print(f"    {token} -> Shape: {repr_tensor.shape}")
    
    print("\n=== Token indices version===")
    cot_only_reprs, cot_only_tokens, questions = get_last_layer_representations(
        model,
        tokenizer,
        question_cot_pairs,
        return_cot_only=True
    )
    print(f"CoT-only processing with questions:")
    for i, (reprs, tokens, question) in enumerate(zip(cot_only_reprs, cot_only_tokens, questions)):
        print(f"  Pair {i}: Question: {question}")
        for j, (token, repr_tensor) in enumerate(zip(tokens, reprs)):
            print(f"    {token} -> Shape: {repr_tensor.shape}")

    # Process with mixed CoT-only flags (some True, some False)
    mixed_pairs = [
        ("Q1: What is AI?", "AI is artificial intelligence.", True),   # CoT-only
        ("Q2: What is ML?", "ML is machine learning.", False),         # Full sequence
        ("Q3: What is DL?", "DL is deep learning.", True)              # CoT-only
    ]
    
    mixed_reprs, mixed_tokens, mixed_questions = get_last_layer_representations(
        model,
        tokenizer,
        mixed_pairs,
        return_token_text=True
    )
    
    print(f"\nMixed CoT-only processing with questions:")
    for i, (reprs, tokens, question) in enumerate(zip(mixed_reprs, mixed_tokens, mixed_questions)):
        flag = mixed_pairs[i][2]
        print(f"  Pair {i} (flag={flag}): Question: {question}")
        for j, (token, repr_tensor) in enumerate(zip(tokens, reprs)):
            print(f"    {token} -> Shape: {repr_tensor.shape}")
    
    print("\n=== Token indices version===")
    
    mixed_reprs, mixed_tokens, mixed_questions = get_last_layer_representations(
        model,
        tokenizer,
        mixed_pairs,
        return_cot_only=True
    )
    
    print(f"\nMixed CoT-only processing with questions:")
    for i, (reprs, tokens, question) in enumerate(zip(mixed_reprs, mixed_tokens, mixed_questions)):
        flag = mixed_pairs[i][2]
        print(f"  Pair {i} (flag={flag}): Question: {question}")
        for j, (token, repr_tensor) in enumerate(zip(tokens, reprs)):
            print(f"    {token} -> Shape: {repr_tensor.shape}")
    
    # Example 9: Demonstrate that full sequence is encoded but only CoT is returned
    print("\n=== Full Sequence Encoding Demo ===")
    question = "What is the capital of France?"
    cot = "The capital of France is Paris. This is a well-known fact about European geography."
    
    # Process with CoT-only
    cot_only_reprs, cot_only_tokens, returned_question = get_last_layer_representations(
        model,
        tokenizer,
        (question, cot),
        return_token_text=True,
        return_cot_only=True
    )
    
    # Process without CoT-only to compare
    full_reprs, full_tokens, returned_question_full = get_last_layer_representations(
        model,
        tokenizer,
        (question, cot),
        return_token_text=True,
        return_cot_only=False
    )
    
    print(f"Question: {returned_question}")
    print(f"Full sequence tokens:")
    for token, repr_tensor in zip(full_tokens[0], full_reprs[0]):
        print(f"    {token} -> Shape: {repr_tensor.shape}")
    print(f"CoT-only tokens:")
    for token, repr_tensor in zip(cot_only_tokens[0], cot_only_reprs[0]):
        print(f"    {token} -> Shape: {repr_tensor.shape}")