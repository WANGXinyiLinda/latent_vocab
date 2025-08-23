import torch
from transformers import AutoTokenizer, AutoModel
from typing import Union, List, Tuple
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention


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
    text: Union[str, List[str]],
    device: str = "cpu",
    max_length: int = 512,
    return_tokens: bool = False,
    split_delimiters: List[str] = ["\n", ".", "!", "?"]
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[str]], List[torch.Tensor], Tuple[List[torch.Tensor], List[List[str]]], List[List[torch.Tensor]], Tuple[List[List[torch.Tensor]], List[List[List[str]]]]]:
    """
    Extract the last layer representations from a Hugging Face pretrained language model.
    
    Args:
        model (AutoModel): Hugging Face pretrained model
        tokenizer (AutoTokenizer): Hugging Face pretrained tokenizer
        text (Union[str, List[str]]): Input text or list of texts to process
        device (str): Device to run the model on ("auto", "cpu", "cuda", or specific device)
        max_length (int): Maximum sequence length for tokenization
        return_tokens (bool): Whether to also return the tokenized tokens
        split_delimiters (List[str]): List of text delimiters to split on (e.g., ["\n", ".", "Answer:"])
        
    Returns:
        For single text input:
            - If split_delimiters=None and return_tokens=False: tensor of shape (num_tokens, hidden_size)
            - If split_delimiters=None and return_tokens=True: tuple of (tensor, token_list)
            - If split_delimiters is provided and return_tokens=False: list of tensors, each of shape (num_tokens_in_segment, hidden_size)
            - If split_delimiters is provided and return_tokens=True: tuple of (list_of_tensors, list_of_token_lists)
        
        For batch input:
            - If split_delimiters=None and return_tokens=False: list of tensors
            - If split_delimiters=None and return_tokens=True: tuple of (list_of_tensors, list_of_token_lists)
            - If split_delimiters is provided and return_tokens=False: list of lists of tensors
            - If split_delimiters is provided and return_tokens=True: tuple of (list_of_lists_of_tensors, list_of_lists_of_token_lists)
    """
    
    # Handle batch processing
    is_batch = isinstance(text, list)
    if is_batch:
        texts = text
        if not texts:
            return [] if not return_tokens else ([], [])
    else:
        texts = [text]
    
    # Auto-detect device if specified
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Tokenize the input texts
    inputs = tokenizer(
        texts,
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
    
    for batch_idx in range(len(texts)):
        # Get the hidden states for this text
        text_hidden_states = last_hidden_states[batch_idx]  # Shape: (seq_len, hidden_size)
        
        # Get the tokens for this text
        text_tokens = tokenizer.convert_ids_to_tokens(input_ids[batch_idx])
        
        # Get the offset mapping for this text
        text_offset_mapping = offset_mappings[batch_idx]
        
        # Filter out special tokens and get meaningful representations
        meaningful_tokens = []
        meaningful_representations = []
        
        for i, (token, offset) in enumerate(zip(text_tokens, text_offset_mapping)):
            # Skip special tokens and padding
            if token in tokenizer.special_tokens_map.values() or offset[0] == offset[1]:
                continue
            
            meaningful_tokens.append(token)
            meaningful_representations.append(text_hidden_states[i])
        
        # Handle splitting for this text
        if split_delimiters:
            # Split representations and tokens based on text-level delimiters
            split_representations = []
            split_token_lists = []
            
            # First, find all split positions in the original text
            split_positions = []
            for delimiter in split_delimiters:
                start = 0
                while True:
                    pos = texts[batch_idx].find(delimiter, start)
                    if pos == -1:
                        break
                    split_positions.append(pos + len(delimiter))  # Split after the delimiter
                    start = pos + 1
            
            split_positions = sorted(split_positions)
            
            # Group tokens by their text positions
            current_reprs = []
            current_tokens = []
            
            for token, repr_tensor, offset in zip(meaningful_tokens, meaningful_representations, 
                    [offset for offset in text_offset_mapping if offset[0] != offset[1]]):
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
        else:
            # No splitting - stack the representations
            if meaningful_representations:
                meaningful_representations = torch.stack(meaningful_representations)
            else:
                # If no meaningful tokens, return empty tensor
                meaningful_representations = torch.empty(0, last_hidden_states.shape[-1])
            
            batch_results.append(meaningful_representations)
            batch_tokens.append(meaningful_tokens)
    
    # Return results based on input type and parameters
    if not is_batch:
        # Single text input - return first (and only) result
        if return_tokens:
            return batch_results[0], batch_tokens[0]
        else:
            return batch_results[0]
    else:
        # Batch input - return all results
        if return_tokens:
            return batch_results, batch_tokens
        else:
            return batch_results


class MultiHeadAttentionQuantizer(nn.Module):
    """
    Multi-head attention based quantizer that maps representations to k vectors.
    Uses learnable query vectors instead of the input representations.
    """
    
    def __init__(self, hidden_size: int, num_heads: int, k: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.k = k
        self.head_dim = hidden_size // num_heads
        
        # Learnable query vectors (k vectors)
        self.learnable_queries = nn.Parameter(torch.randn(k, hidden_size))
        
        # Multi-head attention layer
        self.attention = MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            representations: Input representations of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Quantized vectors of shape (batch_size, k, hidden_size)
        """
        batch_size = representations.shape[0]
        
        # Expand learnable queries to batch size
        # queries: (batch_size, k, hidden_size)
        queries = self.learnable_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply multi-head attention
        # queries as query, representations as key and value
        attended_output, _ = self.attention(
            query=queries,
            key=representations,
            value=representations
        )
        
        # Apply output projection and layer norm
        output = self.output_projection(attended_output)
        output = self.layer_norm(output)
        
        return output


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
            Tuple of (quantized, loss, perplexity)
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
        
        return quantized, loss, perplexity


class VQVAETextReconstructor(nn.Module):
    """
    VQ-VAE for text reconstruction using pretrained language models.
    """
    
    def __init__(
        self,
        encoder_model: AutoModel,
        decoder_model: AutoModel,
        tokenizer: AutoTokenizer,
        num_latent_tokens: int,
        hidden_size: int,
        num_heads: int,
        k: int,
        commitment_cost: float = 0.25,
        explain_token: str = "<EXPLAIN>"
    ):
        super().__init__()
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.tokenizer = tokenizer
        self.num_latent_tokens = num_latent_tokens
        self.hidden_size = hidden_size
        self.k = k
        self.explain_token = explain_token
        
        # Freeze encoder and decoder models
        for param in self.encoder_model.parameters():
            param.requires_grad = False
        for param in self.decoder_model.parameters():
            param.requires_grad = False
        
        # Add new latent tokens to vocabulary
        self._add_latent_tokens()
        
        # Multi-head attention quantizer
        self.quantizer = MultiHeadAttentionQuantizer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            k=k
        )
        
        # Vector quantizer
        self.vq = VectorQuantizer(
            num_embeddings=num_latent_tokens,
            embedding_dim=hidden_size,
            commitment_cost=commitment_cost
        )
        
        # Projection layer to match hidden size
        self.projection = nn.Linear(hidden_size, hidden_size)
        
    def _add_latent_tokens(self):
        """Add new latent tokens to the tokenizer and model vocabulary."""
        # Add latent tokens
        latent_tokens = [f"<LATENT_{i}>" for i in range(self.num_latent_tokens)]
        
        # Add explain token if not already present
        if self.explain_token not in self.tokenizer.get_vocab():
            latent_tokens.append(self.explain_token)
        
        # Add tokens to tokenizer
        self.tokenizer.add_tokens(latent_tokens)
        
        # Resize token embeddings for both encoder and decoder
        if hasattr(self.encoder_model, 'resize_token_embeddings'):
            self.encoder_model.resize_token_embeddings(len(self.tokenizer))
        if hasattr(self.decoder_model, 'resize_token_embeddings'):
            self.decoder_model.resize_token_embeddings(len(self.tokenizer))
        
        # Initialize new token embeddings
        self._initialize_new_embeddings()
    
    def _initialize_new_embeddings(self):
        """Initialize the embeddings for newly added tokens."""
        # Get the token IDs for new tokens
        latent_token_ids = []
        for i in range(self.num_latent_tokens):
            token = f"<LATENT_{i}>"
            if token in self.tokenizer.get_vocab():
                latent_token_ids.append(self.tokenizer.convert_tokens_to_ids(token))
        
        explain_token_id = None
        if self.explain_token in self.tokenizer.get_vocab():
            explain_token_id = self.tokenizer.convert_tokens_to_ids(self.explain_token)
        
        # Initialize encoder embeddings
        if hasattr(self.encoder_model, 'embeddings'):
            for token_id in latent_token_ids:
                if token_id < self.encoder_model.embeddings.word_embeddings.num_embeddings:
                    # Initialize with small random values
                    self.encoder_model.embeddings.word_embeddings.weight.data[token_id] = \
                        torch.randn(self.hidden_size) * 0.02
        
        # Initialize decoder embeddings (same as encoder for consistency)
        if hasattr(self.decoder_model, 'embeddings'):
            for token_id in latent_token_ids:
                if token_id < self.decoder_model.embeddings.word_embeddings.num_embeddings:
                    self.decoder_model.embeddings.word_embeddings.weight.data[token_id] = \
                        self.encoder_model.embeddings.word_embeddings.weight.data[token_id]
        
        # Initialize explain token embeddings
        if explain_token_id is not None:
            if hasattr(self.encoder_model, 'embeddings'):
                if explain_token_id < self.encoder_model.embeddings.word_embeddings.num_embeddings:
                    self.encoder_model.embeddings.word_embeddings.weight.data[explain_token_id] = \
                        torch.randn(self.hidden_size) * 0.02
            
            if hasattr(self.decoder_model, 'embeddings'):
                if explain_token_id < self.decoder_model.embeddings.word_embeddings.num_embeddings:
                    self.decoder_model.embeddings.word_embeddings.weight.data[explain_token_id] = \
                        self.encoder_model.embeddings.word_embeddings.weight.data[explain_token_id]
    
    def encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode text to get quantized representations.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (quantized_vectors, vq_loss, perplexity)
        """
        # Get representations from frozen encoder
        with torch.no_grad():
            representations = get_last_layer_representations(
                self.encoder_model,
                self.tokenizer,
                text,
                device="auto"
            )
        
        # Ensure representations is a tensor
        if isinstance(representations, list):
            representations = representations[0]  # Take first segment if split
        
        # Add batch dimension if needed
        if representations.dim() == 2:
            representations = representations.unsqueeze(0)
        
        # Project to match hidden size if needed
        if representations.shape[-1] != self.hidden_size:
            representations = self.projection(representations)
        
        # Apply multi-head attention quantizer
        quantized_vectors = self.quantizer(representations)
        
        # Apply vector quantization
        quantized, vq_loss, perplexity = self.vq(quantized_vectors)
        
        return quantized, vq_loss, perplexity
    
    def decode(self, quantized_vectors: torch.Tensor, original_text: str) -> torch.Tensor:
        """
        Decode quantized vectors back to text representations.
        
        Args:
            quantized_vectors: Quantized vectors of shape (batch_size, k, hidden_size)
            original_text: Original text for context
            
        Returns:
            Decoded representations
        """
        # Get the token IDs for latent tokens
        latent_token_ids = []
        for i in range(self.k):
            token = f"<LATENT_{i}>"
            if token in self.tokenizer.get_vocab():
                latent_token_ids.append(self.tokenizer.convert_tokens_to_ids(token))
        
        explain_token_id = None
        if self.explain_token in self.tokenizer.get_vocab():
            explain_token_id = self.tokenizer.convert_tokens_to_ids(self.explain_token)
        
        # Create input sequence: [LATENT_0, LATENT_1, ..., LATENT_k-1, EXPLAIN, original_text]
        input_tokens = latent_token_ids[:self.k]
        if explain_token_id is not None:
            input_tokens.append(explain_token_id)
        
        # Add original text tokens
        original_tokens = self.tokenizer.encode(original_text, add_special_tokens=False)
        input_tokens.extend(original_tokens)
        
        # Convert to tensor and add batch dimension
        input_ids = torch.tensor([input_tokens], device=quantized_vectors.device)
        
        # Run decoder model
        with torch.no_grad():
            outputs = self.decoder_model(
                input_ids=input_ids,
                output_hidden_states=True
            )
        
        # Return the last hidden states
        return outputs.hidden_states[-1]
    
    def forward(self, text: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VQ-VAE.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (quantized_vectors, vq_loss, perplexity, decoded_representations)
        """
        # Encode
        quantized, vq_loss, perplexity = self.encode(text)
        
        # Decode
        decoded = self.decode(quantized, text)
        
        return quantized, vq_loss, perplexity, decoded
    
    def get_latent_tokens(self) -> List[str]:
        """Get the list of latent token strings."""
        return [f"<LATENT_{i}>" for i in range(self.num_latent_tokens)]
    
    def get_explain_token(self) -> str:
        """Get the explain token string."""
        return self.explain_token


def create_vq_vae_model(
    model_name: str,
    num_latent_tokens: int,
    k: int,
    num_heads: int = 8,
    commitment_cost: float = 0.25,
    explain_token: str = "<EXPLAIN>"
) -> VQVAETextReconstructor:
    """
    Create a VQ-VAE model with the specified configuration.
    
    Args:
        model_name: HuggingFace model name or path
        num_latent_tokens: Number of latent tokens to add
        k: Number of quantized vectors
        num_heads: Number of attention heads
        commitment_cost: Commitment cost for vector quantization
        explain_token: Token to use as explanation separator
        
    Returns:
        Configured VQ-VAE model
    """
    # Load tokenizer and models
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder_model = AutoModel.from_pretrained(model_name)
    decoder_model = AutoModel.from_pretrained(model_name)
    
    # Get hidden size from model
    if hasattr(encoder_model, 'config'):
        hidden_size = encoder_model.config.hidden_size
    else:
        # Default hidden size for common models
        hidden_size = 768
    
    # Create VQ-VAE model
    model = VQVAETextReconstructor(
        encoder_model=encoder_model,
        decoder_model=decoder_model,
        tokenizer=tokenizer,
        num_latent_tokens=num_latent_tokens,
        hidden_size=hidden_size,
        num_heads=num_heads,
        k=k,
        commitment_cost=commitment_cost,
        explain_token=explain_token
    )
    
    return model


# Example usage
if __name__ == "__main__":
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
        return_tokens=True,
        split_delimiters=["\n", "."]
    )
    print(f"Number of segments: {len(split_reprs)}")
    for i, (reprs, tokens) in enumerate(zip(split_reprs, split_tokens)):
        print(f"Segment {i}: {tokens} -> Shape: {reprs.shape}")
    
    # Example 4: Split by words that might be tokenized into multiple tokens
    print("\n=== Split by Multiple Tokens ===")
    text_multi = "Question: What is AI? Answer: AI is intelligence. Explanation: It's complex."
    split_reprs, split_tokens = get_last_layer_representations(
        model,
        tokenizer,
        text_multi, 
        return_tokens=True,
        split_delimiters=["Question:", "Answer:", "Explanation:"]
    )
    print(f"Number of segments: {len(split_reprs)}")
    for i, (reprs, tokens) in enumerate(zip(split_reprs, split_tokens)):
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
        return_tokens=True,
        split_delimiters=["!", "?", "\n"]
    )
    print(f"Number of segments: {len(split_reprs)}")
    for i, (reprs, tokens) in enumerate(zip(split_reprs, split_tokens)):
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
        return_tokens=True,
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
        return_tokens=True,
        split_delimiters=["."]
    )
    print(f"\nBatch with splitting:")
    for i, (text_reprs, text_tokens) in enumerate(zip(batch_split_reprs, batch_split_tokens)):
        print(f"Text {i}: {len(text_reprs)} segments")
        for j, (reprs, tokens) in enumerate(zip(text_reprs, text_tokens)):
            print(f"  Segment {j}: {clean_tokens_for_display(tokens)} -> Shape: {reprs.shape}")
    
    # Example 7: Demonstrate BPE token cleaning
    print("\n=== BPE Token Cleaning Demo ===")
    demo_text = "Hello world!\nHow are you?"
    print(f"Original text: {repr(demo_text)}")
    
    demo_reprs, demo_tokens = get_last_layer_representations(
        model, 
        tokenizer, 
        demo_text, 
        return_tokens=True,
        split_delimiters=None
    )
    
    print(f"Raw BPE tokens: {demo_tokens}")
    print(f"Clean tokens: {clean_tokens_for_display(demo_tokens)}")
    
    # Show what each special character means
    print(f"\nBPE Character Meanings:")
    print(f"Ġ -> space character")
    print(f"Ċ -> newline character") 
    print(f"▁ -> space character (SentencePiece)")
    
    # Alternative: Use tokenizer's built-in method
    clean_text = tokenizer.convert_tokens_to_string(demo_tokens)
    print(f"Tokenizer clean text: {repr(clean_text)}")
    
    # Example 8: VQ-VAE usage
    print("\n=== VQ-VAE Demo ===")
    try:
        # Create VQ-VAE model
        vq_vae = create_vq_vae_model(
            model_name=model_name,
            num_latent_tokens=64,  # 64 latent tokens
            k=8,                   # 8 quantized vectors
            num_heads=8,           # 8 attention heads
            commitment_cost=0.25
        )
        
        print(f"Created VQ-VAE model with:")
        print(f"  - {vq_vae.num_latent_tokens} latent tokens")
        print(f"  - {vq_vae.k} quantized vectors")
        print(f"  - {vq_vae.hidden_size} hidden size")
        print(f"  - Latent tokens: {vq_vae.get_latent_tokens()[:5]}...")
        print(f"  - Explain token: {vq_vae.get_explain_token()}")
        
        # Test encoding and decoding
        test_text = "This is a test sentence for VQ-VAE."
        print(f"\nTesting with text: {test_text}")
        
        # Encode
        quantized, vq_loss, perplexity = vq_vae.encode(test_text)
        print(f"Encoded shape: {quantized.shape}")
        print(f"VQ loss: {vq_loss.item():.4f}")
        print(f"Perplexity: {perplexity.item():.2f}")
        
        # Decode
        decoded = vq_vae.decode(quantized, test_text)
        print(f"Decoded shape: {decoded.shape}")
        
        # Full forward pass
        quantized, vq_loss, perplexity, decoded = vq_vae(test_text)
        print(f"Full forward pass completed successfully!")
        
    except Exception as e:
        print(f"VQ-VAE demo failed: {e}")
        print("This might be due to model compatibility or missing dependencies.")