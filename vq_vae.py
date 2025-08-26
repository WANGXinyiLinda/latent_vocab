import torch
from transformers import AutoTokenizer, AutoModel
from typing import Union, List, Tuple
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import random
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from encoder import get_last_layer_representations, MultiHeadAttentionQuantizer, VectorQuantizer


class VQVAETextReconstructor(nn.Module):
    """
    VQ-VAE for text reconstruction using pretrained language models.
    """
    
    def __init__(
        self,
        num_latent_tokens: int,
        num_heads: int,
        k: int,
        commitment_cost: float = 0.25,
        explain_token: str = "<EXPLAIN>",
        think_token: str = "<THINK>",
        pretrained_model_name: str = None,
        prompt: str = None,
        compress_cot_only: bool = True,
        max_length: int = 512,
        train_decoder: bool = False,
        previous_segments_mode: str = "text", # "text" or "latent"
        device: str = "auto"
    ):
        super().__init__()
        self.num_latent_tokens = num_latent_tokens
        self.num_heads = num_heads
        self.k = k
        self.commitment_cost = commitment_cost
        self.explain_token = explain_token
        self.think_token = think_token
        self.prompt = prompt
        self.compress_cot_only = compress_cot_only
        self.max_length = max_length
        self.train_decoder = train_decoder
        self.previous_segments_mode = previous_segments_mode
        if pretrained_model_name is None:
            raise ValueError("pretrained_model_name is required")
        self.pretrained_model_name = pretrained_model_name
        
        # Auto-detect device if specified
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load encoder and decoder models, move to device
        self.encoder_model = AutoModel.from_pretrained(pretrained_model_name).to(device)
        self.decoder_model = AutoModel.from_pretrained(pretrained_model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Freeze encoder model
        self.encoder_model.eval()
        for param in self.encoder_model.parameters():
            param.requires_grad = False
        
        # Freeze decoder model if not training
        if self.train_decoder:
            self.decoder_model.train()
        else:
            self.decoder_model.eval()
            for param in self.decoder_model.parameters():
                param.requires_grad = False
        
        # Get hidden size from encoder model
        self.hidden_size = self.encoder_model.config.hidden_size
        
        # Multi-head attention quantizer
        self.quantizer = MultiHeadAttentionQuantizer(
            hidden_size=self.hidden_size,
            num_heads=num_heads,
            k=k
        ).to(device)
        
        # Vector quantizer
        self.vq = VectorQuantizer(
            num_embeddings=num_latent_tokens,
            embedding_dim=self.hidden_size,
            commitment_cost=commitment_cost
        ).to(device) 
        
        # Add explain token embedding if specified
        if self.explain_token:
            self.explain_embedding = nn.Parameter(
                torch.randn(1, 1, self.hidden_size) * 0.02).to(device)
        
        # Add think token embedding if specified
        if self.think_token:
            self.think_embedding = nn.Parameter(
                torch.randn(1, 1, self.hidden_size) * 0.02).to(device)        
            
    def encode(self, 
               inputs: Union[List[Tuple[str, str]], List[Tuple[str, str, bool]]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
               List[str], List[List[List[int]]], List[int]]:
        """
        Encode text to get quantized representations.
        
        Args:
            inputs: Input list of (question, cot) pairs or list of (question, cot, return_cot_only) tuples. Always batched.
            
        Returns:
            Tuple of (quantized_vectors, vq_loss, perplexity)
        """
        # Get representations from frozen encoder
        with torch.no_grad():
            representations, tokens, questions = get_last_layer_representations(
                self.encoder_model,
                self.tokenizer,
                inputs,
                device=self.device,
                max_length=self.max_length,
                return_cot_only=self.compress_cot_only
            )
        
        all_inputs = []
        all_tokens = []
        all_questions = []
        segment_groups = []
        for i, (representation, token, question) in enumerate(zip(representations, tokens, questions)):
            # check if encoded representations are split into multiple segments
            if not isinstance(representation, list):
                representation = [representation]
                token = [token]
                
            # process each segment separately
            for rep, tok in zip(representation, token):
                # Add batch dimension
                all_inputs.append(rep)
                all_tokens.append(tok)
                all_questions.append(question)
                # build segment groups
                segment_groups.append(i)
                
        # pad sequences to the same length and create attention mask
        max_length = min(max([len(input_representation) for input_representation in all_inputs]), self.max_length)        
        padded_inputs = []
        attention_mask = []
        for input_representation in all_inputs:
            num_pad = max_length - len(input_representation)
            # pad input embeddings
            with torch.no_grad():
                pad_embeddings = torch.zeros(num_pad, self.hidden_size, device=self.device)
            padded_inputs.append(torch.cat([input_representation, pad_embeddings], dim=0))
            # create attention mask for padded input embeddings
            attention_mask.append(torch.cat(
                    [torch.zeros(len(input_representation), device=self.device, dtype=torch.bool), 
                    torch.ones(num_pad, device=self.device, dtype=torch.bool)], dim=0)
            )

        # stack padded inputs and attention mask
        padded_inputs = torch.stack(padded_inputs)
        attention_mask = torch.stack(attention_mask)
        
        # apply multi-head attention quantizer
        quantized_vectors = self.quantizer(padded_inputs, attention_mask)
        
        # apply vector quantization
        quantized, vq_loss, perplexity, encoding_indices = self.vq(quantized_vectors)
        
        return quantized, vq_loss, perplexity, encoding_indices, all_questions, all_tokens, segment_groups
    
    def decode(self, 
               quantized_vectors: torch.Tensor, 
               original_tokens: List[List[int]], 
               questions: List[str]=None,
               segment_groups: List[int]=None,
               batch_order: str = "length" # "random", "length", or "none"
    ) -> torch.Tensor:
        """
        Decode batched quantized vectors with corresponding questions back to text representations and compute reconstruction loss.
        
        This method assumes the decoder_model is a LlamaForCausalLM and uses its built-in
        loss computation with the labels argument.
        
        Args:
            quantized_vectors: List of quantized vectors for each item in the batch
            original_tokens: List of tokenized (potentially split) original CoT segments for each item
            questions: List of questions for each CoT in the batch
            segment_groups: List of segment groups for each CoT in the batch
        Returns:
            Average reconstruction loss across the batch
        """
        # Start with the quantized vectors as the initial embeddings
        # These will serve as the "latent token" embeddings
        input_embeddings = quantized_vectors  # Shape: (batch_size, k, hidden_size)
        
        # Add explain token embedding if specified
        if self.explain_token:
            # Expand explain embedding to match batch size
            explain_embedding = self.explain_embedding.expand(input_embeddings.shape[0], 1, self.hidden_size)
            input_embeddings = torch.cat([input_embeddings, explain_embedding], dim=1)
        
        if self.think_token:
            think_embedding = self.think_embedding.expand(input_embeddings.shape[0], 1, self.hidden_size)
            input_embeddings = torch.cat([think_embedding, input_embeddings], dim=1)
        
        embed_layer = self.decoder_model.get_input_embeddings()
        ctx = torch.enable_grad() if self.train_decoder else torch.no_grad()
        
        # Pre-compute all token embeddings to avoid repeated lookups
        all_tokens = []
        # Collect all tokens that need embedding
        if self.prompt:
            all_tokens.append(self.prompt)  # Prompt (only one)
            
        if questions:
            all_tokens.extend(questions)  # Questions
        
        all_tokens.extend(original_tokens)  # Original tokens
        
        # Batch embed all tokens at once (major speedup)
        token_embeddings = []
        with ctx:
            if all_tokens:
                # Create a single tensor with all tokens for batch embedding
                token_lengths = [len(tokens) for tokens in all_tokens]
                max_token_len = max(token_lengths) if token_lengths else 0
                
                if max_token_len > 0:
                    # Pad tokens to same length for batch processing
                    padded_tokens = []
                    for tokens in all_tokens:
                        padded = tokens + [self.tokenizer.pad_token_id] * (max_token_len - len(tokens))
                        padded_tokens.append(padded)
                    
                    # Batch embed all tokens
                    all_token_tensor = torch.tensor(padded_tokens, device=self.device).int()
                    all_embeddings = embed_layer(all_token_tensor)
                    
                    # Unpack embeddings back to original lengths
                    start_idx = 0
                    for length in token_lengths:
                        token_embeddings.append(all_embeddings[start_idx:start_idx + length])
                        start_idx += 1
        
        # upack token embeddings for prompt, questions, and original tokens
        prompt_embedding, question_embeddings, original_embeddings, previous_segments_embeddings = None, [], [], []                
        if self.prompt:
            prompt_embedding = token_embeddings[0]
            if questions:
                question_embeddings = token_embeddings[1:len(questions)+1]
        elif questions:
                question_embeddings = token_embeddings[:len(questions)]
        original_embeddings = token_embeddings[-len(original_tokens):]
        
        # build previous segments embeddings from original token embeddings or quantized vectors
        if self.previous_segments_mode == "text" and len(original_tokens) > 1:
            segments_embeddings = original_embeddings
        elif self.previous_segments_mode == "latent" and len(quantized_vectors) > 1:
            segments_embeddings = quantized_vectors
        else:
            segments_embeddings = []
        
        # concatenate previous segments embeddings for each segment inside the same group
        prev_group = None
        previous_segments_embeddings = torch.Tensor([], 
                            device=self.device, dtype=input_embeddings[0].dtype)
        for group, segment_embedding in zip(segment_groups, segments_embeddings):
            if group != prev_group:
                previous_segments_embeddings.append(None)
                previous_segments_embedding = segment_embedding
                prev_group = group
            else:
                previous_segments_embedding = torch.cat([previous_segments_embedding, segment_embedding], dim=0)
                previous_segments_embeddings.append(previous_segments_embedding)

        # Build modified embeddings efficiently
        modified_embeddings = []
        labels = []
        
        # Build all embeddings in one pass
        for i, input_embedding in enumerate(input_embeddings):
            # Start with input embedding
            current_embedding = input_embedding
            
            # Add previous segments if available
            if len(previous_segments_embeddings) > 0:
                assert len(previous_segments_embeddings) == len(input_embeddings)
                if previous_segments_embeddings[i]:
                    current_embedding = torch.cat([previous_segments_embeddings[i], current_embedding], dim=0)
            
            # Add question embedding if available
            if len(question_embeddings) > 0:
                assert len(question_embeddings) == len(input_embeddings)
                current_embedding = torch.cat([question_embeddings[i], current_embedding], dim=0)
            
            # Add prompt embedding if available
            if prompt_embedding:
                current_embedding = torch.cat([prompt_embedding, current_embedding], dim=0)
            
            current_label = [-100] * len(current_embedding)
            
            # Add original text embedding and labels
            if len(original_embeddings) > 0:
                assert len(original_embeddings) == len(input_embeddings)
                current_embedding = torch.cat([current_embedding, original_embeddings[i]], dim=0)
                current_label += original_tokens[i]
            
            modified_embeddings.append(current_embedding)
            labels.append(current_label)
        
        # reorder modified_embeddings and labels to optimize performance/memory usage
        if batch_order == "random": # possibly performance gain through randomization
            temp = list(zip(modified_embeddings, labels))
            random.shuffle(temp)
            modified_embeddings, labels = zip(*temp)
        elif batch_order == "length": # more efficient for long sequences
            temp = list(zip(modified_embeddings, labels))
            temp.sort(key=lambda x: len(x[0]))
            modified_embeddings, labels = zip(*temp)
        elif batch_order == "none": # segments in the same CoT are grouped together
            pass
        else:
            raise ValueError(f"Invalid batch order: {batch_order}")
        
        # regroup modified_embeddings and labels to match the original batch size
        # prevent possible memory issues    
        batch_size = len(questions)
        i = 0
        batch_input_embeddings = []
        batch_labels = []
        while len(modified_embeddings) > batch_size*i:
            start = batch_size*i
            end = min(len(modified_embeddings), batch_size*(i+1))
            batch_input_embeddings.append(modified_embeddings[start:end])
            batch_labels.append(labels[start:end])
            i += 1
    
        # Batch pad all sequences at once
        reconstruction_loss = 0
        with ctx:
            pad_embedding = embed_layer(torch.tensor([self.tokenizer.pad_token_id], device=self.device).int())
            
        for input_embeddings, labels in zip(batch_input_embeddings, batch_labels):
            batch_max_length = min(max([len(input_embedding) for input_embedding in input_embeddings]), self.max_length)
            
            # Pre-allocate tensors for batch processing
            batch_size = len(input_embeddings)
            padded_input_embeddings = pad_embedding.expand(batch_size, batch_max_length, self.hidden_size)
            attention_mask = torch.zeros(batch_size, batch_max_length, device=self.device, dtype=torch.bool)
            padded_labels = torch.full((batch_size, batch_max_length), -100, device=self.device, dtype=torch.long)
            
            # Fill tensors efficiently
            for i, (input_embedding, label) in enumerate(zip(input_embeddings, labels)):
                seq_len = len(input_embedding)
                if seq_len > batch_max_length:
                    seq_len = batch_max_length
                
                # Copy embeddings
                padded_input_embeddings[i, :seq_len] = input_embedding[:seq_len]
                
                # Set attention mask
                attention_mask[i, :seq_len] = 1
                
                # Copy labels
                padded_labels[i, :seq_len] = torch.tensor(label[:seq_len], device=self.device, dtype=torch.long)
        
            # Run decoder model with custom embeddings and labels using inputs_embeds argument
            # This bypasses the embedding layer and directly uses our custom embeddings
            # LlamaForCausalLM will automatically compute the loss when labels are provided
            with ctx:
                model_outputs = self.decoder_model(
                    inputs_embeds=padded_input_embeddings,
                    attention_mask=attention_mask,
                    labels=padded_labels,
                    output_attentions=False,
                    return_dict=True
                )
                
            # compute reconstruction loss
            reconstruction_loss += model_outputs.loss
        
        reconstruction_loss /= len(batch_input_embeddings)
        
        # Return the reconstruction loss average across segments
        return reconstruction_loss
    
    
    def forward(self, inputs: Union[List[Tuple[str, str]], List[Tuple[str, str, bool]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VQ-VAE.
        
        Args:
            inputs: Input list of (question, cot) pairs or list of (question, cot, return_cot_only) tuples. Always batched.
            
        Returns:
            Tuple of (quantized_vectors, vq_loss, perplexity, reconstruction_loss, total_loss)
        """
        # Encode
        quantized, vq_loss, perplexity, encoding_indices, questions, tokens, segment_groups = self.encode(inputs)
        # Decode
        reconstruction_loss = self.decode(quantized, tokens, questions, segment_groups)
        # Compute the final VQ-VAE loss function
        # VQ loss includes commitment loss and codebook loss
        # Reconstruction loss measures how well the model reconstructs the original tokens
        # Combine VQ loss and reconstruction loss
        total_loss = vq_loss + reconstruction_loss
        
        return total_loss, reconstruction_loss, vq_loss, encoding_indices, quantized, perplexity 
    
        
    def save_checkpoint(self, checkpoint_path: str):
        """
        Save checkpoint with trained components and configuration.
        
        Args:
            checkpoint_path: Path to save the checkpoint
        """
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)

        # Save configuration and trained components
        checkpoint_data = {
            'config': {
                'num_latent_tokens': self.num_latent_tokens,
                'num_heads': self.num_heads,
                'k': self.k,
                'commitment_cost': self.commitment_cost,
                'explain_token': self.explain_token,
                'think_token': self.think_token,
                'pretrained_model_name': self.pretrained_model_name,
                'prompt': self.prompt,
                'compress_cot_only': self.compress_cot_only,
                'max_length': self.max_length,
                'train_decoder': self.train_decoder,
                'previous_segments_mode': self.previous_segments_mode
            },
            'trained_components': {
                'quantizer_state_dict': self.quantizer.state_dict(),
                'vq_state_dict': self.vq.state_dict(),
            },
            'tokenizer': self.tokenizer
        }
        
        # Add explain embedding if it exists
        if hasattr(self, 'explain_embedding'):
            checkpoint_data['trained_components']['explain_embedding'] = self.explain_embedding.data
        
        # Add thinking embedding if it exists
        if hasattr(self, 'think_embedding'):
            checkpoint_data['trained_components']['think_embedding'] = self.think_embedding.data
        
        torch.save(checkpoint_data, checkpoint_path)
        
        print(f"Checkpoint saved to: {checkpoint_path}")
    
    def load_from_checkpoint(self, checkpoint_path: str, pretrained_model_name: str = None):
        """
        Load VQVAETextReconstructor from a saved checkpoint.
        
        Args:
            checkpoint_path: Path to the saved checkpoint
            pretrained_model_name: Optional override for the pretrained model name
            
        Returns:
            Loaded VQVAETextReconstructor instance
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract configuration
        config = checkpoint['config']
        pretrained_model_name = pretrained_model_name or config['pretrained_model_name']
        
        # Get all config parameters
        model_kwargs = {
            'num_latent_tokens': config['num_latent_tokens'],
            'num_heads': config['num_heads'],
            'k': config['k'],
            'commitment_cost': config['commitment_cost'],
            'explain_token': config['explain_token'],
            'think_token': config['think_token'],
            'pretrained_model_name': pretrained_model_name,
            'prompt': config.get('prompt', None),
            'compress_cot_only': config.get('compress_cot_only', True),
            'max_length': config.get('max_length', 512),
            'train_decoder': config.get('train_decoder', False),
            'previous_segments_mode': config.get('previous_segments_mode', 'text')
        }
        
        # Create model instance
        model = VQVAETextReconstructor(**model_kwargs)
        
        # Load trained components
        trained_components = checkpoint['trained_components']
        
        # Load quantizer state
        if 'quantizer_state_dict' in trained_components:
            model.quantizer.load_state_dict(trained_components['quantizer_state_dict'])
        
        # Load VQ state
        if 'vq_state_dict' in trained_components:
            model.vq.load_state_dict(trained_components['vq_state_dict'])
        
        # Load explain embedding if available
        if 'explain_embedding' in trained_components:
            model.explain_embedding = nn.Parameter(trained_components['explain_embedding'])
        
        # Load thinking embedding if available
        if 'think_embedding' in trained_components:
            model.think_embedding = nn.Parameter(trained_components['think_embedding'])
        
        print(f"Model loaded from checkpoint: {checkpoint_path}")
        print(f"Pretrained model: {pretrained_model_name}")
        print(f"Configuration: {config}")
        
        return model
    
    def get_checkpoint_info(self) -> dict:
        """Get information about the current model configuration and components."""
        return {
            'config': {
                'num_latent_tokens': self.num_latent_tokens,
                'num_heads': self.num_heads,
                'k': self.k,
                'commitment_cost': self.commitment_cost,
                'explain_token': self.explain_token,
                'think_token': self.think_token,
                'pretrained_model_name': self.pretrained_model_name,
                'prompt': self.prompt,
                'compress_cot_only': self.compress_cot_only,
                'max_length': self.max_length,
                'train_decoder': self.train_decoder,
                'previous_segments_mode': self.previous_segments_mode
            },
            'model_info': {
                'encoder_model_type': type(self.encoder_model).__name__,
                'decoder_model_type': type(self.decoder_model).__name__,
                'tokenizer_type': type(self.tokenizer).__name__,
                'vocabulary_size': len(self.tokenizer.get_vocab()),
            }
        }
    
    def get_model_size_info(self) -> dict:
        """Get information about the model size and components."""
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        
        # Count parameters in each component
        quantizer_params = sum(p.numel() for p in self.quantizer.parameters())
        vq_params = sum(p.numel() for p in self.vq.parameters())
        
        # Count encoder parameters (frozen)
        encoder_params = sum(p.numel() for p in self.encoder_model.parameters())
        
        # Count decoder parameters
        decoder_params = sum(p.numel() for p in self.decoder_model.parameters())
        if not self.train_decoder:
            frozen_params += decoder_params
        else:
            trainable_params += decoder_params
        
        # Count custom embeddings
        custom_embedding_params = 0
        if hasattr(self, 'explain_embedding'):
            custom_embedding_params += self.explain_embedding.numel()
        if hasattr(self, 'think_embedding'):
            custom_embedding_params += self.think_embedding.numel()
        
        # All our custom components are trainable
        trainable_params += quantizer_params + vq_params + custom_embedding_params
        frozen_params += encoder_params
        
        total_params = trainable_params + frozen_params
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': frozen_params,
            'component_parameters': {
                'quantizer': quantizer_params,
                'vector_quantizer': vq_params,
                'encoder_model': encoder_params,
                'decoder_model': decoder_params,
                'custom_embeddings': custom_embedding_params,
            },
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'trainable_size_mb': trainable_params * 4 / (1024 * 1024),
            'frozen_size_mb': frozen_params * 4 / (1024 * 1024)
        }
    
    def get_trainable_parameters(self) -> dict:
        """
        Get detailed information about all trainable parameters in the model.
        
        Returns:
            Dictionary containing trainable parameter information
        """
        trainable_params = {}
        
        # Quantizer parameters
        quantizer_params = {}
        for name, param in self.quantizer.named_parameters():
            if param.requires_grad:
                quantizer_params[name] = {
                    'shape': list(param.shape),
                    'numel': param.numel(),
                    'dtype': str(param.dtype),
                    'device': str(param.device)
                }
        trainable_params['quantizer'] = quantizer_params
        
        # Vector quantizer parameters
        vq_params = {}
        for name, param in self.vq.named_parameters():
            if param.requires_grad:
                vq_params[name] = {
                    'shape': list(param.shape),
                    'numel': param.numel(),
                    'dtype': str(param.dtype),
                    'device': str(param.device)
                }
        trainable_params['vector_quantizer'] = vq_params
        
        # Custom embeddings
        custom_embeddings = {}
        if hasattr(self, 'explain_embedding') and self.explain_embedding.requires_grad:
            custom_embeddings['explain_embedding'] = {
                'shape': list(self.explain_embedding.shape),
                'numel': self.explain_embedding.numel(),
                'dtype': str(self.explain_embedding.dtype),
                'device': str(self.explain_embedding.device)
            }
        
        if hasattr(self, 'think_embedding') and self.think_embedding.requires_grad:
            custom_embeddings['think_embedding'] = {
                'shape': list(self.think_embedding.shape),
                'numel': self.think_embedding.numel(),
                'dtype': str(self.think_embedding.dtype),
                'device': str(self.think_embedding.device)
            }
        
        if custom_embeddings:
            trainable_params['custom_embeddings'] = custom_embeddings
        
        # Decoder parameters (if training)
        if self.train_decoder:
            decoder_params = {}
            for name, param in self.decoder_model.named_parameters():
                if param.requires_grad:
                    decoder_params[name] = {
                        'shape': list(param.shape),
                        'numel': param.numel(),
                        'dtype': str(param.dtype),
                        'device': str(param.device)
                    }
            trainable_params['decoder_model'] = decoder_params
        
        return trainable_params
    
    def get_all_parameters_summary(self) -> dict:
        """
        Get a comprehensive summary of all parameters in the model.
        
        Returns:
            Dictionary containing complete parameter information
        """
        all_params = {}
        
        # Quantizer parameters
        quantizer_params = {}
        for name, param in self.quantizer.named_parameters():
            quantizer_params[name] = {
                'shape': list(param.shape),
                'numel': param.numel(),
                'dtype': str(param.dtype),
                'device': str(param.device),
                'requires_grad': param.requires_grad
            }
        all_params['quantizer'] = quantizer_params
        
        # Vector quantizer parameters
        vq_params = {}
        for name, param in self.vq.named_parameters():
            vq_params[name] = {
                'shape': list(param.shape),
                'numel': param.numel(),
                'dtype': str(param.dtype),
                'device': str(param.device),
                'requires_grad': param.requires_grad
            }
        all_params['vector_quantizer'] = vq_params
        
        # Encoder model parameters (all frozen)
        encoder_params = {}
        for name, param in self.encoder_model.named_parameters():
            encoder_params[name] = {
                'shape': list(param.shape),
                'numel': param.numel(),
                'dtype': str(param.dtype),
                'device': str(param.device),
                'requires_grad': param.requires_grad
            }
        all_params['encoder_model'] = encoder_params
        
        # Decoder model parameters
        decoder_params = {}
        for name, param in self.decoder_model.named_parameters():
            decoder_params[name] = {
                'shape': list(param.shape),
                'numel': param.numel(),
                'dtype': str(param.dtype),
                'device': str(param.device),
                'requires_grad': param.requires_grad
            }
        all_params['decoder_model'] = decoder_params
        
        # Custom embeddings
        custom_embeddings = {}
        if hasattr(self, 'explain_embedding'):
            custom_embeddings['explain_embedding'] = {
                'shape': list(self.explain_embedding.shape),
                'numel': self.explain_embedding.numel(),
                'dtype': str(self.explain_embedding.dtype),
                'device': str(self.explain_embedding.device),
                'requires_grad': self.explain_embedding.requires_grad
            }
        
        if hasattr(self, 'think_embedding'):
            custom_embeddings['think_embedding'] = {
                'shape': list(self.think_embedding.shape),
                'numel': self.think_embedding.numel(),
                'dtype': str(self.think_embedding.dtype),
                'device': str(self.think_embedding.device),
                'requires_grad': self.think_embedding.requires_grad
            }
        
        if custom_embeddings:
            all_params['custom_embeddings'] = custom_embeddings
        
        return all_params
    
    def print_parameter_summary(self):
        """
        Print a formatted summary of all model parameters.
        """
        print("=" * 80)
        print("VQ-VAE MODEL PARAMETER SUMMARY")
        print("=" * 80)
        
        # Get size info
        size_info = self.get_model_size_info()
        print(f"\nMODEL SIZE OVERVIEW:")
        print(f"  Total Parameters: {size_info['total_parameters']:,}")
        print(f"  Trainable Parameters: {size_info['trainable_parameters']:,}")
        print(f"  Frozen Parameters: {size_info['frozen_parameters']:,}")
        print(f"  Total Size: {size_info['model_size_mb']:.2f} MB")
        print(f"  Trainable Size: {size_info['trainable_size_mb']:.2f} MB")
        print(f"  Frozen Size: {size_info['frozen_size_mb']:.2f} MB")
        
        # Get detailed parameter info
        trainable_params = self.get_trainable_parameters()
        all_params = self.get_all_parameters_summary()
        
        print(f"\nTRAINABLE COMPONENTS:")
        print(f"  {'Component':<20} {'Parameters':<12} {'Size (MB)':<12}")
        print(f"  {'-'*20} {'-'*12} {'-'*12}")
        
        for component, params in trainable_params.items():
            if params:  # Only show components with parameters
                total_params = sum(p['numel'] for p in params.values())
                size_mb = total_params * 4 / (1024 * 1024)
                print(f"  {component:<20} {total_params:<12,} {size_mb:<12.2f}")
        
        print(f"\nFROZEN COMPONENTS:")
        print(f"  {'Component':<20} {'Parameters':<12} {'Size (MB)':<12}")
        print(f"  {'-'*20} {'-'*12} {'-'*12}")
        
        # Show frozen components
        frozen_components = ['encoder_model']
        if not self.train_decoder:
            frozen_components.append('decoder_model')
        
        for component in frozen_components:
            if component in all_params:
                total_params = sum(p['numel'] for p in all_params[component].values())
                size_mb = total_params * 4 / (1024 * 1024)
                print(f"  {component:<20} {total_params:<12,} {size_mb:<12.2f}")
        
        print(f"\nDETAILED PARAMETER BREAKDOWN:")
        for component, params in all_params.items():
            if params:
                print(f"\n  {component.upper()}:")
                total_params = sum(p['numel'] for p in params.values())
                trainable_params_count = sum(p['numel'] for p in params.values() if p['requires_grad'])
                print(f"    Total: {total_params:,} parameters")
                print(f"    Trainable: {trainable_params_count:,} parameters")
                print(f"    Frozen: {total_params - trainable_params_count:,} parameters")
                
                # Show top 5 largest parameter tensors
                sorted_params = sorted(params.items(), key=lambda x: x[1]['numel'], reverse=True)
                for name, param_info in sorted_params[:5]:
                    status = "✓" if param_info['requires_grad'] else "✗"
                    print(f"      {status} {name}: {param_info['shape']} ({param_info['numel']:,})")
        
        print("=" * 80)



def get_checkpoint_info(checkpoint_path: str) -> dict:
    """
    Get information about a saved checkpoint without loading the full model.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Dictionary containing checkpoint information
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    info = {
        'checkpoint_path': checkpoint_path,
        'file_size_mb': os.path.getsize(checkpoint_path) / (1024 * 1024),
        'checkpoint_type': 'full_model' if 'model_state_dict' in checkpoint else 'trained_components_only'
    }
    
    if 'config' in checkpoint:
        info['config'] = checkpoint['config']
    
    if 'trained_components' in checkpoint:
        info['trained_components'] = list(checkpoint['trained_components'].keys())
    
    return info


def demonstrate_checkpoint_workflow():
    """
    Comprehensive demonstration of the checkpoint workflow for VQ-VAE.
    This shows how to save, load, and manage checkpoints in different scenarios.
    """
    print("=== VQ-VAE Checkpoint Workflow Demonstration ===\n")
    
    try:
        # 1. Create and train a model (simulated)
        print("1. Creating VQ-VAE model...")
        model_name = "../models/OLMo-2-0425-1B-Base"
        
        vq_vae = VQVAETextReconstructor(
            pretrained_model_name=model_name,
            num_latent_tokens=32,
            k=4,
            num_heads=4,
            commitment_cost=0.25
        )
        
        print(f"   ✓ Model created with {vq_vae.num_latent_tokens} latent tokens")
        print(f"   ✓ Pretrained model: {vq_vae.pretrained_model_name}")
        
        # 2. Simulate training (just show the process)
        print("\n2. Simulating training process...")
        print("   ✓ Quantizer trained")
        print("   ✓ Vector quantizer trained")
        print("   ✓ Latent token embeddings learned")
        
        # 3. Save different types of checkpoints
        print("\n3. Saving different types of checkpoints...")
        
        # Save checkpoint
        light_checkpoint = "checkpoints/vq_vae_checkpoint.pt"
        vq_vae.save_checkpoint(light_checkpoint)
        print(f"   ✓ Checkpoint saved: {light_checkpoint}")
        
        # 4. Demonstrate loading from checkpoint
        print("\n4. Loading from checkpoint...")
        
        # Load checkpoint
        print("   Loading checkpoint...")
        light_model = VQVAETextReconstructor.load_from_checkpoint(light_checkpoint)
        print("   ✓ Model loaded successfully")
        
        # Create a new model for testing
        print("   Creating new model for testing...")
        new_vq_vae = VQVAETextReconstructor(
            pretrained_model_name=model_name,
            num_latent_tokens=32,
            k=4,
            num_heads=4
        )
        print("   ✓ New model created successfully")
        
        # 5. Test loaded models
        print("\n5. Testing loaded models...")
        test_text = "This is a test sentence for checkpoint verification."
        
        # Test loaded model
        test_text_formatted = [("Test question", test_text)]
        quantized_light, vq_loss_light, perplexity_light, recon_loss_light, total_loss_light = light_model(test_text_formatted)
        print(f"   ✓ Loaded model: VQ loss = {vq_loss_light.item():.4f}, Reconstruction loss = {recon_loss_light.item():.4f}")
        
        # Test new model
        quantized_new, vq_loss_new, perplexity_new, recon_loss_new, total_loss_new = new_vq_vae(test_text_formatted)
        print(f"   ✓ New model: VQ loss = {vq_loss_new.item():.4f}, Reconstruction loss = {recon_loss_new.item():.4f}, Total loss = {total_loss_new.item():.4f}")
        
        # 6. Checkpoint information and analysis
        print("\n6. Checkpoint analysis...")
        
        # Get info about the checkpoint
        light_info = get_checkpoint_info(light_checkpoint)
        
        print(f"   Checkpoint: {light_info['file_size_mb']:.2f} MB")
        
        # 7. Model information
        print("\n7. Model information...")
        size_info = vq_vae.get_model_size_info()
        print(f"   Total parameters: {size_info['total_parameters']:,}")
        print(f"   Model size: {size_info['model_size_mb']:.2f} MB")
        print(f"   Components: {list(size_info['component_parameters'].keys())}")
        
        # 8. Cleanup
        print("\n8. Cleaning up...")
        import shutil
        if os.path.exists("checkpoints"):
            shutil.rmtree("checkpoints")
        print("   ✓ Checkpoint files cleaned up")
        
        print("\n=== Checkpoint Workflow Demonstration Completed Successfully! ===")
        
    except Exception as e:
        print(f"\n❌ Checkpoint workflow demonstration failed: {e}")
        import traceback
        traceback.print_exc()


# Example usage
if __name__ == "__main__":
    # Run the main demo
    print("Running VQ-VAE main demo...")
    
    # Example 13: VQ-VAE usage
    print("\n=== VQ-VAE Demo ===")
    try:
        # Create VQ-VAE model
        vq_vae = VQVAETextReconstructor(
            pretrained_model_name="../models/OLMo-2-0425-1B-Base",
            num_latent_tokens=64,  # 64 latent tokens
            k=8,                   # 8 quantized vectors
            num_heads=8,           # 8 attention heads
            commitment_cost=0.25
        )
        
        print(f"Created VQ-VAE model with:")
        print(f"  - {vq_vae.num_latent_tokens} latent tokens")
        print(f"  - {vq_vae.k} quantized vectors")
        print(f"  - {vq_vae.hidden_size} hidden size")

        # Test encoding and decoding
        test_text = [("Test question", "This is a test sentence for VQ-VAE.")]
        print(f"\nTesting with text: {test_text}")
        
        # Encode
        quantized, vq_loss, perplexity, questions, tokens = vq_vae.encode(test_text)
        print(f"Encoded shape: {quantized[0].shape if quantized else 'empty'}")
        print(f"VQ loss: {vq_loss[0].item():.4f}")
        print(f"Perplexity: {perplexity[0].item():.2f}")
        
        # Decode - note: decode now returns only the loss for batched input
        recon_loss = vq_vae.decode(quantized, tokens, questions)
        print(f"Reconstruction loss: {recon_loss.item():.4f}")
        
        # Full forward pass
        quantized, vq_loss, perplexity, recon_loss, total_loss = vq_vae(test_text)
        print(f"Full forward pass completed successfully!")
        print(f"Average reconstruction loss: {recon_loss.item():.4f}")
        print(f"Total VQ-VAE loss: {total_loss.item():.4f}")
        
        # Demonstrate checkpoint functionality
        print(f"\n=== Checkpoint Demo ===")
        
        # Save checkpoint
        checkpoint_path = "vq_vae_checkpoint.pt"
        vq_vae.save_checkpoint(checkpoint_path)
        print(f"Saved checkpoint to: {checkpoint_path}")
        
        # Get checkpoint info
        checkpoint_info = vq_vae.get_checkpoint_info()
        print(f"Checkpoint info: {checkpoint_info}")
        
        # Load from checkpoint
        print(f"\nLoading from checkpoint...")
        loaded_vq_vae = VQVAETextReconstructor.load_from_checkpoint(checkpoint_path)
        
        # Test loaded model
        test_text_loaded = [("Test question loaded", "Testing the loaded model with this sentence.")]
        quantized_loaded, vq_loss_loaded, perplexity_loaded, recon_loss_loaded, total_loss_loaded = loaded_vq_vae(test_text_loaded)
        print(f"Loaded model test successful!")
        print(f"  - VQ loss: {vq_loss_loaded.item():.4f}")
        print(f"  - Perplexity: {perplexity_loaded.item():.2f}")
        print(f"  - Reconstruction loss: {recon_loss_loaded.item():.4f}")
        print(f"  - Total loss: {total_loss_loaded.item():.4f}")
        
        # Clean up checkpoint files
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        print(f"\nCleaned up checkpoint files.")
        
        # Demonstrate additional checkpoint features
        print(f"\n=== Additional Checkpoint Features ===")
        
        # Show model size information
        size_info = vq_vae.get_model_size_info()
        print(f"Model size info: {size_info}")
        
        # Note: Some advanced checkpoint features are not implemented in this version
        print(f"Advanced checkpoint features (latent embeddings only, inference export) are not implemented.")
        
    except Exception as e:
        print(f"VQ-VAE demo failed: {e}")
        print("This might be due to model compatibility or missing dependencies.")
    
    # Uncomment the line below to run the comprehensive checkpoint workflow demo
    # demonstrate_checkpoint_workflow()