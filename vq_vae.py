import torch
from transformers import AutoTokenizer, AutoModel
from typing import Union, List, Tuple
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
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
        previous_segments_mode: str = "text" # "text" or "latent"
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
        
        # Load encoder and decoder models
        self.encoder_model = AutoModel.from_pretrained(pretrained_model_name)
        self.decoder_model = AutoModel.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        
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
        )
        
        # Vector quantizer
        self.vq = VectorQuantizer(
            num_embeddings=num_latent_tokens,
            embedding_dim=self.hidden_size,
            commitment_cost=commitment_cost
        )
        
        # Add explain token embedding if specified
        if self.explain_token:
            self.explain_embedding = nn.Parameter(torch.randn(1, 1, self.hidden_size) * 0.02)
        
        # Add think token embedding if specified
        if self.think_token:
            self.think_embedding = nn.Parameter(torch.randn(1, 1, self.hidden_size) * 0.02)
            
            
    def encode(self, inputs: Union[List[Tuple[str, str]], List[Tuple[str, str, bool]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
                device="auto",
                max_length=self.max_length,
                return_cot_only=self.compress_cot_only
            )
        
        quantized, vq_loss, perplexity = [], [], []
        for i, (representation, token, question) in enumerate(zip(representations, tokens, questions)):
            # check if encoded representations are split into multiple segments
            if not isinstance(representation, list):
                representation = [representation]
                token = [token]
            
            current_quantized, current_vq_loss, current_perplexity = [], [], []
            # process each segment separately
            for j, (rep, tok) in enumerate(zip(representation, token)):
                # Add batch dimension
                rep = rep.unsqueeze(0)
                
                # Apply multi-head attention quantizer
                quantized_vectors = self.quantizer(rep)
                
                # Apply vector quantization
                _quantized, _vq_loss, _perplexity = self.vq(quantized_vectors)
                
                # Append to lists
                current_quantized.append(_quantized)
                current_vq_loss.append(_vq_loss)
                current_perplexity.append(_perplexity)
            
            # Stack lists and average across segments
            quantized.append(torch.stack(current_quantized).mean(dim=0))
            vq_loss.append(torch.stack(current_vq_loss).mean(dim=0))
            perplexity.append(torch.stack(current_perplexity).mean(dim=0))
        
        return quantized, vq_loss, perplexity, questions, tokens
    
    def decode(self, quantized_vectors: torch.Tensor, original_tokens: List[List[int]], question: str=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode quantized vectors back to text representations and compute reconstruction loss.
        All vectors belong to the same question.
        
        This method assumes the decoder_model is a LlamaForCausalLM and uses its built-in
        loss computation with the labels argument.
        
        Args:
            original_tokens: Tokenized (potentially splited) original CoT segments for reconstruction
            question: Question for the CoT
        Returns:
            Tuple of (decoded_representations, reconstruction_loss)
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
        device = quantized_vectors.device
        ctx = torch.enable_grad() if self.train_decoder else torch.no_grad()
        
        if self.previous_segments_mode == "text":
            # Extract previous segments from original_tokens
            previous_embeddings = None
            if len(original_tokens) > 1:
                # Add previous segments' token embeddings
                # the first segment does not have previous segments, so we skip it
                for segment_tokens, input_embedding in zip(original_tokens[:-1], input_embeddings[1:]): 
                    with ctx:
                        new_embedding = embed_layer(
                            torch.tensor(segment_tokens, device=device).int())
                    if previous_embeddings is None:
                        previous_embeddings = new_embedding
                    else:
                        previous_embeddings = torch.cat([previous_embeddings, new_embedding], dim=0)
                    input_embedding = torch.cat([previous_embeddings, input_embedding], dim=0)

        
        elif self.previous_segments_mode == "latent":
            # Extract previous segments from quantized_vectors
            previous_embeddings = None
            if len(quantized_vectors) > 1:
                # Add previous segments' latent quantized vectors
                # the first segment does not have previous segments, so we skip it
                for segment_embeddings, input_embedding in zip(quantized_vectors[:-1], input_embeddings[1:]): 
                    if previous_embeddings is None:
                        previous_embeddings = segment_embeddings
                    else:
                        previous_embeddings = torch.cat([previous_embeddings, segment_embeddings], dim=0)
                    input_embedding = torch.cat([previous_embeddings, input_embedding], dim=0)
        
        else:
            raise ValueError(f"Invalid previous segments mode: {self.previous_segments_mode}")
        
        # Add question embedding if specified
        if question:
            question_tokens = self.tokenizer.encode(question, add_special_tokens=False)
            with ctx:
                question_embeddings = embed_layer(
                    torch.tensor(question_tokens, device=device).int())
            for input_embedding in input_embeddings:
                input_embedding = torch.cat([question_embeddings, input_embedding], dim=0)
        
        # Add prompt embeddings if specified
        if self.prompt:
            prompt_tokens = self.tokenizer.encode(self.prompt, add_special_tokens=False)
            with ctx:
                prompt_embeddings = embed_layer(
                    torch.tensor(prompt_tokens, device=device).int())
            for input_embedding in input_embeddings:
                input_embedding = torch.cat([prompt_embeddings, input_embedding], dim=0)
        
        # create labels
        labels = [[-100] * len(input_embedding) for input_embedding in input_embeddings]

        # Add original text embeddings
        with ctx:
            original_embeddings = embed_layer(
                torch.tensor(original_tokens, device=device).int())
        for input_embedding, original_embedding, original_token, label in \
                zip(input_embeddings, original_embeddings, original_tokens, labels):
            input_embedding = torch.cat([original_embedding, input_embedding], dim=0)
            # add original tokens to labels
            label += original_token
        
        # pad sequences to the same length and create attention mask
        max_length = min(max([len(input_embedding) for input_embedding in input_embeddings]), self.max_length)
        padded_input_embeddings = []
        attention_mask = []
        for input_embedding, label in zip(input_embeddings, labels):
            num_pad = max_length - len(input_embedding)
            # pad input embeddings
            with ctx:
                pad_embeddings = embed_layer(
                    torch.tensor([self.tokenizer.pad_token_id] * num_pad, device=device).int())
            padded_input_embeddings.append(torch.cat([input_embedding, pad_embeddings], dim=0))
            # create attention mask for padded input embeddings
            attention_mask.append(torch.cat([torch.ones(len(input_embedding), device=device), 
                                             torch.zeros(num_pad, device=device)], dim=0))
            # update labels
            label += [-100] * num_pad
        
        # stack padded input embeddings and attention mask
        padded_input_embeddings = torch.stack(padded_input_embeddings)
        attention_mask = torch.stack(attention_mask)
        labels = torch.tensor(labels, device=device, dtype=torch.long)
        
        # Run decoder model with custom embeddings and labels using inputs_embeds argument
        # This bypasses the embedding layer and directly uses our custom embeddings
        # LlamaForCausalLM will automatically compute the loss when labels are provided
        with ctx:
            model_outputs = self.decoder_model(
                inputs_embeds=padded_input_embeddings,
                attention_mask=attention_mask,
                labels=labels,
                output_attentions=False,
                return_dict=True
            )
        # Return the reconstruction loss average across segments
        return model_outputs.loss
    
    
    def forward(self, inputs: Union[List[Tuple[str, str]], List[Tuple[str, str, bool]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VQ-VAE.
        
        Args:
            inputs: Input list of (question, cot) pairs or list of (question, cot, return_cot_only) tuples. Always batched.
            
        Returns:
            Tuple of (quantized_vectors, vq_loss, perplexity, decoded_representations, reconstruction_loss, total_loss)
        """
        # Encode
        quantized, vq_loss, perplexity, questions, tokens = self.encode(inputs)
        
        # Decode
        reconstruction_losses = []
        for i, (quantized, token, question) in enumerate(zip(quantized, tokens, questions)):
            recon_loss = self.decode(quantized, token, question)
            reconstruction_losses.append(recon_loss)
        
        # Average the losses across the batch
        avg_reconstruction_loss = torch.stack(reconstruction_losses).mean()
        avg_vq_loss = torch.stack(vq_loss).mean()
        avg_perplexity = torch.stack(perplexity).mean()
        
        # Compute the final VQ-VAE loss function
        # Combine VQ loss and reconstruction loss
        # VQ loss includes commitment loss and codebook loss
        # Reconstruction loss measures how well the model reconstructs the original tokens
        total_loss = avg_vq_loss + avg_reconstruction_loss
        
        return quantized, avg_vq_loss, avg_perplexity, avg_reconstruction_loss, total_loss
    
        
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
        
        vq_vae = create_vq_vae_model(
            model_name=model_name,
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
        
        # Lightweight checkpoint (recommended for most use cases)
        light_checkpoint = "checkpoints/vq_vae_light.pt"
        vq_vae.save_checkpoint(light_checkpoint, save_full_model=False)
        print(f"   ✓ Lightweight checkpoint saved: {light_checkpoint}")
        
        # Full model checkpoint (for complete state preservation)
        full_checkpoint = "checkpoints/vq_vae_full.pt"
        vq_vae.save_checkpoint(full_checkpoint, save_full_model=True)
        print(f"   ✓ Full model checkpoint saved: {full_checkpoint}")
        
        # Minimal latent embeddings checkpoint
        latent_checkpoint = "checkpoints/vq_vae_latent.pt"
        vq_vae.save_latent_embeddings_only(latent_checkpoint)
        print(f"   ✓ Latent embeddings checkpoint saved: {latent_checkpoint}")
        
        # Inference export
        inference_checkpoint = "checkpoints/vq_vae_inference.pt"
        vq_vae.export_for_inference(inference_checkpoint)
        print(f"   ✓ Inference model exported: {inference_checkpoint}")
        
        # 4. Demonstrate loading from different checkpoints
        print("\n4. Loading from different checkpoints...")
        
        # Load lightweight checkpoint
        print("   Loading lightweight checkpoint...")
        light_model = VQVAETextReconstructor.load_from_checkpoint(light_checkpoint)
        print("   ✓ Lightweight model loaded successfully")
        
        # Load full checkpoint
        print("   Loading full checkpoint...")
        full_model = VQVAETextReconstructor.load_from_checkpoint(full_checkpoint)
        print("   ✓ Full model loaded successfully")
        
        # Load latent embeddings only
        print("   Loading latent embeddings...")
        new_vq_vae = create_vq_vae_model(
            model_name=model_name,
            num_latent_tokens=32,
            k=4,
            num_heads=4
        )
        new_vq_vae.load_latent_embeddings_only(latent_checkpoint)
        print("   ✓ Latent embeddings loaded successfully")
        
        # 5. Test loaded models
        print("\n5. Testing loaded models...")
        test_text = "This is a test sentence for checkpoint verification."
        
        # Test lightweight model
        quantized_light, vq_loss_light, perplexity_light, decoded_light, recon_loss_light = light_model(test_text)
        print(f"   ✓ Lightweight model: VQ loss = {vq_loss_light.item():.4f}, Reconstruction loss = {recon_loss_light.item():.4f}")
        
        # Test full model
        quantized_full, vq_loss_full, perplexity_full, decoded_full, recon_loss_full = full_model(test_text)
        print(f"   ✓ Full model: VQ loss = {vq_loss_full.item():.4f}, Reconstruction loss = {recon_loss_full.item():.4f}")
        
        # Test model with loaded latent embeddings
        quantized_new, vq_loss_new, perplexity_new, decoded_new, recon_loss_new, total_loss_new = new_vq_vae(test_text)
        print(f"   ✓ New model with loaded embeddings: VQ loss = {vq_loss_new.item():.4f}, Reconstruction loss = {recon_loss_new.item():.4f}, Total loss = {total_loss_new.item():.4f}")
        
        # 6. Checkpoint information and analysis
        print("\n6. Checkpoint analysis...")
        
        # Get info about each checkpoint type
        light_info = get_checkpoint_info(light_checkpoint)
        full_info = get_checkpoint_info(full_checkpoint)
        latent_info = get_checkpoint_info(latent_checkpoint)
        inference_info = get_checkpoint_info(inference_checkpoint)
        
        print(f"   Lightweight checkpoint: {light_info['file_size_mb']:.2f} MB")
        print(f"   Full checkpoint: {full_info['file_size_mb']:.2f} MB")
        print(f"   Latent checkpoint: {latent_info['file_size_mb']:.2f} MB")
        print(f"   Inference checkpoint: {inference_info['file_size_mb']:.2f} MB")
        
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
        test_text = "This is a test sentence for VQ-VAE."
        print(f"\nTesting with text: {test_text}")
        
        # Encode
        quantized, vq_loss, perplexity = vq_vae.encode(test_text)
        print(f"Encoded shape: {quantized.shape}")
        print(f"VQ loss: {vq_loss.item():.4f}")
        print(f"Perplexity: {perplexity.item():.2f}")
        
        # Decode
        decoded, recon_loss = vq_vae.decode(quantized, test_text)
        print(f"Decoded shape: {decoded.shape}")
        print(f"Reconstruction loss: {recon_loss.item():.4f}")
        
        # Full forward pass
        quantized, vq_loss, perplexity, decoded, recon_loss, total_loss = vq_vae(test_text)
        print(f"Full forward pass completed successfully!")
        print(f"Average reconstruction loss: {recon_loss.item():.4f}")
        print(f"Total VQ-VAE loss: {total_loss.item():.4f}")
        
        # Demonstrate checkpoint functionality
        print(f"\n=== Checkpoint Demo ===")
        
        # Save checkpoint with only trained components (recommended)
        checkpoint_path = "vq_vae_checkpoint.pt"
        vq_vae.save_checkpoint(checkpoint_path, save_full_model=False)
        print(f"Saved lightweight checkpoint to: {checkpoint_path}")
        
        # Get checkpoint info
        checkpoint_info = vq_vae.get_checkpoint_info()
        print(f"Checkpoint info: {checkpoint_info}")
        
        # Load from checkpoint
        print(f"\nLoading from checkpoint...")
        loaded_vq_vae = VQVAETextReconstructor.load_from_checkpoint(checkpoint_path)
        
        # Test loaded model
        test_text_loaded = "Testing the loaded model with this sentence."
        quantized_loaded, vq_loss_loaded, perplexity_loaded, decoded_loaded, recon_loss_loaded, total_loss_loaded = loaded_vq_vae(test_text_loaded)
        print(f"Loaded model test successful!")
        print(f"  - VQ loss: {vq_loss_loaded.item():.4f}")
        print(f"  - Perplexity: {perplexity_loaded.item():.2f}")
        print(f"  - Reconstruction loss: {recon_loss_loaded.item():.4f}")
        print(f"  - Total loss: {total_loss_loaded.item():.4f}")
        
        # Demonstrate full model saving (for complete state preservation)
        full_checkpoint_path = "vq_vae_full_checkpoint.pt"
        vq_vae.save_checkpoint(full_checkpoint_path, save_full_model=True)
        print(f"Saved full checkpoint to: {full_checkpoint_path}")
        
        # Load full checkpoint
        print(f"\nLoading from full checkpoint...")
        loaded_full_vq_vae = VQVAETextReconstructor.load_from_checkpoint(full_checkpoint_path)
        
        # Test full loaded model
        test_text_full = "Testing the fully loaded model."
        quantized_full, vq_loss_full, perplexity_full, decoded_full, recon_loss_full, total_loss_full = loaded_full_vq_vae(test_text_full)
        print(f"Full model test successful!")
        print(f"  - VQ loss: {vq_loss_full.item():.4f}")
        print(f"  - Perplexity: {perplexity_full.item():.2f}")
        print(f"  - Reconstruction loss: {recon_loss_full.item():.4f}")
        print(f"  - Total loss: {total_loss_full.item():.4f}")
        
        # Clean up checkpoint files
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        if os.path.exists(full_checkpoint_path):
            os.remove(full_checkpoint_path)
        print(f"\nCleaned up checkpoint files.")
        
        # Demonstrate additional checkpoint features
        print(f"\n=== Additional Checkpoint Features ===")
        
        # Show model size information
        size_info = vq_vae.get_model_size_info()
        print(f"Model size info: {size_info}")
        
        # Save minimal latent embeddings checkpoint
        latent_checkpoint_path = "vq_vae_latent_only.pt"
        vq_vae.save_latent_embeddings_only(latent_checkpoint_path)
        
        # Export for inference
        inference_path = "vq_vae_inference.pt"
        vq_vae.export_for_inference(inference_path)
        
        # Get checkpoint info without loading
        checkpoint_info = get_checkpoint_info(latent_checkpoint_path)
        print(f"Latent checkpoint info: {checkpoint_info}")
        
        inference_info = get_checkpoint_info(inference_path)
        print(f"Inference checkpoint info: {inference_info}")
        
        # Test loading latent embeddings only
        print(f"\nTesting latent embeddings loading...")
        vq_vae.load_latent_embeddings_only(latent_checkpoint_path)
        print(f"Latent embeddings loaded successfully!")
        
        # Clean up additional files
        if os.path.exists(latent_checkpoint_path):
            os.remove(latent_checkpoint_path)
        if os.path.exists(inference_path):
            os.remove(inference_path)
        print(f"Cleaned up additional checkpoint files.")
        
    except Exception as e:
        print(f"VQ-VAE demo failed: {e}")
        print("This might be due to model compatibility or missing dependencies.")
    
    # Uncomment the line below to run the comprehensive checkpoint workflow demo
    # demonstrate_checkpoint_workflow()