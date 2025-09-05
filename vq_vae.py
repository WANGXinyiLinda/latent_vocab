import torch
from torch.xpu.random import current_device
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from typing import Union, List, Tuple
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import random
import copy
import gc
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from encoder import get_last_layer_representations, MultiHeadAttentionQuantizer, VectorQuantizer


def clear_memory():
    """Utility function to clear GPU memory and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


class VQVAETextReconstructor(nn.Module):
    """
    VQ-VAE for text reconstruction using pretrained language models.
    """
    
    def __init__(
        self,
        num_latent_tokens: int,
        num_heads: int,
        k: int,
        pretrained_model_name: str,
        commitment_cost: float = 0.25,
        explain_token: str = "<EXPLAIN>",
        think_token: str = "<THINK>",
        prompt: str = None,
        compress_cot_only: bool = True,
        max_length: int = 512,
        train_decoder: bool = False,
        previous_segments_mode: str = "text", # "text" or "latent" or "none"
        device: str = "auto",
        num_cot_traces: int = 1,
        parallel_token: str = "<PARALLEL>",
        parallel_prediction_mode: str = "concatenate" # "concatenate" or "individual"
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
        self.pretrained_model_name = pretrained_model_name
        self.num_cot_traces = num_cot_traces
        self.parallel_token = parallel_token
        self.parallel_prediction_mode = parallel_prediction_mode
        # Auto-detect device if specified
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load encoder and decoder models, move to device
        self.encoder_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name).to(device)
        if self.train_decoder:
            self.decoder_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name).to(device)
            self.decoder_model.train()
        else:
            self.decoder_model = self.encoder_model
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Freeze encoder model
        self.encoder_model.eval()
        for param in self.encoder_model.parameters():
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
            self.explain_token_embedding = nn.Parameter(
                torch.randn(1, self.hidden_size) * 0.02).to(device)
        
        # Add think token embedding if specified
        if self.think_token:
            self.think_token_embedding = nn.Parameter(
                torch.randn(1, self.hidden_size) * 0.02).to(device)
        
        if self.num_cot_traces > 1:
            self.parallel_token_embedding = nn.Parameter(
                torch.randn(self.num_cot_traces, self.hidden_size) * 0.02).to(device)
            
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, device: str = "auto"):
        """
        Load VQVAETextReconstructor from a saved checkpoint.
        
        Args:
            checkpoint_path: Path to the saved checkpoint
            
        Returns:
            Loaded VQVAETextReconstructor instance
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        device = torch.device(device) if device != "auto" \
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load checkpoint with weights_only=False to handle tokenizer and other objects
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Extract configuration
        config = checkpoint['config']
        
        # Get all config parameters
        model_kwargs = {
            'num_latent_tokens': config['num_latent_tokens'],
            'num_heads': config['num_heads'],
            'k': config['k'],
            'commitment_cost': config['commitment_cost'],
            'explain_token': config['explain_token'],
            'think_token': config['think_token'],
            'pretrained_model_name': config['pretrained_model_name'],
            'prompt': config.get('prompt', None),
            'compress_cot_only': config.get('compress_cot_only', True),
            'max_length': config.get('max_length', 512),
            'train_decoder': config.get('train_decoder', False),
            'previous_segments_mode': config.get('previous_segments_mode', 'text'),
            'num_cot_traces': config.get('num_cot_traces', 1),
            'parallel_token': config.get('parallel_token', '<PARALLEL>'),
            'parallel_prediction_mode': config.get('parallel_prediction_mode', 'concatenate')
        }
        
        # Create model instance
        model = cls(**model_kwargs, device=device)
        
        # Load trained components
        trained_components = checkpoint['trained_components']
        
        # Load quantizer state
        if 'quantizer_state_dict' in trained_components:
            model.quantizer.load_state_dict(trained_components['quantizer_state_dict'])
        
        # Load VQ state
        if 'vq_state_dict' in trained_components:
            model.vq.load_state_dict(trained_components['vq_state_dict'])
        
        # Load explain token embedding if available
        if 'explain_token_embedding' in trained_components:
            model.explain_token_embedding = nn.Parameter(trained_components['explain_token_embedding'].to(model.device))
        
        # Load think token embedding if available
        if 'think_token_embedding' in trained_components:
            model.think_token_embedding = nn.Parameter(trained_components['think_token_embedding'].to(model.device))
        
        # Load parallel token embedding if available
        if 'parallel_token_embedding' in trained_components:
            model.parallel_token_embedding = nn.Parameter(trained_components['parallel_token_embedding'].to(model.device))
        
        print(f"Model loaded from checkpoint: {checkpoint_path}")
        print(f"Pretrained model: {config['pretrained_model_name']}")
        print(f"Configuration: {config}")
        
        return model      
            
    def encode(self, 
               inputs: Union[List[Tuple[str, str]], List[Tuple[str, str, bool]], List[Tuple[str, List[str]]]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
               List[str], List[List[List[int]]], List[int]]:
        """
        Encode text to get quantized representations.
        
        Args:
            inputs: Input list of (question, cot) pairs, list of (question, cot, return_cot_only) tuples,
                   or list of (question, [CoT1, CoT2, ..., CoTn]) tuples for multi-CoT encoding. Always batched.
            
        Returns:
            Tuple of (quantized_vectors, vq_loss, perplexity, encoding_indices, questions, tokens, example_groups, step_groups)
        """
        # Get representations from frozen encoder
        with torch.no_grad():
            # representations are on cpu
            representations, tokens, questions, example_groups, step_groups = get_last_layer_representations(
                self.encoder_model,
                self.tokenizer,
                inputs,
                device=self.device,
                max_length=self.max_length,
                return_cot_only=self.compress_cot_only,
                num_cot_traces=self.num_cot_traces
            )
            
            # Clear cache after encoder forward pass
            clear_memory()
        
        if self.num_cot_traces > 1:
            all_inputs = []
            all_tokens = []
            all_questions = []
            all_step_groups = []
            all_example_groups = []
            current_example_group = 0
            max_num_steps = max(step_groups) + 1
            current_inputs = [[] for _ in range(max_num_steps)]
            current_tokens = [[] for _ in range(self.num_cot_traces)]
            current_question = questions[0]
            current_trace_group = -1
            for i in range(len(representations)):
                if example_groups[i] != current_example_group:
                    for j in range(max_num_steps):
                        if len(current_inputs[j]) > 0:
                            all_inputs.append(torch.cat(current_inputs[j], dim=0))
                            all_example_groups.append(current_example_group)
                            all_step_groups.append(j)
                            if j == 0:
                                all_tokens.append(current_tokens)
                            else:
                                all_tokens.append(None)
                            all_questions.append(current_question)
                    current_inputs = [[] for _ in range(max_num_steps)]
                    current_tokens = [[] for _ in range(self.num_cot_traces)]
                    current_question = questions[i]
                    current_example_group = example_groups[i]
                    current_trace_group = -1
                assert current_question == questions[i]
                current_inputs[step_groups[i]].append(representations[i])
                current_inputs[step_groups[i]].append(self.parallel_token_embedding.unsqueeze(0))
                if step_groups[i] == 0:
                    current_trace_group += 1
                current_tokens[current_trace_group].append(tokens[i])
        else:
            all_inputs = representations
            all_tokens = tokens
            all_questions = questions
            all_example_groups = example_groups
            all_step_groups = step_groups
                
        # pad sequences to the same length and create attention mask
        batch_size = len(all_inputs)
        max_length = min(max([len(input_representation) for input_representation in all_inputs]), self.max_length)   
        # input sequence tensor: [N, S, D]
        padded_inputs = torch.zeros(batch_size, max_length, self.hidden_size, device=self.device)
        # attention mask tensor: [N, T, S]
        attention_mask = torch.zeros(batch_size, 1, max_length, device=self.device, dtype=torch.bool)
        for i, input_representation in enumerate(all_inputs):
            padded_inputs[i, :len(input_representation)] = input_representation
            # create attention mask for padded input embeddings. True represents masked tokens.
            attention_mask[i, :, len(input_representation):] = 1
        # attention mask tensor: [N * H, T, S]
        attention_mask = torch.repeat_interleave(attention_mask, self.num_heads, dim=0)
        
        # apply multi-head attention quantizer
        quantized_vectors = self.quantizer(padded_inputs, attention_mask)
        
        # apply vector quantization
        quantized, vq_loss, perplexity, encoding_indices = self.vq(quantized_vectors)
        
        # Clean up intermediate tensors
        del padded_inputs, attention_mask, quantized_vectors
        clear_memory()
        
        return quantized, vq_loss, perplexity, encoding_indices, all_questions, all_tokens, all_example_groups, all_step_groups
    
    def decode(self, 
               quantized_vectors: torch.Tensor, 
               original_tokens: List[List[int]], 
               questions: List[str]=None,
               example_groups: List[int]=None,
               step_groups: List[int]=None,
               batch_order: str = "length" # "random", "length", or "none"
    ) -> torch.Tensor:
        """
        Decode batched quantized vectors with corresponding questions back to text representations and compute reconstruction loss.
        
        This method assumes the decoder_model is a LlamaForCausalLM and uses its built-in
        loss computation with the labels argument.
        
        Args:
            quantized_vectors: all quantized vectors, shape: (num_segments, k, hidden_size)
            original_tokens: List of tokenized (potentially split) original CoT segments, shape: (num_segments, num_tokens)
            questions: List of questions for each CoT in the batch, shape: (num_segments, num_tokens)
            example_groups: List of example groups for each CoT in the batch, shape: (num_segments,)
            step_groups: List of step groups for each CoT in the batch, shape: (num_segments,)
        Returns:
            Average reconstruction loss across the batch
        """
        embed_layer = self.decoder_model.get_input_embeddings()
        ctx = torch.enable_grad() if self.train_decoder else torch.no_grad()
        num_segments = len(quantized_vectors)
        batch_size = max(example_groups) + 1
        max_num_steps = max(step_groups) + 1
        
        # Pre-compute all token embeddings to avoid repeated lookups
        all_tokens = []
        # Collect all tokens that need embedding and record their positions
        prompt_positions = []
        question_positions = []
        original_positions = []
        quantized_positions = []
        
        # add pad token
        with ctx:
            all_tokens.append(self.tokenizer.pad_token_id)
            pad_token_position = 0
        
        # add prompt tokens
        if self.prompt:
            prompt_tokens = self.tokenizer(self.prompt, add_special_tokens=False)['input_ids']
            all_tokens += prompt_tokens
            prompt_positions = list(range(len(all_tokens)-len(prompt_tokens), len(all_tokens)))
            
        # add question tokens
        if questions:
            question_tokens = self.tokenizer(questions, add_special_tokens=False)['input_ids']
            for question in question_tokens:
                all_tokens += question
                question_positions.append(list(range(len(all_tokens)-len(question), len(all_tokens))))
        
        # add original tokens
        if self.num_cot_traces > 1:
            for original_token in original_tokens:
                if original_token:
                    positions = []
                    for token in original_token:
                        position = []
                        for t in token:
                            if len(t) > 0:
                                all_tokens += t
                                position.append(list(range(len(all_tokens)-len(t), len(all_tokens))))
                        positions.append(position)
                    original_positions.append(positions)
                else:
                    original_positions.append(None)
        else:
            for original_token in original_tokens:
                all_tokens += original_token
                original_positions.append(list(range(len(all_tokens)-len(original_token), len(all_tokens))))
            
        # get positions of quantized vectors
        start_idx = len(all_tokens)
        for _ in range(num_segments):
            quantized_positions.append(list(range(start_idx, 
                                        start_idx+quantized_vectors.shape[1])))
            start_idx += quantized_vectors.shape[1]
        
        # batch embed all tokens at once (major speedup)
        with ctx:
            if all_tokens:
                all_token_tensor = torch.tensor(all_tokens, device=self.device).int()
                token_embeddings = embed_layer(all_token_tensor)

        # add quantized vectors to token embeddings
        token_embeddings = torch.cat([token_embeddings, quantized_vectors.view(-1, self.hidden_size)], dim=0)
        
        # add explain token embedding if specified
        if self.explain_token:
            # add explain token embedding to token embeddings
            token_embeddings = torch.cat([token_embeddings, self.explain_token_embedding], dim=0)
            # get position of explain token
            explain_token_position = len(token_embeddings) - 1
        
        # add think token embedding if specified
        if self.think_token:
            # add think token embedding to token embeddings
            token_embeddings = torch.cat([token_embeddings, self.think_token_embedding], dim=0)
            # get position of think token
            think_token_position = len(token_embeddings) - 1
        
        # add parallel token embedding if specified
        if self.num_cot_traces > 1:
            # add parallel token embedding to token embeddings
            token_embeddings = torch.cat([token_embeddings, self.parallel_token_embedding], dim=0)
            # get position of parallel token
            parallel_token_position = len(token_embeddings) - 1
        
        if self.previous_segments_mode == "text":
            segments_positions = copy.deepcopy(original_positions)
        elif self.previous_segments_mode == "latent":
            segments_positions = copy.deepcopy(quantized_positions)
        elif self.previous_segments_mode == "none":
            segments_positions = []
        else:
            raise ValueError(f"Invalid previous segments mode: {self.previous_segments_mode}")
        
        # Build input batch with positions
        input_positions = []
        labels = []
        if self.num_cot_traces > 1:
            # for multi-CoT setting, each quantized vector is associated with multiple original tokens
            # condition on different CoT traces, the quantized vectors should predict different original tokens
            current_example_group = None
            current_quantized_positions = []
            current_step_group = []
            for i in range(num_segments):
                if example_groups[i] != current_example_group:
                    if current_example_group:
                        for original_position in current_segment_positions:
                            current_previous_positions = []
                            for j, original_step_position in enumerate(original_position):
                                if len(original_step_position) == 0:
                                    continue
                                current_input_positions = []
                                current_labels = []
                                assert current_step_group[j] == j, \
                                    f"Step group mismatch: {current_step_group[j]} != {j}"
                                if self.prompt:
                                    current_input_positions += prompt_positions
                                if questions:
                                    current_input_positions += current_question_positions
                                if current_previous_positions:
                                    current_input_positions += current_previous_positions
                                current_previous_positions += original_step_position
                                if self.think_token: # ignore the previous segments mode. Always use text mode.
                                    current_input_positions.append(think_token_position)
                                current_input_positions += current_quantized_positions[j]
                                if self.explain_token:
                                    current_input_positions.append(explain_token_position)
                                current_labels += [-100] * len(current_input_positions)
                                current_input_positions += original_step_position
                                current_labels += original_step_position
                                assert len(current_input_positions) == len(current_labels), \
                                    f"Length mismatch: positions={len(current_input_positions)}, labels={len(current_labels)}"
                                input_positions.append(current_input_positions)
                                labels.append(current_labels)
                    current_example_group = example_groups[i]
                    current_quantized_positions = []
                    current_step_group = []
                    
                if original_tokens[i]:
                    current_segment_positions = original_positions[i]
                current_question_positions = question_positions[i]
                current_quantized_positions.append(quantized_positions[i])
                current_step_group.append(step_groups[i])
            
        else:
            # get positions of previous segments first
            prev_group = None
            previous_segments_positions = []
            for group, segment_position in zip(example_groups, segments_positions):
                if group != prev_group:
                    previous_segments_positions.append([])
                    previous_segments_position = segment_position
                    prev_group = group
                else:
                    previous_segments_position += segment_position
                    previous_segments_positions.append(previous_segments_position)
            
            # build input batch with positions
            for i in range(num_segments):
                cur_segment_positions = []
                cur_labels = []
                if self.prompt:
                    cur_segment_positions += prompt_positions
                if questions:
                    cur_segment_positions += question_positions[i]
                if self.think_token and self.previous_segments_mode == "latent":
                    cur_segment_positions.append(think_token_position)
                if previous_segments_positions:
                    cur_segment_positions += previous_segments_positions[i]
                if self.think_token and self.previous_segments_mode == "text":
                    cur_segment_positions.append(think_token_position)
                cur_segment_positions += quantized_positions[i]
                if self.explain_token:
                    cur_segment_positions.append(explain_token_position)
                cur_labels += [-100] * len(cur_segment_positions)
                
                # # Ensure original_positions[i] and original_tokens[i] have matching lengths
                # if len(original_positions[i]) != len(original_tokens[i]):
                #     print(f"Error: Length mismatch at segment {i}: positions={len(original_positions[i])}, tokens={len(original_tokens[i])}")
                #     print("positions: ", original_positions[i])
                #     print("tokens: ", original_tokens[i])
                #     exit(1)
                # else:
                cur_segment_positions += original_positions[i]
                cur_labels += original_tokens[i]
                # Ensure input_positions and labels have the same length
                assert len(cur_segment_positions) == len(cur_labels), \
                    f"Length mismatch: positions={len(cur_segment_positions)}, labels={len(cur_labels)}"
                input_positions.append(cur_segment_positions)
                labels.append(cur_labels)
            
        # reorder modified_embeddings and labels to optimize performance/memory usage
        if batch_order == "random": # possibly performance gain through randomization
            temp = list(zip(input_positions, labels))
            random.shuffle(temp)
            input_positions, labels = zip(*temp)
        elif batch_order == "length": # more efficient for long sequences
            temp = list(zip(input_positions, labels))
            temp.sort(key=lambda x: len(x[0]))
            input_positions, labels = zip(*temp)
        elif batch_order == "none": # segments in the same CoT are grouped together
            pass
        else:
            raise ValueError(f"Invalid batch order: {batch_order}")
        
        # regroup modified_embeddings and labels to match the original batch size
        # prevent possible memory issues by batching
        i = 0
        batch_input_positions = []
        batch_labels = []
        while len(input_positions) > batch_size*i:
            start = batch_size*i
            end = min(len(input_positions), batch_size*(i+1))
            batch_input_positions.append(input_positions[start:end])
            batch_labels.append(labels[start:end])
            i += 1
    
        # Batch pad all sequences at once
        reconstruction_loss = 0
        for input_positions, labels in zip(batch_input_positions, batch_labels):
            # # Validate that input_positions and labels have matching lengths
            # for pos, label in zip(input_positions, labels):
            #     if len(pos) != len(label):
            #         print(f"Error: Length mismatch detected - positions: {len(pos)}, labels: {len(label)}")
            #         print(f"Position sample: {pos[:5]}...")
            #         print(f"Label sample: {label[:5]}...")
            #         exit(1)
            batch_max_length = min(max([len(input_position) for input_position in input_positions]), self.max_length)
            
            # Pre-allocate tensors for batch processing
            batch_size = len(input_positions)
            # zero is the default value for padding positions
            padded_input_positions = [[0] * batch_max_length for _ in range(batch_size)]
            attention_mask = [[False] * batch_max_length for _ in range(batch_size)]
            padded_labels = [[-100] * batch_max_length for _ in range(batch_size)]

            for i, (input_position, label) in enumerate(zip(input_positions, labels)):
                assert len(input_position) == len(label), \
                    f"Length mismatch: positions={len(input_position)}, labels={len(label)}"
                seq_len = len(input_position)
                # Ensure sequences are truncated to the same length
                if seq_len > batch_max_length:
                    seq_len = batch_max_length
                
                # Copy embeddings
                padded_input_positions[i][:seq_len] = input_position[:seq_len]
                
                # Set attention mask
                for j in range(seq_len):
                    attention_mask[i][j] = True
                
                # Copy labels
                padded_labels[i][:seq_len] = label[:seq_len]
            
            # Convert to tensors
            padded_input_positions = torch.tensor(padded_input_positions, device=self.device, dtype=torch.int)
            attention_mask = torch.tensor(attention_mask, device=self.device, dtype=torch.bool)
            padded_labels = torch.tensor(padded_labels, device=self.device, dtype=torch.long)
        
            padded_input_positions_flat = padded_input_positions.flatten()
            
            # Memory-efficient embedding lookup using advanced indexing instead of one-hot
            # This avoids creating a massive one-hot tensor
            padded_input_embeddings = token_embeddings[padded_input_positions_flat]
            padded_input_embeddings = padded_input_embeddings.view(batch_size, batch_max_length, self.hidden_size)
            
            # Clean up intermediate tensors
            del padded_input_positions_flat
        
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
            if hasattr(model_outputs, 'loss') and model_outputs.loss is not None:
                reconstruction_loss += model_outputs.loss
            else:
                print(f"Error: Model output has no loss attribute. Output type: {type(model_outputs)}")
                print(f"Available attributes: {dir(model_outputs)}")
                exit(1)
            
            # Clean up large tensors to prevent memory accumulation
            del padded_input_embeddings, attention_mask, padded_labels, model_outputs
            clear_memory()
        
        reconstruction_loss /= len(batch_input_positions)
        
        # Return the reconstruction loss average across segments
        return reconstruction_loss
    
    
    def forward(self, inputs: Union[List[Tuple[str, str]], List[Tuple[str, str, bool]], List[Tuple[str, List[str]]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VQ-VAE.
        
        Args:
            inputs: Input list of (question, cot) pairs, list of (question, cot, return_cot_only) tuples,
                   or list of (question, [CoT1, CoT2, ..., CoTn]) tuples for multi-CoT encoding. Always batched.
            
        Returns:
            Tuple of (vq_loss, perplexity, reconstruction_loss, total_loss)
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
        
        return vq_loss, perplexity, reconstruction_loss, total_loss 
    
        
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
                'previous_segments_mode': self.previous_segments_mode,
                'num_cot_traces': self.num_cot_traces
            },
            'trained_components': {
                'quantizer_state_dict': self.quantizer.state_dict(),
                'vq_state_dict': self.vq.state_dict(),
            },
            'tokenizer': self.tokenizer
        }
        
        # Add explain embedding if it exists
        if hasattr(self, 'explain_token_embedding'):
            checkpoint_data['trained_components']['explain_token_embedding'] = self.explain_token_embedding.data
        
        # Add thinking embedding if it exists
        if hasattr(self, 'think_token_embedding'):
            checkpoint_data['trained_components']['think_token_embedding'] = self.think_token_embedding.data
            
        # Add parallel token embedding if it exists
        if hasattr(self, 'parallel_token_embedding'):
            checkpoint_data['trained_components']['parallel_token_embedding'] = self.parallel_token_embedding.data
        
        torch.save(checkpoint_data, checkpoint_path)
        
        print(f"Checkpoint saved to: {checkpoint_path}")
    
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
                'previous_segments_mode': self.previous_segments_mode,
                'num_cot_traces': self.num_cot_traces
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
        if hasattr(self, 'explain_token_embedding'):
            custom_embedding_params += self.explain_token_embedding.numel()
        if hasattr(self, 'think_token_embedding'):
            custom_embedding_params += self.think_token_embedding.numel()
        if hasattr(self, 'parallel_token_embedding'):
            custom_embedding_params += self.parallel_token_embedding.numel()
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
        if hasattr(self, 'explain_token_embedding') and self.explain_token_embedding.requires_grad:
            custom_embeddings['explain_token_embedding'] = {
                'shape': list(self.explain_token_embedding.shape),
                'numel': self.explain_token_embedding.numel(),
                'dtype': str(self.explain_token_embedding.dtype),
                'device': str(self.explain_token_embedding.device)
            }
        
        if hasattr(self, 'think_token_embedding') and self.think_token_embedding.requires_grad:
            custom_embeddings['think_token_embedding'] = {
                'shape': list(self.think_token_embedding.shape),
                'numel': self.think_token_embedding.numel(),
                'dtype': str(self.think_token_embedding.dtype),
                'device': str(self.think_token_embedding.device)
            }
        
        if hasattr(self, 'parallel_token_embedding') and self.parallel_token_embedding.requires_grad:
            custom_embeddings['parallel_token_embedding'] = {
                'shape': list(self.parallel_token_embedding.shape),
                'numel': self.parallel_token_embedding.numel(),
                'dtype': str(self.parallel_token_embedding.dtype),
                'device': str(self.parallel_token_embedding.device)
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
        if hasattr(self, 'explain_token_embedding'):
            custom_embeddings['explain_token_embedding'] = {
                'shape': list(self.explain_token_embedding.shape),
                'numel': self.explain_token_embedding.numel(),
                'dtype': str(self.explain_token_embedding.dtype),
                'device': str(self.explain_token_embedding.device),
                'requires_grad': self.explain_token_embedding.requires_grad
            }
        
        if hasattr(self, 'think_token_embedding'):
            custom_embeddings['think_token_embedding'] = {
                'shape': list(self.think_token_embedding.shape),
                'numel': self.think_token_embedding.numel(),
                'dtype': str(self.think_token_embedding.dtype),
                'device': str(self.think_token_embedding.device),
                'requires_grad': self.think_token_embedding.requires_grad
            }
        
        if hasattr(self, 'parallel_token_embedding'):
            custom_embeddings['parallel_token_embedding'] = {
                'shape': list(self.parallel_token_embedding.shape),
                'numel': self.parallel_token_embedding.numel(),
                'dtype': str(self.parallel_token_embedding.dtype),
                'device': str(self.parallel_token_embedding.device),
                'requires_grad': self.parallel_token_embedding.requires_grad
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
    test_text = [("Test question", "This is a test sentence for VQ-VAE."), 
                    ("Test question 2", "This is should be splitted into two segments: This is the first segment. This is the second segment.")]
    print(f"\nTesting with text: {test_text}")
    
    # Encode
    quantized, vq_loss, perplexity, encoding_indices, questions, tokens, segment_groups = vq_vae.encode(test_text)
    print(f"Encoded shape: {quantized.shape}")
    print(f"VQ loss: {vq_loss.item():.4f}")
    print(f"Perplexity: {perplexity.item():.2f}")
    print(f"Encoding indices: {encoding_indices}")
    print(f"Questions: {questions}")
    print(f"Tokens: {tokens}")
    print(f"Segment groups: {segment_groups}")
    
    # Decode - note: decode now returns only the loss for batched input
    recon_loss = vq_vae.decode(quantized, tokens, questions, segment_groups)
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    
    # Full forward pass
    vq_loss, perplexity, recon_loss, total_loss = vq_vae(test_text)
    print(f"Full forward pass completed successfully!")
    print(f"Average reconstruction loss: {recon_loss.item():.4f}")
    print(f"Total VQ-VAE loss: {total_loss.item():.4f}")
    
    # Demonstrate checkpoint functionality
    print(f"\n=== Checkpoint Demo ===")
    
    # Save checkpoint
    checkpoint_path = "vq_vae_checkpoint.pt"
    vq_vae.save_checkpoint(checkpoint_path)
    print(f"Saved checkpoint to: {checkpoint_path}")
    
    # Load from checkpoint
    print(f"\nLoading from checkpoint...")
    # Create a temporary instance to call the load method
    temp_vq_vae = VQVAETextReconstructor(
        pretrained_model_name="../models/OLMo-2-0425-1B-Base",
        num_latent_tokens=64,
        k=8,
        num_heads=8,
        commitment_cost=0.25
    )
    loaded_vq_vae = temp_vq_vae.load_from_checkpoint(checkpoint_path)
    
    # Test loaded model
    quantized_loaded, vq_loss_loaded, perplexity_loaded, recon_loss_loaded, total_loss_loaded = loaded_vq_vae(test_text)
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