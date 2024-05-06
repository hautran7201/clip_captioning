import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import List, Optional
from utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from model.clip_encoder import CLIPEncoder
    

class CLIPTextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        embed_dim = config.hidden_size
        
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embedding, embed_dim)
        
        self.register_buffer(
            'position_ids', torch.arange(config.max_position_embedding).expand(1, -1), persistent=False
        )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor],
        position_ids: Optional[torch.LongTensor]=None,
        input_embeds: Optional[torch.FloatTensor]=None
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else input_embeds[-2]
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
            
        if input_embeds is None:
            input_embeds = self.token_embedding(input_ids)
            
        position_embeddings = self.position_embedding(position_ids)
        embeddings = input_embeds + position_embeddings
        
        return embeddings


class CLIPTextTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = CLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        
        self.eos_token_id = config.eos_token_id
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict : Optional[bool] = False
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Shape
        input_shape = input_ids.shape
        
        # Embedding
        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)
        
        if input_ids is None:
            raise ValueError("You have to specify input_ids")
            
        # Causal attention mask
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )        
        
        # Attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
            
        encoder_outputs = self.encoder(
            input_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)
        
        if self.eos_token_id == 2:
            # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
            # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, sequence_length, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]
            
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        
        return (
            last_hidden_state,
            pooled_output,
            encoder_outputs.hidden_states,
            encoder_outputs.attentions
        )        