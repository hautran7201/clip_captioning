import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import List, Optional


class CLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = nn.GELU()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
    

class CLIPAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.embed_dim = config.hidden_size
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.dropout = config.attention_dropout
        self.scale = self.head_dim**-0.5
        
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        # Hidden states shape
        bsz, tgt_len, embed_dim = hidden_states.size()
        
        # Get projection
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self.k_proj(hidden_states)
        value_states= self.v_proj(hidden_states)
        
        # Reshape: 
        # (bsz, tgt_len, embed_dim)
        # --> (bsz, tgt_len, num_heads, head_dim)
        # --> (bsz, num_heads, tgt_len, head_dim)
        # --> (bsz*num_heads, tgt_len, head_dim)
        proj_shape = (bsz*self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = self._shape(key_states, -1, bsz).view(*proj_shape)
        value_states = self._shape(value_states, -1, bsz).view(*proj_shape)
        
        # Attention weight
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        
        # Check attention weight shape
        src_len = key_states.size(1)
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )
            
        # Apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz*self.num_heads, tgt_len, src_len)
            
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        
        # Apply softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz*self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
        
        # Dropout
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Attention output
        attn_outputs = torch.bmm(attn_probs, value_states)
        
        # Check attention output shape
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )
            
        # Output reshape
        attn_outputs = attn_outputs.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_outputs = attn_outputs.transpose(1, 2)
        attn_outputs = attn_outputs.reshape(bsz, tgt_len, embed_dim)
        
        # Projection output
        attn_outputs = self.out_proj(attn_outputs)
        
        return attn_outputs, attn_weights_reshaped


class CLIPEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
    
    def forward(
        self, 
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[torch.Tensor] = None
    ):
        residual = hidden_states
        
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weight = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions
        )
        hidden_states += residual
        
        residual = hidden_states
        
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states += residual
        
        outputs = (hidden_states, )
        
        if output_attentions:
            outputs += (attn_weight,)
    
        return outputs
    

class CLIPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        
    def forward(
        self, 
        input_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None, 
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ):
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_layers
        return_dict = return_dict if return_dict is not None else self.return_dict
        
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        hidden_states = input_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states += (hidden_states,)
                
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )
                
            hidden_states = layer_outputs[0]
            
            if output_hidden_states:
                all_attentions += (layer_outputs[1], )
                
        if encoder_states:
            encoder_states += (hidden_states, )
        
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        
        return hidden_states, encoder_states, all_attentions
