import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import List, Optional
from model.clip_encoder import CLIPEncoder
from config import CLIPVisionConfig


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()        
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        
        self.class_emmbedding = nn.Parameter(torch.randn(self.embed_dim))
        
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False
        )
        
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            'position_ids', torch.arange(self.num_positions).expand((1, -1)), persistent=False
        )
        
    def forward(
        self, 
        pixel_values: torch.FloatTensor
    ) -> torch.Tensor:
        
        batch_size = pixel_values.shape[0]
        
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        
        # Reshape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        
        class_embeds = self.class_emmbedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        
        return embeddings
    

class CLIPVisionTransformer(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        
    def forward(
        self, 
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Embedding patches and normalize
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layernorm(hidden_states)
        
        # Encoder
        encoder_outputs = self.encoder(
            input_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)
        
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return (
            last_hidden_state,
            pooled_output,
            encoder_outputs.hidden_states,
            encoder_outputs.attentions
        )
    

