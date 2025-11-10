# Copyright 2024-2025 Flux PAB Implementation
"""
Flux Model with PAB Support

This module wraps the Flux model to pass timestep information to all blocks
for PAB caching.
"""

import torch
from torch import Tensor
import types
import logging

from .pab_manager import enable_pab
from utils.timer import get_timer


def wrap_flux_forward_with_pab(model):
    """
    Wrap Flux model's forward method to pass timestep to blocks
    
    Args:
        model: Flux model instance
        
    Returns:
        model: Modified model with PAB-enabled forward
    """
    # Save original forward
    original_forward = model.forward
    
    @get_timer("dit_pab")
    def forward_with_pab(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor = None,
        pooled_projections: Tensor = None,
        timestep: Tensor = None,
        img_ids: Tensor = None,
        txt_ids: Tensor = None,
        guidance: Tensor = None,
        joint_attention_kwargs: dict = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Tensor:
        """
        Forward pass with timestep passed to blocks for PAB
        This is for diffusers FluxTransformer2DModel
        """
        from diffusers.models.modeling_outputs import Transformer2DModelOutput
        from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
        
        # Handle LoRA scaling
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        
        # Embed inputs
        hidden_states = self.x_embedder(hidden_states)
        
        # Process timestep for embedding (keep original for PAB)
        timestep_for_embed = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance_for_embed = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance_for_embed = None
        
        # Extract timestep value for PAB (after embedding processing)
        timestep_value = None
        if enable_pab() and timestep is not None:
            try:
                # timestep_for_embed is already in 0-1000 scale after * 1000
                timestep_value = float(timestep_for_embed[0].item()) if timestep_for_embed.numel() > 0 else 0.0
            except (IndexError, ValueError, TypeError):
                timestep_value = None
        
        temb = (
            self.time_text_embed(timestep_for_embed, pooled_projections)
            if guidance_for_embed is None
            else self.time_text_embed(timestep_for_embed, guidance_for_embed, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        
        # Handle txt_ids and img_ids
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        
        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)
        
        # Get the correct blocks attribute (diffusers)
        if hasattr(self, 'transformer_blocks'):
            double_blocks = self.transformer_blocks
            single_blocks = self.single_transformer_blocks
        else:
            double_blocks = self.double_blocks
            single_blocks = self.single_blocks

        # Pass through transformer blocks (double stream blocks)
        for index_block, block in enumerate(double_blocks):
            if hasattr(block, 'pab_enabled'):  # PAB-enabled block
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                    timestep=timestep_value
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs
                )
            
            # Handle controlnet residuals if present
            if controlnet_block_samples is not None:
                import numpy as np
                interval_control = len(double_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                if controlnet_blocks_repeat:
                    hidden_states = hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
        
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        
        # Pass through single transformer blocks
        for index_block, block in enumerate(single_blocks):
            if hasattr(block, 'pab_enabled'):  # PAB-enabled block
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                    timestep=timestep_value
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs
                )
            
            # Handle controlnet residuals if present
            if controlnet_single_block_samples is not None:
                import numpy as np
                interval_control = len(single_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1]:, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1]:, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )
        
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]
        
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        
        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)
        
        if not return_dict:
            return (output,)
        
        return Transformer2DModelOutput(sample=output)
    
    # Get timestep_embedding function from flux
    # from flux.modules.layers import timestep_embedding
    # global timestep_embedding_fn
    # timestep_embedding_fn = timestep_embedding
    
    # Replace forward method
    model.forward = types.MethodType(forward_with_pab, model)
    
    logging.info("✓ Wrapped Flux model forward method with PAB timestep passing")
    
    return model

