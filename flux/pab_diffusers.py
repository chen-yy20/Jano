# Copyright 2024-2025 Flux PAB Implementation
"""
PAB implementation for diffusers FluxTransformer2DModel

This module replaces attention processors with PAB-enabled processors that cache
ONLY attention outputs, not entire block outputs.

Reference: STDit implementation which uses custom AttentionProcessor to intercept attention computation.
"""

import torch
from torch import Tensor
import types
import logging
from typing import Optional, Dict, Any, Tuple

from .pab_manager import enable_pab, if_broadcast_self, if_broadcast_cross
from .pab_attention_processor import PABFluxAttnProcessor2_0, reset_pab_stats_processor, get_pab_stats_processor, reset_global_timestep_count


# Global PAB statistics (for backward compatibility)
class PABStats:
    """PAB Statistics Tracker"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.self_compute = 0
        self.self_cached = 0
        self.cross_compute = 0
        self.cross_cached = 0
    
    def get_stats(self):
        # Get stats from processor
        proc_stats = get_pab_stats_processor()
        return {
            'self_compute': proc_stats['self_compute'],
            'self_cached': proc_stats['self_cached'],
            'self_cache_ratio': (proc_stats['self_cached'] / (proc_stats['self_compute'] + proc_stats['self_cached']) * 100) if (proc_stats['self_compute'] + proc_stats['self_cached']) > 0 else 0,
            'cross_compute': proc_stats['cross_compute'],
            'cross_cached': proc_stats['cross_cached'],
            'cross_cache_ratio': (proc_stats['cross_cached'] / (proc_stats['cross_compute'] + proc_stats['cross_cached']) * 100) if (proc_stats['cross_compute'] + proc_stats['cross_cached']) > 0 else 0,
        }


# Global statistics instance
_pab_stats = PABStats()


def get_pab_stats():
    """Get PAB statistics"""
    return _pab_stats.get_stats()


def reset_pab_stats():
    """Reset PAB statistics"""
    _pab_stats.reset()
    reset_pab_stats_processor()


# Global timestep storage for processors
_global_timestep = None

def set_global_timestep(timestep: Optional[float]):
    """Set global timestep for PAB processors"""
    global _global_timestep
    _global_timestep = timestep


def create_pab_double_stream_forward(original_forward, block_idx):
    """
    Create a PAB-enabled forward method for FluxTransformerBlock (double stream)
    
    This wraps the original forward and passes timestep to attention processor via joint_attention_kwargs.
    """
    
    def forward_with_pab(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        temb: Tensor,
        image_rotary_emb: Optional[Tuple[Tensor, Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        timestep: float = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward with PAB - passes timestep to attention processor
        """
        # Pass timestep to attention processor via joint_attention_kwargs
        if enable_pab() and timestep is not None:
            if joint_attention_kwargs is None:
                joint_attention_kwargs = {}
            joint_attention_kwargs = joint_attention_kwargs.copy()
            joint_attention_kwargs['timestep'] = timestep
            # Also set global timestep for processor access
            set_global_timestep(timestep)
        
        # Call original forward
        result = original_forward(
            self,
            hidden_states,
            encoder_hidden_states,
            temb,
            image_rotary_emb,
            joint_attention_kwargs
        )
        
        return result
    
    return forward_with_pab


def create_pab_single_stream_forward(original_forward, block_idx):
    """
    Create a PAB-enabled forward method for FluxSingleTransformerBlock
    
    This wraps the original forward and passes timestep to attention processor via joint_attention_kwargs.
    """
    
    def forward_with_pab(
        self,
        hidden_states: Tensor,
        temb: Tensor,
        image_rotary_emb: Optional[Tuple[Tensor, Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        timestep: float = None,
    ) -> Tensor:
        """
        Forward with PAB - passes timestep to attention processor
        """
        # Pass timestep to attention processor via joint_attention_kwargs
        if enable_pab() and timestep is not None:
            if joint_attention_kwargs is None:
                joint_attention_kwargs = {}
            joint_attention_kwargs = joint_attention_kwargs.copy()
            joint_attention_kwargs['timestep'] = timestep
            # Also set global timestep for processor access
            set_global_timestep(timestep)
        
        # Call original forward
        result = original_forward(
            self,
            hidden_states,
            temb,
            image_rotary_emb,
            joint_attention_kwargs
        )
        
        return result
    
    return forward_with_pab


def apply_pab_to_model(model):
    """
    Apply PAB to diffusers FluxTransformer2DModel by replacing attention processors
    
    Args:
        model: FluxTransformer2DModel instance
    """
    import logging
    
    # Replace attention processors with PAB-enabled processors
    processors = {}
    
    # Get all attention processors
    def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict):
        if hasattr(module, "get_processor"):
            processors[f"{name}.processor"] = (module, module.get_processor())
        
        for sub_name, child in module.named_children():
            fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)
        
        return processors
    
    for name, module in model.named_children():
        fn_recursive_add_processors(name, module, processors)
    
    # Debug: Log all found processors to identify single stream blocks
    logging.info(f"[PAB] Found {len(processors)} processors total")
    single_processor_names = [name for name in processors.keys() if 'single' in name.lower()]
    logging.info(f"[PAB] Processors with 'single' in name: {len(single_processor_names)}")
    if single_processor_names:
        for name in single_processor_names[:5]:  # Log first 5
            logging.info(f"[PAB] Single processor path: {name}")
    else:
        # Log all processor names to debug
        all_names = list(processors.keys())
        logging.info(f"[PAB] All processor paths (first 20): {all_names[:20]}")
        logging.info(f"[PAB] Checking for patterns...")
        transformer_blocks = [name for name in all_names if 'transformer' in name.lower()]
        logging.info(f"[PAB] Processors with 'transformer' in name: {len(transformer_blocks)}")
        if transformer_blocks:
            logging.info(f"[PAB] Sample transformer processor paths: {transformer_blocks[:5]}")
    
    # Replace processors
    num_double = 0
    num_single = 0
    
    for name, (module, original_processor) in processors.items():
        # Determine block type and layer index from name
        if 'transformer_blocks' in name and 'single_transformer_blocks' not in name:
            # Double stream block
            # Extract layer index from name like "transformer_blocks.0.attn.processor"
            try:
                parts = name.split('.')
                layer_idx = int(parts[1]) if len(parts) > 1 else -1
                
                # Create PAB processor
                pab_processor = PABFluxAttnProcessor2_0(
                    layer_idx=layer_idx,
                    block_type='double',
                    original_processor=original_processor
                )
                module.set_processor(pab_processor)
                num_double += 1
                logging.debug(f"[PAB] Replaced processor: {name} (double stream, layer {layer_idx})")
            except (ValueError, IndexError) as e:
                logging.warning(f"[PAB] Could not extract layer index from {name}: {e}")
        
        elif 'single_transformer_blocks' in name:
            # Single stream block
            try:
                parts = name.split('.')
                # Extract layer index from name like "transformer.single_transformer_blocks.0.attn.processor"
                # or "single_transformer_blocks.0.attn.processor"
                layer_idx = -1
                for i, part in enumerate(parts):
                    if part == 'single_transformer_blocks' and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                            break
                        except (ValueError, IndexError):
                            pass
                
                if layer_idx == -1:
                    logging.warning(f"[PAB] Could not extract layer index from {name}")
                    continue
                
                # Create PAB processor
                pab_processor = PABFluxAttnProcessor2_0(
                    layer_idx=layer_idx,
                    block_type='single',
                    original_processor=original_processor
                )
                module.set_processor(pab_processor)
                num_single += 1
                logging.debug(f"[PAB] Replaced processor: {name} (single stream, layer {layer_idx})")
            except (ValueError, IndexError) as e:
                logging.warning(f"[PAB] Could not extract layer index from {name}: {e}")
    
    logging.info(f"✓ Applied PAB to {num_double} double stream processors and {num_single} single stream processors")
    
    # Wrap forward methods to pass timestep
    if hasattr(model, 'transformer_blocks'):
        for i, block in enumerate(model.transformer_blocks):
            original_forward = block.forward
            block.forward = types.MethodType(
                create_pab_double_stream_forward(original_forward.__func__, i),
                block
            )
            block.pab_enabled = True
            block.pab_block_idx = i
        
        logging.info(f"✓ Wrapped {len(model.transformer_blocks)} double stream block forwards")
    
    if hasattr(model, 'single_transformer_blocks'):
        for i, block in enumerate(model.single_transformer_blocks):
            original_forward = block.forward
            block.forward = types.MethodType(
                create_pab_single_stream_forward(original_forward.__func__, i),
                block
            )
            block.pab_enabled = True
            block.pab_block_idx = i
        
        logging.info(f"✓ Wrapped {len(model.single_transformer_blocks)} single stream block forwards")
    
    return model


def reset_model_pab_cache(model):
    """Reset all PAB caches in the model"""
    
    # Reset global timestep counter
    reset_global_timestep_count()
    
    # Reset processor caches
    processors = {}
    
    def fn_recursive_get_processors(name: str, module: torch.nn.Module, processors: Dict):
        if hasattr(module, "get_processor"):
            proc = module.get_processor()
            if isinstance(proc, PABFluxAttnProcessor2_0):
                processors[f"{name}.processor"] = proc
        
        for sub_name, child in module.named_children():
            fn_recursive_get_processors(f"{name}.{sub_name}", child, processors)
        
        return processors
    
    for name, module in model.named_children():
        fn_recursive_get_processors(name, module, processors)
    
    for name, proc in processors.items():
        proc.reset_cache()
    
    logging.info(f"✓ Reset PAB caches for {len(processors)} processors")
