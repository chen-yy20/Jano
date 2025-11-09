# Copyright 2024-2025 Flux PAB Implementation
"""
PAB Attention Processor for Flux

This module implements PAB (Pyramid Attention Broadcast) by creating custom attention processors
that cache only attention outputs, not the entire block output.

Reference: STDit implementation which uses custom AttentionProcessor to intercept attention computation.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any
import logging

from .pab_manager import enable_pab, if_broadcast_self, if_broadcast_cross

# Global PAB statistics
_pab_stats_processor = {
    'self_compute': 0,
    'self_cached': 0,
    'cross_compute': 0,
    'cross_cached': 0,
}

# Global timestep counter for PAB (shared across all processors)
# Key insight from wan_pab: self and cross attention share the SAME counter,
# but use different ranges (self_range vs cross_range) for broadcast decisions
_global_timestep_count = 0  # Single counter shared by both self and cross
_global_last_timestep = None

def reset_pab_stats_processor():
    """Reset PAB statistics for processors"""
    global _pab_stats_processor, _global_timestep_count, _global_last_timestep
    _pab_stats_processor = {
        'self_compute': 0,
        'self_cached': 0,
        'cross_compute': 0,
        'cross_cached': 0,
    }
    _global_timestep_count = 0
    _global_last_timestep = None

def get_pab_stats_processor():
    """Get PAB statistics for processors"""
    return _pab_stats_processor.copy()

def reset_global_timestep_count():
    """Reset global timestep counter (call at start of each sampling step)"""
    global _global_timestep_count, _global_last_timestep
    _global_timestep_count = 0
    _global_last_timestep = None

def get_global_timestep_count():
    """Get global timestep counter (shared by both self and cross)"""
    global _global_timestep_count
    return _global_timestep_count

def increment_global_timestep_count():
    """Increment global timestep counter (once per timestep, shared by self and cross)"""
    global _global_timestep_count
    _global_timestep_count += 1
    return _global_timestep_count


class PABFluxAttnProcessor2_0:
    """
    PAB-enabled Attention Processor for Flux
    
    This processor caches ONLY attention outputs, allowing MLP to still be computed
    with current input when attention is reused.
    
    Reference: Based on diffusers FluxAttnProcessor2_0, but adds PAB caching.
    """
    
    def __init__(self, layer_idx, block_type='single', original_processor=None):
        """
        Args:
            layer_idx: Layer index for this attention
            block_type: 'single' for SingleStreamBlock, 'double' for DoubleStreamBlock
            original_processor: Original processor to use when PAB is disabled
        """
        self.layer_idx = layer_idx
        self.block_type = block_type
        self.original_processor = original_processor
        
        # PAB cache - stores only the last computed attention output
        # Reference: wan_pab stores cache_self and cache_cross, which are the last computed attention
        # We reuse this cache across timesteps when broadcast is enabled
        self.cached_attention = None  # Last computed attention output
        self.last_compute_timestep = None  # Timestep where we last computed (not cached)
        
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("PABFluxAttnProcessor2_0 requires PyTorch 2.0")
    
    def __call__(
        self,
        attn,  # Attention module
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        timestep: Optional[float] = None,
        **kwargs,  # Accept additional kwargs (like timestep from cross_attention_kwargs)
    ) -> torch.FloatTensor:
        """
        Compute attention with PAB caching
        
        Key behavior:
        - Caches only attention output (not MLP)
        - When reusing cache, returns cached attention output
        - MLP in block forward will still use current input
        """
        
        # Get timestep from parameter, kwargs, or global variable
        if timestep is None:
            timestep = kwargs.get('timestep', None)
        
        if timestep is None:
            try:
                from pab_diffusers import _global_timestep
                timestep = _global_timestep
            except:
                timestep = None
        
        # Debug: Always log processor calls for single stream blocks (layer 0 only to avoid spam)
        # This helps debug why self-attention statistics are 0
        if enable_pab() and self.layer_idx == 0 and self.block_type == 'single':
            if timestep is None:
                logging.warning(
                    f"[PAB] Single stream processor layer {self.layer_idx}: "
                    f"called but timestep is None. encoder_hidden_states={encoder_hidden_states is not None}, kwargs.keys()={list(kwargs.keys())}"
                )
            else:
                logging.info(
                    f"[PAB] Single stream processor layer {self.layer_idx}: "
                    f"called with timestep={timestep}"
                )
        
        # If PAB is disabled or no timestep, use original processor or compute normally
        # BUT: Still update statistics even if timestep is missing (to debug why it's missing)
        if not enable_pab():
            if self.original_processor is not None:
                return self.original_processor(
                    attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb, **kwargs
                )
            else:
                return self._compute_attention(
                    attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb
                )
        
        if timestep is None:
            # PAB is enabled but timestep is missing - log this and still compute (but don't cache)
            if self.layer_idx == 0:
                logging.warning(
                    f"[PAB] Processor layer {self.layer_idx} ({self.block_type}): "
                    f"PAB enabled but timestep is None. Computing without cache."
                )
            # Compute attention but don't cache (no timestep = no caching)
            attn_output = self._compute_attention(
                attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb
            )
            # Still update statistics to track that processor was called
            if self.block_type == 'double':
                _pab_stats_processor['cross_compute'] += 1
            else:
                _pab_stats_processor['self_compute'] += 1
            return attn_output
        
        timestep_key = int(timestep)
        
        # Manage global timestep counter (shared by both self and cross attention)
        # Reference: wan_pab increments timestep_count once per timestep in block forward
        # (line 114-116), before checking self or cross. All processors in the same timestep
        # should see the same counter value.
        global _global_last_timestep, _global_timestep_count
        if _global_last_timestep is not None and _global_last_timestep != timestep_key:
            # New timestep detected - increment counter once (shared by both self and cross)
            _global_timestep_count = increment_global_timestep_count()
        elif _global_last_timestep is None:
            # First timestep - initialize counter to 0, then increment to 1
            # This matches wan_pab's behavior: counter starts at 0, then increments to 1 on first timestep
            _global_timestep_count = increment_global_timestep_count()
        _global_last_timestep = timestep_key
        
        # Get current counter value (same for all processors in this timestep)
        # Note: This counter is shared by both self and cross attention, but they use
        # different ranges (self_range vs cross_range) to decide when to broadcast
        current_count = get_global_timestep_count()
        
        # Decide whether to broadcast (reuse cache) or compute
        # Reference: wan_pab calls if_broadcast_* with (timestep_key, self.timestep_count - 1)
        # (line 131) because the counter was already incremented above (line 115).
        # We need to pass (current_count - 1) to match wan_pab's behavior.
        # Example: if current_count = 1 (first timestep), pass 0; if current_count = 2, pass 1.
        count_for_broadcast = current_count - 1
        if self.block_type == 'double':
            # Cross attention: uses cross_range for broadcast decision
            broadcast, _ = if_broadcast_cross(timestep_key, count_for_broadcast)
        else:
            # Self attention: uses self_range for broadcast decision  
            broadcast, _ = if_broadcast_self(timestep_key, count_for_broadcast)
        
        # Check if we should reuse cached attention
        # Reference: wan_pab reuses cache_self/cache_cross from last computation
        if broadcast and self.cached_attention is not None:
            if self.block_type == 'double':
                _pab_stats_processor['cross_cached'] += 1
            else:
                _pab_stats_processor['self_cached'] += 1
            
            if self.layer_idx == 0:
                logging.info(
                    f"[PAB] Processor layer {self.layer_idx} ({self.block_type}): "
                    f"CACHE ATTENTION (t={timestep_key}, count={current_count}, reusing from t={self.last_compute_timestep})"
                )
            
            # Return cached attention output
            return self.cached_attention
        
        # Compute attention
        attn_output = self._compute_attention(
            attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb
        )
        
        # Cache the attention output for potential reuse in future timesteps
        # Reference: wan_pab stores cache_self/cache_cross after computation
        self.cached_attention = attn_output
        self.last_compute_timestep = timestep_key
        
        if self.block_type == 'double':
            _pab_stats_processor['cross_compute'] += 1
        else:
            _pab_stats_processor['self_compute'] += 1
        
        if self.layer_idx == 0:
            logging.info(
                f"[PAB] Processor layer {self.layer_idx} ({self.block_type}): COMPUTE ATTENTION (t={timestep_key}, count={current_count})"
            )
        
        return attn_output
    
    def _compute_attention(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        """
        Compute attention (original logic from FluxAttnProcessor2_0)
        """
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        
        # `sample` projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        # For double stream blocks, encoder_hidden_states is not None
        if encoder_hidden_states is not None:
            # `context` projections
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
            
            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            
            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
            
            # Concatenate context and sample queries/keys/values
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
        
        # Apply rotary embedding if provided
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
        
        # Compute attention
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        # For double stream, split back into context and sample
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )
            
            # Linear projection (double stream)
            # Double stream always has to_out and to_add_out
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)  # dropout
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            
            return hidden_states, encoder_hidden_states
        else:
            # Single stream - return attention output directly without to_out projection
            # Reference: STDit implementation returns hidden_states directly for single stream
            # The to_out projection is NOT done here - it's done in the block's forward
            # via proj_out which combines attention and MLP outputs
            return hidden_states
    
    def reset_cache(self):
        """Reset PAB cache (for new generation task)"""
        self.cached_attention = None
        self.last_compute_timestep = None

