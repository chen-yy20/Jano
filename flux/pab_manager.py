# Copyright 2024-2025 Flux PAB Implementation
# PAB (Pyramid Attention Broadcast) Manager for Flux
# Based on VideoSys PAB: https://github.com/NUS-HPC-AI-Lab/VideoSys

"""
Pyramid Attention Broadcast (PAB) Manager for Flux

PAB is a training-free acceleration technique for diffusion models that reuses
attention outputs during stable phases of the diffusion process.

Core Idea:
- The middle 70% of diffusion steps have relatively stable attention outputs
- Self-Attention changes more (high-frequency signal)
- Cross-Attention is most stable (low-frequency signal)
"""

import logging
from typing import Optional

__all__ = ['PABConfig', 'PABManager', 'set_pab_manager', 'enable_pab', 
           'update_steps', 'if_broadcast_self', 'if_broadcast_cross']

# Global PAB manager instance
PAB_MANAGER = None


class PABConfig:
    """
    PAB Configuration Class
    
    Parameters:
        self_broadcast (bool): Enable self-attention broadcast
        self_threshold (list): Timestep range for self-attention broadcast [min, max]
        self_range (int): Self-attention broadcast interval (reuse every N steps)
        
        cross_broadcast (bool): Enable cross-attention broadcast
        cross_threshold (list): Timestep range for cross-attention broadcast [min, max]
        cross_range (int): Cross-attention broadcast interval (reuse every N steps)
    
    Note: For Flux, "cross-attention" refers to the interaction between image and text streams.
    
    Example:
        >>> # Default configuration
        >>> config = PABConfig()
        >>> 
        >>> # Custom configuration for Flux
        >>> config = PABConfig(
        ...     self_broadcast=True,
        ...     self_threshold=[100, 800],
        ...     self_range=2,
        ...     cross_broadcast=True,
        ...     cross_threshold=[100, 900],
        ...     cross_range=5
        ... )
    """
    
    def __init__(
        self,
        # Self-Attention Configuration
        self_broadcast: bool = True,
        self_threshold: Optional[list] = None,
        self_range: Optional[int] = None,
        # Cross-Attention Configuration (img-txt interaction)
        cross_broadcast: bool = True,
        cross_threshold: Optional[list] = None,
        cross_range: Optional[int] = None,
    ):
        self.steps = None  # Total sampling steps, set at runtime
        
        # Self-Attention Configuration
        self.self_broadcast = self_broadcast
        self.self_threshold = self_threshold if self_threshold is not None else [100, 800]
        self.self_range = self_range if self_range is not None else 2
        
        # Cross-Attention Configuration
        self.cross_broadcast = cross_broadcast
        self.cross_threshold = cross_threshold if cross_threshold is not None else [100, 900]
        self.cross_range = cross_range if cross_range is not None else 5


class PABManager:
    """PAB Manager"""
    
    def __init__(self, config: PABConfig):
        self.config: PABConfig = config
        
        # Print initialization info
        init_prompt = "Initializing Pyramid Attention Broadcast (PAB) for Flux"
        init_prompt += f"\n  Self-Attention Broadcast: {config.self_broadcast}"
        init_prompt += f", Range: {config.self_range}, Threshold: {config.self_threshold}"
        init_prompt += f"\n  Cross-Attention Broadcast: {config.cross_broadcast}"
        init_prompt += f", Range: {config.cross_range}, Threshold: {config.cross_threshold}"
        logging.info(init_prompt)
    
    def if_broadcast_self(self, timestep: float, count: int):
        """
        Determine if self-attention should broadcast (reuse cache)
        
        Args:
            timestep: Current diffusion timestep (0-1000 scale)
            count: Current counter (number of calls within current sampling step)
            
        Returns:
            (broadcast, new_count): Whether to broadcast and updated counter
        """
        if (
            self.config.self_broadcast
            and (timestep is not None)
            and (count % self.config.self_range != 0)
            and (self.config.self_threshold[0] < timestep < self.config.self_threshold[1])
        ):
            flag = True
        else:
            flag = False
        # Increment counter (don't modulo, as it resets per sampling step)
        count = count + 1
        return flag, count
    
    def if_broadcast_cross(self, timestep: float, count: int):
        """
        Determine if cross-attention should broadcast (reuse cache)
        
        Args:
            timestep: Current diffusion timestep (0-1000 scale)
            count: Current counter (number of calls within current sampling step)
            
        Returns:
            (broadcast, new_count): Whether to broadcast and updated counter
        """
        if (
            self.config.cross_broadcast
            and (timestep is not None)
            and (count % self.config.cross_range != 0)
            and (self.config.cross_threshold[0] < timestep < self.config.cross_threshold[1])
        ):
            flag = True
        else:
            flag = False
        # Increment counter (don't modulo, as it resets per sampling step)
        count = count + 1
        return flag, count


# ==================== Global Interface Functions ====================

def set_pab_manager(config: PABConfig):
    """Set global PAB manager"""
    global PAB_MANAGER
    PAB_MANAGER = PABManager(config)


def enable_pab():
    """Check if PAB is enabled"""
    if PAB_MANAGER is None:
        return False
    return (
        PAB_MANAGER.config.self_broadcast
        or PAB_MANAGER.config.cross_broadcast
    )


def update_steps(steps: int):
    """Update total sampling steps"""
    if PAB_MANAGER is not None:
        PAB_MANAGER.config.steps = steps


def if_broadcast_self(timestep: float, count: int):
    """Check if self-attention should broadcast"""
    if not enable_pab():
        return False, count
    return PAB_MANAGER.if_broadcast_self(timestep, count)


def if_broadcast_cross(timestep: float, count: int):
    """Check if cross-attention should broadcast"""
    if not enable_pab():
        return False, count
    return PAB_MANAGER.if_broadcast_cross(timestep, count)

