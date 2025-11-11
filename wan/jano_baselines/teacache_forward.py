# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
from datetime import datetime
import logging
import os
import sys
import warnings

warnings.filterwarnings('ignore')

import torch, random
import torch.distributed as dist
from PIL import Image

import gc
from contextlib import contextmanager
import torchvision.transforms.functional as TF
import torch.cuda.amp as amp
import numpy as np
import math
from wan.modules.model import sinusoidal_embedding_1d

from utils.timer import get_timer
from jano.mask_manager.wan_mask_manager import get_mask_manager
from jano.modules.wan.wan_t2v import WanT2V_jano
from jano.stuff import get_timestep, get_masked_timer
from utils.envs import GlobalEnv


def wrap_model_with_teacache(wan_t2v: WanT2V_jano, args):
    # TeaCache
    # wan_t2v.__class__.generate = t2v_generate
    wan_t2v.model.__class__.enable_teacache = True
    wan_t2v.model.__class__.forward = jano_teacache_forward
    wan_t2v.model.__class__.cnt = 0
    wan_t2v.model.__class__.num_steps = args.sample_steps*2
    wan_t2v.model.__class__.teacache_thresh = args.teacache_thresh
    wan_t2v.model.__class__.accumulated_rel_l1_distance_even = 0
    wan_t2v.model.__class__.accumulated_rel_l1_distance_odd = 0
    wan_t2v.model.__class__.previous_e0_even = None
    wan_t2v.model.__class__.previous_e0_odd = None
    wan_t2v.model.__class__.previous_residual_even = None
    wan_t2v.model.__class__.previous_residual_odd = None
    wan_t2v.model.__class__.use_ref_steps = args.use_ret_steps
    if args.use_ret_steps:
        if '1.3B' in args.ckpt_dir:
            wan_t2v.model.__class__.coefficients = [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02]
        if '14B' in args.ckpt_dir:
            wan_t2v.model.__class__.coefficients = [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01]
        wan_t2v.model.__class__.ret_steps = 5*2
        wan_t2v.model.__class__.cutoff_steps = args.sample_steps*2
    else:
        if '1.3B' in args.ckpt_dir:
            wan_t2v.model.__class__.coefficients = [2.39676752e+03, -1.31110545e+03,  2.01331979e+02, -8.29855975e+00, 1.37887774e-01]
        if '14B' in args.ckpt_dir:
            wan_t2v.model.__class__.coefficients = [-5784.54975374,  5449.50911966, -1811.16591783,   256.27178429, -13.02252404]
        wan_t2v.model.__class__.ret_steps = 1*2
        wan_t2v.model.__class__.cutoff_steps = args.sample_steps*2 - 2


def get_active_hidden_states(hidden_states, use_jano):
    if use_jano:
        mask_manager = get_mask_manager()
        if mask_manager.step_level == 3:
            active_hidden_states = hidden_states[:, mask_manager.active_bool_mask, ...]
        elif mask_manager.step_level == 2:
            active_hidden_states = hidden_states[:, mask_manager.active_bool_mask_in_l2, ...]
        else:
            active_hidden_states = hidden_states
    else:
        active_hidden_states = hidden_states
    return active_hidden_states

def teacache_forward(
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None,
):
    r"""
    Forward pass through the diffusion model

    Args:
        x (List[Tensor]):
            List of input video tensors, each with shape [C_in, F, H, W]
        t (Tensor):
            Diffusion timesteps tensor of shape [B]
        context (List[Tensor]):
            List of text embeddings each with shape [L, C]
        seq_len (`int`):
            Maximum sequence length for positional encoding
        clip_fea (Tensor, *optional*):
            CLIP image features for image-to-video mode
        y (List[Tensor], *optional*):
            Conditional video inputs for image-to-video mode, same shape as x

    Returns:
        List[Tensor]:
            List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
    """
    if self.model_type == 'i2v':
        assert clip_fea is not None and y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                    dim=1) for u in x
    ])

    # time embeddings
    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat(
                [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        context = torch.concat([context_clip, context], dim=1)

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens)
        
    if self.enable_teacache:
        modulated_inp = e0 if self.use_ref_steps else e
        # teacache
        if self.cnt%2==0: # even -> conditon
            self.is_even = True
            if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                    should_calc_even = True
                    self.accumulated_rel_l1_distance_even = 0
            else:
                rescale_func = np.poly1d(self.coefficients)
                self.accumulated_rel_l1_distance_even += rescale_func(((modulated_inp-self.previous_e0_even).abs().mean() / self.previous_e0_even.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance_even < self.teacache_thresh:
                    should_calc_even = False
                    print(f"Teacache step {self.cnt // 2}: skip!, error: {self.accumulated_rel_l1_distance_even}/{self.teacache_thresh} ", flush=True)

                else:
                    should_calc_even = True
                    self.accumulated_rel_l1_distance_even = 0
                    print(f"Teacache step {self.cnt // 2}: should calc! ", flush=True)
                    
            self.previous_e0_even = modulated_inp.clone()

        else: # odd -> unconditon
            self.is_even = False
            if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                    should_calc_odd = True
                    self.accumulated_rel_l1_distance_odd = 0
            else: 
                rescale_func = np.poly1d(self.coefficients)
                self.accumulated_rel_l1_distance_odd += rescale_func(((modulated_inp-self.previous_e0_odd).abs().mean() / self.previous_e0_odd.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance_odd < self.teacache_thresh:
                    should_calc_odd = False
                else:
                    should_calc_odd = True
                    self.accumulated_rel_l1_distance_odd = 0
            self.previous_e0_odd = modulated_inp.clone()

    with get_masked_timer("DiT-fw"):
        if self.enable_teacache: 
            if self.is_even:
                if not should_calc_even:
                    x += self.previous_residual_even
                else:
                    ori_x = x.clone()
                    for block in self.blocks:
                        x = block(x, **kwargs)
                    self.previous_residual_even = x - ori_x
            else:
                if not should_calc_odd:
                    x += self.previous_residual_odd
                else:
                    ori_x = x.clone()
                    for block in self.blocks:
                        x = block(x, **kwargs)
                    self.previous_residual_odd = x - ori_x
        
        else:
            for block in self.blocks:
                x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    self.cnt += 1
    if self.cnt >= self.num_steps:
        self.cnt = 0
    return [u.float() for u in x]



def jano_teacache_forward(
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None,
):
    r"""
    Forward pass through the diffusion model

    Args:
        x (List[Tensor]):
            List of input video tensors, each with shape [C_in, F, H, W]
        t (Tensor):
            Diffusion timesteps tensor of shape [B]
        context (List[Tensor]):
            List of text embeddings each with shape [L, C]
        seq_len (`int`):
            Maximum sequence length for positional encoding
        clip_fea (Tensor, *optional*):
            CLIP image features for image-to-video mode
        y (List[Tensor], *optional*):
            Conditional video inputs for image-to-video mode, same shape as x

    Returns:
        List[Tensor]:
            List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
    """
    if self.model_type == 'i2v':
        assert clip_fea is not None and y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                    dim=1) for u in x
    ])

    # time embeddings
    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat(
                [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
        context = torch.concat([context_clip, context], dim=1)

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens)
    
    mask_manager = get_mask_manager()
    use_jano = GlobalEnv.get_envs("enable_stdit")
    if mask_manager is None:
        GlobalEnv.set_envs("timer_prefix", "full")
    else:
        mask_manager.update_step_level()
        level = mask_manager.step_level
        if level == 0:
            GlobalEnv.set_envs("timer_prefix", "full")
        elif level == 1:
            GlobalEnv.set_envs("timer_prefix", "1")
        elif level == 2:
            GlobalEnv.set_envs("timer_prefix", "2")
        elif level == 3:
            GlobalEnv.set_envs("timer_prefix", "3")
        if level != 0:   
            print(f"{get_timestep()} | level {level}.", flush=True)
        
    if self.enable_teacache:
        modulated_inp = e0 if self.use_ref_steps else e
        if use_jano:
            should_store = mask_manager.step_level != 0
        else:
            should_store = True
            
        # teacache
        if self.cnt%2==0: # even -> conditon
            self.is_even = True
            if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                    should_calc_even = True
                    self.accumulated_rel_l1_distance_even = 0
            else:
                rescale_func = np.poly1d(self.coefficients)
                self.accumulated_rel_l1_distance_even += rescale_func(((modulated_inp-self.previous_e0_even).abs().mean() / self.previous_e0_even.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance_even < self.teacache_thresh:
                    # jano 判定
                    if use_jano:
                        if mask_manager.step_level == 1: # 只有仅仅active的时候全部跳过
                            should_calc_even = False
                        else:
                            should_calc_even = True
                            self.accumulated_rel_l1_distance_even = 0
                    else:
                        should_calc_even = False
                        
                    if should_calc_even == False:
                        print(f"Teacache step {self.cnt // 2}: skip!, error: {self.accumulated_rel_l1_distance_even}/{self.teacache_thresh} ", flush=True)

                else:
                    should_calc_even = True
                    self.accumulated_rel_l1_distance_even = 0
                    print(f"Teacache step {self.cnt // 2}: should calc! ", flush=True)
                    
            self.previous_e0_even = modulated_inp.clone()

        else: # odd -> unconditon
            self.is_even = False
            if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                should_calc_odd = True
                self.accumulated_rel_l1_distance_odd = 0
            else: 
                rescale_func = np.poly1d(self.coefficients)
                self.accumulated_rel_l1_distance_odd += rescale_func(((modulated_inp-self.previous_e0_odd).abs().mean() / self.previous_e0_odd.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance_odd < self.teacache_thresh:
                    # jano 判定
                    if use_jano:
                        if mask_manager.step_level == 1: # 只有仅仅active的时候全部跳过
                            should_calc_odd = False
                        else:
                            should_calc_odd = True
                            self.accumulated_rel_l1_distance_odd = 0
                    else:
                        should_calc_odd = False
                        
                    if should_calc_odd == False:
                        print(f"Teacache step {self.cnt // 2}: skip!, error: {self.accumulated_rel_l1_distance_odd}/{self.teacache_thresh} ", flush=True)
                else:
                    should_calc_odd = True
                    self.accumulated_rel_l1_distance_odd = 0
            self.previous_e0_odd = modulated_inp.clone()

    with get_masked_timer("tea_dit"):
        if use_jano:
            x = mask_manager.apply_sequence_mask(x)

        if self.enable_teacache: 
            if self.is_even:
                if not should_calc_even and self.previous_residual_even is not None:
                    print(f"{get_timestep()} | Teacache: SKIP even.", flush=True)
                    x += self.previous_residual_even
                else:
                    # ori_x = x.clone()
                    ori_x = get_active_hidden_states(x, use_jano).clone()
                    for block in self.blocks:
                        x = block(x, **kwargs)
                    if should_store:
                        self.previous_residual_even = get_active_hidden_states(x, use_jano) - ori_x
                        print(f"{get_timestep()} | Teacache: stored {self.previous_residual_even.shape=}", flush=True)

            else:
                if not should_calc_odd and self.previous_residual_odd is not None:
                    print(f"{get_timestep()} | Teacache: SKIP odd.", flush=True)
                    x += self.previous_residual_odd
                else:
                    ori_x = get_active_hidden_states(x, use_jano).clone()
                    for block in self.blocks:
                        x = block(x, **kwargs)
                    if should_store:
                        self.previous_residual_odd = get_active_hidden_states(x, use_jano) - ori_x
                        print(f"{get_timestep()} | Teacache: stored {self.previous_residual_odd.shape=}", flush=True)
        
        else:
            for block in self.blocks:
                x = block(x, **kwargs)

        # head
        x = self.head(x, e)
        if use_jano:
            x = mask_manager.process_masked_output(x, f"output_c{GlobalEnv.get_envs('cond')}", -1)


    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    self.cnt += 1
    if self.cnt >= self.num_steps:
        self.cnt = 0
    return [u.float() for u in x]

