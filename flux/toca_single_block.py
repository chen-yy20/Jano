"""
ToCa Single Block Implementation for Flux - CORRECT VERSION
基于flux-ToCa的正确逻辑：
1. Full步骤：完整计算attention和MLP，保存输出到cache
2. ToCa步骤：直接复用cache的attention输出，选择性计算MLP
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
from .toca_functions import cache_init, cal_type, cache_cutfresh, update_cache, force_init


class ToCaSingleBlockForward:
    """
    完整的ToCa single block forward实现
    正确实现attention cache复用逻辑
    """
    def __init__(self, block, block_idx, original_forward, cache_dic, current):
        self.block = block
        self.block_idx = block_idx
        self.original_forward = original_forward
        self.cache_dic = cache_dic
        self.current = current
    
    def __call__(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        ToCa-enhanced forward for FluxSingleTransformerBlock
        """
        # 设置当前层信息
        self.current['layer'] = self.block_idx
        self.current['stream'] = 'single_stream'
        
        # 获取当前计算类型
        calc_type = self.current.get('type', 'full')
        
        if calc_type == 'full':
            # ===== FULL模式：完整计算并存储cache =====
            residual = hidden_states
            norm_hidden_states, gate = self.block.norm(hidden_states, emb=temb)
            
            # 1. 计算Attention（完整计算）
            joint_attention_kwargs = joint_attention_kwargs or {}
            attn_output = self.block.attn(
                hidden_states=norm_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **joint_attention_kwargs,
            )
            
            # 2. 计算MLP（完整计算）
            mlp_hidden_states = self.block.act_mlp(self.block.proj_mlp(norm_hidden_states))
            
            # 3. Concat并输出
            hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
            gate = gate.unsqueeze(1)
            hidden_states = gate * self.block.proj_out(hidden_states)
            hidden_states = residual + hidden_states
            
            if hidden_states.dtype == torch.float16:
                hidden_states = hidden_states.clip(-65504, 65504)
            
            # ===== 存储cache =====
            try:
                # 确保cache结构存在
                if 'single_stream' not in self.cache_dic['cache'][-1]:
                    self.cache_dic['cache'][-1]['single_stream'] = {}
                if self.block_idx not in self.cache_dic['cache'][-1]['single_stream']:
                    self.cache_dic['cache'][-1]['single_stream'][self.block_idx] = {}
                
                # 存储Attention输出（关键！）
                self.cache_dic['cache'][-1]['single_stream'][self.block_idx]['attn'] = attn_output.detach().clone()
                
                # 调试信息（仅第一层）
                if self.block_idx == 0 and self.current['step'] < 3:
                    print(f"  [Debug] Block 0 Full: attn_output.shape={attn_output.shape}, hidden_states.shape={hidden_states.shape}")
                
                # 存储MLP中间结果（linear1的mlp部分输出，激活前）
                mlp_before_act = self.block.proj_mlp(norm_hidden_states)
                self.cache_dic['cache'][-1]['single_stream'][self.block_idx]['mlp'] = mlp_before_act.detach().clone()
                
                # 存储最终输出（用于FORA模式）
                self.cache_dic['cache'][-1]['single_stream'][self.block_idx]['total'] = hidden_states.detach().clone()
                
                # 存储attention map（用于计算importance score）
                # 使用norm_hidden_states的norm作为proxy
                if self.cache_dic['cache_type'] == 'attention':
                    # 计算一个简单的attention score proxy
                    attn_map_proxy = torch.linalg.norm(attn_output, dim=-1)  # [B, L]
                    self.cache_dic['attn_map'][-1]['single_stream'][self.block_idx]['total'] = attn_map_proxy.detach()
                
                # 初始化cache_index
                self.current['module'] = 'mlp'
                force_init(cache_dic=self.cache_dic, current=self.current, tokens=norm_hidden_states)
                
            except Exception as e:
                print(f"⚠ Cache storage failed at block {self.block_idx}: {e}")
                import traceback
                traceback.print_exc()
            
            return hidden_states
        
        elif calc_type == 'ToCa':
            # ===== ToCa模式：复用attention cache，选择性计算MLP =====
            residual = hidden_states
            norm_hidden_states, gate = self.block.norm(hidden_states, emb=temb)
            
            try:
                # 1. 从cache直接读取Attention输出（不重新计算！）
                attn_output = self.cache_dic['cache'][-1]['single_stream'][self.block_idx]['attn']
                
                # 检查shape是否匹配
                if attn_output.shape[1] != norm_hidden_states.shape[1]:
                    # Shape不匹配，回退到完整计算
                    print(f"  [Warning] Block {self.block_idx} ToCa: Cache shape mismatch! "
                          f"cached={attn_output.shape}, current={norm_hidden_states.shape}, "
                          f"step={self.current['step']}")
                    raise RuntimeError(f"Cache shape mismatch")
                
                # 2. MLP: 选择重要token进行fresh计算
                self.current['module'] = 'mlp'
                
                # 选择需要fresh的tokens
                fresh_indices, fresh_tokens_input = cache_cutfresh(
                    self.cache_dic,
                    norm_hidden_states,  # 基于当前的norm_hidden_states选择
                    self.current
                )
                
                # 只对fresh tokens计算MLP（proj_mlp部分）
                mlp_fresh = self.block.proj_mlp(fresh_tokens_input)
                
                # 更新cache（只更新fresh部分）
                update_cache(
                    fresh_indices,
                    mlp_fresh,
                    self.cache_dic,
                    self.current
                )
                
                # 从cache读取完整的MLP结果（包含fresh和stale）
                mlp_before_act = self.cache_dic['cache'][-1]['single_stream'][self.block_idx]['mlp']
                
                # 应用激活函数
                mlp_hidden_states = self.block.act_mlp(mlp_before_act)
                
                # 3. Concat并输出
                hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
                gate = gate.unsqueeze(1)
                hidden_states = gate * self.block.proj_out(hidden_states)
                hidden_states = residual + hidden_states
                
                if hidden_states.dtype == torch.float16:
                    hidden_states = hidden_states.clip(-65504, 65504)
                
                return hidden_states
                
            except Exception as e:
                # ToCa失败，回退到full computation
                if self.block_idx == 0:
                    print(f"⚠ ToCa failed at block {self.block_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                
                # 回退：完整计算
                residual = hidden_states
                norm_hidden_states, gate = self.block.norm(hidden_states, emb=temb)
                
                joint_attention_kwargs = joint_attention_kwargs or {}
                attn_output = self.block.attn(
                    hidden_states=norm_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                    **joint_attention_kwargs,
                )
                
                mlp_hidden_states = self.block.act_mlp(self.block.proj_mlp(norm_hidden_states))
                
                hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
                gate = gate.unsqueeze(1)
                hidden_states = gate * self.block.proj_out(hidden_states)
                hidden_states = residual + hidden_states
                
                if hidden_states.dtype == torch.float16:
                    hidden_states = hidden_states.clip(-65504, 65504)
                
                return hidden_states
        
        elif calc_type == 'FORA':
            # ===== FORA模式：完全复用cache（aggressive caching）=====
            try:
                # 直接从cache读取最终输出
                hidden_states = self.cache_dic['cache'][-1]['single_stream'][self.block_idx]['total']
                return hidden_states
            except:
                # 回退到原始forward
                return self.original_forward(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs
                )
        
        else:
            # 其他模式：使用原始forward
            return self.original_forward(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs
            )


class ToCaDoubleBlockForward:
    """
    ToCa double stream block forward实现
    处理text和image两个流
    """
    def __init__(self, block, block_idx, original_forward, cache_dic, current):
        self.block = block
        self.block_idx = block_idx
        self.original_forward = original_forward
        self.cache_dic = cache_dic
        self.current = current
    
    def __call__(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        ToCa-enhanced forward for FluxTransformerBlock (double stream)
        """
        self.current['layer'] = self.block_idx
        self.current['stream'] = 'double_stream'
        
        calc_type = self.current.get('type', 'full')
        
        if calc_type == 'full':
            # ===== FULL模式 =====
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.block.norm1(hidden_states, emb=temb)
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.block.norm1_context(
                encoder_hidden_states, emb=temb
            )
            
            # Attention
            joint_attention_kwargs = joint_attention_kwargs or {}
            attn_output, context_attn_output = self.block.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **joint_attention_kwargs,
            )
            
            # Image stream
            hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
            norm_hidden_states = self.block.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            ff_output = self.block.ff(norm_hidden_states)
            hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
            
            # Text stream
            encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
            norm_encoder_hidden_states = self.block.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            context_ff_output = self.block.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
            
            # ===== 存储cache =====
            try:
                if 'double_stream' not in self.cache_dic['cache'][-1]:
                    self.cache_dic['cache'][-1]['double_stream'] = {}
                if self.block_idx not in self.cache_dic['cache'][-1]['double_stream']:
                    self.cache_dic['cache'][-1]['double_stream'][self.block_idx] = {}
                
                # 存储attention输出
                # 注意：需要分别存储img和txt的attention
                # 合并后的attn包含了两部分
                combined_attn = torch.cat([context_attn_output, attn_output], dim=1)
                self.cache_dic['cache'][-1]['double_stream'][self.block_idx]['attn'] = combined_attn.detach().clone()
                self.cache_dic['txt_shape'] = context_attn_output.shape[1]
                
                # 存储MLP输出
                self.cache_dic['cache'][-1]['double_stream'][self.block_idx]['img_mlp'] = ff_output.detach().clone()
                self.cache_dic['cache'][-1]['double_stream'][self.block_idx]['txt_mlp'] = context_ff_output.detach().clone()
                
                # 存储attention map
                if self.cache_dic['cache_type'] == 'attention':
                    img_attn_map = torch.linalg.norm(attn_output, dim=-1)
                    txt_attn_map = torch.linalg.norm(context_attn_output, dim=-1)
                    combined_map = torch.cat([txt_attn_map, img_attn_map], dim=1)
                    self.cache_dic['attn_map'][-1]['double_stream'][self.block_idx]['total'] = combined_map.detach()
                    self.cache_dic['attn_map'][-1]['double_stream'][self.block_idx]['img_mlp'] = img_attn_map.detach()
                    self.cache_dic['attn_map'][-1]['double_stream'][self.block_idx]['txt_mlp'] = txt_attn_map.detach()
                
                # 初始化cache_index
                self.current['module'] = 'img_mlp'
                force_init(cache_dic=self.cache_dic, current=self.current, tokens=hidden_states)
                self.current['module'] = 'txt_mlp'
                force_init(cache_dic=self.cache_dic, current=self.current, tokens=encoder_hidden_states)
                
            except Exception as e:
                print(f"⚠ Cache storage failed at double block {self.block_idx}: {e}")
            
            # 注意：diffusers的FluxTransformerBlock返回 (encoder_hidden_states, hidden_states)
            # 即 (txt, img)，必须保持这个顺序！
            return encoder_hidden_states, hidden_states
        
        elif calc_type == 'ToCa':
            # ===== ToCa模式 =====
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.block.norm1(hidden_states, emb=temb)
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.block.norm1_context(
                encoder_hidden_states, emb=temb
            )
            
            try:
                # 1. 从cache读取attention输出
                combined_attn = self.cache_dic['cache'][-1]['double_stream'][self.block_idx]['attn']
                txt_shape = self.cache_dic.get('txt_shape', encoder_hidden_states.shape[1])
                context_attn_output = combined_attn[:, :txt_shape]
                attn_output = combined_attn[:, txt_shape:]
                
                # 检查shape是否匹配
                if (context_attn_output.shape[1] != encoder_hidden_states.shape[1] or
                    attn_output.shape[1] != hidden_states.shape[1]):
                    raise RuntimeError(f"Cache shape mismatch in double stream")
                
                # 2. Image MLP: 选择性计算
                hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
                norm_hidden_states = self.block.norm2(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                
                self.current['module'] = 'img_mlp'
                fresh_indices, fresh_tokens = cache_cutfresh(
                    self.cache_dic,
                    norm_hidden_states,
                    self.current
                )
                
                fresh_ff_output = self.block.ff(fresh_tokens)
                update_cache(fresh_indices, fresh_ff_output, self.cache_dic, self.current)
                
                ff_output = self.cache_dic['cache'][-1]['double_stream'][self.block_idx]['img_mlp']
                hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
                
                # 3. Text MLP: 选择性计算
                encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
                norm_encoder_hidden_states = self.block.norm2_context(encoder_hidden_states)
                norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
                
                self.current['module'] = 'txt_mlp'
                fresh_indices, fresh_tokens = cache_cutfresh(
                    self.cache_dic,
                    norm_encoder_hidden_states,
                    self.current
                )
                
                fresh_context_ff_output = self.block.ff_context(fresh_tokens)
                update_cache(fresh_indices, fresh_context_ff_output, self.cache_dic, self.current)
                
                context_ff_output = self.cache_dic['cache'][-1]['double_stream'][self.block_idx]['txt_mlp']
                encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
                
                # 返回顺序必须是 (txt, img)
                return encoder_hidden_states, hidden_states
                
            except Exception as e:
                print(f"⚠ ToCa failed at double block {self.block_idx}: {e}")
                import traceback
                traceback.print_exc()
                
                # 回退到原始forward
                return self.original_forward(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs
                )
        
        else:
            # 其他模式：使用原始forward
            return self.original_forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs
            )


class ToCaTransformerPatcher:
    """
    为Flux Transformer添加ToCa支持的补丁类
    """
    
    def __init__(self, transformer, num_inference_steps):
        self.transformer = transformer
        self.num_inference_steps = num_inference_steps
        self.cache_dic = None
        self.current = None
        self.is_enabled = False
        
        # 保存原始的forward方法
        self.original_single_block_forwards = {}
        self.original_double_block_forwards = {}
        self.original_transformer_forward = None
        
    def enable_toca(self):
        """启用ToCa加速"""
        if self.is_enabled:
            print("⚠ ToCa已经启用")
            return
        
        # 初始化cache
        timesteps_list = list(range(self.num_inference_steps, 0, -1))
        self.cache_dic, self.current = cache_init(timesteps_list)
        self.current['num_steps'] = self.num_inference_steps - 1
        self.current['step'] = 0
        
        print(f"✓ ToCa cache初始化完成")
        print(f"  Cache类型: {self.cache_dic['cache_type']}")
        print(f"  Fresh ratio: {self.cache_dic['fresh_ratio']}")
        print(f"  Fresh threshold: {self.cache_dic['fresh_threshold']}")
        print(f"  Force fresh: {self.cache_dic['force_fresh']}")
        
        # Patch blocks
        self._patch_double_blocks()  # 先patch double stream (19层)
        self._patch_single_blocks()  # 再patch single stream (38层)
        
        # Hook transformer的forward来管理步骤
        self._hook_transformer_forward()
        
        self.is_enabled = True
        print(f"✓ ToCa已启用（完整加速：double + single stream）")
        
    def _patch_single_blocks(self):
        """为single transformer blocks添加ToCa支持"""
        if not hasattr(self.transformer, 'single_transformer_blocks'):
            print("⚠ Warning: transformer没有single_transformer_blocks")
            return
        
        blocks = self.transformer.single_transformer_blocks
        num_blocks = len(blocks)
        
        for block_idx in range(num_blocks):
            block = blocks[block_idx]
            
            # 保存原始forward
            self.original_single_block_forwards[block_idx] = block.forward
            
            # 创建新的forward callable对象
            new_forward = ToCaSingleBlockForward(
                block=block,
                block_idx=block_idx,
                original_forward=block.forward,
                cache_dic=self.cache_dic,
                current=self.current
            )
            
            # 替换forward
            block.forward = new_forward
        
        print(f"✓ 已patch {num_blocks} 个single transformer blocks")
    
    def _patch_double_blocks(self):
        """为double transformer blocks添加ToCa支持"""
        if not hasattr(self.transformer, 'transformer_blocks'):
            print("⚠ Warning: transformer没有transformer_blocks (double stream)")
            return
        
        blocks = self.transformer.transformer_blocks
        num_blocks = len(blocks)
        
        for block_idx in range(num_blocks):
            block = blocks[block_idx]
            
            # 保存原始forward
            self.original_double_block_forwards[block_idx] = block.forward
            
            # 创建新的forward callable对象
            new_forward = ToCaDoubleBlockForward(
                block=block,
                block_idx=block_idx,
                original_forward=block.forward,
                cache_dic=self.cache_dic,
                current=self.current
            )
            
            # 替换forward
            block.forward = new_forward
        
        print(f"✓ 已patch {num_blocks} 个double transformer blocks")
    
    def _hook_transformer_forward(self):
        """Hook transformer的forward来管理ToCa步骤"""
        original_forward = self.transformer.forward
        self.original_transformer_forward = original_forward
        
        cache_dic = self.cache_dic
        current = self.current
        
        def wrapped_forward(*args, **kwargs):
            # 在每次forward之前调用cal_type
            if cache_dic is not None and current is not None:
                cal_type(cache_dic, current)
                
                # Debug信息
                if current['step'] % 5 == 0:
                    print(f"  Step {current['step']}: type={current.get('type', 'unknown')}")
            
            # 调用原始forward
            result = original_forward(*args, **kwargs)
            
            # 步骤递增在pipeline中处理
            
            return result
        
        self.transformer.forward = wrapped_forward
    
    def disable_toca(self):
        """完全禁用ToCa，恢复到原始diffusers状态"""
        if not self.is_enabled:
            return
        
        # 恢复transformer的forward
        if self.original_transformer_forward is not None:
            self.transformer.forward = self.original_transformer_forward
        
        # 恢复single blocks的forward
        if hasattr(self.transformer, 'single_transformer_blocks'):
            blocks = self.transformer.single_transformer_blocks
            for block_idx in range(len(blocks)):
                if block_idx in self.original_single_block_forwards:
                    blocks[block_idx].forward = self.original_single_block_forwards[block_idx]
        
        # 恢复double blocks的forward
        if hasattr(self.transformer, 'transformer_blocks'):
            blocks = self.transformer.transformer_blocks
            for block_idx in range(len(blocks)):
                if block_idx in self.original_double_block_forwards:
                    blocks[block_idx].forward = self.original_double_block_forwards[block_idx]
        
        # 清理ToCa添加的属性
        if hasattr(self.transformer, 'toca_cache_dic'):
            delattr(self.transformer, 'toca_cache_dic')
        if hasattr(self.transformer, 'toca_current'):
            delattr(self.transformer, 'toca_current')
        
        # 清理cache
        self.cache_dic = None
        self.current = None
        self.is_enabled = False
        
        # 清空CUDA缓存，防止编译缓存影响
        import torch
        torch.cuda.empty_cache()
        
        # 如果有torch.compile，清理编译缓存
        if hasattr(torch, '_dynamo'):
            torch._dynamo.reset()
        
        print("✓ ToCa已禁用，已恢复到原始diffusers状态")
    
    def update_step(self, step_idx):
        """更新当前推理步骤"""
        if self.current is not None:
            self.current['step'] = step_idx


def apply_toca_to_pipeline(pipe, num_inference_steps, enable=True):
    """
    为FluxPipeline应用ToCa支持
    
    Args:
        pipe: FluxPipeline实例
        num_inference_steps: 推理步骤数
        enable: 是否启用ToCa
    
    Returns:
        ToCaTransformerPatcher实例（用于后续控制）
    """
    patcher = ToCaTransformerPatcher(pipe.transformer, num_inference_steps)
    
    if enable:
        patcher.enable_toca()
    
    # 存储到pipeline以便后续访问
    pipe.toca_patcher = patcher
    
    return patcher

