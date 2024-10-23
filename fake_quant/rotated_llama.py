import warnings
from typing import List, Optional, Tuple, Union

import math
import copy

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import (
    LlamaMLP, LlamaAttention, LlamaDecoderLayer, LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding, LlamaDynamicNTKScalingRotaryEmbedding,
    apply_rotary_pos_emb, repeat_kv, LlamaPreTrainedModel, LlamaRMSNorm,
    LLAMA_INPUTS_DOCSTRING, _CONFIG_FOR_DOC
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)


logger = logging.get_logger(__name__)


class RotatedEmbedding(nn.Embedding):
    
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        _freeze=False,
        device=None,
        dtype=None):
        super().__init__(num_embeddings, embedding_dim, padding_idx,
                         max_norm, norm_type, scale_grad_by_freq, sparse,
                         _weight, _freeze, device, dtype)
    
    def forward(self, x, Q=None):
        W = self.weight
        dtype = W.dtype
        if Q is not None:
            W_ = torch.matmul(W.to(dtype=torch.float64), Q.to(W.device, dtype=torch.float64)).to(dtype=dtype)
        else:
            W_ = W
        
        return torch.nn.functional.embedding(
            x, W_,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse)


class RotatedLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        output=False):
        super().__init__(in_features, out_features, bias, device, dtype)
        
        # output: is Q applied to the output side?
        self.output = output
    
    def forward(self, x, Q=None):
        W = self.weight
        b = self.bias
        dtype = W.dtype
        
        if Q is None:
            W_ = W
            b_ = b

        else:
            
            if self.output:
                W_ = torch.matmul(Q.to(W.device, dtype=torch.float64).T, W.to(dtype=torch.float64)).to(dtype=dtype)
            else:
                W_ = torch.matmul(W.to(dtype=torch.float64), Q.to(W.device, dtype=torch.float64)).to(dtype=dtype)
            
            if b is not None:
                if self.output:
                    b_ = torch.matmul(Q.to(W.device, dtype=torch.float64).T, b.to(dtype=torch.float64)).to(dtype=dtype)
                else:
                    b_ = torch.matmul(b.to(dtype=torch.float64), Q.to(W.device, dtype=torch.float64)).to(dtype=dtype)
            else:
                b_ = b
        
        x = torch.nn.functional.linear(x, W_, b_)
        return x


class RotatedHead(nn.Linear):
    
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
    
    def forward(self, x, Q=None):
        W = self.weight
        dtype = W.dtype
        
        if Q is not None:
            W_ = torch.matmul(W.to(dtype=torch.float64), Q.to(W.device, dtype=torch.float64)).to(dtype=dtype)
        else:
            W_ = W
        
        return torch.nn.functional.linear(
            x, W_,
        )


class RotatedOVProj(nn.Linear):
    
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        output=False,
        nheads=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        
        # output: is R2 applied to the output side?
        # True -> Qin is R1, Qout is R2
        self.output = output
        self.nheads = nheads
    
    def forward(self, x, Qin=None, Qout=None):
        W = self.weight
        dtype = W.dtype
        
        if Qin is not None:
            if self.output:
                W_ = torch.matmul(W.to(dtype=torch.float64), Qin.to(W.device, dtype=torch.float64)).to(dtype=dtype)
            else:
                W_ = W.to(dtype=torch.float64).reshape(W.size(0), self.nheads, -1)
                W_ = torch.einsum('inh,hj->inj', W_, Qin.to(W.device, dtype=torch.float64)).reshape(W.size(0), -1).to(dtype=dtype)
        else:
            W_ = W
        
        if Qout is not None:
            if self.output:
                W_ = W_.to(dtype=torch.float64).reshape(self.nheads, -1, W.size(1))
                W_ = torch.einsum('ih,nhj->nij', Qout.to(W.device, dtype=torch.float64).T, W_).reshape(W.size(0), -1).to(dtype=dtype)
            else:
                W_ = torch.matmul(Qout.to(W.device, dtype=torch.float64).T, W_.to(dtype=torch.float64)).to(dtype=dtype)
        else:
            pass
        
        x = torch.nn.functional.linear(x, W_)
        return x


class RotatedLlamaMLP(nn.Module):
    def __init__(self, config, module: LlamaMLP):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = RotatedLinear(self.hidden_size, self.intermediate_size, bias=False, output=False)
        self.up_proj = RotatedLinear(self.hidden_size, self.intermediate_size, bias=False, output=False)
        self.down_proj = RotatedLinear(self.intermediate_size, self.hidden_size, bias=False, output=True)
        self.act_fn = ACT2FN[config.hidden_act]
        
        self.gate_proj.weight.data = module.gate_proj.weight.data.detach().clone()
        self.up_proj.weight.data = module.up_proj.weight.data.detach().clone()
        self.down_proj.weight.data = module.down_proj.weight.data.detach().clone()
        
        # with torch.no_grad():
        #     self.gate_proj.weight.copy_(module.gate_proj.weight)
        #     self.up_proj.weight.copy_(module.up_proj.weight)
        #     self.down_proj.weight.copy_(module.down_proj.weight)

    def forward(self, x, R1=None):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x, Q=R1)) * self.up_proj(x, Q=R1), Q=R1)

        return down_proj


class RotatedLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None, module: LlamaAttention=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = RotatedLinear(
            self.hidden_size, self.num_heads * self.head_dim,
            config.attention_bias, module.q_proj.weight.data.device,
            module.q_proj.weight.data.dtype, output=False)
        #print('before')
        #print(self.q_proj.weight)
        self.k_proj = RotatedLinear(
            self.hidden_size, self.num_key_value_heads * self.head_dim,
            config.attention_bias, module.k_proj.weight.data.device,
            module.k_proj.weight.data.dtype, output=False)
        self.v_proj = RotatedOVProj(
            self.hidden_size, self.num_key_value_heads * self.head_dim,
            config.attention_bias, module.v_proj.weight.data.device,
            module.v_proj.weight.data.dtype, output=True, nheads=self.num_heads)
        self.o_proj = RotatedOVProj(
            self.num_heads * self.head_dim, self.hidden_size,
            config.attention_bias, module.o_proj.weight.data.device,
            module.o_proj.weight.data.dtype, output=False, nheads=self.num_heads)
        self._init_rope()
        
        attention_bias = config.attention_bias
        
        self.q_proj.weight.data = module.q_proj.weight.data.detach().clone()
        #print('after')
        #print(self.q_proj.weight)
        self.k_proj.weight.data = module.k_proj.weight.data.detach().clone()
        self.v_proj.weight.data = module.v_proj.weight.data.detach().clone()
        self.o_proj.weight.data = module.o_proj.weight.data.detach().clone()
        
        if attention_bias:
            self.q_proj.bias.data = module.q_proj.bias.data.detach().clone()
            self.k_proj.bias.data = module.k_proj.bias.data.detach().clone()
            self.v_proj.bias.data = module.v_proj.bias.data.detach().clone()
            self.o_proj.bias.data = module.o_proj.bias.data.detach().clone()
        # with torch.no_grad():
        #     self.q_proj.weight.copy_(module.q_proj.weight)
        #     self.k_proj.weight.copy_(module.k_proj.weight)
        #     self.v_proj.weight.copy_(module.v_proj.weight)
        #     self.o_proj.weight.copy_(module.o_proj.weight)
            
        #     if attention_bias:
        #         self.q_proj.bias.copy_(module.q_proj.bias)
        #         self.k_proj.bias.copy_(module.k_proj.bias)
        #         self.v_proj.bias.copy_(module.v_proj.bias)
        #         self.o_proj.bias.copy_(module.o_proj.bias)

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # !!! Hardcoded SPDA forward (specific to Llama2)
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        R1=None, R2=None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states, Q=R1)
        key_states = self.k_proj(hidden_states, Q=R1)
        value_states = self.v_proj(hidden_states, Qin=R1, Qout=R2)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, position_ids)#, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output, Qin=R2, Qout=R1)

        return attn_output, None, past_key_value

    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_value: Optional[Cache] = None,
    #     output_attentions: bool = False,
    #     use_cache: bool = False,
    #     R1=None, R2=None,
    #     **kwargs,
    # ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    #     if "padding_mask" in kwargs:
    #         warnings.warn(
    #             "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
    #         )

    #     bsz, q_len, _ = hidden_states.size()

    #     if self.config.pretraining_tp > 1:
    #         key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
    #         query_slices = self.q_proj.weight.split(
    #             (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
    #         )
    #         key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
    #         value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

    #         query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
    #         query_states = torch.cat(query_states, dim=-1)

    #         key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
    #         key_states = torch.cat(key_states, dim=-1)

    #         value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
    #         value_states = torch.cat(value_states, dim=-1)

    #     else:
    #         query_states = self.q_proj(hidden_states, Q=R1)
    #         key_states = self.k_proj(hidden_states, Q=R1)
    #         value_states = self.v_proj(hidden_states, Qin=R1, Qout=R2)

    #     query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    #     key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    #     value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    #     kv_seq_len = key_states.shape[-2]
    #     if past_key_value is not None:
    #         if self.layer_idx is None:
    #             raise ValueError(
    #                 f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
    #                 "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
    #                 "with a layer index."
    #             )
    #         kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    #     cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    #     query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    #     if past_key_value is not None:
    #         cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
    #         key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    #     key_states = repeat_kv(key_states, self.num_key_value_groups)
    #     value_states = repeat_kv(value_states, self.num_key_value_groups)

    #     attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    #     if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
    #         raise ValueError(
    #             f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
    #             f" {attn_weights.size()}"
    #         )

    #     if attention_mask is not None:
    #         if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
    #             raise ValueError(
    #                 f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
    #             )
    #         attn_weights = attn_weights + attention_mask

    #     # upcast attention to fp32
    #     attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    #     attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    #     attn_output = torch.matmul(attn_weights, value_states)

    #     if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
    #         raise ValueError(
    #             f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
    #             f" {attn_output.size()}"
    #         )

    #     attn_output = attn_output.transpose(1, 2).contiguous()

    #     attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    #     if self.config.pretraining_tp > 1:
    #         attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
    #         o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
    #         attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    #     else:
    #         attn_output = self.o_proj(attn_output, Qin=R2, Qout=R1)

    #     if not output_attentions:
    #         attn_weights = None

    #     return attn_output, attn_weights, past_key_value


class RotatedLlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int, layer: LlamaDecoderLayer):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.self_attn = RotatedLlamaAttention(config, layer_idx, layer.self_attn)
        
        self.mlp = RotatedLlamaMLP(config, layer.mlp)
        self.input_layernorm = copy.deepcopy(layer.input_layernorm)
        self.post_attention_layernorm = copy.deepcopy(layer.post_attention_layernorm)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        R1=None, R2=None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            R1=R1, R2=R2,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, R1=R1)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class RotatedLlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, llama):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self._no_split_modules = ["RotatedLlamaDecoderLyaer"]

        self.embed_tokens = RotatedEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            llama.embed_tokens.max_norm,
            llama.embed_tokens.norm_type,
            llama.embed_tokens.scale_grad_by_freq,
            llama.embed_tokens.sparse,
            llama.embed_tokens.weight.data,
            not llama.embed_tokens.weight.requires_grad,
            llama.embed_tokens.weight.data.device,
            llama.embed_tokens.weight.data.dtype)
        
        layers = []
        for layer_idx in range(len(llama.layers)):
            layers.append(RotatedLlamaDecoderLayer(config, layer_idx, llama.layers[layer_idx]))
        self.layers = nn.ModuleList(layers)
        # self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # self.layers = nn.ModuleList(
        #     [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        # )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        #self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.norm = copy.deepcopy(llama.norm)
        self.gradient_checkpointing = True
        # Initialize weights and apply final processing
        #self.post_init()
        self._backward_compatibility_gradient_checkpointing()

    def convert_modules(self):
        config = self.config
        rotated_embed_tokens = RotatedEmbedding(config.vocab_size, config.hidden_size, self.padding_idx)
        rotated_embed_tokens.weight.data.copy_(self.embed_tokens.weight.data)
        del self.embed_tokens
        self.embed_tokens = rotated_embed_tokens
        
        rotated_layers = []
        
        for layer_idx, layer in enumerate(self.layers):
            new_rotated_layer = RotatedLlamaDecoderLayer(config, layer_idx, layer)
            rotated_layers.append(new_rotated_layer)
        
        for layer_idx in range(len(self.layers)):
            del self.layers[0]

        del self.layers
        self.layers = nn.ModuleList(rotated_layers)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        R1=None, R2s=None,
        forward_until=32,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids, Q=R1)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds,
                past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            #print(f'Forward {layer_idx}')
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    R1,
                    R2s[layer_idx] if R2s is not None else None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    R1=R1,
                    R2=R2s[layer_idx] if R2s is not None else None,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            
            if layer_idx == forward_until:
                break

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class RotatedLlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, llama):
        super().__init__(config)
        self.config = config
        self.model = RotatedLlamaModel(config, llama.model)
        #self.model.convert_modules()
        self.vocab_size = config.vocab_size
        self.lm_head = RotatedHead(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=llama.lm_head.weight.data.device,
            dtype=llama.lm_head.weight.data.dtype)
        self.lm_head.weight.data = llama.lm_head.weight.data.detach().clone()
        # Initialize weights and apply final processing
        #self.post_init()
        self._backward_compatibility_gradient_checkpointing()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        R1=None, R2s=None,
        forward_until=32
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            R1=R1, R2s=R2s,
            forward_until=forward_until
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states, Q=R1)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past