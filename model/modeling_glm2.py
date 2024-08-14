"""PyTorch gLM2 model.

Requires flash attention.
Some modules adapted from:
https://github.com/meta-llama/llama/blob/main/llama/model.py
"""

import torch
from einops import rearrange
from typing import Optional, Tuple, Union
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

try:
    from flash_attn.ops.activations import swiglu
    from flash_attn.layers.rotary import apply_rotary_emb_func
    from flash_attn import (
        flash_attn_kvpacked_func,
        flash_attn_varlen_kvpacked_func,
    )
    from flash_attn.bert_padding import pad_input, unpad_input
except ImportError:
    raise ImportError(
        "gLM2 requires flash attention: `pip install flash-attn --no-build-isolation`")

from .configuration_glm2 import gLM2Config


logger = logging.get_logger(__name__)


class RotaryEmbedding(torch.nn.Module):
    """
    Copied from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/layers/rotary.py.
    Changed to only support passing in q or k individually, so that we can use varlen rotary.
    """

    def __init__(
        self,
        dim: int,
        base=10000.0,
        interleaved=False,
        scale_base=None,
        pos_idx_in_fp32=True,
        device=None,
    ):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.interleaved = interleaved
        self.scale_base = scale_base
        scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim)
            / (1.4 * dim)
            if scale_base is not None
            else None
        )
        self.register_buffer("scale", scale, persistent=False)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _compute_inv_freq(self, device=None):
        return 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device,
                             dtype=torch.float32)
                / self.dim
            )
        )

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                # We want fp32 here as well since inv_freq will be multiplied with t, and the output
                # will be large. Having it in bf16 will lose a lot of precision and cause the
                # cos & sin output to change significantly.
                # We want to recompute self.inv_freq if it was not loaded in fp32
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self._compute_inv_freq(device=device)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device,
                                 dtype=self.inv_freq.dtype)
                inv_freq = self.inv_freq
            # Don't do einsum, it converts fp32 to fp16 under AMP
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, inv_freq)
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (
                    torch.arange(
                        seqlen, dtype=self.scale.dtype, device=self.scale.device
                    )
                    - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** rearrange(
                    power, "s -> s 1"
                )
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seqlen_offset: Union[int, torch.Tensor] = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        q: (batch, seqlen, nheads, headdim). If cu_seqlens is not None,
            shape (total_seqlen, nheads, headdim).
        k: (batch, seqlen, nheads, headdim). If cu_seqlens is not None,
            shape (total_seqlen, nheads, headdim).
        seqlen_offset: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
            If it's a tensor of shape (batch_size,), then to update the cos / sin cache, one
            should pass in max_seqlen, which will update the cos / sin cache up to that length.
        Apply rotary embedding *inplace* to qkv and / or kv.
        """
        if cu_seqlens is not None:
            assert max_seqlen is not None
        seqlen = q.shape[1] if max_seqlen is None else max_seqlen
        if max_seqlen is not None:
            self._update_cos_sin_cache(
                max_seqlen, device=q.device, dtype=q.dtype)
        elif isinstance(seqlen_offset, int):
            self._update_cos_sin_cache(
                seqlen + seqlen_offset, device=q.device, dtype=q.dtype
            )
        q = apply_rotary_emb_func(
            q,
            self._cos_cached,
            self._sin_cached,
            interleaved=self.interleaved,
            inplace=True,
            seqlen_offsets=seqlen_offset,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        if self.scale is None:
            k = apply_rotary_emb_func(
                k,
                self._cos_cached,
                self._sin_cached,
                interleaved=self.interleaved,
                inplace=True,
                seqlen_offsets=seqlen_offset,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        else:
            k = apply_rotary_emb_func(
                k,
                self._cos_k_cached,
                self._sin_k_cached,
                interleaved=self.interleaved,
                inplace=True,
                seqlen_offsets=seqlen_offset,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        return q, k


# @torch.jit.script
def rmsnorm_func(hidden_states, weight, variance_epsilon):
    """Apply the root mean square normalization."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return (weight * hidden_states).to(input_dtype)


class RMSNorm(nn.Module):
    """Root mean square normalization."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.register_buffer(
            "variance_epsilon",
            torch.tensor(eps),
            persistent=False,
        )

    def forward(self, hidden_states):
        return rmsnorm_func(hidden_states, self.weight, self.variance_epsilon)


class Attention(nn.Module):
    """Multi-head attention module."""

    def __init__(self, config: gLM2Config):
        super().__init__()
        self.n_heads = config.heads
        self.head_dim = config.dim // config.heads

        self.wqkv = nn.Linear(config.dim, self.n_heads *
                              self.head_dim * 3, bias=False)
        self.wo = nn.Linear(config.heads * self.head_dim,
                            config.dim, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def _forward_varlen(
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seq_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        total_seqlen, h_size = x.shape
        qkv = self.wqkv(x)
        q, k, v = torch.split(qkv, self.n_heads * self.head_dim, dim=-1)

        q = q.view(total_seqlen, self.n_heads, self.head_dim)
        k = k.view(total_seqlen, self.n_heads, self.head_dim)
        v = v.view(total_seqlen, self.n_heads, self.head_dim)

        q, k = self.rotary_emb(
            q, k, cu_seqlens=cu_seqlens, max_seqlen=max_seq_len)

        # (seqlen, 2, n_heads, head_dim)
        kv = torch.stack([k, v], 1)

        # (seqlen, n_heads, head_dim)
        output = flash_attn_varlen_kvpacked_func(
            q,
            kv,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seq_len,
            max_seqlen_k=max_seq_len,
            dropout_p=0.0,
            causal=False,
        )
        output = output.view(total_seqlen, h_size)
        return self.wo(output)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seq_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cu_seqlens is not None:
            assert max_seq_len is not None
            return self._forward_varlen(x, cu_seqlens, max_seq_len)

        bsz, seqlen, h_size = x.shape
        qkv = self.wqkv(x)
        q, k, v = torch.split(qkv, self.n_heads * self.head_dim, dim=-1)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim)

        q, k = self.rotary_emb(q, k)
        # (bs, seqlen, 2, n_heads, head_dim)
        kv = torch.stack([k, v], 2)

        output = flash_attn_kvpacked_func(
            q,
            kv,
            dropout_p=0.0,
            causal=False,
        )
        output = output.view(bsz, seqlen, h_size)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        SwiGLU FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.
        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * \
            ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(swiglu(self.w1(x), self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: gLM2Config):
        super().__init__()
        self.n_heads = config.heads
        self.dim = config.dim
        self.head_dim = config.dim // config.heads
        self.attention = Attention(config)
        self.feed_forward = FeedForward(
            dim=config.dim,
            hidden_dim=4 * config.dim,
            multiple_of=config.swiglu_multiple_of,
            ffn_dim_multiplier=config.ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seq_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r = self.attention(
            self.attention_norm(x), cu_seqlens, max_seq_len
        )
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r
        return out


class TransformerLayers(nn.Module):
    def __init__(self, config: gLM2Config):
        super().__init__()
        self.config = config
        self.layers = torch.nn.ModuleList(
            [TransformerBlock(config=config) for _ in range(config.depth)]
        )

    def forward(
        self,
        x: torch.FloatTensor,
        attention_mask: Optional[torch.BoolTensor] = None,
        return_all_hiddens: bool = False,
    ):
        if x.shape[-1] != self.config.dim:
            raise ValueError(
                f"Input feature dim should be {self.config.dim}, but input has shape {x.shape}"
            )
        batch_size, seq_len = x.shape[:2]
        should_unpad = attention_mask is not None and not attention_mask.all()
        if should_unpad:
            x, indices, cu_seqlens, max_seq_len_in_batch = unpad_input(
                x, attention_mask
            )
        else:
            indices, cu_seqlens, max_seq_len_in_batch = None, None, None
        hiddens = []
        for layer in self.layers:
            x = layer(x, cu_seqlens, max_seq_len_in_batch)
            if return_all_hiddens:
                hiddens.append(x)

        if should_unpad:
            x = pad_input(x, indices, batch_size, seq_len)
            if return_all_hiddens:
                hiddens = [pad_input(h, indices, batch_size, seq_len)
                           for h in hiddens]

        if return_all_hiddens:
            return x, hiddens
        return x


class gLM2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = gLM2Config
    base_model_prefix = "glm2"
    supports_gradient_checkpointing = False

    # https://github.com/huggingface/transformers/blob/7032e0203262ebb2ebf55da8d2e01f873973e835/src/transformers/models/bert/modeling_bert.py#L748
    def _init_weights(module, initializer_range=0.02):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])


class gLM2Model(gLM2PreTrainedModel):
    """gLM2 Model."""

    def __init__(self, config: gLM2Config):
        super().__init__(config)
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.encoder = TransformerLayers(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutput]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        h = self.tok_embeddings(input_ids)
        if output_hidden_states:
            sequence_output, all_hidden_states = self.encoder(
                h, attention_mask, return_all_hiddens=True)
        else:
            sequence_output = self.encoder(h, attention_mask)
            all_hidden_states = None

        if not return_dict:
            return (sequence_output, all_hidden_states)

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=all_hidden_states,

        )


class gLM2ForMaskedLM(gLM2PreTrainedModel):

    def __init__(self, config: gLM2Config):
        super().__init__(config)

        self.glm2 = gLM2Model(config)
        self.lm_head = gLM2LMHead(config)
        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.glm2(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            labels = labels.to(prediction_scores.device)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class gLM2LMHead(nn.Module):
    """gLM2 head for masked language modeling."""

    def __init__(self, config):
        super().__init__()

        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.proj_output = nn.Linear(
            config.dim, config.vocab_size, bias=False)

    def forward(self, features):
        return self.proj_output(self.norm(features))
