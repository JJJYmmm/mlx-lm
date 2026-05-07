# Copyright © 2026 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import create_attention_mask
from .cache import KVCache
from .qwen3_next import Qwen3NextAttention as Attention
from .qwen3_next import Qwen3NextMLP as MLP
from .qwen3_next import Qwen3NextSparseMoeBlock as SparseMoeBlock


class MTPDecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.self_attn = Attention(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

        if args.num_experts > 0:
            self.mlp = SparseMoeBlock(args)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pre_fc_norm_hidden = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_fc_norm_embedding = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.fc = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
        self.layers = [MTPDecoderLayer(args) for _ in range(args.mtp_num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def make_cache(self):
        return [KVCache() for _ in self.layers]

    def __call__(
        self,
        hidden_states,
        next_token_ids,
        embed_tokens,
        lm_head,
        cache=None,
        return_hidden=False,
    ):
        embeddings = embed_tokens(next_token_ids)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        embeddings = self.pre_fc_norm_embedding(embeddings)
        hidden_states = self.fc(mx.concatenate([embeddings, hidden_states], axis=-1))

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(hidden_states, cache[0] if cache else None)
        for layer, c in zip(self.layers, cache):
            hidden_states = layer(hidden_states, mask=mask, cache=c)

        hidden_states = self.norm(hidden_states)
        if self.args.tie_word_embeddings:
            logits = embed_tokens.as_linear(hidden_states)
        else:
            logits = lm_head(hidden_states)
        if return_hidden:
            return logits, hidden_states
        return logits
