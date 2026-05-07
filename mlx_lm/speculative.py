# Copyright © 2026 Apple Inc.

import math
from dataclasses import dataclass
from typing import List, Optional

import mlx.core as mx

from .models import cache
from .sample_utils import (
    apply_min_p,
    apply_top_k,
    apply_top_p,
    categorical_sampling,
)


@dataclass
class SampleResult:
    token: mx.array
    logprobs: mx.array
    accept_logprobs: mx.array
    xtc_draw: Optional[mx.array] = None


@dataclass
class DraftBatch:
    tokens: mx.array
    samples: List[SampleResult]


@dataclass(frozen=True)
class SpeculativeCacheFamily:
    name: str


TARGET_CACHE_FAMILY = SpeculativeCacheFamily("target")
EXTERNAL_DRAFT_CACHE_FAMILY = SpeculativeCacheFamily("external_draft")
NATIVE_MTP_CACHE_FAMILY = SpeculativeCacheFamily("native_mtp")


def speculative_cache_family(
    *, draft_model=None, native_mtp: bool = False, method: Optional[str] = None
):
    if method is not None:
        return SpeculativeCacheFamily(method)
    if draft_model is not None:
        return EXTERNAL_DRAFT_CACHE_FAMILY
    if native_mtp:
        return NATIVE_MTP_CACHE_FAMILY
    return TARGET_CACHE_FAMILY


def speculative_prompt_cache_key(model_key, family):
    return (model_key, family)


class SpeculativeStats:
    def __init__(self, num_draft_tokens: int):
        self.num_draft_tokens = num_draft_tokens
        self.draft_tokens = 0
        self.accepted_draft_tokens = 0
        self.draft_position_counts = [0] * num_draft_tokens
        self.accepted_draft_position_counts = [0] * num_draft_tokens
        self.draft_position_accept_rates = [0.0] * num_draft_tokens

    @property
    def draft_accept_rate(self):
        return (
            self.accepted_draft_tokens / self.draft_tokens
            if self.draft_tokens
            else 0.0
        )

    def update(self, n_draft: int, accepted: int):
        self.draft_tokens += n_draft
        self.accepted_draft_tokens += accepted
        for i in range(n_draft):
            self.draft_position_counts[i] += 1
            if i < accepted:
                self.accepted_draft_position_counts[i] += 1
            self.draft_position_accept_rates[i] = (
                self.accepted_draft_position_counts[i]
                / self.draft_position_counts[i]
            )

    def to_dict(self):
        return {
            "draft_tokens": self.draft_tokens,
            "accepted_draft_tokens": self.accepted_draft_tokens,
            "draft_accept_rate": self.draft_accept_rate,
            "draft_position_counts": self.draft_position_counts,
            "accepted_draft_position_counts": self.accepted_draft_position_counts,
            "draft_position_accept_rates": self.draft_position_accept_rates,
        }


class SpeculativeSampler:
    def __init__(
        self,
        *,
        temp: float = 0.0,
        top_p: float = 0.0,
        min_p: float = 0.0,
        min_tokens_to_keep: int = 1,
        top_k: int = 0,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        xtc_special_tokens: Optional[List[int]] = None,
    ):
        self.temp = temp
        self.top_p = top_p
        self.min_p = min_p
        self.min_tokens_to_keep = min_tokens_to_keep
        self.top_k = top_k
        self.xtc_probability = xtc_probability
        self.xtc_threshold = xtc_threshold
        self.xtc_special_tokens = []
        for token in xtc_special_tokens or []:
            if isinstance(token, list):
                self.xtc_special_tokens.extend(token)
            else:
                self.xtc_special_tokens.append(token)

    @property
    def is_greedy(self):
        return self.temp == 0

    def _apply_xtc(self, logprobs, xtc_draw):
        if self.xtc_probability <= 0.0:
            return logprobs, None
        if xtc_draw is None:
            xtc_draw = mx.random.uniform(0, 1)
        probs = mx.softmax(logprobs, -1)
        mask = probs > mx.where(probs > self.xtc_threshold, probs, mx.inf).min()
        if self.xtc_special_tokens:
            mask[..., self.xtc_special_tokens] = False
        filtered = mx.where(
            xtc_draw > self.xtc_probability,
            logprobs,
            mx.where(mask, -mx.inf, logprobs),
        )
        return filtered, xtc_draw

    def _filter(self, logprobs, xtc_draw=None):
        filtered = logprobs
        if self.top_p > 0 and self.top_p < 1.0:
            filtered = apply_top_p(filtered, self.top_p)
        if self.min_p != 0.0:
            filtered = apply_min_p(filtered, self.min_p, self.min_tokens_to_keep)
        filtered, xtc_draw = self._apply_xtc(filtered, xtc_draw)
        if self.top_k > 0:
            filtered = apply_top_k(filtered, self.top_k)
        return filtered, xtc_draw

    def sample(self, logits, xtc_draw=None, *, logprobs=True):
        if self.temp == 0 and not logprobs:
            token = mx.argmax(logits, axis=-1)
            return SampleResult(token.astype(mx.uint32), None, None, xtc_draw)

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        logprobs = logprobs.squeeze(0)
        filtered, xtc_draw = self._filter(logprobs, xtc_draw)
        if self.temp == 0:
            token = mx.argmax(logprobs, axis=-1)
            accept_logprobs = logprobs
        else:
            token = categorical_sampling(filtered, self.temp)
            scaled = filtered / self.temp
            accept_logprobs = scaled - mx.logsumexp(
                scaled, axis=-1, keepdims=True
            )
        return SampleResult(
            token.astype(mx.uint32),
            logprobs,
            accept_logprobs,
            xtc_draw,
        )

    def greedy_batch(self, logits):
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        logprobs = logprobs.squeeze(0)
        token = mx.argmax(logprobs, axis=-1)
        return SampleResult(token.astype(mx.uint32), logprobs, logprobs)


class HybridMTPPolicy:
    n_confirmed = 1

    def __init__(self, model_cache, mtp_cache):
        self.model_cache = model_cache
        self.mtp_cache = mtp_cache

    def clear_rollback(self):
        for c in self.model_cache:
            if hasattr(c, "rollback_state"):
                c.rollback_state = None

    def rollback(self, draft_tokens, n_draft, accepted, target_forward):
        for c in self.model_cache:
            rollback = getattr(c, "rollback_state", None)
            if rollback is not None:
                mx.eval(rollback)
                c[0], c[1] = rollback
                c.rollback_state = None
            else:
                c.trim(n_draft)

        hidden = None
        for token in draft_tokens[:accepted]:
            logits, hidden = target_forward(token.reshape(1))
            mx.eval(logits, hidden, [c.state for c in self.model_cache])

        cache.trim_prompt_cache(self.mtp_cache, max(n_draft - accepted - 1, 0))
        return hidden


class NativeMTPProposer:
    def __init__(self, model, mtp_cache, sampler, stream):
        self.model = model
        self.mtp_cache = mtp_cache
        self.sampler = sampler
        self.stream = stream

    def prefill(self, input_tokens, next_tokens, target_hidden):
        self.model.mtp_forward(target_hidden, next_tokens[None], self.mtp_cache)

    def draft(self, hidden_at_position, next_token, n_tokens):
        draft_tokens, samples = [], []
        for _ in range(n_tokens):
            with mx.stream(self.stream):
                logits, hidden_at_position = self.model.mtp_forward(
                    hidden_at_position,
                    next_token.reshape(1, 1),
                    self.mtp_cache,
                    return_hidden=True,
                )
                sample = self.sampler.sample(
                    logits[:, -1, :],
                    logprobs=not self.sampler.is_greedy,
                )
            if self.sampler.is_greedy:
                mx.async_eval(sample.token)
            else:
                mx.async_eval(sample.token, sample.logprobs, sample.accept_logprobs)
            draft_tokens.append(sample.token.reshape(-1))
            samples.append(sample)
            next_token = sample.token.reshape(-1)
        return DraftBatch(mx.concatenate(draft_tokens), samples)

    def accept_all(self, hidden, draft_tokens, bonus_token):
        self.model.mtp_forward(
            hidden[:, len(draft_tokens) - 1 : len(draft_tokens), :],
            draft_tokens[-1].reshape(1, 1),
            self.mtp_cache,
        )


def residual_sample(target_accept_logprobs, draft_accept_logprobs):
    p_target = mx.exp(target_accept_logprobs)
    p_draft = mx.exp(draft_accept_logprobs)
    residual = mx.maximum(p_target - p_draft, 0.0)
    z = residual.sum(keepdims=True)
    dist = mx.where(z > 0, residual / z, p_target)
    return mx.random.categorical(mx.log(dist).reshape(1, -1)).astype(mx.uint32)


def speculative_generate_loop(
    prompt,
    target_forward,
    proposer,
    cache_policy,
    sampler,
    stats,
    *,
    max_tokens,
    prefill_step_size,
    prompt_cache,
    prompt_progress_callback,
):
    total_prompt_tokens = len(prompt)
    processed = 0
    prompt_progress_callback(processed, total_prompt_tokens)
    while total_prompt_tokens - processed > 1:
        n_to_process = min(prefill_step_size, total_prompt_tokens - processed - 1)
        input_tokens = prompt[processed : processed + n_to_process]
        next_tokens = prompt[processed + 1 : processed + n_to_process + 1]
        logits, hidden = target_forward(input_tokens)
        proposer.prefill(input_tokens, next_tokens, hidden)
        mx.eval([c.state for c in prompt_cache])
        processed += n_to_process
        prompt_progress_callback(processed, total_prompt_tokens)
        mx.clear_cache()

    with mx.stream(target_forward.stream):
        y = prompt[-1:].astype(mx.uint32)
        logits, hidden = target_forward(y)
        sample = sampler.sample(logits[:, -1, :])
        next_hidden = hidden[:, -1:, :]
        draft = proposer.draft(
            next_hidden, sample.token.reshape(-1), stats.num_draft_tokens
        )
    mx.eval(sample.token, draft.tokens)
    prompt_progress_callback(len(prompt), len(prompt))

    ntoks = 0
    if ntoks < max_tokens:
        ntoks += 1
        yield sample.token.item(), sample.logprobs, False
    next_main = sample.token.reshape(-1)

    while ntoks < max_tokens:
        n_draft = min(stats.num_draft_tokens, max_tokens - ntoks)
        verify_input = mx.concatenate([next_main, draft.tokens[:n_draft]])
        logits, hidden = target_forward(
            verify_input, n_confirmed=cache_policy.n_confirmed
        )
        logits = logits[:, : n_draft + 1, :]

        if sampler.is_greedy:
            target_sample_batch = sampler.greedy_batch(logits)
            target_samples = [
                SampleResult(
                    target_sample_batch.token[i],
                    target_sample_batch.logprobs[i],
                    target_sample_batch.accept_logprobs[i],
                )
                for i in range(n_draft + 1)
            ]
        else:
            target_samples = []
            for i in range(n_draft + 1):
                xtc_draw = draft.samples[i].xtc_draw if i < n_draft else None
                target_samples.append(sampler.sample(logits[:, i, :], xtc_draw))

        mx.eval(
            [s.token for s in target_samples],
            draft.tokens,
        )

        accepted = 0
        while accepted < n_draft:
            draft_token = draft.tokens[accepted].item()
            if sampler.is_greedy:
                accept = target_samples[accepted].token.item() == draft_token
            else:
                u = mx.random.uniform()
                log_accept = (
                    target_samples[accepted].accept_logprobs[draft_token]
                    - draft.samples[accepted].accept_logprobs[draft_token]
                ).item()
                accept = log_accept >= 0 or u.item() < math.exp(log_accept)
            if not accept:
                break
            accepted += 1

        stats.update(n_draft, accepted)

        for i in range(accepted):
            ntoks += 1
            yield draft.tokens[i].item(), target_samples[i].logprobs, True
            if ntoks == max_tokens:
                break
        if ntoks == max_tokens:
            break

        if accepted == n_draft:
            cache_policy.clear_rollback()
            bonus = target_samples[n_draft]
            proposer.accept_all(hidden, draft.tokens[:n_draft], bonus.token)
            next_main = bonus.token.reshape(-1)
            next_hidden = (
                hidden[:, n_draft : n_draft + 1, :] if hidden is not None else None
            )
            next_logprobs = bonus.logprobs
        else:
            rollback_hidden = cache_policy.rollback(
                draft.tokens, n_draft, accepted, target_forward
            )
            if sampler.is_greedy:
                next_main = target_samples[accepted].token.reshape(-1)
            else:
                next_main = residual_sample(
                    target_samples[accepted].accept_logprobs,
                    draft.samples[accepted].accept_logprobs,
                )
            next_hidden = rollback_hidden
            if next_hidden is None:
                next_hidden = hidden[:, :1, :] if hidden is not None else None
            next_logprobs = target_samples[accepted].logprobs

        draft = proposer.draft(next_hidden, next_main, stats.num_draft_tokens)
        ntoks += 1
        yield next_main.item(), next_logprobs, False
        mx.clear_cache()
