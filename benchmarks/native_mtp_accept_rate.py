# Copyright © 2026 Apple Inc.

import argparse
import time
from dataclasses import dataclass

import mlx.core as mx

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


QUICKSORT_PROMPT = """Implement quicksort in Python.

Requirements:
- write a function named quicksort(xs)
- return a new sorted list
- handle duplicates and negative numbers
- include a short example
"""

MIXED_PROMPTS = [
    QUICKSORT_PROMPT,
    "Write a short explanation of why binary search runs in logarithmic time.",
    "Draft a concise email asking a teammate to review a pull request.",
    "Translate this sentence to Chinese: The benchmark should be reproducible.",
]

GSM8K_FALLBACK_PROMPTS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "Weng earns $12 an hour for babysitting. Yesterday, she babysat for 50 minutes. How much did she earn?",
    "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents gave her $15 for that purpose, and her grandparents gave her twice as much as her parents. How much more money does Betty need?",
    "Julie is reading a 120-page book. Yesterday she read 12 pages and today she read twice as many pages as yesterday. If she wants to finish the book tomorrow, how many pages must she read tomorrow?",
    "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write in a year?",
]


@dataclass
class RunStats:
    seconds: float
    tokens: int
    accepted: int = 0
    draft: int = 0
    position_accepted: list[int] | None = None
    position_draft: list[int] | None = None
    greedy_match: bool = True

    @property
    def tok_s(self):
        return self.tokens / self.seconds

    @property
    def accept_rate(self):
        return self.accepted / self.draft if self.draft else 0.0


def _load_gsm8k_prompts(limit, offset):
    try:
        from datasets import load_dataset

        ds = load_dataset("gsm8k", "main", split=f"test[{offset}:{offset + limit}]")
        return [row["question"] for row in ds]
    except Exception as e:
        print(f"warning: failed to load GSM8K via datasets ({e}); using fallback")
        return GSM8K_FALLBACK_PROMPTS[offset : offset + limit]


def _load_prompts(args):
    if args.prompt is not None:
        return [args.prompt]
    if args.prompt_file is not None:
        with open(args.prompt_file) as f:
            return [p.strip() for p in f.read().split("\n\n") if p.strip()][
                args.offset : args.offset + args.limit
            ]
    if args.prompt_set == "gsm8k":
        return _load_gsm8k_prompts(args.limit, args.offset)
    if args.prompt_set == "mixed":
        return MIXED_PROMPTS[args.offset : args.offset + args.limit]
    return [QUICKSORT_PROMPT]


def _run(
    model,
    tokenizer,
    prompt,
    max_tokens,
    sampler,
    native_mtp=False,
    draft=1,
    temp=0.0,
    seed=None,
):
    if seed is not None:
        mx.random.seed(seed)
    start = time.perf_counter()
    responses = list(
        stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            native_mtp=native_mtp,
            num_draft_tokens=draft,
            temp=temp,
        )
    )
    dt = time.perf_counter() - start
    tokens = [r.token for r in responses]
    if native_mtp:
        stats = responses[-1]
        return RunStats(
            seconds=dt,
            tokens=len(responses),
            accepted=stats.accepted_draft_tokens,
            draft=stats.draft_tokens,
            position_accepted=stats.accepted_draft_position_counts,
            position_draft=stats.draft_position_counts,
        ), tokens
    return RunStats(seconds=dt, tokens=len(responses)), tokens


def main():
    parser = argparse.ArgumentParser(description="Native MTP accept-rate probe")
    parser.add_argument(
        "--model",
        default="/Users/jjjymmm/Code/checkpoints/Qwen3.5-0.8B",
        help="Path to a Qwen3.5/Qwen3.6 checkpoint with native MTP weights.",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--draft", type=int, nargs="+", default=[1, 2, 3, 5])
    parser.add_argument(
        "--prompt-set",
        choices=["quicksort", "gsm8k", "mixed"],
        default="quicksort",
        help="Prompt suite to use when --prompt/--prompt-file is not provided.",
    )
    parser.add_argument("--prompt", default=None)
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Text file with prompts separated by blank lines.",
    )
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load(args.model, lazy=True)
    sampler = make_sampler(temp=args.temp)
    prompts = _load_prompts(args)
    print(
        f"model={args.model} prompt_set={args.prompt_set} "
        f"prompts={len(prompts)} max_tokens={args.max_tokens} "
        f"temp={args.temp} seed={args.seed}"
    )

    baseline_runs = []
    mtp_runs = {n: [] for n in args.draft}

    for i, prompt in enumerate(prompts):
        base, base_tokens = _run(
            model,
            tokenizer,
            prompt,
            args.max_tokens,
            sampler,
            native_mtp=False,
            temp=args.temp,
            seed=args.seed + i,
        )
        baseline_runs.append(base)
        if not args.quiet:
            print(f"prompt={i} baseline tok/s={base.tok_s:.2f}")

        for n in args.draft:
            mtp, mtp_tokens = _run(
                model,
                tokenizer,
                prompt,
                args.max_tokens,
                sampler,
                native_mtp=True,
                draft=n,
                temp=args.temp,
                seed=args.seed + i,
            )
            mtp.greedy_match = mtp_tokens == base_tokens
            mtp_runs[n].append(mtp)
            if not args.quiet:
                pos_rates = " ".join(
                    f"p{j + 1}={100 * a / d:.1f}%"
                    for j, (a, d) in enumerate(
                        zip(mtp.position_accepted, mtp.position_draft)
                    )
                    if d
                )
                print(
                    f"prompt={i} n={n} tok/s={mtp.tok_s:.2f} "
                    f"speedup={mtp.tok_s / base.tok_s:.2f}x "
                    f"accepted={mtp.accepted}/{mtp.draft} "
                    f"accept={100 * mtp.accept_rate:.1f}% "
                    f"greedy_match={mtp.greedy_match} {pos_rates}"
                )

    base_tokens = sum(r.tokens for r in baseline_runs)
    base_seconds = sum(r.seconds for r in baseline_runs)
    baseline_tps = base_tokens / base_seconds
    print(
        f"summary baseline prompts={len(prompts)} tokens={base_tokens} "
        f"tok/s={baseline_tps:.2f} seconds={base_seconds:.2f}"
    )

    for n, runs in mtp_runs.items():
        tokens = sum(r.tokens for r in runs)
        seconds = sum(r.seconds for r in runs)
        accepted = sum(r.accepted for r in runs)
        draft = sum(r.draft for r in runs)
        pos_accepted = [0] * n
        pos_draft = [0] * n
        for r in runs:
            for i, value in enumerate(r.position_accepted):
                pos_accepted[i] += value
            for i, value in enumerate(r.position_draft):
                pos_draft[i] += value
        pos_rates = " ".join(
            f"p{i + 1}={100 * a / d:.1f}%"
            for i, (a, d) in enumerate(zip(pos_accepted, pos_draft))
            if d
        )
        print(
            f"summary n={n} tokens={tokens} tok/s={tokens / seconds:.2f} "
            f"speedup={tokens / seconds / baseline_tps:.2f}x "
            f"accepted={accepted}/{draft} "
            f"accept={100 * accepted / draft:.1f}% "
            f"greedy_match={sum(r.greedy_match for r in runs)}/{len(runs)} "
            f"{pos_rates}"
        )


if __name__ == "__main__":
    main()
