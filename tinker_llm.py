"""Thinking Machines' Tinker adapters for llm.py.

    from tinker_llm import TinkerPrefill
    from llm import generate

    model = TinkerPrefill.create()                 # Qwen/Qwen3.6-35B-A3B, thinking off
    answer, root = generate("What is 12 * 15?", model)

Needs `tinker` and `tinker-cookbook` (`uv add tinker tinker-cookbook`) and
TINKER_API_KEY in the environment (or `tinker auth login`).

`TinkerPrefill` runs a chat model with assistant prefill, for
`generate` and `continue_from`, and reads the log-prob of each <follow_proposal>
choice.

`TinkerLLM` is a plain chat call, `llm(system, prompt) -> str` (same Qwen model
with thinking off by default), used by the CoT baseline in eval_aime.py.

Both keep the raw tokens and per-token log-probs of their last call in `last`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from llm import FOLLOW_TAG

if TYPE_CHECKING:
    from llm import Reply

QWEN_HYBRID = "Qwen/Qwen3.6-35B-A3B"
QWEN_NO_THINKING = "qwen3_5_disable_thinking"  # cookbook renderer: empty, closed <think></think>


def _resolve(x: Any) -> Any:
    """Tinker calls return futures; accept either a future or a plain value."""
    return x.result() if hasattr(x, "result") and callable(x.result) else x


@dataclass
class Sample:
    """What one call produced, kept for inspection and log-prob work."""

    text: str  # reply text (what llm.py decodes)
    tokens: list[int]
    logprobs: Optional[list[float]]
    clean: bool  # False if the reply did not end with the expected stop token


@dataclass
class TinkerLLM:
    sampling_client: Any
    renderer: Any
    sampling_params: Any
    last: Optional[Sample] = field(default=None, repr=False)

    @classmethod
    def create(
        cls,
        model: str = QWEN_HYBRID,
        renderer_name: str = QWEN_NO_THINKING,
        max_tokens: int = 16384,
        temperature: float = 1.0,
        model_path: Optional[str] = None,
    ) -> "TinkerLLM":
        """Connect to Tinker. Pass `model_path` to sample from your own saved
        weights (`model` still picks the tokenizer and renderer)."""
        import tinker
        from tinker import types
        from tinker_cookbook import renderers, tokenizer_utils

        service = tinker.ServiceClient()
        if model_path is not None:
            client = _resolve(service.create_sampling_client(model_path=model_path))
        else:
            client = _resolve(service.create_sampling_client(base_model=model))
        renderer = renderers.get_renderer(renderer_name, tokenizer_utils.get_tokenizer(model))
        params = types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=renderer.get_stop_sequences(),
        )
        return cls(client, renderer, params)

    def __call__(self, system: str, prompt: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        model_input = self.renderer.build_generation_prompt(messages)
        response = _resolve(self.sampling_client.sample(
            prompt=model_input, num_samples=1, sampling_params=self.sampling_params,
        ))
        seq = response.sequences[0]
        message, termination = self.renderer.parse_response(seq.tokens)
        text = _text_content(message.get("content", ""))
        self.last = Sample(
            text=text,
            tokens=list(seq.tokens),
            logprobs=list(seq.logprobs) if getattr(seq, "logprobs", None) is not None else None,
            clean=bool(getattr(termination, "is_clean", True)),
        )
        return text


def _text_content(content: Any) -> str:
    """Text of a cookbook Message content: a string, or a list of parts (only
    the "text" parts are kept)."""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if p.get("type") == "text")


PROMPT_TAIL = 32  # prompt tokens to get top-k log-probs for (covers the open <node><follow_proposal idx="k"/>)


def _is_idx(s: str) -> bool:
    """A bare ASCII number (top-k lists can hold e.g. subscript digits)."""
    return bool(re.fullmatch(r"[0-9]+", s))


def _at_follow_idx(text: str) -> bool:
    """True right before a proposal idx digit, i.e. after '<follow_proposal idx="'."""
    return bool(re.search(r"""<follow_proposal\s+idx\s*=\s*["']$""", text))


@dataclass
class TinkerPrefill:
    """Chat model with assistant prefill, the `prefill(system, user, prefix) -> str`
    for llm.generate and llm.continue_from.

    The cookbook renderer builds the chat prompt up to the opened assistant turn
    and appends `prefix` as plain text (`build_generation_prompt(..., prefill=)`).
    With `qwen3_5_disable_thinking` that turn starts with an empty, closed
    <think></think> block, so the model continues the trace directly, without
    reasoning. Each call returns an llm.Reply: the continuation text plus, for
    every <follow_proposal> in it and for the one the prefill ends with, the log-prob of
    each idx digit (from top-k log-probs, so no extra calls). `last` keeps the
    sampled tokens and their log-probs.

        model = TinkerPrefill.create()   # Qwen/Qwen3.6-35B-A3B, thinking off
        answer, root = generate("What is 12 * 15?", model)
    """

    sampling_client: Any
    renderer: Any
    tokenizer: Any
    sampling_params: Any
    topk: int = 20  # top-k log-probs per position, used to read <follow_proposal> choices
    last: Optional[Sample] = field(default=None, repr=False)

    @classmethod
    def create(
        cls,
        model: str = QWEN_HYBRID,
        renderer_name: str = QWEN_NO_THINKING,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        model_path: Optional[str] = None,
    ) -> "TinkerPrefill":
        import tinker
        from tinker import types
        from tinker_cookbook import renderers, tokenizer_utils

        service = tinker.ServiceClient()
        if model_path is not None:
            client = _resolve(service.create_sampling_client(model_path=model_path))
        else:
            client = _resolve(service.create_sampling_client(base_model=model))
        tokenizer = tokenizer_utils.get_tokenizer(model)
        renderer = renderers.get_renderer(renderer_name, tokenizer)
        params = types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=renderer.get_stop_sequences(),
        )
        return cls(client, renderer, tokenizer, params)

    def __call__(self, system: str, user: str, prefill: str) -> "Reply":
        from llm import Reply

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        model_input = self.renderer.build_generation_prompt(messages, prefill=prefill)
        response = _resolve(self.sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=self.sampling_params,
            topk_sample_logprobs=self.topk,
            include_prompt_logprobs=True,
            topk_prompt_logprobs=self.topk,
            prompt_logprobs_last_n=PROMPT_TAIL,
        ))
        seq = response.sequences[0]
        tokens = list(seq.tokens)
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        stops = getattr(self.sampling_params, "stop", None) or []
        self.last = Sample(
            text=text,
            tokens=tokens,
            logprobs=list(seq.logprobs) if getattr(seq, "logprobs", None) is not None else None,
            clean=not stops or (bool(tokens) and tokens[-1] in stops),
        )
        return Reply(
            text=text,
            choice_logprobs=self._sampled_choices(tokens, getattr(seq, "topk_logprobs", None), text),
            prefill_choice=self._prefill_choice(
                model_input.to_ints(), getattr(response, "topk_prompt_logprobs", None)
            ),
        )

    # --- reading <follow_proposal> choices from top-k log-probs -------------
    def _digit_dist(self, entries: Any) -> Optional[dict[int, float]]:
        """{idx: logprob} from a position's top-k (token_id, logprob) list,
        keeping only tokens that decode to a bare number."""
        if not entries:
            return None
        dist: dict[int, float] = {}
        for token_id, lp in entries:
            s = self.tokenizer.decode([token_id])
            if _is_idx(s):
                k = int(s)
                dist[k] = max(lp, dist.get(k, float("-inf")))
        return dist

    def _sampled_choices(self, tokens: list[int], topk: Any, text: str) -> list:
        """One entry per <follow_proposal> tag in the generated text: the distribution at
        the token right after it (None if that token isn't a number or top-k is
        missing). If tags can't be lined up with tokens, all entries are None."""
        pieces = [self.tokenizer.decode([t]) for t in tokens]
        n_tags = len(FOLLOW_TAG.findall("".join(pieces)))
        choices: list = []
        tail = ""
        for i, piece in enumerate(pieces):
            if _at_follow_idx(tail):
                entries = topk[i] if topk is not None and i < len(topk) else None
                choices.append(self._digit_dist(entries) if _is_idx(piece) else None)
            tail = (tail + piece)[-32:]
        return choices if len(choices) == n_tags else [None] * n_tags

    def _prefill_choice(self, prompt_tokens: list[int], topk: Any) -> Optional[dict[int, float]]:
        """Distribution at the idx digit of the last <follow_proposal> in the prompt (the
        choice the pre-started reply ends with), from top-k prompt log-probs."""
        if not topk:
            return None
        n = len(prompt_tokens)
        offset = 0 if len(topk) == n else n - len(topk)
        tail_start = max(0, n - PROMPT_TAIL)
        pieces = [self.tokenizer.decode([t]) for t in prompt_tokens[tail_start:]]
        found = None
        tail = ""
        for j, piece in enumerate(pieces):
            if _at_follow_idx(tail) and _is_idx(piece):
                found = tail_start + j
            tail = (tail + piece)[-32:]
        if found is None or not 0 <= found - offset < len(topk):
            return None
        return self._digit_dist(topk[found - offset])
