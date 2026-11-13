"""gpt-oss on Thinking Machines' Tinker as the `llm(system, prompt) -> str` for llm.py.

    from tinker_llm import TinkerLLM
    from llm import search
    from tree import Node

    llm = TinkerLLM.create()                       # openai/gpt-oss-120b, medium reasoning
    result = search(Node.new_root("What is 12 * 15?"), llm, max_calls=10)

Needs `tinker` and `tinker-cookbook` (`uv add tinker tinker-cookbook`) and
TINKER_API_KEY in the environment (or `tinker auth login`).

The prompt is rendered with the cookbook's gpt-oss renderer (harmony format):
the system prompt becomes the developer instructions, the model reasons in its
analysis channel, and only the final channel is returned as text. The raw
tokens and per-token log-probs of the last call are kept in `last` for later
use (e.g. the log-prob of each <follow_proposal> choice).

`TinkerLLM.inkling()` sets up thinkingmachines/Inkling-Small with low thinking
effort, for `search(root, llm, mode="continue")`.

`TinkerPrefill` runs a chat model with assistant prefill (default
Qwen/Qwen3.6-35B-A3B with thinking off), for `search(root, prefill=...)`.

`TinkerCompleter` is the completion-mode counterpart: raw text in, raw text
out, for `search(root, complete=...)` with a base model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from llm import FOLLOW_TAG

DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_RENDERER = "gpt_oss_medium_reasoning"  # also: gpt_oss_low_reasoning, gpt_oss_high_reasoning


def _resolve(x: Any) -> Any:
    """Tinker calls return futures; accept either a future or a plain value."""
    return x.result() if hasattr(x, "result") and callable(x.result) else x


@dataclass
class Sample:
    """What one call produced, kept for inspection and log-prob work."""

    text: str  # final-channel text (what llm.py decodes)
    thinking: str  # analysis-channel text
    tokens: list[int]
    logprobs: Optional[list[float]]
    clean: bool  # False if the reply did not end with the expected stop token


INKLING_SMALL = "thinkingmachines/Inkling-Small"


@dataclass
class TinkerLLM:
    sampling_client: Any
    renderer: Any
    sampling_params: Any
    prompt_kwargs: dict = field(default_factory=dict)  # extra build_generation_prompt args, e.g. effort
    last: Optional[Sample] = field(default=None, repr=False)

    @classmethod
    def create(
        cls,
        model: str = DEFAULT_MODEL,
        renderer_name: Optional[str] = DEFAULT_RENDERER,
        max_tokens: int = 16384,
        temperature: float = 1.0,
        model_path: Optional[str] = None,
        prompt_kwargs: Optional[dict] = None,
    ) -> "TinkerLLM":
        """Connect to Tinker. `renderer_name=None` uses the cookbook's recommended
        renderer for `model`. Pass `model_path` to sample from your own saved
        weights (`model` still picks the tokenizer and renderer)."""
        import tinker
        from tinker import types
        from tinker_cookbook import model_info, renderers, tokenizer_utils

        service = tinker.ServiceClient()
        if model_path is not None:
            client = _resolve(service.create_sampling_client(model_path=model_path))
        else:
            client = _resolve(service.create_sampling_client(base_model=model))
        if renderer_name is None:
            renderer_name = model_info.get_recommended_renderer_name(model)
        renderer = renderers.get_renderer(renderer_name, tokenizer_utils.get_tokenizer(model))
        params = types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=renderer.get_stop_sequences(),
        )
        return cls(client, renderer, params, dict(prompt_kwargs or {}))

    @classmethod
    def inkling(
        cls,
        model: str = INKLING_SMALL,
        effort: float = 0.0,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        model_path: Optional[str] = None,
    ) -> "TinkerLLM":
        """Inkling with its recommended renderer (tml_v0). `effort` is in [0, 1);
        0.0 conditions the model toward no reasoning but does not forbid it.
        Inkling can't continue a partially written reply, so use it with
        search(..., mode="continue") for the cut-trace workflow."""
        return cls.create(
            model=model,
            renderer_name=None,
            max_tokens=max_tokens,
            temperature=temperature,
            model_path=model_path,
            prompt_kwargs={"effort": effort},
        )

    def __call__(self, system: str, prompt: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        model_input = self.renderer.build_generation_prompt(messages, **self.prompt_kwargs)
        response = _resolve(self.sampling_client.sample(
            prompt=model_input, num_samples=1, sampling_params=self.sampling_params,
        ))
        seq = response.sequences[0]
        message, termination = self.renderer.parse_response(seq.tokens)
        text, thinking = _split_content(message.get("content", ""))
        self.last = Sample(
            text=text,
            thinking=thinking,
            tokens=list(seq.tokens),
            logprobs=list(seq.logprobs) if getattr(seq, "logprobs", None) is not None else None,
            clean=bool(getattr(termination, "is_clean", True)),
        )
        return text


def _split_content(content: Any) -> tuple[str, str]:
    """(final text, thinking) from a cookbook Message content: a string, or a
    list of parts typed "text" / "thinking"."""
    if isinstance(content, str):
        return content, ""
    text = "".join(p.get("text", "") for p in content if p.get("type") == "text")
    thinking = "".join(p.get("thinking", "") for p in content if p.get("type") == "thinking")
    return text, thinking


DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-35B-A3B-Base"


@dataclass
class TinkerCompleter:
    """Raw text completion on Tinker, the `complete(prefix) -> str` for
    llm.search(..., complete=...). No chat template: the prefix is tokenized as
    is and the model continues it, so a base model fits best.

        completer = TinkerCompleter.create()   # Qwen/Qwen3.5-35B-A3B-Base
        result = search(Node.new_root("What is 12 * 15?"), complete=completer)
    """

    sampling_client: Any
    tokenizer: Any
    sampling_params: Any
    last: Optional[Sample] = field(default=None, repr=False)

    @classmethod
    def create(
        cls,
        model: str = DEFAULT_BASE_MODEL,
        max_tokens: int = 2048,
        temperature: float = 1.0,
        stop: Optional[list[str]] = None,
        model_path: Optional[str] = None,
    ) -> "TinkerCompleter":
        import tinker
        from tinker import types

        service = tinker.ServiceClient()
        if model_path is not None:
            client = _resolve(service.create_sampling_client(model_path=model_path))
        else:
            client = _resolve(service.create_sampling_client(base_model=model))
        params = types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop if stop is not None else ["</trace>"],
        )
        return cls(client, client.get_tokenizer(), params)

    def __call__(self, prefix: str) -> str:
        from tinker import types

        model_input = types.ModelInput.from_ints(self.tokenizer.encode(prefix))
        return self._sample(model_input)

    def _sample(self, model_input: Any) -> str:
        response = _resolve(self.sampling_client.sample(
            prompt=model_input, num_samples=1, sampling_params=self.sampling_params,
        ))
        seq = response.sequences[0]
        text = self.tokenizer.decode(seq.tokens)
        self.last = Sample(
            text=text,
            thinking="",
            tokens=list(seq.tokens),
            logprobs=list(seq.logprobs) if getattr(seq, "logprobs", None) is not None else None,
            clean=True,
        )
        return text


QWEN_HYBRID = "Qwen/Qwen3.6-35B-A3B"
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
    for llm.search(..., prefill=...).

    The cookbook renderer builds the chat prompt up to the opened assistant turn
    and appends `prefix` as plain text (`build_generation_prompt(..., prefill=)`).
    With `qwen3_5_disable_thinking` that turn starts with an empty, closed
    <think></think> block, so the model continues the trace directly, without
    reasoning. Each call returns an llm.Reply: the continuation text plus, for
    every <follow_proposal> in it and for the one the prefill ends with, the log-prob of
    each idx digit (from top-k log-probs, so no extra calls). `last` keeps the
    sampled tokens and their log-probs.

        prefill = TinkerPrefill.create()   # Qwen/Qwen3.6-35B-A3B, thinking off
        result = search(Node.new_root("What is 12 * 15?"), prefill=prefill)
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
        renderer_name: str = "qwen3_5_disable_thinking",
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
            thinking="",
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
