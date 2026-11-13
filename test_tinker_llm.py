"""Tests for the Tinker adapter with fake client and renderer (no network, no tinker install)."""

import unittest
from types import SimpleNamespace

from llm import search
from tinker_llm import TinkerCompleter, TinkerLLM, TinkerPrefill, _split_content
from tree import Node

REPLY = "<node><thought>12*15 = 180</thought></node>"


class FakeFuture:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


class FakeRenderer:
    """Stands in for the cookbook gpt-oss renderer: tokens are just a list of ints
    and parse_response returns pre-set content."""

    def __init__(self, content, clean=True):
        self.content = content
        self.clean = clean
        self.messages = None

    def build_generation_prompt(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return "MODEL_INPUT"

    def parse_response(self, tokens):
        return {"role": "assistant", "content": self.content}, SimpleNamespace(is_clean=self.clean)


class FakeSamplingClient:
    def __init__(self, future=True):
        self.future = future
        self.calls = []

    def sample(self, prompt, num_samples, sampling_params):
        self.calls.append((prompt, num_samples, sampling_params))
        seq = SimpleNamespace(tokens=[1, 2, 3], logprobs=[-0.1, -0.2, -0.3])
        response = SimpleNamespace(sequences=[seq])
        return FakeFuture(response) if self.future else response


class TestTinkerLLM(unittest.TestCase):
    def make(self, content, **kw):
        client = FakeSamplingClient(**kw)
        renderer = FakeRenderer(content)
        return TinkerLLM(client, renderer, "PARAMS"), client, renderer

    def test_returns_final_text_and_keeps_thinking(self):
        content = [{"type": "thinking", "thinking": "let me think"}, {"type": "text", "text": REPLY}]
        llm, client, renderer = self.make(content)
        self.assertEqual(llm("SYS", "PROMPT"), REPLY)
        self.assertEqual(renderer.messages, [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "PROMPT"},
        ])
        self.assertEqual(client.calls, [("MODEL_INPUT", 1, "PARAMS")])
        self.assertEqual(llm.last.thinking, "let me think")
        self.assertEqual(llm.last.tokens, [1, 2, 3])
        self.assertEqual(llm.last.logprobs, [-0.1, -0.2, -0.3])
        self.assertTrue(llm.last.clean)

    def test_prompt_kwargs_passed_to_renderer(self):
        client, renderer = FakeSamplingClient(), FakeRenderer(REPLY)
        llm = TinkerLLM(client, renderer, "PARAMS", {"effort": 0.0})
        llm("s", "p")
        self.assertEqual(renderer.kwargs, {"effort": 0.0})

    def test_continue_mode_search(self):
        llm, _, renderer = self.make([{"type": "text", "text": "  <thought>12*15 = 180</thought>\n</node>"}])
        result = search(Node.new_root("What is 12 * 15?"), llm, max_calls=3, mode="continue")
        self.assertEqual(result.answer.thought, "12*15 = 180")
        self.assertIn("<trace>", renderer.messages[1]["content"])

    def test_plain_string_content_and_non_future(self):
        llm, _, _ = self.make(REPLY, future=False)
        self.assertEqual(llm("s", "p"), REPLY)

    def test_split_content(self):
        self.assertEqual(_split_content("x"), ("x", ""))
        parts = [{"type": "text", "text": "a"}, {"type": "thinking", "thinking": "t"}, {"type": "text", "text": "b"}]
        self.assertEqual(_split_content(parts), ("ab", "t"))

    def test_plugs_into_search(self):
        llm, _, _ = self.make([{"type": "text", "text": REPLY}])
        result = search(Node.new_root("What is 12 * 15?"), llm, max_calls=3)
        self.assertEqual(result.answer.thought, "12*15 = 180")


class FakeTokenizer:
    def decode(self, tokens):
        return "decoded:" + ",".join(map(str, tokens))


class TestTinkerCompleter(unittest.TestCase):
    def test_sample_decodes_and_keeps_logprobs(self):
        client = FakeSamplingClient()
        completer = TinkerCompleter(client, FakeTokenizer(), "PARAMS")
        self.assertEqual(completer._sample("MODEL_INPUT"), "decoded:1,2,3")
        self.assertEqual(client.calls, [("MODEL_INPUT", 1, "PARAMS")])
        self.assertEqual(completer.last.logprobs, [-0.1, -0.2, -0.3])


class PrefillRenderer:
    def __init__(self):
        self.calls = []

    def build_generation_prompt(self, messages, prefill=None):
        self.calls.append((messages, prefill))
        return "MODEL_INPUT"


class TextTokenizer:
    """decode() returns a fixed continuation, like a tokenizer decoding sampled tokens."""

    def __init__(self, text):
        self.text = text
        self.kwargs = None

    def decode(self, tokens, **kwargs):
        self.kwargs = kwargs
        return self.text


class TestTinkerPrefill(unittest.TestCase):
    def test_passes_prefill_and_returns_continuation(self):
        client, renderer = FakeSamplingClient(), PrefillRenderer()
        tok = TextTokenizer("  <thought>t</thought>\n</node>")
        llm = TinkerPrefill(client, renderer, tok, SimpleNamespace(stop=[3]))
        self.assertEqual(llm("SYS", "USER", "<trace>\n<node>\n"), "  <thought>t</thought>\n</node>")
        messages, prefill = renderer.calls[0]
        self.assertEqual(messages, [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}])
        self.assertEqual(prefill, "<trace>\n<node>\n")
        self.assertEqual(tok.kwargs, {"skip_special_tokens": True})
        self.assertTrue(llm.last.clean)  # fake tokens end with 3, the stop token
        self.assertEqual(llm.last.logprobs, [-0.1, -0.2, -0.3])

    def test_plugs_into_search(self):
        tok = TextTokenizer("  <thought>12*15 = 180</thought>\n</node>\n</trace>")
        llm = TinkerPrefill(FakeSamplingClient(), PrefillRenderer(), tok, SimpleNamespace(stop=[]))
        result = search(Node.new_root("What is 12 * 15?"), prefill=llm, max_calls=3)
        self.assertEqual(result.answer.thought, "12*15 = 180")


if __name__ == "__main__":
    unittest.main()
