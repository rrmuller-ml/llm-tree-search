"""Tests for the Tinker adapter with fake client and renderer (no network, no tinker install)."""

import math
import unittest
from types import SimpleNamespace

from llm import expand_prefill, search
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


class FakeModelInput:
    def __init__(self, ids):
        self.ids = ids

    def to_ints(self):
        return list(self.ids)


class VocabTokenizer:
    """Tokens are indices into a vocab of text pieces."""

    def __init__(self):
        self.vocab: list[str] = []

    def ids(self, pieces):
        out = []
        for piece in pieces:
            if piece not in self.vocab:
                self.vocab.append(piece)
            out.append(self.vocab.index(piece))
        return out

    def decode(self, ids, **kwargs):
        return "".join(self.vocab[i] for i in ids)


PROMPT = ["SYS USER", "<trace>", "\n", "<node>", "\n  ", "<follow_proposal", " idx", '="', "1", '"/>', "\n"]
GEN = ["  ", "<thought>", "via B", "</thought>", "<proposal>", "a", "</proposal>", "<proposal>", "b",
       "</proposal>", "</node>", "\n", "<node>", "<follow_proposal", " idx", '="', "1", '"/>', "<thought>", "b gives 180",
       "</thought>", "</node>"]


class PrefillRenderer:
    def __init__(self, prompt_ids):
        self.prompt_ids = prompt_ids
        self.calls = []

    def build_generation_prompt(self, messages, prefill=None):
        self.calls.append((messages, prefill))
        return FakeModelInput(self.prompt_ids)


class PrefillClient:
    def __init__(self, gen_ids, gen_topk, prompt_topk):
        self.gen_ids, self.gen_topk, self.prompt_topk = gen_ids, gen_topk, prompt_topk
        self.kwargs = None

    def sample(self, prompt, num_samples, sampling_params, **kwargs):
        self.kwargs = kwargs
        seq = SimpleNamespace(tokens=self.gen_ids, logprobs=[-0.5] * len(self.gen_ids), topk_logprobs=self.gen_topk)
        return FakeFuture(SimpleNamespace(sequences=[seq], topk_prompt_logprobs=self.prompt_topk))


def make_prefill(prompt_topk_full=True):
    tok = VocabTokenizer()
    prompt_ids = tok.ids(PROMPT)
    gen_ids = tok.ids(GEN)
    d0, d1, d2, x = tok.ids(["0", "1", "2", "x"])
    gen_topk = [None] * len(gen_ids)
    gen_topk[GEN.index("1")] = [(d1, -0.1), (d0, -2.5), (x, -6.0)]
    digit_pos = PROMPT.index("1")
    prompt_topk = [None] * len(prompt_ids)
    prompt_topk[digit_pos] = [(d1, -0.2), (d0, -1.8), (d2, -7.0)]
    if not prompt_topk_full:  # only the last n positions returned
        prompt_topk = prompt_topk[-4:]
    client = PrefillClient(gen_ids, gen_topk, prompt_topk)
    llm = TinkerPrefill(client, PrefillRenderer(prompt_ids), tok, SimpleNamespace(stop=[]))
    return llm, client


class TestIdxDigits(unittest.TestCase):
    def test_unicode_digits_are_ignored(self):
        tok = VocabTokenizer()
        d1, sub0, sup2 = tok.ids(["1", "₀", "²"])
        llm = TinkerPrefill(None, None, tok, SimpleNamespace(stop=[]))
        self.assertEqual(llm._digit_dist([(d1, -0.1), (sub0, -2.0), (sup2, -3.0)]), {1: -0.1})


class TestFollowProposalDetection(unittest.TestCase):
    def test_reads_digit_after_follow_proposal(self):
        prompt = ["SYS", "<trace>", "\n", "<node>", "\n  ", "<follow", "_proposal", " idx", '="', "1", '"/>', "\n"]
        gen = ["  ", "<thought>", "x", "</thought>", "<proposal>", "a", "</proposal>", "<proposal>", "b",
               "</proposal>", "</node>", "<node>", "<follow", "_proposal", " idx", '="', "0", '"/>', "<thought>",
               "y", "</thought>", "</node>"]
        tok = VocabTokenizer()
        p_ids, g_ids = tok.ids(prompt), tok.ids(gen)
        d0, d1 = tok.ids(["0", "1"])
        gen_topk = [None] * len(g_ids)
        gen_topk[gen.index("0")] = [(d0, -0.2), (d1, -1.7)]
        prompt_topk = [None] * len(p_ids)
        prompt_topk[prompt.index("1")] = [(d1, -0.1), (d0, -2.4)]
        llm = TinkerPrefill(PrefillClient(g_ids, gen_topk, prompt_topk), PrefillRenderer(p_ids), tok,
                            SimpleNamespace(stop=[]))
        reply = llm("SYS", "USER", "<trace>\n")
        self.assertEqual(reply.choice_logprobs, [{0: -0.2, 1: -1.7}])
        self.assertEqual(reply.prefill_choice, {1: -0.1, 0: -2.4})


class TestTinkerPrefill(unittest.TestCase):
    def test_reply_text_and_request(self):
        llm, client = make_prefill()
        reply = llm("SYS", "USER", "<trace>\n")
        self.assertEqual(reply.text, "".join(GEN))
        self.assertEqual(client.kwargs["topk_sample_logprobs"], 20)
        self.assertEqual(client.kwargs["topk_prompt_logprobs"], 20)
        self.assertIn("prompt_logprobs_last_n", client.kwargs)
        self.assertTrue(client.kwargs["include_prompt_logprobs"])  # required by the SDK with last_n

    def test_reads_choices_from_topk(self):
        llm, _ = make_prefill()
        reply = llm("SYS", "USER", "<trace>\n")
        self.assertEqual(reply.choice_logprobs, [{1: -0.1, 0: -2.5}])  # "x" is not a number
        self.assertEqual(reply.prefill_choice, {1: -0.2, 0: -1.8, 2: -7.0})

    def test_prompt_topk_for_last_n_only(self):
        llm, _ = make_prefill(prompt_topk_full=False)
        reply = llm("SYS", "USER", "<trace>\n")
        self.assertEqual(reply.prefill_choice, {1: -0.2, 0: -1.8, 2: -7.0})

    def test_logprobs_land_on_the_right_proposals(self):
        llm, _ = make_prefill()
        root = Node.new_root("Q?", "Split.")
        root.first.propose("A")
        root.first.propose("B")
        created = expand_prefill(root.first, "B", llm)
        a, b = root.first.proposals
        self.assertEqual((a.logprob, b.logprob), (-1.8, -0.2))  # from the prompt (our choice)
        self.assertAlmostEqual(b.prob, math.exp(-0.2))
        pa, pb = created[0].proposals
        self.assertEqual((pa.logprob, pb.logprob), (-2.5, -0.1))  # from the sampled <follow_proposal>
        self.assertTrue(created[1].is_answer)
        self.assertEqual(Node.from_dict(root.to_dict()).first.proposals[0].logprob, -1.8)

    def test_no_topk_means_no_logprobs(self):
        llm, client = make_prefill()
        client.gen_topk, client.prompt_topk = None, None
        reply = llm("SYS", "USER", "<trace>\n")
        self.assertEqual(reply.choice_logprobs, [None])
        self.assertIsNone(reply.prefill_choice)

    def test_plugs_into_search(self):
        llm, _ = make_prefill()
        result = search(Node.new_root("Q?"), prefill=llm, max_calls=3)
        self.assertEqual(result.answer.thought, "b gives 180")


if __name__ == "__main__":
    unittest.main()
