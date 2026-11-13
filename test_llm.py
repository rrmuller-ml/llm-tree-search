import unittest

from llm import (
    COMPLETION_HEADER,
    EXAMPLE_TRACE,
    SYSTEM_PROMPT,
    DecodeError,
    apply_chain,
    completion_prefix,
    continue_prompt,
    decode_continuation,
    decode_chain,
    depth_first,
    encode_chain,
    encode_context,
    encode_trace,
    expand,
    expand_completion,
    expand_continue,
    expand_prefill,
    prefill_parts,
    pick_next,
    search,
    validate_chain,
    valid_prefix,
    PartialChain,
)
from tree import Node

Q = "What is 12 * 15?"

CHAIN = """
Sure, here is my reasoning:
<node>
  <thought>Split into easier parts.</thought>
  <proposal idx="0">Distribute over 15: 12*10 + 12*5</proposal>
  <proposal idx="1">Double and halve: 6*30</proposal>
</node>
<node>
  <parent>0</parent>
  <thought>12*10 = 120 and 12*5 = 60.</thought>
  <proposal idx="0">Sum: 120 + 60</proposal>
</node>
<node>
  <parent>0</parent>
  <thought>120 + 60 = 180</thought>
</node>
"""


class FakeLLM:
    """Returns canned replies in order and records the prompts it got."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        return self.replies.pop(0)


class TestDecode(unittest.TestCase):
    def test_decode_chain(self):
        first, second, last = decode_chain(CHAIN)
        self.assertIsNone(first.parent)
        self.assertEqual([p.title for p in first.proposals],
                         ["Distribute over 15: 12*10 + 12*5", "Double and halve: 6*30"])
        self.assertEqual(second.parent, 0)
        self.assertFalse(second.is_answer)
        self.assertTrue(last.is_answer)  # no proposals = answer

    def test_ignores_text_outside_nodes_and_truncated_tail(self):
        truncated = CHAIN + "<node><parent>0</parent><thought>half writ"
        self.assertEqual(len(decode_chain(truncated)), 3)

    def test_unescapes(self):
        chain = decode_chain("<node><thought>a &lt; b</thought><proposal>x &amp; y</proposal></node>")
        self.assertEqual((chain[0].thought, chain[0].proposals[0].title), ("a < b", "x & y"))

    def test_extra_attributes_ignored(self):
        chain = decode_chain('<node><thought>t</thought><proposal value="0.9" idx="2">a</proposal></node>')
        self.assertEqual(chain[0].proposals[0].idx, 2)

    def test_idx_defaults_to_position(self):
        chain = decode_chain(
            "<node><thought>t</thought><proposal>a</proposal><proposal>b</proposal></node>"
            "<node><parent>1</parent><thought>u</thought></node>"
        )
        self.assertEqual(chain[0].position_of(1), 1)

    def test_explicit_idx(self):
        chain = decode_chain(
            '<node><thought>t</thought><proposal idx="5">a</proposal><proposal idx="7">b</proposal></node>'
        )
        self.assertEqual(chain[0].position_of(7), 1)
        self.assertIsNone(chain[0].position_of(1))

    def test_bad_idx(self):
        with self.assertRaises(DecodeError):
            decode_chain('<node><thought>t</thought><proposal idx="first">a</proposal></node>')

    def test_missing_thought(self):
        with self.assertRaises(DecodeError):
            decode_chain("<node><proposal>a</proposal></node>")

    def test_encode_decode_roundtrip(self):
        chain = decode_chain(CHAIN)
        self.assertEqual(decode_chain(encode_chain(chain)), chain)


class TestApply(unittest.TestCase):
    def test_apply_from_question(self):
        root = Node.new_root(Q)
        created = apply_chain(root, Q, decode_chain(CHAIN))
        self.assertEqual(len(created), 3)
        first = root.first
        self.assertEqual(first.thought, "Split into easier parts.")
        self.assertEqual(len(first.proposals), 2)
        self.assertIs(first.get_proposal("Distribute over 15: 12*10 + 12*5").node, created[1])
        self.assertFalse(first.get_proposal("Double and halve: 6*30").is_explored)
        self.assertTrue(created[2].is_answer)
        self.assertEqual(created[2].title, "Sum: 120 + 60")

    def test_no_scores_written(self):
        root = Node.new_root(Q)
        apply_chain(root, Q, decode_chain(CHAIN))
        for _, p in root.unexplored():
            self.assertIsNone(p.prob)
            self.assertIsNone(p.value)

    def test_parent_picks_non_first_proposal(self):
        root = Node.new_root(Q)
        chain = decode_chain(
            "<node><thought>t</thought><proposal>a</proposal><proposal>b</proposal></node>"
            "<node><parent>1</parent><thought>via b</thought></node>"
        )
        apply_chain(root, Q, chain)
        self.assertEqual(root.first.get_proposal("b").node.thought, "via b")
        self.assertFalse(root.first.get_proposal("a").is_explored)

    def test_chain_may_stop_with_open_proposals(self):
        root = Node.new_root(Q)
        apply_chain(root, Q, decode_chain(CHAIN.split("<node>\n  <parent>0</parent>")[0]))
        self.assertEqual(len(root.first.proposals), 2)
        self.assertEqual(root.answer_nodes(), [])

    def test_apply_from_existing_proposal(self):
        root = Node.new_root(Q, "Split.")
        root.first.propose("Double and halve: 6*30")
        created = apply_chain(root.first, "Double and halve: 6*30",
                              decode_chain("<node><thought>6*30 = 180</thought></node>"))
        self.assertIs(created[0].parent, root.first)
        self.assertTrue(created[0].is_answer)

    def test_invalid_chains_leave_tree_unchanged(self):
        bad = {
            "empty": "no nodes here",
            "missing parent": "<node><thought>t</thought><proposal>a</proposal></node>"
                              "<node><thought>u</thought></node>",
            "parent out of range": "<node><thought>t</thought><proposal>a</proposal></node>"
                                   "<node><parent>3</parent><thought>u</thought></node>",
            "follows an answer": "<node><thought>t</thought></node>"
                                 "<node><parent>0</parent><thought>u</thought></node>",
            "duplicate titles": "<node><thought>t</thought><proposal>a</proposal><proposal>a</proposal></node>",
            "duplicate idx": '<node><thought>t</thought><proposal idx="0">a</proposal>'
                             '<proposal idx="0">b</proposal></node>',
            "empty thought": "<node><thought>  </thought></node>",
        }
        for name, text in bad.items():
            with self.subTest(name):
                root = Node.new_root(Q)
                before = root.to_dict()
                with self.assertRaises(DecodeError):
                    apply_chain(root, Q, decode_chain(text))
                self.assertEqual(root.to_dict(), before)

    def test_cannot_apply_to_explored_proposal(self):
        root = Node.new_root(Q, "already")
        with self.assertRaises(ValueError):
            apply_chain(root, Q, decode_chain(CHAIN))


class TestContext(unittest.TestCase):
    def test_context_for_question(self):
        ctx = encode_context(Node.new_root(Q), Q)
        self.assertIn(f"<question>{Q}</question>", ctx)
        self.assertNotIn("<path>", ctx)
        self.assertNotIn("<develop>", ctx)

    def test_context_has_path_and_develop(self):
        root = Node.new_root(Q, "Split.")
        root.first.propose("A & B")
        mid = root.first.develop("A & B", "x < y")
        mid.propose("Next step")
        ctx = encode_context(mid, "Next step")
        self.assertIn('<step title="What is 12 * 15?">Split.</step>', ctx)
        self.assertIn('<step title="A &amp; B">x &lt; y</step>', ctx)
        self.assertIn("<develop>Next step</develop>", ctx)


class TestLoop(unittest.TestCase):
    def test_expand_calls_llm(self):
        root = Node.new_root(Q)
        llm = FakeLLM(CHAIN)
        expand(root, Q, llm)
        self.assertEqual(llm.calls[0][0], SYSTEM_PROMPT)
        self.assertEqual(len(root.answer_nodes()), 1)

    def test_depth_first_default(self):
        root = Node.new_root(Q, "Split.")
        root.first.propose("shallow")
        deep = root.first.propose("deep path")
        child = root.first.develop(deep, "going deeper")
        child.propose("x")
        child.propose("y")
        self.assertEqual(pick_next(root)[1].title, "x")  # deepest node, earliest proposal

    def test_custom_score_and_skip(self):
        root = Node.new_root(Q, "Split.")
        a = root.first.propose("a")
        b = root.first.propose("b")
        a.prob, b.prob = 0.2, 0.8  # e.g. from log-probs of <parent>
        by_prob = lambda n, p: p.prob or 0.0
        self.assertEqual(pick_next(root, by_prob)[1].title, "b")
        self.assertEqual(pick_next(root, by_prob, frozenset({id(b)}))[1].title, "a")
        self.assertEqual(pick_next(root, depth_first)[1].title, "a")

    def test_search_backtracks(self):
        first = (
            "<node><thought>Split.</thought><proposal>Try A</proposal><proposal>Try B</proposal></node>"
            "<node><parent>0</parent><thought>A looks hard.</thought><proposal>Push on A</proposal></node>"
        )
        second = "<node><thought>Pushing on A: 180</thought></node>"
        llm = FakeLLM(first, second)
        root = Node.new_root(Q)
        result = search(root, llm, max_calls=5)
        self.assertEqual(result.calls, 2)
        self.assertEqual(result.answer.title, "Push on A")  # DFS resumes the deepest open proposal
        self.assertIn("<develop>Push on A</develop>", llm.calls[1][1])

    def test_search_skips_failed_proposal(self):
        root = Node.new_root(Q)
        llm = FakeLLM("garbage")
        result = search(root, llm, max_calls=5)
        self.assertIsNone(result.answer)
        self.assertEqual(result.calls, 1)  # question failed; nothing else to try
        self.assertEqual(len(result.errors), 1)

    def test_search_stops_at_budget(self):
        root = Node.new_root(Q, "Split.")
        for t in "abc":
            root.first.propose(t)
        reply = "<node><thought>try</thought><proposal>more</proposal></node>"
        result = search(root, FakeLLM(reply, reply, reply), max_calls=2)
        self.assertEqual(result.calls, 2)
        self.assertIsNone(result.answer)


class TestContinueMode(unittest.TestCase):
    def tree(self) -> Node:
        root = Node.new_root(Q)
        apply_chain(root, Q, decode_chain(CHAIN))
        return root

    def test_prompt_has_instruction_and_cut_trace(self):
        root = self.tree()
        prompt, opener = continue_prompt(root.first, "Double and halve: 6*30")
        self.assertTrue(prompt.endswith("<node>\n  <parent>1</parent>\n"))
        self.assertIn("Continue it", prompt)
        self.assertNotIn("12*10 = 120", prompt)
        self.assertNotIn("14 * 25", prompt)  # no few-shot header; the system prompt has the example

    def test_reply_continuing_inside_open_node(self):
        chain = decode_continuation("  <thought>6*30 = 180</thought>\n</node>", "<node>\n  <parent>1</parent>\n")
        self.assertEqual(chain[0].thought, "6*30 = 180")

    def test_reply_rewriting_open_node(self):
        opener = "<node>\n  <parent>1</parent>\n"
        ok = decode_continuation("<node><parent>1</parent><thought>t</thought></node>", opener)
        self.assertEqual(ok[0].thought, "t")
        no_parent = decode_continuation("<node><thought>t</thought></node>", opener)
        self.assertEqual(no_parent[0].thought, "t")
        with self.assertRaises(DecodeError):
            decode_continuation("<node><parent>0</parent><thought>t</thought></node>", opener)

    def test_expand_continue(self):
        root = self.tree()
        llm = FakeLLM("<thought>6*30 = 180</thought></node>")
        created = expand_continue(root.first, "Double and halve: 6*30", llm)
        self.assertEqual(created[0].title, "Double and halve: 6*30")
        self.assertTrue(created[0].is_answer)
        self.assertEqual(llm.calls[0][0], SYSTEM_PROMPT)

    def test_search_continue_mode(self):
        first = ("<thought>Split.</thought><proposal>Try A</proposal><proposal>Try B</proposal></node>"
                 "<node><parent>0</parent><thought>A looks hard.</thought><proposal>Push on A</proposal></node>")
        second = "<thought>Pushing on A: 180</thought></node>"
        llm = FakeLLM(first, second)
        result = search(Node.new_root(Q), llm, max_calls=5, mode="continue")
        self.assertEqual(result.answer.title, "Push on A")
        self.assertTrue(llm.calls[1][1].endswith("<node>\n  <parent>0</parent>\n"))


class TestPrefillMode(unittest.TestCase):
    def tree(self) -> Node:
        root = Node.new_root(Q)
        apply_chain(root, Q, decode_chain(CHAIN))
        return root

    def test_parts(self):
        root = self.tree()
        user, prefill, opener = prefill_parts(root.first, "Double and halve: 6*30")
        self.assertIn(f"<question>{Q}</question>", user)
        self.assertTrue(prefill.startswith("<trace>\n<node>\n"))
        self.assertTrue(prefill.endswith("<node>\n  <parent>1</parent>\n"))
        self.assertNotIn("12*10 = 120", prefill)
        self.assertEqual(opener, "<node>\n  <parent>1</parent>\n")

    def test_parts_for_question(self):
        user, prefill, opener = prefill_parts(Node.new_root(Q), Q)
        self.assertEqual(prefill, "<trace>\n<node>\n")

    def test_expand_and_search(self):
        calls = []

        def model(system, user, prefill):
            calls.append(prefill)
            if len(calls) == 1:
                return ("  <thought>Split.</thought>\n  <proposal>Try A</proposal>\n  <proposal>Try B</proposal>\n"
                        "</node>\n<node>\n  <parent>0</parent>\n  <thought>A looks hard.</thought>\n"
                        "  <proposal>Push on A</proposal>\n</node>\n")
            return "  <thought>Pushing on A: 180</thought>\n</node>\n</trace>"

        result = search(Node.new_root(Q), prefill=model, max_calls=5)
        self.assertEqual(result.answer.title, "Push on A")
        self.assertTrue(calls[1].endswith("<node>\n  <parent>0</parent>\n"))

    def test_search_rejects_two_models(self):
        with self.assertRaises(ValueError):
            search(Node.new_root(Q), FakeLLM(), prefill=lambda s, u, p: "")


class TestRepairAndSalvage(unittest.TestCase):
    def test_single_proposal_forces_parent(self):
        chain = decode_chain(
            "<node><thought>t</thought><proposal>only</proposal></node>"
            "<node><parent>1</parent><thought>u</thought></node>"
        )
        good, err = valid_prefix(chain)
        self.assertIsNone(err)
        self.assertEqual(good[1].parent, 0)
        self.assertEqual(chain[1].parent, 1)  # input not mutated

    def test_keeps_valid_start(self):
        chain = decode_chain(
            "<node><thought>t</thought><proposal>a</proposal><proposal>b</proposal></node>"
            "<node><parent>1</parent><thought>u</thought><proposal>x</proposal><proposal>y</proposal></node>"
            "<node><parent>5</parent><thought>bad</thought></node>"
        )
        good, err = valid_prefix(chain)
        self.assertEqual(len(good), 2)
        self.assertIn("node 2", err)

    def test_nothing_valid(self):
        good, err = valid_prefix([])
        self.assertEqual(good, [])
        self.assertTrue(err)

    def test_partial_reply_is_applied_and_reported(self):
        reply = ("  <thought>Split.</thought>\n  <proposal>A</proposal>\n  <proposal>B</proposal>\n</node>\n"
                 "<node>\n  <parent>0</parent>\n  <thought>via A</thought>\n  <proposal>A1</proposal>\n"
                 "  <proposal>A2</proposal>\n</node>\n<node>\n  <parent>7</parent>\n  <thought>bad</thought>\n</node>")
        root = Node.new_root(Q)
        with self.assertRaises(PartialChain) as ctx:
            expand_prefill(root, Q, lambda s, u, p: reply)
        self.assertEqual(len(ctx.exception.created), 2)
        self.assertEqual(root.first.get_proposal("A").node.thought, "via A")

    def test_search_continues_after_partial(self):
        first = ("  <thought>Split.</thought>\n  <proposal>A</proposal>\n  <proposal>B</proposal>\n</node>\n"
                 "<node>\n  <parent>9</parent>\n  <thought>bad</thought>\n</node>")
        second = "  <thought>A gives 180</thought>\n</node>"
        replies = [first, second]
        result = search(Node.new_root(Q), prefill=lambda s, u, p: replies.pop(0), max_calls=5)
        self.assertEqual(result.answer.title, "A")
        self.assertEqual(result.calls, 2)
        self.assertEqual(len(result.errors), 1)


class TestExample(unittest.TestCase):
    def test_example_is_a_valid_chain(self):
        chain = decode_chain(EXAMPLE_TRACE)
        validate_chain(chain)
        self.assertEqual(len(chain), 3)
        self.assertTrue(chain[-1].is_answer)
        self.assertIn(EXAMPLE_TRACE, SYSTEM_PROMPT)
        self.assertIn(EXAMPLE_TRACE, COMPLETION_HEADER)


class FakeCompleter:
    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.prefixes: list[str] = []

    def __call__(self, prefix: str) -> str:
        self.prefixes.append(prefix)
        return self.replies.pop(0)


class TestCompletion(unittest.TestCase):
    def tree(self) -> Node:
        root = Node.new_root(Q)
        apply_chain(root, Q, decode_chain(CHAIN))
        return root

    def test_encode_trace_roundtrips_path(self):
        root = self.tree()
        leaf = root.answer_nodes()[0]
        self.assertEqual(decode_chain(encode_trace(leaf)), decode_chain(CHAIN))

    def test_prefix_for_question(self):
        root = Node.new_root(Q)
        prefix, opener = completion_prefix(root, Q)
        self.assertTrue(prefix.startswith(COMPLETION_HEADER))
        self.assertTrue(prefix.endswith(f"<question>{Q}</question>\n<trace>\n<node>\n"))
        self.assertEqual(opener, "<node>\n")

    def test_prefix_cuts_after_owner_and_picks_proposal(self):
        root = self.tree()
        first = root.first
        prefix, opener = completion_prefix(first, "Double and halve: 6*30")
        self.assertEqual(opener, "<node>\n  <parent>1</parent>\n")
        trace = prefix.split("<trace>\n")[-1]
        self.assertIn("Distribute over 15", trace)        # sibling proposal still listed
        self.assertNotIn("12*10 = 120", trace)            # explored continuation removed
        self.assertTrue(trace.endswith(opener))

    def test_expand_completion(self):
        root = self.tree()
        complete = FakeCompleter("  <thought>6*30 = 180</thought>\n</node>\n</trace>\n<question>junk")
        created = expand_completion(root.first, "Double and halve: 6*30", complete)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].title, "Double and halve: 6*30")
        self.assertTrue(created[0].is_answer)

    def test_expand_completion_multi_node(self):
        root = Node.new_root(Q)
        reply = ("  <thought>Split.</thought>\n  <proposal idx=\"0\">A</proposal>\n  <proposal idx=\"1\">B</proposal>\n"
                 "</node>\n<node>\n  <parent>1</parent>\n  <thought>via B: 180</thought>\n</node>\n")
        expand_completion(root, Q, FakeCompleter(reply))
        self.assertEqual(root.first.get_proposal("B").node.thought, "via B: 180")
        self.assertFalse(root.first.get_proposal("A").is_explored)

    def test_search_with_completion(self):
        first = ("  <thought>Split.</thought>\n  <proposal>Try A</proposal>\n  <proposal>Try B</proposal>\n"
                 "</node>\n<node>\n  <parent>0</parent>\n  <thought>A looks hard.</thought>\n"
                 "  <proposal>Push on A</proposal>\n</node>\n")
        second = "  <thought>Pushing on A: 180</thought>\n</node>\n</trace>"
        complete = FakeCompleter(first, second)
        result = search(Node.new_root(Q), complete=complete, max_calls=5)
        self.assertEqual(result.answer.title, "Push on A")
        self.assertTrue(complete.prefixes[1].endswith("<node>\n  <parent>0</parent>\n"))

    def test_search_needs_exactly_one_model(self):
        with self.assertRaises(ValueError):
            search(Node.new_root(Q))
        with self.assertRaises(ValueError):
            search(Node.new_root(Q), FakeLLM(), complete=FakeCompleter())


if __name__ == "__main__":
    unittest.main()
