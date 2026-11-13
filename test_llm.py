import math
import unittest

from llm import (
    COMPLETION_HEADER,
    EXAMPLE_TRACE,
    SYSTEM_PROMPT,
    DecodeError,
    NodeSpec,
    ProposalSpec,
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
    question_node,
    pick_next,
    search,
    validate_chain,
    valid_prefix,
    PartialChain,
    Reply,
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
  <follow_proposal idx='0'/>
  <thought>12*10 = 120 and 12*5 = 60.</thought>
  <proposal idx="0">Sum: 120 + 60</proposal>
</node>
<node>
  <follow_proposal idx='0'/>
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
        self.assertIsNone(first.follow)
        self.assertEqual([p.title for p in first.proposals],
                         ["Distribute over 15: 12*10 + 12*5", "Double and halve: 6*30"])
        self.assertEqual(second.follow, 0)
        self.assertFalse(second.is_answer)
        self.assertTrue(last.is_answer)  # no proposals = answer

    def test_ignores_text_outside_nodes_and_truncated_tail(self):
        truncated = CHAIN + "<node><follow_proposal idx='0'/><thought>half writ"
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
            "<node><follow_proposal idx='1'/><thought>u</thought></node>"
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

    def test_follow_picks_non_first_proposal(self):
        root = Node.new_root(Q)
        chain = decode_chain(
            "<node><thought>t</thought><proposal>a</proposal><proposal>b</proposal></node>"
            "<node><follow_proposal idx='1'/><thought>via b</thought></node>"
        )
        apply_chain(root, Q, chain)
        self.assertEqual(root.first.get_proposal("b").node.thought, "via b")
        self.assertFalse(root.first.get_proposal("a").is_explored)

    def test_chain_may_stop_with_open_proposals(self):
        root = Node.new_root(Q)
        apply_chain(root, Q, decode_chain(CHAIN.split("<node>\n  <follow_proposal idx='0'/>")[0]))
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
            "missing follow_proposal": "<node><thought>t</thought><proposal>a</proposal></node>"
                              "<node><thought>u</thought></node>",
            "follow_proposal out of range": "<node><thought>t</thought><proposal>a</proposal></node>"
                                   "<node><follow_proposal idx='3'/><thought>u</thought></node>",
            "follows an answer": "<node><thought>t</thought></node>"
                                 "<node><follow_proposal idx='0'/><thought>u</thought></node>",
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
        self.assertIn('<step title="A &amp; B">x < y</step>', ctx)  # math stays raw
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
        a.prob, b.prob = 0.2, 0.8  # e.g. from log-probs of <follow_proposal>
        by_prob = lambda n, p: p.prob or 0.0
        self.assertEqual(pick_next(root, by_prob)[1].title, "b")
        self.assertEqual(pick_next(root, by_prob, frozenset({id(b)}))[1].title, "a")
        self.assertEqual(pick_next(root, depth_first)[1].title, "a")

    def test_search_backtracks(self):
        first = (
            "<node><thought>Split.</thought><proposal>Try A</proposal><proposal>Try B</proposal></node>"
            "<node><follow_proposal idx='0'/><thought>A looks hard.</thought><proposal>Push on A</proposal></node>"
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
        self.assertTrue(prompt.endswith('<node>\n  <follow_proposal idx="1"/>\n'))
        self.assertIn("Continue it", prompt)
        self.assertNotIn("12*10 = 120", prompt)
        self.assertNotIn("14 * 25", prompt)  # no few-shot header; the system prompt has the example

    def test_reply_continuing_inside_open_node(self):
        chain = decode_continuation("  <thought>6*30 = 180</thought>\n</node>", "<node>\n  <follow_proposal idx='1'/>\n")
        self.assertEqual(chain[0].thought, "6*30 = 180")

    def test_reply_rewriting_open_node(self):
        opener = "<node>\n  <follow_proposal idx='1'/>\n"
        ok = decode_continuation("<node><follow_proposal idx='1'/><thought>t</thought></node>", opener)
        self.assertEqual(ok[0].thought, "t")
        no_follow = decode_continuation("<node><thought>t</thought></node>", opener)
        self.assertEqual(no_follow[0].thought, "t")
        with self.assertRaises(DecodeError):
            decode_continuation("<node><follow_proposal idx='0'/><thought>t</thought></node>", opener)

    def test_expand_continue(self):
        root = self.tree()
        llm = FakeLLM("<thought>6*30 = 180</thought></node>")
        created = expand_continue(root.first, "Double and halve: 6*30", llm)
        self.assertEqual(created[0].title, "Double and halve: 6*30")
        self.assertTrue(created[0].is_answer)
        self.assertEqual(llm.calls[0][0], SYSTEM_PROMPT)

    def test_search_continue_mode(self):
        first = ("<thought>Split.</thought><proposal>Try A</proposal><proposal>Try B</proposal></node>"
                 "<node><follow_proposal idx='0'/><thought>A looks hard.</thought><proposal>Push on A</proposal></node>")
        second = "<thought>Pushing on A: 180</thought></node>"
        llm = FakeLLM(first, second)
        result = search(Node.new_root(Q), llm, max_calls=5, mode="continue")
        self.assertEqual(result.answer.title, "Push on A")
        self.assertTrue(llm.calls[1][1].endswith('<node>\n  <follow_proposal idx="0"/>\n'))


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
        self.assertTrue(prefill.endswith('<node>\n  <follow_proposal idx="1"/>\n'))
        self.assertNotIn("12*10 = 120", prefill)
        self.assertEqual(opener, '<node>\n  <follow_proposal idx="1"/>\n')

    def test_parts_for_question(self):
        user, prefill, opener = prefill_parts(Node.new_root(Q), Q)
        self.assertEqual(prefill, f'<trace>\n{question_node(Q)}\n<node>\n  <follow_proposal idx="0"/>\n')
        self.assertEqual(opener, '<node>\n  <follow_proposal idx="0"/>\n')

    def test_expand_and_search(self):
        calls = []

        def model(system, user, prefill):
            calls.append(prefill)
            if len(calls) == 1:
                return ("  <thought>Split.</thought>\n  <proposal>Try A</proposal>\n  <proposal>Try B</proposal>\n"
                        "</node>\n<node>\n  <follow_proposal idx='0'/>\n  <thought>A looks hard.</thought>\n"
                        "  <proposal>Push on A</proposal>\n</node>\n")
            return "  <thought>Pushing on A: 180</thought>\n</node>\n</trace>"

        result = search(Node.new_root(Q), prefill=model, max_calls=5)
        self.assertEqual(result.answer.title, "Push on A")
        self.assertTrue(calls[1].endswith('<node>\n  <follow_proposal idx="0"/>\n'))

    def test_search_rejects_two_models(self):
        with self.assertRaises(ValueError):
            search(Node.new_root(Q), FakeLLM(), prefill=lambda s, u, p: "")


class TestRepairAndSalvage(unittest.TestCase):
    def test_single_proposal_forces_follow(self):
        chain = decode_chain(
            "<node><thought>t</thought><proposal>only</proposal></node>"
            "<node><follow_proposal idx='1'/><thought>u</thought></node>"
        )
        good, err = valid_prefix(chain)
        self.assertIsNone(err)
        self.assertEqual(good[1].follow, 0)
        self.assertEqual(chain[1].follow, 1)  # input not mutated

    def test_keeps_valid_start(self):
        chain = decode_chain(
            "<node><thought>t</thought><proposal>a</proposal><proposal>b</proposal></node>"
            "<node><follow_proposal idx='1'/><thought>u</thought><proposal>x</proposal><proposal>y</proposal></node>"
            "<node><follow_proposal idx='5'/><thought>bad</thought></node>"
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
                 "<node>\n  <follow_proposal idx='0'/>\n  <thought>via A</thought>\n  <proposal>A1</proposal>\n"
                 "  <proposal>A2</proposal>\n</node>\n<node>\n  <follow_proposal idx='7'/>\n  <thought>bad</thought>\n</node>")
        root = Node.new_root(Q)
        with self.assertRaises(PartialChain) as ctx:
            expand_prefill(root, Q, lambda s, u, p: reply)
        self.assertEqual(len(ctx.exception.created), 2)
        self.assertEqual(root.first.get_proposal("A").node.thought, "via A")

    def test_search_continues_after_partial(self):
        first = ("  <thought>Split.</thought>\n  <proposal>A</proposal>\n  <proposal>B</proposal>\n</node>\n"
                 "<node>\n  <follow_proposal idx='9'/>\n  <thought>bad</thought>\n</node>")
        second = "  <thought>A gives 180</thought>\n</node>"
        replies = [first, second]
        result = search(Node.new_root(Q), prefill=lambda s, u, p: replies.pop(0), max_calls=5)
        self.assertEqual(result.answer.title, "A")
        self.assertEqual(result.calls, 2)
        self.assertEqual(len(result.errors), 1)


class TestChoiceLogprobs(unittest.TestCase):
    REPLY = ("  <thought>Split.</thought>\n  <proposal>A</proposal>\n  <proposal>B</proposal>\n</node>\n"
             "<node>\n  <follow_proposal idx='1'/>\n  <thought>via B: 180</thought>\n</node>\n")

    def test_first_call_scores_generated_choices(self):
        root = Node.new_root(Q)
        reply = Reply(self.REPLY, choice_logprobs=[{0: -1.2, 1: -0.4}])
        expand_prefill(root, Q, lambda s, u, p: reply)
        a, b = root.first.proposals
        self.assertEqual((a.logprob, b.logprob), (-1.2, -0.4))
        self.assertAlmostEqual(a.prob, math.exp(-1.2))  # raw, not renormalized
        self.assertIsNone(root.proposals[0].logprob)  # the question has no choice

    def test_prefill_choice_scores_owner_node(self):
        root = Node.new_root(Q, "Split.")
        root.first.propose("A")
        root.first.propose("B")
        reply = Reply("  <thought>B: 180</thought>\n</node>", prefill_choice={0: -0.3, 1: -1.5})
        expand_prefill(root.first, "B", lambda s, u, p: reply)
        self.assertEqual([p.logprob for p in root.first.proposals], [-0.3, -1.5])

    def test_missing_idx_and_unknown_dist(self):
        root = Node.new_root(Q)
        reply = Reply(self.REPLY, choice_logprobs=[{1: -0.1}])
        expand_prefill(root, Q, lambda s, u, p: reply)
        a, b = root.first.proposals
        self.assertIsNone(a.logprob)
        self.assertIsNone(a.prob)
        self.assertEqual(b.logprob, -0.1)
        root2 = Node.new_root(Q)
        expand_prefill(root2, Q, lambda s, u, p: Reply(self.REPLY, choice_logprobs=[None]))
        self.assertIsNone(root2.first.proposals[0].logprob)

    def test_plain_string_reply_has_no_logprobs(self):
        root = Node.new_root(Q)
        expand_prefill(root, Q, lambda s, u, p: self.REPLY)
        self.assertTrue(all(p.logprob is None for p in root.first.proposals))

    def test_rewritten_open_node_uses_reply_choices(self):
        root = Node.new_root(Q, "Split.")
        root.first.propose("A")
        root.first.propose("B")
        text = "<node>\n  <follow_proposal idx='1'/>\n  <thought>B: 180</thought>\n</node>"
        reply = Reply(text, choice_logprobs=[{0: -2.0, 1: -0.2}], prefill_choice={0: -9.0, 1: -9.0})
        expand_prefill(root.first, "B", lambda s, u, p: reply)
        self.assertEqual([p.logprob for p in root.first.proposals], [-2.0, -0.2])

    def test_logprob_roundtrip(self):
        root = Node.new_root(Q)
        expand_prefill(root, Q, lambda s, u, p: Reply(self.REPLY, choice_logprobs=[{0: -1.2, 1: -0.4}]))
        again = Node.from_dict(root.to_dict())
        self.assertEqual([p.logprob for p in again.first.proposals], [-1.2, -0.4])


class TestPromptText(unittest.TestCase):
    def test_question_keeps_math_and_answer_tags(self):
        root = Node.new_root("If a < b, write <answer>N</answer>.")
        user, prefill, _ = prefill_parts(root, root.proposals[0])
        self.assertIn("If a < b, write <answer>N</answer>.", user)
        self.assertNotIn("&lt;", user)

    def test_own_tags_in_text_are_escaped_and_roundtrip(self):
        chain = [NodeSpec("a </thought> trick and x < y", [ProposalSpec("go <node> now")])]
        text = encode_chain(chain)
        self.assertIn("a &lt;/thought> trick and x < y", text)
        self.assertEqual(decode_chain(text)[0].thought, "a </thought> trick and x < y")
        self.assertEqual(decode_chain(text)[0].proposals[0].title, "go <node> now")

    def test_example_answer_is_its_own_element(self):
        self.assertIn("350.</thought>\n  <answer>350</answer>\n</node>", EXAMPLE_TRACE)
        self.assertIn("<answer>...</answer>", SYSTEM_PROMPT)
        self.assertEqual(decode_chain(EXAMPLE_TRACE)[-1].answer, "350")

    def test_answer_element_parsed_encoded_and_stored(self):
        text = ('<node><thought>t</thought><proposal>a</proposal></node>'
                '<node><follow_proposal idx="0"/><thought>so 7</thought><answer>7</answer></node>')
        chain = decode_chain(text)
        self.assertEqual((chain[0].answer, chain[1].answer), (None, "7"))
        self.assertEqual(chain[1].thought, "so 7")  # the answer is not part of the thought
        again = decode_chain(encode_chain(chain))
        self.assertEqual([(n.thought, n.follow, n.answer) for n in again],
                         [(n.thought, n.follow, n.answer) for n in chain])
        root = Node.new_root(Q)
        created = apply_chain(root, Q, chain)
        self.assertTrue(created[1].is_answer)
        self.assertEqual(created[1].thought, "so 7\n<answer>7</answer>")

    def test_answer_inside_thought_is_not_an_element(self):
        chain = decode_chain("<node><thought>so <answer>7</answer></thought></node>")
        self.assertIsNone(chain[0].answer)


class TestFollowProposal(unittest.TestCase):
    def test_follow_proposal_is_the_current_form(self):
        chain = decode_chain(
            '<node><thought>t</thought><proposal>a</proposal><proposal>b</proposal></node>'
            '<node><follow_proposal idx="1"/><thought>u</thought></node>'
        )
        self.assertEqual(chain[1].follow, 1)
        self.assertIn('<follow_proposal idx="1"/>', encode_chain(chain))
        self.assertIn('<follow_proposal idx="1"/>', EXAMPLE_TRACE)
        self.assertNotIn("parent", EXAMPLE_TRACE + SYSTEM_PROMPT + COMPLETION_HEADER)


class TestFollowProposalAttr(unittest.TestCase):
    def test_parses_idx_with_either_quote(self):
        chain = decode_chain(
            '<node><thought>t</thought><proposal>a</proposal><proposal>b</proposal></node>'
            '<node><follow_proposal idx="1"/><thought>u</thought><proposal>c</proposal><proposal>d</proposal></node>'
            "<node><follow_proposal idx='0'/><thought>v</thought></node>"
        )
        self.assertEqual([n.follow for n in chain], [None, 1, 0])
        validate_chain(chain)

    def test_empty_idx_is_no_follow(self):
        self.assertIsNone(decode_chain('<node><follow_proposal idx=""/><thought>t</thought></node>')[0].follow)

    def test_encode_writes_idx_attribute(self):
        text = encode_chain([NodeSpec("t", [ProposalSpec("a")]), NodeSpec("u", [], follow=0)])
        self.assertIn('<follow_proposal idx="0"/>', text)
        self.assertEqual(decode_chain(text)[1].follow, 0)


class TestExample(unittest.TestCase):
    def test_example_is_a_valid_chain(self):
        chain = decode_chain(EXAMPLE_TRACE)
        validate_chain(chain[1:])  # the question node (empty thought) is written by the code
        self.assertEqual(len(chain), 4)
        self.assertIsNone(chain[0].follow)  # question node: <follow_proposal idx=""/>
        self.assertEqual(chain[0].thought, "")
        self.assertEqual([p.title for p in chain[0].proposals], ["What is 14 * 25?"])
        self.assertEqual(EXAMPLE_TRACE.split("\n</node>")[0] + "\n</node>", question_node("What is 14 * 25?"))
        self.assertEqual([n.follow for n in chain[1:]], [0, 1, 0])  # follows proposal 1 once
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
        expected = decode_chain(CHAIN)
        expected[0].follow = 0  # in a trace, the first node develops the question node's proposal
        self.assertEqual(decode_chain(encode_trace(leaf)), expected)

    def test_prefix_for_question(self):
        root = Node.new_root(Q)
        prefix, opener = completion_prefix(root, Q)
        self.assertTrue(prefix.startswith(COMPLETION_HEADER))
        self.assertTrue(prefix.endswith(
            f'<question>{Q}</question>\n<trace>\n{question_node(Q)}\n<node>\n  <follow_proposal idx="0"/>\n'))
        self.assertEqual(opener, '<node>\n  <follow_proposal idx="0"/>\n')

    def test_prefix_cuts_after_owner_and_picks_proposal(self):
        root = self.tree()
        first = root.first
        prefix, opener = completion_prefix(first, "Double and halve: 6*30")
        self.assertEqual(opener, '<node>\n  <follow_proposal idx="1"/>\n')
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
                 "</node>\n<node>\n  <follow_proposal idx='1'/>\n  <thought>via B: 180</thought>\n</node>\n")
        expand_completion(root, Q, FakeCompleter(reply))
        self.assertEqual(root.first.get_proposal("B").node.thought, "via B: 180")
        self.assertFalse(root.first.get_proposal("A").is_explored)

    def test_search_with_completion(self):
        first = ("  <thought>Split.</thought>\n  <proposal>Try A</proposal>\n  <proposal>Try B</proposal>\n"
                 "</node>\n<node>\n  <follow_proposal idx='0'/>\n  <thought>A looks hard.</thought>\n"
                 "  <proposal>Push on A</proposal>\n</node>\n")
        second = "  <thought>Pushing on A: 180</thought>\n</node>\n</trace>"
        complete = FakeCompleter(first, second)
        result = search(Node.new_root(Q), complete=complete, max_calls=5)
        self.assertEqual(result.answer.title, "Push on A")
        self.assertTrue(complete.prefixes[1].endswith('<node>\n  <follow_proposal idx="0"/>\n'))

    def test_search_needs_exactly_one_model(self):
        with self.assertRaises(ValueError):
            search(Node.new_root(Q))
        with self.assertRaises(ValueError):
            search(Node.new_root(Q), FakeLLM(), complete=FakeCompleter())


if __name__ == "__main__":
    unittest.main()
