"""Tests for rl_train.py helpers (no network)."""

import unittest

from rl_train import decision_kinds, group_advantages, make_datum, reward, token_mask

# Qwen3.6 token pieces for a continuation after the prefill '...<follow_proposal idx="0"/>\n'
# (checked with the real tokenizer).
PREFILL = '<trace>\n<node>\n  <follow_proposal idx=""/>\n  <thought></thought>\n  <proposal idx="0">Q</proposal>\n</node>\n<node>\n  <follow_proposal idx="0"/>\n'
PIECES = [' ', ' <', 'thought', '>x', '</', 'thought', '>', '\n', ' ', ' <', 'proposal', ' idx', '="', '0', '">', 'a',
          '</', 'proposal', '>', '\n', ' ', ' <', 'proposal', ' idx', '="', '1', '">', 'b', '</', 'proposal', '>', '\n',
          '</', 'node', '>', '\n', '<', 'node', '>', '\n', ' ', ' <', 'follow', '_prop', 'osal', ' idx', '="', '1',
          '"/>', '\n', ' ', ' <', 'thought', '>', 'y', '</', 'thought', '>', '\n', ' ', ' <', 'answer', '>', '3',
          '</', 'answer', '>', '\n', '</', 'node', '>', '\n', '</', 'trace', '>']
REPLY = "".join(PIECES)


class TestDecisions(unittest.TestCase):
    def test_kinds_at_each_decision(self):
        kinds = decision_kinds(PIECES, PREFILL)
        marked = [(i, PIECES[i], k) for i, k in enumerate(kinds) if k]
        self.assertEqual([(p, k) for _, p, k in marked], [
            ("proposal", "kind"),   # first node: propose, not answer
            (" ", "more"),          # after proposal 0: another proposal
            ("</", "more"),         # after proposal 1: close the node
            ("<", "next"),          # after </node>: another node
            ("1", "follow"),        # follow proposal 1
            ("answer", "kind"),     # second node: answer
            ("</", "next"),         # after the answer node: end the trace
        ])

    def test_proposal_idx_digits_are_not_decisions(self):
        kinds = decision_kinds(PIECES, PREFILL)
        idx_positions = [i for i in range(1, len(PIECES)) if PIECES[i - 1] == '="' and PIECES[i - 2] == ' idx'
                         and PIECES[i - 3] == 'proposal']
        self.assertTrue(idx_positions)
        self.assertTrue(all(kinds[i] is None for i in idx_positions))

    def test_follow_needs_a_digit(self):
        pieces = ['<follow_proposal idx="', 'x']
        self.assertEqual(decision_kinds(pieces), [None, None])


class TestReward(unittest.TestCase):
    def test_valid_and_correct(self):
        self.assertEqual(reward(REPLY, "3"), (2.0, True, True, 3))
        self.assertEqual(reward(REPLY, 4), (1.0, True, False, 3))  # format only

    def test_invalid_format_gets_nothing(self):
        r, valid, correct, pred = reward(REPLY.replace("</trace>", ""), 3)
        self.assertEqual((r, valid, correct), (0.0, False, False))

    def test_group_advantages(self):
        self.assertEqual(group_advantages([1.0, 0.0, 0.0, 1.0]), [0.5, -0.5, -0.5, 0.5])
        self.assertEqual(group_advantages([1.0, 1.0]), [0.0, 0.0])


class FakeTokenizer:
    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids, **kw):
        return "".join(self.pieces[i] for i in ids)


class TestDatum(unittest.TestCase):
    def rollout(self):
        from tinker import types

        prompt = types.ModelInput.from_ints([101, 102, 103, 104])
        return {"prompt": prompt, "prefill": PREFILL, "tokens": list(range(len(PIECES))),
                "logprobs": [-0.5] * len(PIECES)}

    def test_mask_and_datum_line_up(self):
        r = self.rollout()
        mask, counts = token_mask(r, FakeTokenizer(PIECES), "decisions")
        self.assertEqual(counts, {"kind": 2, "more": 2, "next": 2, "follow": 1})
        d = make_datum(r, 0.5, mask)
        adv = d.loss_fn_inputs["advantages"].to_torch().tolist()
        target = d.loss_fn_inputs["target_tokens"].to_torch().tolist()
        self.assertEqual(len(adv), d.model_input.length)
        self.assertEqual(target[3:], r["tokens"])  # prompt length 4 -> 3 padded positions
        self.assertEqual(sum(1 for a in adv if a), 7)
        follow_pos = 3 + PIECES.index("1", PIECES.index("follow"))
        self.assertEqual(adv[follow_pos], 0.5)

    def test_all_tokens(self):
        r = self.rollout()
        mask, _ = token_mask(r, FakeTokenizer(PIECES), "all")
        self.assertEqual(mask, [1.0] * len(PIECES))


if __name__ == "__main__":
    unittest.main()
