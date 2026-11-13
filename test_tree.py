import unittest

from tree import Node, Proposal

Q = "What is 12 * 15?"
FIRST = "Split into easier parts."


def make_tree() -> Node:
    root = Node.new_root(Q, FIRST)
    root.first.propose_many([
        {"title": "Distribute", "prob": 0.6, "value": 0.8},
        {"title": "Double/halve", "prob": 0.3, "value": 0.9},
        {"title": "Square", "prob": 0.1},
    ])
    return root


def make_branch() -> Node:
    root = Node.new_root(Q, FIRST)
    root.first.then("Tens", "10*15 + 2*15", prob=0.5).then("Sum", "150 + 30", value=0.9, answer=True)
    return root


class TestRoot(unittest.TestCase):
    def test_root_has_single_question_proposal_and_no_parent(self):
        root = Node.new_root(Q)
        self.assertTrue(root.is_root)
        self.assertIsNone(root.parent)
        self.assertIsNone(root.thought)
        self.assertEqual([p.title for p in root.proposals], [Q])
        self.assertEqual(root.question, Q)
        self.assertIsNone(root.first)

    def test_root_with_thought_develops_first_node(self):
        root = Node.new_root(Q, FIRST)
        self.assertEqual(root.first.thought, FIRST)
        self.assertIs(root.first.parent, root)
        self.assertEqual(root.first.title, Q)
        self.assertEqual(root.first.question, Q)

    def test_root_cannot_get_more_proposals_or_answer(self):
        root = Node.new_root(Q)
        with self.assertRaises(ValueError):
            root.propose("another question")
        with self.assertRaises(ValueError):
            root.mark_answer()

    def test_develop_root_question_later(self):
        root = Node.new_root(Q)
        first = root.develop(Q, FIRST)
        self.assertIs(root.first, first)


class TestProposeDevelopAnswer(unittest.TestCase):
    def test_propose_creates_unexplored_proposal(self):
        p = Node.new_root(Q, FIRST).first.propose("Idea", value=0.4, prob=0.5)
        self.assertIsInstance(p, Proposal)
        self.assertEqual((p.title, p.value, p.prob), ("Idea", 0.4, 0.5))
        self.assertFalse(p.is_explored)

    def test_duplicate_proposal_rejected(self):
        first = Node.new_root(Q, FIRST).first
        first.propose("Idea")
        with self.assertRaises(ValueError):
            first.propose("Idea")

    def test_develop_links_proposal_to_child(self):
        first = make_tree().first
        child = first.develop("Distribute", "12*10 + 12*5", value=0.7)
        p = first.get_proposal("Distribute")
        self.assertIs(p.node, child)
        self.assertIs(child.parent, first)
        self.assertIs(child.incoming, p)
        self.assertEqual(child.title, "Distribute")
        self.assertEqual(p.value, 0.7)
        with self.assertRaises(ValueError):
            first.develop(p, "again")

    def test_develop_unknown_proposal(self):
        first = make_tree().first
        with self.assertRaises(KeyError):
            first.develop("nope", "x")
        with self.assertRaises(ValueError):
            first.develop(Proposal("foreign"), "x")

    def test_answer_is_terminal(self):
        child = make_tree().first.develop("Square", "12*12 + 36", answer=True)
        self.assertTrue(child.is_answer)
        self.assertFalse(child.is_expandable)
        with self.assertRaises(ValueError):
            child.propose("more")

    def test_cannot_answer_with_proposals(self):
        with self.assertRaises(ValueError):
            make_tree().first.mark_answer()

    def test_prob_validated(self):
        with self.assertRaises(ValueError):
            Node.new_root(Q, FIRST).first.propose("x", prob=1.5)

    def test_then_chains(self):
        leaf = make_branch().first.children[0].children[0]
        self.assertTrue(leaf.answer)
        self.assertEqual(leaf.depth, 3)
        self.assertEqual([n.title for n in leaf.path()], [None, Q, "Tens", "Sum"])


class TestPolicyAndValue(unittest.TestCase):
    def test_propose_many_normalizes(self):
        first = Node.new_root(Q, FIRST).first
        first.propose_many([{"title": "a", "prob": 0.2}, {"title": "b", "prob": 0.6}])
        self.assertAlmostEqual(first.get_proposal("b").prob, 0.75)
        self.assertAlmostEqual(sum(p.prob for p in first.proposals), 1.0)

    def test_missing_probs_are_uniform(self):
        first = Node.new_root(Q, FIRST).first
        first.propose_many([{"title": t} for t in "abcd"])
        for p in first.proposals:
            self.assertAlmostEqual(p.prob, 0.25)

    def test_policy_does_not_mutate(self):
        first = Node.new_root(Q, FIRST).first
        first.propose("a", prob=0.2)
        first.propose("b", prob=0.2)
        self.assertEqual(first.policy(), {"a": 0.5, "b": 0.5})
        self.assertEqual(first.get_proposal("a").prob, 0.2)

    def test_best_proposal(self):
        first = make_tree().first
        self.assertEqual(first.best_proposal(by="value").title, "Double/halve")
        self.assertEqual(first.best_proposal(by="prob").title, "Distribute")
        first.develop("Double/halve", "t")
        self.assertEqual(first.best_proposal(by="value", only_unexplored=True).title, "Distribute")

    def test_best_proposal_none_without_scores(self):
        first = Node.new_root(Q, FIRST).first
        first.propose("a")
        self.assertIsNone(first.best_proposal())

    def test_best_answer(self):
        tree = make_tree()
        tree.first.develop("Distribute", "t", answer=True, value=0.7)
        tree.first.develop("Double/halve", "t", answer=True, value=0.95)
        self.assertEqual(tree.best_answer().title, "Double/halve")


class TestQueries(unittest.TestCase):
    def test_unexplored_frontier_answers(self):
        tree = make_tree()
        tree.first.develop("Distribute", "t")
        tree.first.develop("Square", "t", answer=True)
        self.assertEqual([p.title for _, p in tree.unexplored()], ["Double/halve"])
        self.assertEqual([n.title for n in tree.frontier()], ["Distribute"])
        self.assertEqual([n.title for n in tree.answer_nodes()], ["Square"])

    def test_dict_roundtrip(self):
        tree = make_tree()
        tree.first.develop("Distribute", "t", answer=True, value=0.9)
        again = Node.from_dict(tree.to_dict())
        self.assertEqual(again.to_dict(), tree.to_dict())
        self.assertIs(again.first.parent, again)
        self.assertEqual(again.first.children[0].title, "Distribute")

    def test_render(self):
        out = make_tree().render(show_thoughts=True)
        self.assertIn(f"Q: {Q}", out)
        self.assertIn(f"~ {FIRST}", out)
        self.assertIn("Distribute [p=0.60 v=0.80] (proposed)", out)


class TestBranches(unittest.TestCase):
    def test_branch_is_a_node(self):
        b = make_branch()
        self.assertIsInstance(b, Node)
        self.assertTrue(b.is_branch)
        self.assertFalse(make_tree().is_branch)

    def test_to_branch_copies_path(self):
        tree = make_tree()
        leaf = tree.first.develop("Distribute", "t1").then("Sum", "t2", value=0.9, answer=True)
        b = leaf.to_branch()
        self.assertTrue(b.is_branch)
        self.assertIsNot(b, tree)
        titles = [n.title for n in b.walk()]
        self.assertEqual(titles, [None, Q, "Distribute", "Sum"])
        self.assertTrue(b.answer_nodes()[0].answer)
        self.assertEqual(b.first.get_proposal("Distribute").value, 0.8)

    def test_merge_reuses_matching_proposals(self):
        tree = make_tree()
        b = Node.new_root(Q, FIRST)
        b.first.then("Distribute", "12*10 + 12*5", answer=True)
        tree.merge(b)
        self.assertEqual(len(tree.first.proposals), 3)
        node = tree.first.get_proposal("Distribute").node
        self.assertTrue(node.answer)
        self.assertEqual(tree.first.get_proposal("Distribute").value, 0.8)  # kept existing score

    def test_merge_adds_new_paths(self):
        tree = make_tree()
        tree.merge(make_branch())
        self.assertEqual(len(tree.first.proposals), 4)
        self.assertTrue(tree.first.get_proposal("Tens").node.children[0].answer)

    def test_merge_conflicts(self):
        with self.assertRaises(ValueError):
            make_tree().merge(Node.new_root("other question", FIRST))
        with self.assertRaises(ValueError):
            make_tree().merge(Node.new_root(Q, "different first thought"))
        tree = make_tree()
        tree.first.develop("Square", "original")
        b = Node.new_root(Q, FIRST)
        b.first.then("Square", "conflicting")
        with self.assertRaises(ValueError):
            tree.merge(b)
        with self.assertRaises(ValueError):
            tree.merge(tree.first)  # not a root

    def test_branches_and_from_branches(self):
        tree = make_tree()
        tree.first.develop("Distribute", "a", answer=True)
        tree.first.develop("Double/halve", "b")
        branches = tree.branches()
        self.assertEqual(len(branches), 2)
        self.assertTrue(all(b.is_branch for b in branches))
        rebuilt = Node.from_branches(branches)
        self.assertEqual({p.title for p in rebuilt.first.proposals}, {"Distribute", "Double/halve"})


if __name__ == "__main__":
    unittest.main()
