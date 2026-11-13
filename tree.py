"""Thought-tree for LLM tree search, where everything is a Node.

A Node has:
  - `thought`    the developed reasoning at this step (None only for the root)
  - `proposals`  possible next paths: (title, value, prob, node)
                 `node` points to the child Node once that path is developed
  - `parent`     None for the root
  - `answer`     bool; True means the thought contains the final answer, so the
                 node is terminal and cannot propose anything

The root has no parent and no thought, and exactly one proposal whose title is
the question; that proposal's node holds the initial thought.

Generation loop:  think (develop a node)  ->  propose many paths (title, value, prob)
                  ->  pick a proposal  ->  develop it  ->  ...  ->  answer

A "branch" is just a root where every node has at most one proposal, so
branches and full trees share the same type and can be merged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Literal, Optional

SelectBy = Literal["value", "prob"]


@dataclass(eq=False)
class Proposal:
    title: str
    value: Optional[float] = None  # estimated quality of this path
    prob: Optional[float] = None  # policy probability of exploring this path
    node: Optional[Node] = field(default=None, repr=False)  # set once developed
    logprob: Optional[float] = None  # raw log-prob of the LLM choosing this path (prob = exp(logprob))

    def __post_init__(self) -> None:
        _check_prob(self.prob)

    @property
    def is_explored(self) -> bool:
        return self.node is not None

    def label(self) -> str:
        scores = []
        if self.prob is not None:
            scores.append(f"p={self.prob:.2f}")
        if self.value is not None:
            scores.append(f"v={self.value:.2f}")
        parts = [self.title]
        if scores:
            parts.append(f"[{' '.join(scores)}]")
        if self.node is None:
            parts.append("(proposed)")
        elif self.node.is_answer:
            parts.append("(answer)")
        return " ".join(parts)


@dataclass(eq=False)
class Node:
    thought: Optional[str] = None
    proposals: list[Proposal] = field(default_factory=list)
    parent: Optional[Node] = field(default=None, repr=False)
    answer: bool = False  # thought holds the final answer; no proposals allowed

    # --- construction ------------------------------------------------------
    @classmethod
    def new_root(cls, question: str, thought: Optional[str] = None) -> Node:
        """Root with a single proposal titled with the question.

        If `thought` is given, that proposal is developed into the first node.
        """
        root = cls()
        root.proposals.append(Proposal(title=question))
        if thought is not None:
            root.develop(question, thought)
        return root

    # --- identity / position ----------------------------------------------
    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def root(self) -> Node:
        node = self
        while node.parent is not None:
            node = node.parent
        return node

    @property
    def question(self) -> str:
        return self.root.proposals[0].title

    @property
    def first(self) -> Optional[Node]:
        """The node holding the initial thought (root's single proposal)."""
        return self.root.proposals[0].node

    @property
    def incoming(self) -> Optional[Proposal]:
        """The parent's proposal that points to this node."""
        if self.parent is None:
            return None
        return next(p for p in self.parent.proposals if p.node is self)

    @property
    def title(self) -> Optional[str]:
        return self.incoming.title if self.incoming else None

    @property
    def depth(self) -> int:
        return 0 if self.parent is None else self.parent.depth + 1

    @property
    def is_answer(self) -> bool:
        return self.answer

    @property
    def children(self) -> list[Node]:
        return [p.node for p in self.proposals if p.node is not None]

    @property
    def is_expandable(self) -> bool:
        """Has a thought, has not answered: can propose more paths."""
        return not self.is_root and not self.is_answer

    @property
    def is_branch(self) -> bool:
        """True if no node in this subtree has more than one proposal."""
        return all(len(n.proposals) <= 1 for n in self.walk())

    # --- propose / develop / answer ---------------------------------------
    def propose(self, title: str, value: Optional[float] = None, prob: Optional[float] = None) -> Proposal:
        if self.is_root:
            raise ValueError("The root has exactly one proposal: the question.")
        if self.is_answer:
            raise ValueError("Cannot propose on a node that gave an answer.")
        if self.get_proposal(title) is not None:
            raise ValueError(f"Proposal {title!r} already exists.")
        proposal = Proposal(title=title, value=value, prob=prob)
        self.proposals.append(proposal)
        return proposal

    def propose_many(self, proposals: list[dict], normalize: bool = True) -> list[Proposal]:
        """Add several proposals (dicts with title, optional value/prob), e.g. from one LLM call."""
        new = [self.propose(**p) for p in proposals]
        if normalize:
            self.normalize_probs()
        return new

    def develop(
        self,
        proposal: Proposal | str,
        thought: str,
        answer: bool = False,
        value: Optional[float] = None,
    ) -> Node:
        """Develop one of this node's proposals into a child node with a thought.

        `answer=True` marks the thought as the final answer (terminal node).
        """
        p = self._resolve(proposal)
        if p.node is not None:
            raise ValueError(f"Proposal {p.title!r} is already developed.")
        if value is not None:
            p.value = value
        p.node = Node(thought=thought, parent=self)
        if answer:
            p.node.mark_answer()
        return p.node

    def then(
        self,
        title: str,
        thought: str,
        value: Optional[float] = None,
        prob: Optional[float] = None,
        answer: bool = False,
    ) -> Node:
        """Propose and develop in one go; returns the new child (handy for building branches)."""
        p = self.propose(title, value=value, prob=prob)
        return self.develop(p, thought, answer=answer)

    def mark_answer(self) -> Node:
        """Declare this node's thought as the final answer; it can no longer propose."""
        if self.is_root:
            raise ValueError("The root cannot answer.")
        if self.proposals:
            raise ValueError("Cannot answer on a node that has proposals.")
        self.answer = True
        return self

    # --- policy / selection ------------------------------------------------
    def normalize_probs(self) -> None:
        """Rescale proposal probs to sum to 1. Missing probs count as uniform."""
        for p, q in zip(self.proposals, self.policy().values()):
            p.prob = q

    def policy(self) -> dict[str, float]:
        """Proposal exploration probabilities, normalized (does not mutate)."""
        raw = [p.prob if p.prob is not None else 1.0 for p in self.proposals]
        total, n = sum(raw), len(raw)
        return {p.title: (r / total if total > 0 else 1.0 / n) for p, r in zip(self.proposals, raw)}

    def best_proposal(self, by: SelectBy = "value", only_unexplored: bool = False) -> Optional[Proposal]:
        """Proposal with the highest value or prob; proposals missing that score are ignored."""
        candidates = [p for p in self.proposals if getattr(p, by) is not None]
        if only_unexplored:
            candidates = [p for p in candidates if not p.is_explored]
        return max(candidates, key=lambda p: getattr(p, by), default=None)

    def get_proposal(self, title: str) -> Optional[Proposal]:
        return next((p for p in self.proposals if p.title == title), None)

    def _resolve(self, proposal: Proposal | str) -> Proposal:
        if isinstance(proposal, str):
            found = self.get_proposal(proposal)
            if found is None:
                raise KeyError(f"No proposal titled {proposal!r}.")
            return found
        if proposal not in self.proposals:
            raise ValueError("Proposal does not belong to this node.")
        return proposal

    # --- tree queries (from this node down) -------------------------------
    def walk(self) -> Iterator[Node]:
        """Depth-first, pre-order traversal over developed nodes."""
        yield self
        for child in self.children:
            yield from child.walk()

    def path(self) -> list[Node]:
        """Nodes from the root down to this one."""
        node, out = self, []
        while node is not None:
            out.append(node)
            node = node.parent
        return out[::-1]

    def unexplored(self) -> list[tuple[Node, Proposal]]:
        """(node, proposal) pairs not developed yet: candidates to pick next."""
        return [(n, p) for n in self.walk() for p in n.proposals if not p.is_explored]

    def frontier(self) -> list[Node]:
        """Nodes with a thought but no proposals and no answer yet: need to propose."""
        return [n for n in self.walk() if n.is_expandable and not n.proposals]

    def answer_nodes(self) -> list[Node]:
        return [n for n in self.walk() if n.is_answer]

    def best_answer(self) -> Optional[Node]:
        """Answer node whose incoming proposal has the highest value."""
        scored = [n for n in self.answer_nodes() if n.incoming.value is not None]
        return max(scored, key=lambda n: n.incoming.value, default=None)

    # --- branches ----------------------------------------------------------
    def to_branch(self) -> Node:
        """Copy the root-to-this-node path as a standalone branch (a new root)."""
        path = self.path()
        if len(path) < 2:
            raise ValueError("Need at least the first node to make a branch.")
        branch = Node.new_root(self.question)
        cur = branch
        for parent, child in zip(path, path[1:]):
            src = child.incoming
            if cur.is_root:
                cur = cur.develop(cur.proposals[0], child.thought, value=src.value)
                cur.parent.proposals[0].prob = src.prob
            else:
                cur = cur.then(src.title, child.thought, value=src.value, prob=src.prob)
        if self.answer:
            cur.mark_answer()
        return branch

    def branches(self) -> list[Node]:
        """Every path from the root to a node with no developed children, as branches."""
        return [n.to_branch() for n in self.root.walk() if not n.is_root and not n.children]

    def merge(self, other: Node) -> Node:
        """Merge another tree/branch (same question) into this one, matching proposals by title."""
        if self.is_root != other.is_root:
            raise ValueError("Can only merge nodes at the same position (root with root).")
        if self.is_root and self.question != other.question:
            raise ValueError("Cannot merge trees with different questions.")
        _merge_into(self, other)
        return self

    @classmethod
    def from_branches(cls, branches: list[Node]) -> Node:
        if not branches:
            raise ValueError("Need at least one branch.")
        root = Node.new_root(branches[0].question)
        for b in branches:
            root.merge(b)
        return root

    # --- (de)serialization / display --------------------------------------
    def to_dict(self) -> dict:
        d: dict = {}
        if self.thought is not None:
            d["thought"] = self.thought
        if self.answer:
            d["answer"] = True
        if self.proposals:
            d["proposals"] = [
                {k: v for k, v in (("title", p.title), ("value", p.value), ("prob", p.prob),
                                   ("logprob", p.logprob),
                                   ("node", p.node.to_dict() if p.node else None)) if v is not None}
                for p in self.proposals
            ]
        return d

    @classmethod
    def from_dict(cls, d: dict, parent: Optional[Node] = None) -> Node:
        node = cls(thought=d.get("thought"), answer=bool(d.get("answer", False)), parent=parent)
        for pd in d.get("proposals", []):
            p = Proposal(title=pd["title"], value=pd.get("value"), prob=pd.get("prob"),
                         logprob=pd.get("logprob"))
            if "node" in pd:
                p.node = cls.from_dict(pd["node"], parent=node)
            node.proposals.append(p)
        return node

    def render(self, show_thoughts: bool = False) -> str:
        lines: list[str] = []

        def visit(node: Node, prefix: str) -> None:
            for i, p in enumerate(node.proposals):
                last = i == len(node.proposals) - 1
                lines.append(f"{prefix}{'└── ' if last else '├── '}{p.label()}")
                child_prefix = prefix + ("    " if last else "│   ")
                if p.node is not None:
                    if show_thoughts and p.node.thought:
                        lines.append(f"{child_prefix}  ~ {p.node.thought}")
                    visit(p.node, child_prefix)

        if self.is_root:
            first = self.proposals[0]
            lines.append(f"Q: {first.title}")
            if first.node is not None:
                if show_thoughts and first.node.thought:
                    lines.append(f"  ~ {first.node.thought}")
                visit(first.node, "")
        else:
            lines.append(self.title or "")
            visit(self, "")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()


def _check_prob(prob: Optional[float]) -> None:
    if prob is not None and not 0.0 <= prob <= 1.0:
        raise ValueError(f"prob must be in [0, 1], got {prob}")


def _merge_into(dst: Node, src: Node) -> None:
    if dst.thought is None:
        dst.thought = src.thought
    elif src.thought is not None and dst.thought != src.thought:
        raise ValueError(f"Conflicting thought: {dst.thought!r} vs {src.thought!r}")
    dst.answer = dst.answer or src.answer
    if dst.answer and src.proposals:
        raise ValueError("Cannot merge proposals into a node that gave an answer.")
    for sp in src.proposals:
        dp = dst.get_proposal(sp.title)
        if dp is None:
            dp = Proposal(title=sp.title, value=sp.value, prob=sp.prob, logprob=sp.logprob)
            dst.proposals.append(dp)
        else:
            dp.value = dp.value if dp.value is not None else sp.value
            dp.prob = dp.prob if dp.prob is not None else sp.prob
            dp.logprob = dp.logprob if dp.logprob is not None else sp.logprob
        if sp.node is not None:
            if dp.node is None:
                dp.node = Node(parent=dst)
            _merge_into(dp.node, sp.node)
    if dst.answer and dst.proposals:
        raise ValueError("Merged node has both an answer and proposals.")
