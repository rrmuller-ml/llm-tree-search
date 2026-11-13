"""LLM-in-the-loop for the thought tree: XML codec + search.

The LLM writes a depth-first chain of nodes. Each node first lists candidate
proposals; the *next* node then says, with <parent>, which of those proposals
it develops. Proposals not picked stay unexplored in the tree.

    <node>
      <thought>Split into easier parts.</thought>
      <proposal idx="0">Distribute over 15: 12*10 + 12*5</proposal>
      <proposal idx="1">Double and halve: 6*30</proposal>
    </node>
    <node>
      <parent>0</parent>
      <thought>12*10 = 120 and 12*5 = 60, so 12*15 = 180.</thought>
    </node>

- The first node of a reply develops the proposal the code asked for (no <parent>).
- <parent>k</parent> refers to the proposal with idx k of the previous node
  (idx defaults to the proposal's 0-based position).
- A node with no proposals is the answer: its thought is the final answer and
  nothing follows it.
- A reply may stop on a node with open proposals; search resumes there later.
- A truncated reply keeps every complete <node> and drops the unfinished one.

The LLM does not write scores. A proposal's `prob` is meant to come from the
log-prob of the LLM writing that <parent>, and its `value` from a trained
critic; both are left unset here. `search` takes a `score` function so either
can be plugged in later.

The LLM is any function `llm(system, prompt) -> str`, so any provider works.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from tree import Node, Proposal

LLM = Callable[[str, str], str]
Score = Callable[[Node, Proposal], float]

SYSTEM_PROMPT = """\
You solve problems by writing a chain of reasoning steps as XML nodes.

Each node has:
- <thought>: your reasoning for this step.
- <proposal idx="i">title</proposal>: two to four possible next steps, each a
  short title. idx counts from 0.
When a thought states the final answer, write that node with no proposals.

After a node with proposals, write the next node starting with
<parent>i</parent>: the idx of the proposal it develops. Keep going depth-first
until a node gives the final answer. Write only <node> elements.

Example:
<node>
  <thought>Split into easier parts.</thought>
  <proposal idx="0">Distribute over 15: 12*10 + 12*5</proposal>
  <proposal idx="1">Double and halve: 6*30</proposal>
</node>
<node>
  <parent>0</parent>
  <thought>12*10 = 120 and 12*5 = 60, so 12*15 = 180.</thought>
</node>
"""


class DecodeError(ValueError):
    """The LLM output could not be decoded or breaks the chain rules."""


# --- parsed form -----------------------------------------------------------
@dataclass
class ProposalSpec:
    title: str
    idx: Optional[int] = None


@dataclass
class NodeSpec:
    thought: str
    proposals: list[ProposalSpec] = field(default_factory=list)
    parent: Optional[int] = None  # idx of the previous node's proposal this node develops

    @property
    def is_answer(self) -> bool:
        return not self.proposals

    def idx_of(self, k: int) -> int:
        p = self.proposals[k]
        return p.idx if p.idx is not None else k

    def position_of(self, idx: int) -> Optional[int]:
        """0-based position of the proposal with this idx, or None."""
        return next((k for k in range(len(self.proposals)) if self.idx_of(k) == idx), None)


# --- decode ----------------------------------------------------------------
_NODE = re.compile(r"<node\b[^>]*>(.*?)</node\s*>", re.S | re.I)
_PARENT = re.compile(r"<parent\b[^>]*>\s*(-?\d+)\s*</parent\s*>", re.S | re.I)
_THOUGHT = re.compile(r"<thought\b[^>]*>(.*?)</thought\s*>", re.S | re.I)
_PROPOSAL = re.compile(r"<proposal\b([^>]*)>(.*?)</proposal\s*>", re.S | re.I)
_IDX = re.compile(r"\bidx\s*=\s*[\"']\s*([^\"']*?)\s*[\"']", re.I)


def _text(s: str) -> str:
    return html.unescape(s).strip()


def decode_chain(text: str) -> list[NodeSpec]:
    """Parse every complete <node> in `text`. Text outside nodes and an unfinished
    trailing node (truncated output) are ignored."""
    chain: list[NodeSpec] = []
    for body in _NODE.findall(text):
        m = _THOUGHT.search(body)
        if not m:
            raise DecodeError(f"node {len(chain)} has no <thought>")
        proposals = []
        for attrs, title in _PROPOSAL.findall(body):
            idx = _IDX.search(attrs)
            if idx and not re.fullmatch(r"-?\d+", idx.group(1)):
                raise DecodeError(f"idx={idx.group(1)!r} is not an integer")
            proposals.append(ProposalSpec(_text(title), int(idx.group(1)) if idx else None))
        parent = _PARENT.search(body)
        chain.append(NodeSpec(_text(m.group(1)), proposals, int(parent.group(1)) if parent else None))
    return chain


def validate_chain(chain: list[NodeSpec]) -> None:
    """Check the chain rules; raises DecodeError on the first violation."""
    if not chain:
        raise DecodeError("no complete <node> in the output")
    for i, n in enumerate(chain):
        if not n.thought:
            raise DecodeError(f"node {i} has an empty <thought>")
        titles = [p.title for p in n.proposals]
        if any(not t for t in titles):
            raise DecodeError(f"node {i} has a proposal with an empty title")
        if len(set(titles)) != len(titles):
            raise DecodeError(f"node {i} has duplicate proposal titles")
        idxs = [n.idx_of(k) for k in range(len(n.proposals))]
        if len(set(idxs)) != len(idxs):
            raise DecodeError(f"node {i} has duplicate proposal idx")
        if i == 0:
            continue
        prev = chain[i - 1]
        if prev.is_answer:
            raise DecodeError(f"node {i} follows node {i - 1}, which has no proposals (it is the answer)")
        if n.parent is None:
            raise DecodeError(f"node {i} has no <parent>")
        if prev.position_of(n.parent) is None:
            raise DecodeError(f"node {i}: <parent>{n.parent}</parent> is not a proposal of node {i - 1}")


# --- encode ----------------------------------------------------------------
def encode_chain(chain: list[NodeSpec]) -> str:
    """Inverse of decode_chain (useful for examples, logging and tests)."""
    out = []
    for n in chain:
        lines = ["<node>"]
        if n.parent is not None:
            lines.append(f"  <parent>{n.parent}</parent>")
        lines.append(f"  <thought>{html.escape(n.thought, quote=False)}</thought>")
        for k, p in enumerate(n.proposals):
            lines.append(f'  <proposal idx="{n.idx_of(k)}">{html.escape(p.title, quote=False)}</proposal>')
        lines.append("</node>")
        out.append("\n".join(lines))
    return "\n".join(out)


def encode_context(node: Node, proposal: Proposal | str) -> str:
    """Prompt for one call: the question, the path so far, and the proposal to develop."""
    p = node._resolve(proposal)
    lines = [f"<question>{html.escape(node.question, quote=False)}</question>"]
    steps = [n for n in node.path() if not n.is_root]
    if steps:
        lines.append("<path>")
        for n in steps:
            title = html.escape(n.title or "", quote=True)
            lines.append(f'  <step title="{title}">{html.escape(n.thought or "", quote=False)}</step>')
        lines.append("</path>")
    if node.is_root:
        lines.append("Start reasoning about the question. Your first node has no <parent>.")
    else:
        lines.append(f"<develop>{html.escape(p.title, quote=False)}</develop>")
        lines.append("Continue from the path by developing the step in <develop>. "
                     "Your first node develops it and has no <parent>.")
    return "\n".join(lines)


# --- apply -----------------------------------------------------------------
def apply_chain(node: Node, proposal: Proposal | str, chain: list[NodeSpec]) -> list[Node]:
    """Develop `proposal` of `node` with the chain. Validates everything first,
    so a bad chain leaves the tree unchanged. A node without proposals is marked
    as the answer. Returns the new nodes in order."""
    p = node._resolve(proposal)
    if p.is_explored:
        raise ValueError(f"Proposal {p.title!r} is already developed.")
    validate_chain(chain)
    created: list[Node] = []
    cur_node, cur_prop = node, p
    for i, spec in enumerate(chain):
        child = cur_node.develop(cur_prop, spec.thought, answer=spec.is_answer)
        created.append(child)
        if spec.proposals:
            new = [child.propose(s.title) for s in spec.proposals]
            if i + 1 < len(chain):
                cur_prop = new[spec.position_of(chain[i + 1].parent)]
        cur_node = child
    return created


# --- loop ------------------------------------------------------------------
def expand(node: Node, proposal: Proposal | str, llm: LLM) -> list[Node]:
    """One LLM call: develop `proposal` with a depth-first chain of nodes."""
    text = llm(SYSTEM_PROMPT, encode_context(node, proposal))
    return apply_chain(node, proposal, decode_chain(text))


def depth_first(node: Node, proposal: Proposal) -> float:
    """Default score: deepest node first, then the earliest proposal of that node.
    Gives plain DFS backtracking when nothing better is available."""
    return node.depth - node.proposals.index(proposal) / (len(node.proposals) + 1)


def pick_next(root: Node, score: Score = depth_first, skip: frozenset[int] = frozenset()) -> Optional[tuple[Node, Proposal]]:
    """Unexplored (node, proposal) with the highest score. `skip` holds id()s of
    proposals to leave out (e.g. ones whose call failed)."""
    candidates = [(n, p) for n, p in root.unexplored() if id(p) not in skip]
    if not candidates:
        return None
    return max(candidates, key=lambda np: score(*np))


@dataclass
class SearchResult:
    answer: Optional[Node]
    calls: int
    errors: list[str]


def search(root: Node, llm: LLM, max_calls: int = 10, score: Score = depth_first) -> SearchResult:
    """Call the LLM until a node gives the answer or the call budget runs out.

    Each call starts from the unexplored proposal with the highest `score`
    (the question on the first call). The default is DFS backtracking; pass a
    score built from log-probs or a critic to change the order. A proposal whose
    call fails to decode is not retried.
    """
    errors: list[str] = []
    failed: set[int] = set()
    calls = 0
    while calls < max_calls:
        if root.answer_nodes():
            break
        nxt = pick_next(root, score, frozenset(failed))
        if nxt is None:
            break
        calls += 1
        try:
            expand(*nxt, llm)
        except DecodeError as e:
            failed.add(id(nxt[1]))
            errors.append(f"{nxt[1].title!r}: {e}")
    answers = root.answer_nodes()
    return SearchResult(answers[0] if answers else None, calls, errors)
