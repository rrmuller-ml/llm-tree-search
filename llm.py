"""LLM-in-the-loop for the thought tree: XML codec + search.

The LLM writes a depth-first chain of nodes. Each node first lists candidate
proposals; the *next* node then says, with <follow_proposal>, which of those proposals
it develops. Proposals not picked stay unexplored in the tree.

    <node>
      <follow_proposal idx=""/>
      <thought></thought>
      <proposal idx="0">What is 14 * 25?</proposal>
    </node>
    <node>
      <follow_proposal idx="0"/>
      <thought>14 * 25 is easier to compute in parts.</thought>
      <proposal idx="0">Distribute over 25: 10*25 + 4*25</proposal>
      <proposal idx="1">Use 4*25 = 100: 14*25 = 3.5*100</proposal>
    </node>
    <node>
      <follow_proposal idx="1"/>
      <thought>4*25 = 100 and 14 = 3.5*4, so 14*25 = 3.5*100.</thought>
      <proposal idx="0">Multiply: 3.5*100</proposal>
      <proposal idx="1">Check another way: 7*50</proposal>
    </node>
    <node>
      <follow_proposal idx="0"/>
      <thought>3.5*100 = 350, so 14 * 25 = 350.</thought>
      <answer>350</answer>
    </node>

- Every trace starts with a question node written by the code, mirroring the
  tree's root: <follow_proposal idx=""/>, an empty thought, and the question as its only
  proposal.
- Every node after it starts with <follow_proposal idx="k"/>: the proposal with idx k of
  the previous node (idx defaults to the proposal's 0-based position). The code
  writes the <follow_proposal> of the first node of each reply (the proposal it chose to
  develop); the model writes the rest.
- A node with no proposals is the answer: its thought is the final answer and
  nothing follows it.
- A reply may stop on a node with open proposals; search resumes there later.
- A truncated reply keeps every complete <node> and drops the unfinished one.

The LLM does not write scores. A proposal's `prob` is meant to come from the
log-prob of the LLM writing that <follow_proposal>, and its `value` from a trained
critic; both are left unset here. `search` takes a `score` function so either
can be plugged in later.

The model is called with assistant prefill: `prefill(system, user, prefix)`
continues a reply that is pre-started with the trace so far, cut after the node
that owns the proposal to explore, plus an open `<node><follow_proposal idx="k"/>`
choosing that proposal; the model writes the rest (`expand_prefill`).
"""

from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from tree import Node, Proposal

PrefillLLM = Callable[[str, str, str], "str | Reply"]  # (system, user, assistant prefill) -> continuation
Score = Callable[[Node, Proposal], float]

EXAMPLE_QUESTION = "What is 14 * 25?"


def question_node(question: str) -> str:
    """The node every trace starts with, mirroring the tree's root: an empty
    <follow_proposal>, an empty thought, and the question as its only proposal. Written by
    the code, not the model."""
    return (
        "<node>\n"
        '  <follow_proposal idx=""/>\n'
        "  <thought></thought>\n"
        f'  <proposal idx="0">{_xml_text(question)}</proposal>\n'
        "</node>"
    )


EXAMPLE_TRACE = """\
<node>
  <follow_proposal idx=""/>
  <thought></thought>
  <proposal idx="0">What is 14 * 25?</proposal>
</node>
<node>
  <follow_proposal idx="0"/>
  <thought>14 * 25 is easier to compute in parts.</thought>
  <proposal idx="0">Distribute over 25: 10*25 + 4*25</proposal>
  <proposal idx="1">Use 4*25 = 100: 14*25 = 3.5*100</proposal>
</node>
<node>
  <follow_proposal idx="1"/>
  <thought>4*25 = 100 and 14 = 3.5*4, so 14*25 = 3.5*100.</thought>
  <proposal idx="0">Multiply: 3.5*100</proposal>
  <proposal idx="1">Check another way: 7*50</proposal>
</node>
<node>
  <follow_proposal idx="0"/>
  <thought>3.5*100 = 350, so 14 * 25 = 350.</thought>
  <answer>350</answer>
</node>"""

SYSTEM_PROMPT = f"""\
You solve problems by writing a chain of reasoning steps as XML nodes.

Each node has:
- <thought>: one short reasoning step, at most a few sentences. Do one thing
  per node (one computation, one case, one idea) and move the rest of the work
  into the next nodes.
- <proposal idx="i">title</proposal>: at least two (two to four) different
  possible next steps, each a short title. idx counts from 0.
When a thought reaches the final answer, write that node with no proposals and
add the answer after its thought as <answer>...</answer>.

The trace starts with a question node: <follow_proposal idx=""/>, an empty thought, and
the question as its only proposal. Every node after it starts with
<follow_proposal idx="i"/>: the idx of the proposal it develops, chosen among the
proposals of the node directly above it (not a node number). Keep going
depth-first until a node gives the final answer. Write only <node> elements.

Example for "{EXAMPLE_QUESTION}":
{EXAMPLE_TRACE}
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
    follow: Optional[int] = None  # <follow_proposal idx>: the previous node's proposal this node develops
    answer: Optional[str] = None  # text of the node's <answer> element (outside <thought>)

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
_ANSWER_EL = re.compile(r"<answer\b[^>]*>(.*?)</answer\s*>", re.S | re.I)
_FOLLOW = re.compile(r"<follow_proposal\b[^>]*?\bidx\s*=\s*[\"']\s*(-?\d+)\s*[\"']", re.I)
FOLLOW_TAG = re.compile(r"<follow_proposal\b", re.I)


def _follow_of(text: str) -> Optional[int]:
    """The idx in <follow_proposal idx="k"/>; None if absent or empty (the
    question node has <follow_proposal idx=""/>)."""
    m = _FOLLOW.search(text)
    return int(m.group(1)) if m else None
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
        follow = _follow_of(body)
        outside = body[: m.start()] + body[m.end():]
        ans = _ANSWER_EL.search(outside)
        chain.append(NodeSpec(_text(m.group(1)), proposals, follow, _text(ans.group(1)) if ans else None))
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
        if n.follow is None:
            raise DecodeError(f"node {i} has no <follow_proposal>")
        if prev.position_of(n.follow) is None:
            raise DecodeError(f'node {i}: <follow_proposal idx="{n.follow}"/> is not a proposal of node {i - 1}')


class PartialChain(DecodeError):
    """Only the start of the reply was valid; that part was added to the tree."""

    def __init__(self, message: str, created: list):
        super().__init__(message)
        self.created = created


def _node_error(chain: list[NodeSpec], i: int) -> Optional[str]:
    try:
        validate_chain(chain[: i + 1])
    except DecodeError as e:
        return str(e)
    return None


def valid_prefix(chain: list[NodeSpec]) -> tuple[list[NodeSpec], Optional[str]]:
    """Repair what is unambiguous, then keep the longest valid start of the chain.

    Repair: after a node with exactly one proposal there is nothing to choose,
    so the next node's <follow_proposal> is set to that proposal whatever it said.
    Returns (valid nodes, error of the first invalid node or None)."""
    chain = [NodeSpec(n.thought, list(n.proposals), n.follow, n.answer) for n in chain]
    for i in range(1, len(chain)):
        prev = chain[i - 1]
        if len(prev.proposals) == 1:
            chain[i].follow = prev.idx_of(0)
    for i in range(len(chain)):
        err = _node_error(chain, i)
        if err:
            return chain[:i], err
    return chain, None if chain else "no complete <node> in the output"


_OWN_TAG = re.compile(r"<(?=/?(?:node|thought|proposal|follow_proposal|question|trace|step|develop)\b)", re.I)


def _xml_text(s: str) -> str:
    """Text placed inside our XML: kept as the model writes it (math like
    `a < b` and <answer> tags stay readable), except that our own structural
    tags are escaped so they can't break the structure. decode_chain unescapes."""
    return _OWN_TAG.sub("&lt;", s)


# --- encode ----------------------------------------------------------------
def encode_chain(chain: list[NodeSpec]) -> str:
    """Inverse of decode_chain (useful for examples, logging and tests)."""
    out = []
    for n in chain:
        lines = ["<node>"]
        if n.follow is not None:
            lines.append(f'  <follow_proposal idx="{n.follow}"/>')
        lines.append(f"  <thought>{_xml_text(n.thought)}</thought>")
        for k, p in enumerate(n.proposals):
            lines.append(f'  <proposal idx="{n.idx_of(k)}">{_xml_text(p.title)}</proposal>')
        if n.answer is not None:
            lines.append(f"  <answer>{_xml_text(n.answer)}</answer>")
        lines.append("</node>")
        out.append("\n".join(lines))
    return "\n".join(out)


def encode_trace(node: Node) -> str:
    """The path from the first node down to `node` as XML nodes. Each node lists
    all its proposals (idx = position), and <follow_proposal> links each to the previous;
    the first node has <follow_proposal idx="0"/> (the question node's only proposal)."""
    steps = [n for n in node.path() if not n.is_root]
    chain = [
        NodeSpec(
            thought=n.thought or "",
            proposals=[ProposalSpec(p.title) for p in n.proposals],
            follow=0 if i == 0 else n.parent.proposals.index(n.incoming),
        )
        for i, n in enumerate(steps)
    ]
    return encode_chain(chain)


def prefill_parts(node: Node, proposal: Proposal | str) -> tuple[str, str, str]:
    """For chat models that can continue a pre-started reply: (user message,
    assistant prefill, opener). The prefill is the trace cut after `node`, ending
    in an open node whose <follow_proposal> picks `proposal`."""
    p = node._resolve(proposal)
    opener = f'<node>\n  <follow_proposal idx="{node.proposals.index(p)}"/>\n'
    trace = question_node(node.question) + "\n" + (encode_trace(node) + "\n" if not node.is_root else "")
    user = (
        f"<question>{_xml_text(node.question)}</question>\n"
        "Write the trace inside <trace> ... </trace>."
    )
    prefill = "<trace>\n" + trace + opener
    return user, prefill, opener


def decode_continuation(reply: str, opener: str) -> list[NodeSpec]:
    """Decode a continuation of `opener`. Accepts a reply that continues inside
    the open node, or one that rewrites the open node from <node> (its <follow_proposal>,
    if any, must match the opener's)."""
    reply = reply.split("</trace>")[0]
    if not reply.lstrip().lower().startswith("<node"):
        return decode_chain(opener + reply)
    chain = decode_chain(reply)
    wanted = _follow_of(opener)
    if chain and wanted is not None and chain[0].follow is not None and chain[0].follow != wanted:
        raise DecodeError(f"reply develops proposal {chain[0].follow}, expected {wanted}")
    return chain


# --- apply -----------------------------------------------------------------
def apply_chain(node: Node, proposal: Proposal | str, chain: list[NodeSpec]) -> list[Node]:
    """Develop `proposal` of `node` with the chain. Validates everything first,
    so a bad chain leaves the tree unchanged. A node without proposals is marked
    as the answer; its <answer> element, if any, is stored at the end of its
    thought. Returns the new nodes in order."""
    p = node._resolve(proposal)
    if p.is_explored:
        raise ValueError(f"Proposal {p.title!r} is already developed.")
    validate_chain(chain)
    created: list[Node] = []
    cur_node, cur_prop = node, p
    for i, spec in enumerate(chain):
        thought = spec.thought
        if spec.answer is not None:  # the tree has no answer field: keep it with the thought
            thought = f"{thought}\n<answer>{spec.answer}</answer>"
        child = cur_node.develop(cur_prop, thought, answer=spec.is_answer)
        created.append(child)
        if spec.proposals:
            new = [child.propose(s.title) for s in spec.proposals]
            if i + 1 < len(chain):
                cur_prop = new[spec.position_of(chain[i + 1].follow)]
        cur_node = child
    return created


# --- loop ------------------------------------------------------------------
ChoiceLogprobs = dict[int, float]  # proposal idx -> log-prob of the model writing that idx


@dataclass
class Reply:
    """A model reply with the log-probs of its <follow_proposal> choices.

    Models may return a plain string instead; then no log-probs are recorded.
    `choice_logprobs` has one entry per <follow_proposal> tag in `text`, in order: the
    distribution over idx digits at the position right after that tag (None
    where unknown). `prefill_choice` is the same for the <follow_proposal> that a
    pre-started reply or prefix ends with (the choice the code made)."""

    text: str
    choice_logprobs: list[Optional[ChoiceLogprobs]] = field(default_factory=list)
    prefill_choice: Optional[ChoiceLogprobs] = None


def _text_of(reply: str | Reply) -> str:
    return reply if isinstance(reply, str) else reply.text


def _chain_choices(chain: list[NodeSpec], reply: str | Reply, opener: Optional[str]) -> Optional[list[Optional[ChoiceLogprobs]]]:
    """Distribution behind each chain node's <follow_proposal> (None for nodes without one)."""
    if isinstance(reply, str):
        return None
    rewrote = reply.text.split("</trace>")[0].lstrip().lower().startswith("<node")
    in_order = list(reply.choice_logprobs)
    if opener is not None and not rewrote and _follow_of(opener) is not None:
        in_order.insert(0, reply.prefill_choice)
    it = iter(in_order)
    return [next(it, None) if spec.follow is not None else None for spec in chain]


def attach_choice_logprobs(
    node: Node, created: list[Node], chain: list[NodeSpec], choices: list[Optional[ChoiceLogprobs]]
) -> None:
    """Store each choice distribution on the proposals it chose among: chain[0]'s
    choice is among `node`'s proposals, chain[i]'s among created[i-1]'s.
    Sets `logprob` and the raw `prob = exp(logprob)`; an idx missing from a known
    distribution (outside the top-k) gets None."""
    for i, dist in enumerate(choices[: len(created)]):
        if dist is None:
            continue
        owner = node if i == 0 else created[i - 1]
        for k, prop in enumerate(owner.proposals):
            idx = k if i == 0 else chain[i - 1].idx_of(k)
            lp = dist.get(idx)
            prop.logprob = lp
            prop.prob = None if lp is None else min(1.0, math.exp(lp))


def _apply_reply(
    node: Node, proposal: Proposal | str, reply: str | Reply, opener: Optional[str] = None
) -> list[Node]:
    """Decode a reply (a continuation of `opener` when given), apply its valid
    start and store any <follow_proposal> log-probs. Raises DecodeError if nothing is
    valid (tree unchanged), or PartialChain after applying a valid start."""
    text = _text_of(reply)
    chain = decode_continuation(text, opener) if opener is not None else decode_chain(text)
    choices = _chain_choices(chain, reply, opener)
    good, err = valid_prefix(chain)
    if not good:
        raise DecodeError(err or "no complete <node> in the output")
    created = apply_chain(node, proposal, good)
    if choices is not None:
        attach_choice_logprobs(node, created, good, choices)
    if err:
        raise PartialChain(f"kept {len(good)} of {len(chain)} nodes; {err}", created)
    return created


def expand_prefill(node: Node, proposal: Proposal | str, llm: PrefillLLM) -> list[Node]:
    """One chat call with assistant prefill: the reply is pre-started with the cut
    trace and the model continues it (see prefill_parts). If the model returns a
    Reply with log-probs, every <follow_proposal> choice in the call gets them."""
    user, prefill, opener = prefill_parts(node, proposal)
    return _apply_reply(node, proposal, llm(SYSTEM_PROMPT, user, prefill), opener)


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


def search(
    root: Node,
    prefill: PrefillLLM,
    max_calls: int = 10,
    score: Score = depth_first,
) -> SearchResult:
    """Call the model until a node gives the answer or the call budget runs out.

    `prefill(system, user, prefix)` is a chat model with assistant prefill
    (see expand_prefill).

    Each call starts from the unexplored proposal with the highest `score`
    (the question on the first call). The default is DFS backtracking; pass a
    score built from log-probs or a critic to change the order. A proposal whose
    call fails to decode is not retried. When only the start of a reply is
    valid, that start is kept and the error is recorded.
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
            expand_prefill(*nxt, prefill)
        except PartialChain as e:
            errors.append(f"{nxt[1].title!r}: {e}")
        except DecodeError as e:
            failed.add(id(nxt[1]))
            errors.append(f"{nxt[1].title!r}: {e}")
    answers = root.answer_nodes()
    return SearchResult(answers[0] if answers else None, calls, errors)
