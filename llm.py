"""LLM-in-the-loop for the thought tree: XML codec + search.

The LLM writes a depth-first chain of nodes. Each node first lists candidate
proposals; the *next* node then says, with <parent>, which of those proposals
it develops. Proposals not picked stay unexplored in the tree.

    <node>
      <thought>14 * 25 is easier to compute in parts.</thought>
      <proposal idx="0">Distribute over 25: 10*25 + 4*25</proposal>
      <proposal idx="1">Use 4*25 = 100: 14*25 = 3.5*100</proposal>
    </node>
    <node>
      <parent>0</parent>
      <thought>10*25 = 250 and 4*25 = 100.</thought>
      <proposal idx="0">Add the parts: 250 + 100</proposal>
      <proposal idx="1">Check another way: 7*50</proposal>
    </node>
    <node>
      <parent>0</parent>
      <thought>250 + 100 = 350, so 14 * 25 = 350.</thought>
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

Two ways to call a model, both using the same format:
- chat: `llm(system, prompt) -> str` gets the instructions and the path so far
  and writes new nodes (`expand`).
- completion: `complete(prefix) -> str` continues a text document. The prefix
  is the trace so far, cut after the node that owns the proposal to explore,
  plus an open `<node><parent>k</parent>` choosing that proposal; the model
  writes the rest (`expand_completion`). Suits base / autocomplete models.
"""

from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

from tree import Node, Proposal

LLM = Callable[[str, str], str]  # (system, prompt) -> reply
Complete = Callable[[str], str]  # prefix -> continuation
PrefillLLM = Callable[[str, str, str], "str | Reply"]  # (system, user, assistant prefill) -> continuation
Score = Callable[[Node, Proposal], float]

EXAMPLE_QUESTION = "What is 14 * 25?"
EXAMPLE_TRACE = """\
<node>
  <thought>14 * 25 is easier to compute in parts.</thought>
  <proposal idx="0">Distribute over 25: 10*25 + 4*25</proposal>
  <proposal idx="1">Use 4*25 = 100: 14*25 = 3.5*100</proposal>
</node>
<node>
  <parent>0</parent>
  <thought>10*25 = 250 and 4*25 = 100.</thought>
  <proposal idx="0">Add the parts: 250 + 100</proposal>
  <proposal idx="1">Check another way: 7*50</proposal>
</node>
<node>
  <parent>0</parent>
  <thought>250 + 100 = 350, so 14 * 25 = 350.</thought>
</node>"""

SYSTEM_PROMPT = f"""\
You solve problems by writing a chain of reasoning steps as XML nodes.

Each node has:
- <thought>: your reasoning for this step.
- <proposal idx="i">title</proposal>: two to four possible next steps, each a
  short title. idx counts from 0.
When a thought states the final answer, write that node with no proposals.

After a node with proposals, write the next node starting with
<parent>i</parent>: the idx of the proposal it develops, chosen among the
proposals of the node directly above it (not a node number). A node with one
proposal is followed by <parent>0</parent>. Keep going depth-first until a node
gives the final answer. Write only <node> elements.

Example for "{EXAMPLE_QUESTION}":
{EXAMPLE_TRACE}
"""

# Header of the completion-mode document: a short description and one solved
# example, so a base model sees the pattern before the real question.
COMPLETION_HEADER = f"""\
Each problem below is solved with a depth-first trace of XML nodes. A node has
a <thought> and two to four <proposal idx="i"> possible next steps. The next
node starts with <parent>i</parent> to say which proposal it develops. A node
with no proposals gives the final answer and ends the trace.

<question>{EXAMPLE_QUESTION}</question>
<trace>
{EXAMPLE_TRACE}
</trace>

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
    so the next node's <parent> is set to that proposal whatever it said.
    Returns (valid nodes, error of the first invalid node or None)."""
    chain = [NodeSpec(n.thought, list(n.proposals), n.parent) for n in chain]
    for i in range(1, len(chain)):
        prev = chain[i - 1]
        if len(prev.proposals) == 1:
            chain[i].parent = prev.idx_of(0)
    for i in range(len(chain)):
        err = _node_error(chain, i)
        if err:
            return chain[:i], err
    return chain, None if chain else "no complete <node> in the output"


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


def encode_trace(node: Node) -> str:
    """The path from the first node down to `node` as XML nodes. Each node lists
    all its proposals (idx = position), and <parent> links each to the previous."""
    steps = [n for n in node.path() if not n.is_root]
    chain = [
        NodeSpec(
            thought=n.thought or "",
            proposals=[ProposalSpec(p.title) for p in n.proposals],
            parent=None if i == 0 else n.parent.proposals.index(n.incoming),
        )
        for i, n in enumerate(steps)
    ]
    return encode_chain(chain)


def _open_trace(node: Node, proposal: Proposal | str) -> tuple[str, str]:
    """(question + trace cut after `node` + open node, opener). The opener's
    <parent> picks `proposal` (no <parent> when developing the question)."""
    p = node._resolve(proposal)
    if node.is_root:
        opener = "<node>\n"
    else:
        opener = f"<node>\n  <parent>{node.proposals.index(p)}</parent>\n"
    trace = encode_trace(node)
    body = (
        f"<question>{html.escape(node.question, quote=False)}</question>\n<trace>\n"
        + (trace + "\n" if trace else "")
        + opener
    )
    return body, opener


def completion_prefix(node: Node, proposal: Proposal | str) -> tuple[str, str]:
    """Document for a completion model, cut so the model's continuation develops
    `proposal`. Returns (prefix, opener): opener is the open-node text at the end
    of the prefix, which belongs to the first node the model writes."""
    body, opener = _open_trace(node, proposal)
    return COMPLETION_HEADER + body, opener


CONTINUE_INSTRUCTION = """\
The trace below is in progress. It stops inside an open <node> whose <parent>
is already chosen. Continue it: write that node's <thought> and proposals,
close it, and keep going depth-first until a node gives the final answer.
Reply with only the continuation, starting with that node's <thought>."""


def continue_prompt(node: Node, proposal: Proposal | str) -> tuple[str, str]:
    """Chat-mode version of the cut trace, for models that can't continue a
    partially written reply: the cut trace goes in the user message and the
    model writes the continuation as its reply. Returns (prompt, opener)."""
    body, opener = _open_trace(node, proposal)
    return CONTINUE_INSTRUCTION + "\n\n" + body, opener


def prefill_parts(node: Node, proposal: Proposal | str) -> tuple[str, str, str]:
    """For chat models that can continue a pre-started reply: (user message,
    assistant prefill, opener). The prefill is the trace cut after `node`, ending
    in an open node whose <parent> picks `proposal`."""
    p = node._resolve(proposal)
    if node.is_root:
        opener = "<node>\n"
    else:
        opener = f"<node>\n  <parent>{node.proposals.index(p)}</parent>\n"
    trace = encode_trace(node)
    user = (
        f"<question>{html.escape(node.question, quote=False)}</question>\n"
        "Write the trace inside <trace> ... </trace>."
    )
    prefill = "<trace>\n" + (trace + "\n" if trace else "") + opener
    return user, prefill, opener


def decode_continuation(reply: str, opener: str) -> list[NodeSpec]:
    """Decode a continuation of `opener`. Accepts a reply that continues inside
    the open node, or one that rewrites the open node from <node> (its <parent>,
    if any, must match the opener's)."""
    reply = reply.split("</trace>")[0]
    if not reply.lstrip().lower().startswith("<node"):
        return decode_chain(opener + reply)
    chain = decode_chain(reply)
    wanted = _PARENT.search(opener)
    if chain and wanted and chain[0].parent is not None and chain[0].parent != int(wanted.group(1)):
        raise DecodeError(
            f"reply develops proposal {chain[0].parent}, expected {wanted.group(1)}"
        )
    return chain


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
ChoiceLogprobs = dict[int, float]  # proposal idx -> log-prob of the model writing that idx


@dataclass
class Reply:
    """A model reply with the log-probs of its <parent> choices.

    Models may return a plain string instead; then no log-probs are recorded.
    `choice_logprobs` has one entry per <parent> tag in `text`, in order: the
    distribution over idx digits at the position right after that tag (None
    where unknown). `prefill_choice` is the same for the <parent> that a
    pre-started reply or prefix ends with (the choice the code made)."""

    text: str
    choice_logprobs: list[Optional[ChoiceLogprobs]] = field(default_factory=list)
    prefill_choice: Optional[ChoiceLogprobs] = None


def _text_of(reply: str | Reply) -> str:
    return reply if isinstance(reply, str) else reply.text


def _chain_choices(chain: list[NodeSpec], reply: str | Reply, opener: Optional[str]) -> Optional[list[Optional[ChoiceLogprobs]]]:
    """Distribution behind each chain node's <parent> (None for nodes without one)."""
    if isinstance(reply, str):
        return None
    rewrote = reply.text.split("</trace>")[0].lstrip().lower().startswith("<node")
    in_order = list(reply.choice_logprobs)
    if opener is not None and not rewrote and "<parent>" in opener:
        in_order.insert(0, reply.prefill_choice)
    it = iter(in_order)
    return [next(it, None) if spec.parent is not None else None for spec in chain]


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
    start and store any <parent> log-probs. Raises DecodeError if nothing is
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


def expand(node: Node, proposal: Proposal | str, llm: LLM) -> list[Node]:
    """One chat call: develop `proposal` with a depth-first chain of nodes."""
    return _apply_reply(node, proposal, llm(SYSTEM_PROMPT, encode_context(node, proposal)))


def expand_completion(node: Node, proposal: Proposal | str, complete: Complete) -> list[Node]:
    """One completion call: the model continues the trace from an open node whose
    <parent> picks `proposal`. Anything after </trace> is ignored."""
    prefix, opener = completion_prefix(node, proposal)
    return _apply_reply(node, proposal, complete(prefix), opener)


def expand_continue(node: Node, proposal: Proposal | str, llm: LLM) -> list[Node]:
    """One chat call in continue mode: the cut trace is the user message and the
    model's reply continues it (see continue_prompt)."""
    prompt, opener = continue_prompt(node, proposal)
    return _apply_reply(node, proposal, llm(SYSTEM_PROMPT, prompt), opener)


def expand_prefill(node: Node, proposal: Proposal | str, llm: PrefillLLM) -> list[Node]:
    """One chat call with assistant prefill: the reply is pre-started with the cut
    trace and the model continues it (see prefill_parts). If the model returns a
    Reply with log-probs, every <parent> choice in the call gets them."""
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
    llm: Optional[LLM] = None,
    max_calls: int = 10,
    score: Score = depth_first,
    *,
    complete: Optional[Complete] = None,
    prefill: Optional[PrefillLLM] = None,
    mode: Literal["develop", "continue"] = "develop",
) -> SearchResult:
    """Call the model until a node gives the answer or the call budget runs out.

    Pass one model: a chat `llm(system, prompt)`, a completion
    `complete(prefix)`, or a chat model with assistant prefill
    `prefill(system, user, prefill)`. With
    `llm`, `mode` picks the prompt: "develop" (path + <develop>, see
    encode_context) or "continue" (cut trace with an open node, see
    continue_prompt).

    Each call starts from the unexplored proposal with the highest `score`
    (the question on the first call). The default is DFS backtracking; pass a
    score built from log-probs or a critic to change the order. A proposal whose
    call fails to decode is not retried. When only the start of a reply is
    valid, that start is kept and the error is recorded.
    """
    if sum(x is not None for x in (llm, complete, prefill)) != 1:
        raise ValueError("Pass exactly one of llm=, complete= or prefill=.")
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
            if llm is not None and mode == "continue":
                expand_continue(*nxt, llm)
            elif llm is not None:
                expand(*nxt, llm)
            elif complete is not None:
                expand_completion(*nxt, complete)
            else:
                expand_prefill(*nxt, prefill)
        except PartialChain as e:
            errors.append(f"{nxt[1].title!r}: {e}")
        except DecodeError as e:
            failed.add(id(nxt[1]))
            errors.append(f"{nxt[1].title!r}: {e}")
    answers = root.answer_nodes()
    return SearchResult(answers[0] if answers else None, calls, errors)
