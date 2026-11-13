# llm-tree-search

A small Python data structure for tree-of-thought style LLM reasoning, plus an
HTML viewer for the trees it produces.

The generation loop it is shaped for:

1. **think**: develop a node by writing its `thought`
2. **propose**: suggest several possible next paths, each with a short `title`
3. **pick**: choose a proposal and develop it (back to 1)
4. **answer**: a node whose thought contains the final answer has no proposals
   and is marked `answer = True`

Each proposal can also carry a `value` (from a critic) and a policy
probability `prob` (e.g. from the LLM's log-probs); the tree stores them but
does not require them.

[`llm.py`](llm.py) connects an LLM to the tree through a small XML format
([below](#llm-in-the-loop)), and **it now runs against real models**: open
models on Thinking Machines' [Tinker](https://tinker-docs.thinkingmachines.ai/)
grow the tree, and the result opens in the viewer.

## Quick start: search with a real LLM

```bash
uv sync
```

```bash
uv run tinker auth login
```

```bash
uv run python run_search.py "What is 17 * 23?" --out runs/17x23.yaml --max-calls 3
```

To develop one unexplored proposal of a saved tree (one call):

```bash
uv run python run_search.py --load runs/17x23.yaml \
    --expand "Use the difference of squares method: (20 - 3) * (20 + 3) = 20^2 - 3^2." \
    --out runs/17x23_expanded.yaml
```

This runs `Qwen/Qwen3.6-35B-A3B` on Tinker with thinking off and assistant
prefill ([details](#qwen36-35b-a3b-with-assistant-prefill)), prints the tree
and saves it as YAML. To view it, serve the repo and open the file in the UI:

```bash
python3 -m http.server 8000
```

<http://localhost:8000/ui/?file=../runs/17x23.yaml>

Saved runs, both from these commands:

- [`runs/17x23.yaml`](runs/17x23.yaml): one call, answer 391 by distributing
  (340 + 51). The first choice was "Calculate 17 * 20 and 17 * 3" (p = 0.93);
  "Use the difference of squares method" (p = 0.07) is left unexplored.
- [`runs/17x23_expanded.yaml`](runs/17x23_expanded.yaml): the same tree after
  expanding the difference-of-squares proposal: three new nodes and a second
  answer, 400 - 9 = 391. "Calculate 20^2 which is 400." (p = 0.86) is still
  unexplored.

`run_search.py` options: `--max-calls N`, `--model NAME`, and `--mode`:

| `--mode` | Model | How a proposal is developed |
| --- | --- | --- |
| `prefill` (default) | `Qwen/Qwen3.6-35B-A3B` | Assistant reply pre-started with the cut trace |
| `inkling` | `thinkingmachines/Inkling-Small` | Cut trace in the user message ([continue mode](#inkling-small-continue-mode)) |
| `gpt-oss` | `openai/gpt-oss-120b` | Path + `<develop>` in the user message |

Each call costs Tinker credits; `--max-calls` caps them.

## Files

| Path | What it is |
| --- | --- |
| [`tree.py`](tree.py) | `Node` and `Proposal` classes |
| [`llm.py`](llm.py) | XML codec for LLM output, plus `expand` and `search` |
| [`tinker_llm.py`](tinker_llm.py) | Tinker adapters: `TinkerPrefill` (Qwen), `TinkerLLM` (gpt-oss, Inkling), `TinkerCompleter` (base models) |
| [`run_search.py`](run_search.py) | Command line: run a search on Tinker and save the tree |
| [`runs/`](runs/) | Saved search trees |
| [`test_tree.py`](test_tree.py), [`test_llm.py`](test_llm.py), [`test_tinker_llm.py`](test_tinker_llm.py) | Unit tests (`unittest`), with fake models |
| [`example.yaml`](example.yaml) | Example tree in `Node.to_dict()` format |
| [`ui/`](ui/) | HTML viewer for trees ([README](ui/README.md)) |
| [`pyproject.toml`](pyproject.toml), [`uv.lock`](uv.lock) | Dependencies (`uv`) |

## Data model

Everything is a `Node`:

```
Node
├─ thought:   str | None        # None only on the root
├─ answer:    bool              # True ⇒ thought is the final answer, no proposals
├─ parent:    Node | None       # None on the root
└─ proposals: list[Proposal]
     Proposal
     ├─ title: str              # short description of the path
     ├─ value: float | None     # estimated quality of the path
     ├─ prob:  float | None     # policy probability of exploring it, in [0, 1]
     └─ node:  Node | None      # the child, once the proposal is developed
```

- The **root** has no parent and no thought, and exactly one proposal whose
  title is the question. That proposal's node (`root.first`) holds the
  initial thought.
- A node has no title of its own: `node.title` is the title of the parent's
  proposal pointing to it (`node.incoming`).
- A **branch** is just a root where every node has at most one proposal
  (`is_branch`). Branches and full trees are the same type and can be merged.

## Usage

```python
from tree import Node

root = Node.new_root("What is 12 * 15?", "Split the multiplication into easier parts.")

# propose several paths (probs are normalized to sum to 1)
root.first.propose_many([
    {"title": "Distribute over 15: 12*10 + 12*5", "prob": 0.6, "value": 0.8},
    {"title": "Double and halve: 6*30",           "prob": 0.4, "value": 0.7},
])

# pick one and develop it
pick = root.first.best_proposal(by="value")
node = root.first.develop(pick, "12*10 = 120 and 12*5 = 60.")

# propose + develop in one call, and mark the answer
node.then("Sum: 120 + 60", "120 + 60 = 180", prob=1.0, value=0.95, answer=True)

print(root.render(show_thoughts=True))
```

```
Q: What is 12 * 15?
  ~ Split the multiplication into easier parts.
├── Distribute over 15: 12*10 + 12*5 [p=0.60 v=0.80]
│     ~ 12*10 = 120 and 12*5 = 60.
│   └── Sum: 120 + 60 [p=1.00 v=0.95] (answer)
│         ~ 120 + 60 = 180
└── Double and halve: 6*30 [p=0.40 v=0.70] (proposed)
```

### Main methods

| Method | Purpose |
| --- | --- |
| `Node.new_root(question, thought=None)` | Create a root (and optionally the first node) |
| `propose(title, value, prob)` / `propose_many([...])` | Add proposals to a node |
| `develop(proposal_or_title, thought, answer=False, value=None)` | Turn a proposal into a child node |
| `then(title, thought, value, prob, answer)` | Propose and develop in one call |
| `mark_answer()` | Mark a node's thought as the final answer |
| `policy()` / `normalize_probs()` | Normalized exploration probabilities |
| `best_proposal(by="value" \| "prob", only_unexplored=False)` | Pick the next proposal |
| `unexplored()` / `frontier()` / `answer_nodes()` | Query the tree |
| `best_answer()` | Answer node whose incoming proposal has the highest value |
| `to_branch()` / `branches()` | Extract root-to-node paths as branches |
| `merge(other)` / `Node.from_branches([...])` | Combine trees and branches (matched by proposal title) |
| `to_dict()` / `Node.from_dict(d)` | Serialize, e.g. to YAML or JSON |
| `render(show_thoughts=False)` | ASCII tree |

### Loading the example

```python
import yaml
from tree import Node

root = Node.from_dict(yaml.safe_load(open("example.yaml")))
```

`tree.py` has no dependencies. PyYAML is only needed to read YAML files.

## LLM in the loop

Each LLM call writes a **depth-first chain of nodes**. A node lists candidate
proposals; the next node says with `<parent>` which of those it develops. The
proposals not picked stay unexplored in the tree, so the search can come back
to them later.

```xml
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
```

Rules:

- The first node of a reply develops the proposal the code asked for, so it has no `<parent>`.
- `<parent>k</parent>` is the `idx` of a proposal of the **previous** node.
  `idx` defaults to the proposal's 0-based position.
- A node with **no proposals is the answer**: its thought holds the final
  answer and no node follows it. It is stored with `answer = True`.
- A reply may stop on a node with open proposals; the search resumes from there.
- Text outside `<node>` tags and unknown attributes are ignored. A reply cut
  off mid-node keeps every complete node and drops the unfinished one.
- After a node with exactly one proposal there is nothing to choose, so the
  next node's `<parent>` is set to it whatever number the model wrote.
- If a reply breaks a rule partway, its valid start is kept and added to the
  tree, and the error is recorded in `result.errors` (`PartialChain`). A reply
  with no valid node raises `DecodeError` and leaves the tree unchanged.
  `apply_chain` on its own stays strict.

The LLM writes no scores. `Proposal.value` is meant to come from a trained
critic and is left unset. `Proposal.logprob` is the log-prob of the LLM
writing that proposal's idx right after `<parent>`, and `Proposal.prob` is the
raw `exp(logprob)` (not renormalized over siblings).

They come for free from generation: a model may return an `llm.Reply` with the
idx distribution at each `<parent>` (`choice_logprobs`) and at the `<parent>`
a pre-started reply ends with (`prefill_choice`). `TinkerPrefill` reads both
from top-k log-probs in the same sampling call, so the first call fills every
choice it writes, and an expansion fills the choice the code made. A choice
nothing has followed yet (a reply that stopped on open proposals) has no
`prob` until it is expanded. The UI shows `prob`; `logprob` is in the YAML.

The prompt for each call (`encode_context`) holds the question, the path so
far (each step's title and thought) and the proposal to develop:

```xml
<question>What is 12 * 15?</question>
<path>
  <step title="What is 12 * 15?">Split into easier parts.</step>
</path>
<develop>Double and halve: 6*30</develop>
```

### Running a search

The LLM is any function `llm(system, prompt) -> str`:

```python
from llm import search
from tree import Node

def llm(system: str, prompt: str) -> str:
    ...  # call your model (gpt-oss, Claude, a local model, ...) and return its text

root = Node.new_root("What is 12 * 15?")
result = search(root, llm, max_calls=10)
print(result.answer.thought if result.answer else "no answer", result.calls, result.errors)
```

`search` repeats until a node gives the answer or `max_calls` runs out:

1. Pick the unexplored proposal with the highest `score(node, proposal)`.
   The first call picks the question.
2. Ask the LLM to develop it (`expand`) and add the chain to the tree.

The default score, `depth_first`, takes the deepest open proposal, which is
plain DFS backtracking. Once log-probs or a critic are available, pass your
own, e.g. `score=lambda node, p: p.prob or 0.0`. A proposal whose call fails to
decode is not retried. Save the tree with `root.to_dict()` to view it in the UI.

| Function | Purpose |
| --- | --- |
| `SYSTEM_PROMPT` | Explains the format to the model |
| `encode_context(node, proposal)` | Prompt for one call |
| `decode_chain(text)` / `encode_chain(chain)` | XML text ⇄ list of `NodeSpec` |
| `validate_chain(chain)` | Check the rules |
| `apply_chain(node, proposal, chain)` | Add the chain to the tree |
| `expand(node, proposal, llm)` | One call: encode, call, decode, apply |
| `pick_next(root, score)` / `depth_first` | Choose the next proposal |
| `search(root, llm, max_calls, score)` | The full loop |

### Completion (autocomplete) mode

Instead of a chat call, a completion model can continue a text document. The
document is a short header with one solved example, the question, and the
trace so far, cut right after the node that owns the proposal to explore, with
an open node whose `<parent>` picks that proposal:

```xml
<question>What is 12 * 15?</question>
<trace>
<node>
  <thought>Split into easier parts.</thought>
  <proposal idx="0">Distribute over 15: 12*10 + 12*5</proposal>
  <proposal idx="1">Double and halve: 6*30</proposal>
</node>
<node>
  <parent>1</parent>
```

The model writes the rest of that node (and possibly more nodes). Text after
`</trace>` is ignored. Explored branches below the owner node are left out, but
every proposal of each node on the path is still listed, so `idx` values stay
the same as in the tree.

```python
result = search(root, complete=my_complete)  # my_complete(prefix: str) -> str
```

| Function | Purpose |
| --- | --- |
| `encode_trace(node)` | Path down to `node` as XML nodes |
| `completion_prefix(node, proposal)` | Document to continue, and its open node |
| `expand_completion(node, proposal, complete)` | One completion call |

### gpt-oss-120b on Tinker

[`tinker_llm.py`](tinker_llm.py) runs `openai/gpt-oss-120b` through
[Tinker](https://tinker-docs.thinkingmachines.ai/). Install and log in:

```bash
uv add tinker tinker-cookbook
```

```bash
uv run tinker auth login
```

(or set `TINKER_API_KEY`). Then:

```python
from llm import search
from tinker_llm import TinkerLLM
from tree import Node

llm = TinkerLLM.create()  # openai/gpt-oss-120b, renderer gpt_oss_medium_reasoning
result = search(Node.new_root("What is 12 * 15?"), llm, max_calls=10)
```

- The prompt is rendered with the cookbook's gpt-oss renderer (harmony format).
  `SYSTEM_PROMPT` becomes the developer instructions.
- gpt-oss reasons in its analysis channel first; only the final channel is
  returned to `llm.py`. Reasoning tokens count toward `max_tokens` (default 16384).
- `llm.last` keeps the last call's final text, thinking, sampled tokens,
  per-token log-probs and whether it ended cleanly.
- Options: `TinkerLLM.create(model=..., renderer_name="gpt_oss_low_reasoning" | "gpt_oss_high_reasoning", max_tokens=..., temperature=..., model_path=...)`.
  `model_path` samples from your own saved weights instead of the base model.
- `TinkerCompleter.create()` is the completion-mode adapter: plain text in and
  out, no chat template, default model `Qwen/Qwen3.5-35B-A3B-Base`, stops at
  `</trace>`. Use it as `search(root, complete=TinkerCompleter.create())`.

### Qwen3.6-35B-A3B with assistant prefill

The cleanest way to run the cut-trace workflow on a chat model: the assistant
reply is **pre-started** with the trace, so choosing a proposal is just cutting
the trace and setting `<parent>`, and the model continues its own reply.

```python
from llm import search
from tinker_llm import TinkerPrefill
from tree import Node

prefill = TinkerPrefill.create()  # Qwen/Qwen3.6-35B-A3B, renderer qwen3_5_disable_thinking
result = search(Node.new_root("What is 12 * 15?"), prefill=prefill, max_calls=10)
```

- System message: `SYSTEM_PROMPT`. User message: the question. Assistant
  prefill: `<trace>`, the trace cut after the node that owns the proposal,
  and an open `<node><parent>k</parent>` (`prefill_parts`).
- The `qwen3_5_disable_thinking` renderer opens the assistant turn with an
  empty, closed `<think></think>` block, so the model answers without hidden
  reasoning; the reasoning lives in each `<thought>`.
- The prefill is added with the cookbook's `build_generation_prompt(..., prefill=...)`.
- `prefill.last` holds only the sampled continuation: its text, tokens and
  per-token log-probs, so the log-prob of each `<parent>` the model writes can
  become that proposal's `prob`.
- Other Qwen hybrids work the same way: `TinkerPrefill.create(model="Qwen/Qwen3.5-397B-A17B")`.

### Inkling-Small, continue mode

`thinkingmachines/Inkling-Small` can't continue a partially written reply (its
renderer rejects a prefill), so the cut trace goes in the **user message**
instead, and the reply is the continuation (`mode="continue"`):

```python
from llm import search
from tinker_llm import TinkerLLM
from tree import Node

llm = TinkerLLM.inkling()  # Inkling-Small, renderer tml_v0, effort=0.0
result = search(Node.new_root("What is 12 * 15?"), llm, max_calls=10, mode="continue")
```

- The user message is `CONTINUE_INSTRUCTION` followed by the question and the
  trace cut after the node that owns the proposal, ending in an open
  `<node><parent>k</parent>`. `SYSTEM_PROMPT` (with the example) stays the
  system message.
- The reply may continue inside the open node or rewrite it from `<node>`; a
  rewritten node must keep the chosen `<parent>`.
- `effort` is Inkling's thinking-effort control in [0, 1). `0.0` steers it
  toward no reasoning but doesn't forbid it; any reasoning it does write is
  kept in `llm.last.thinking`, not in the tree.
- Continue mode works with any chat model: `search(root, llm, mode="continue")`.

## Tests

```bash
python3 -m unittest -v test_tree test_llm test_tinker_llm
```

The `example.yaml` tests are skipped if PyYAML is not installed.

## Viewer

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000/ui/>. See [`ui/README.md`](ui/README.md).
