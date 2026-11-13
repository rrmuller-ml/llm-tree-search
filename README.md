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
([below](#llm-in-the-loop)); any provider works.

## Files

| Path | What it is |
| --- | --- |
| [`tree.py`](tree.py) | `Node` and `Proposal` classes |
| [`llm.py`](llm.py) | XML codec for LLM output, plus `expand` and `search` |
| [`test_tree.py`](test_tree.py), [`test_llm.py`](test_llm.py) | Unit tests (`unittest`) |
| [`example.yaml`](example.yaml) | Example tree in `Node.to_dict()` format |
| [`ui/`](ui/) | HTML viewer for trees ([README](ui/README.md)) |

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
  <thought>Split into easier parts.</thought>
  <proposal idx="0">Distribute over 15: 12*10 + 12*5</proposal>
  <proposal idx="1">Double and halve: 6*30</proposal>
</node>
<node>
  <parent>0</parent>
  <thought>12*10 = 120 and 12*5 = 60, so 12*15 = 180.</thought>
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
- A reply that breaks a rule raises `DecodeError` and leaves the tree unchanged.

The LLM writes no scores. `Proposal.prob` is meant to come from the log-prob
of the LLM writing a given `<parent>` after a node, and `Proposal.value` from
a trained critic; the codec leaves both unset.

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

## Tests

```bash
python3 -m unittest -v test_tree test_llm
```

The `example.yaml` tests are skipped if PyYAML is not installed.

## Viewer

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000/ui/>. See [`ui/README.md`](ui/README.md).
