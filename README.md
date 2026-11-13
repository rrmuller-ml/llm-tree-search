# llm-tree-search

A small Python data structure for tree-of-thought style LLM reasoning, plus an
HTML viewer for the trees it produces.

The generation loop it is shaped for:

1. **think**: develop a node by writing its `thought`
2. **propose**: suggest several possible next paths, each with a short `title`,
   a `value` estimate and a policy probability `prob`
3. **pick**: choose a proposal (by value or prob) and develop it (back to 1)
4. **answer**: a node whose thought contains the final answer is marked
   `answer = True` and can no longer propose anything

No LLM calls are included; this repo is only the structure.

## Files

| Path | What it is |
| --- | --- |
| [`tree.py`](tree.py) | `Node` and `Proposal` classes |
| [`test_tree.py`](test_tree.py) | Unit tests (`unittest`) |
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

## Tests

```bash
python3 -m unittest -v test_tree
```

The `example.yaml` tests are skipped if PyYAML is not installed.

## Viewer

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000/ui/>. See [`ui/README.md`](ui/README.md).
