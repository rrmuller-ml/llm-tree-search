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
([below](#llm-in-the-loop)), and **it runs against real models**: open models
on Thinking Machines' [Tinker](https://tinker-docs.thinkingmachines.ai/) grow
the tree, and the result opens in the viewer. [`eval_aime.py`](eval_aime.py)
compares the tree format with plain chain-of-thought on AIME
([results](#aime-evaluation)).

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
    --expand "Direct multiplication: 17 * 20 + 17 * 3" \
    --out runs/17x23_expanded.yaml
```

This runs `Qwen/Qwen3.6-35B-A3B` on Tinker with thinking off and assistant
prefill ([details](#assistant-prefill)), prints the tree
and saves it as YAML. To view it, serve the repo and open the file in the UI:

```bash
python3 serve.py
```

<http://localhost:8000/ui/?file=../runs/17x23.yaml>

Saved runs, both from these commands:

- [`runs/17x23.yaml`](runs/17x23.yaml): one call, answer 391 by difference of
  squares (400 - 9). The first choice was "Use difference of squares:
  (20-3)(20+3)" (p = 0.99); "Direct multiplication: 17 * 20 + 17 * 3"
  (p = 0.01) is left unexplored.
- [`runs/17x23_expanded.yaml`](runs/17x23_expanded.yaml): the same tree after
  expanding "Direct multiplication": three new nodes and a second answer, 391.
  The model's thought switches back to difference of squares instead of
  following the proposal's title, so the branch repeats the same method.

`run_search.py` options: `--max-calls N` and `--model NAME` (default
`Qwen/Qwen3.6-35B-A3B`, with [assistant prefill](#assistant-prefill)).
Each call costs Tinker credits; `--max-calls` caps them.

## Files

| Path | What it is |
| --- | --- |
| [`tree.py`](tree.py) | `Node` and `Proposal` classes |
| [`llm.py`](llm.py) | XML codec for LLM output, plus `expand` and `search` |
| [`tinker_llm.py`](tinker_llm.py) | Tinker adapters: `TinkerPrefill` (assistant prefill), `TinkerLLM` (plain chat, used by the CoT baseline) |
| [`run_search.py`](run_search.py) | Command line: run a search on Tinker and save the tree |
| [`eval_aime.py`](eval_aime.py) | AIME 2024 + 2025 evaluation: CoT vs one tree chain, strict format scoring |
| [`expand_aime.py`](expand_aime.py) | Grow saved AIME trees with more expansions and record every answer they reach |
| [`data/aime_2024_2025.jsonl`](data/aime_2024_2025.jsonl) | The 60 AIME evaluation problems with answers |
| [`build_dataset.py`](build_dataset.py), [`data/aime_1983_2023.jsonl`](data/aime_1983_2023.jsonl) | RL training set: 918 AIME problems from 1983-2023, no eval problems |
| [`runs/`](runs/) | Saved search trees (17 * 23) and AIME results ([`runs/aime_eval/`](runs/aime_eval/)) |
| [`serve.py`](serve.py) | Serves the repo for the viewer with caching turned off |
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
proposals; the next node says with `<follow_proposal>` which of those it develops. The
proposals not picked stay unexplored in the tree, so the search can come back
to them later.

```xml
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
```

Rules:

- Every trace starts with a **question node** written by the code, mirroring the
  tree's root: `<follow_proposal idx=""/>`, an empty thought, and the question as its only proposal.
- The first node of a reply develops the proposal the code asked for; the code
  writes its `<follow_proposal idx="k"/>` (0 when developing the question).
- `<follow_proposal idx="k"/>` points to the proposal with `idx` k of the
  **previous** node. `idx` defaults to the proposal's 0-based position.
- A node with **no proposals is the answer**. Its answer goes in its own
  `<answer>...</answer>` element after `</thought>`; nothing follows it. It is
  stored with `answer = True`, and the `<answer>` element is kept at the end of
  its stored thought (the tree has no separate answer field).
- A reply may stop on a node with open proposals; the search resumes from there.
- Text outside `<node>` tags and unknown attributes are ignored. A reply cut
  off mid-node keeps every complete node and drops the unfinished one.
- After a node with exactly one proposal there is nothing to choose, so the
  next node's `<follow_proposal>` is set to it whatever number the model wrote.
- If a reply breaks a rule partway, its valid start is kept and added to the
  tree, and the error is recorded in `result.errors` (`PartialChain`). A reply
  with no valid node raises `DecodeError` and leaves the tree unchanged.
  `apply_chain` on its own stays strict.

The LLM writes no scores. `Proposal.value` is meant to come from a trained
critic and is left unset. `Proposal.logprob` is the log-prob of the LLM
writing that proposal's idx right after `<follow_proposal>`, and `Proposal.prob` is the
raw `exp(logprob)` (not renormalized over siblings).

They come for free from generation: a model may return an `llm.Reply` with the
idx distribution at each `<follow_proposal>` (`choice_logprobs`) and at the `<follow_proposal>`
a pre-started reply ends with (`prefill_choice`). `TinkerPrefill` reads both
from top-k log-probs in the same sampling call, so the first call fills every
choice it writes, and an expansion fills the choice the code made. A choice
nothing has followed yet (a reply that stopped on open proposals) has no
`prob` until it is expanded. The UI shows `prob`; `logprob` is in the YAML.

### Assistant prefill

Every call uses **assistant prefill**: the assistant reply is pre-started with
the trace, so choosing a proposal is just cutting the trace and setting
`<follow_proposal>`, and the model continues its own reply (`prefill_parts`):

- System message: `SYSTEM_PROMPT` (the format and the example).
- User message: `<question>…</question>` and "Write the trace inside
  `<trace> ... </trace>`."
- Assistant prefill: `<trace>`, the question node, the path cut right after the
  node that owns the proposal to explore, and an open node whose
  `<follow_proposal>` picks that proposal:

```xml
<trace>
<node>
  <follow_proposal idx=""/>
  <thought></thought>
  <proposal idx="0">What is 12 * 15?</proposal>
</node>
<node>
  <follow_proposal idx="0"/>
  <thought>Split into easier parts.</thought>
  <proposal idx="0">Distribute over 15: 12*10 + 12*5</proposal>
  <proposal idx="1">Double and halve: 6*30</proposal>
</node>
<node>
  <follow_proposal idx="1"/>
```

The model writes the rest of that node (and possibly more nodes). Text after
`</trace>` is ignored. Explored branches below the owner node are left out, but
every proposal of each node on the path is still listed, so `idx` values stay
the same as in the tree.

### Running a search

The model is any function `prefill(system, user, prefix)` that returns the
continuation (a `str`, or an `llm.Reply` with log-probs):

```python
from llm import search
from tinker_llm import TinkerPrefill
from tree import Node

prefill = TinkerPrefill.create()  # Qwen/Qwen3.6-35B-A3B, thinking off
root = Node.new_root("What is 12 * 15?")
result = search(root, prefill, max_calls=10)
print(result.answer.thought if result.answer else "no answer", result.calls, result.errors)
```

`search` repeats until a node gives the answer or `max_calls` runs out:

1. Pick the unexplored proposal with the highest `score(node, proposal)`.
   The first call picks the question.
2. Ask the model to develop it (`expand_prefill`) and add the chain to the tree.

The default score, `depth_first`, takes the deepest open proposal, which is
plain DFS backtracking. Once log-probs or a critic are available, pass your
own, e.g. `score=lambda node, p: p.prob or 0.0`. A proposal whose call fails to
decode is not retried. Save the tree with `root.to_dict()` to view it in the UI.

| Function | Purpose |
| --- | --- |
| `SYSTEM_PROMPT` | Explains the format to the model |
| `prefill_parts(node, proposal)` | User message, assistant prefill and open node for one call |
| `encode_trace(node)` | Path down to `node` as XML nodes |
| `decode_chain(text)` / `encode_chain(chain)` | XML text ⇄ list of `NodeSpec` |
| `validate_chain(chain)` | Check the rules |
| `apply_chain(node, proposal, chain)` | Add the chain to the tree |
| `expand_prefill(node, proposal, prefill)` | One call: build the prefill, call, decode, apply |
| `pick_next(root, score)` / `depth_first` | Choose the next proposal |
| `search(root, prefill, max_calls, score)` | The full loop |

### Qwen3.6-35B-A3B on Tinker

[`tinker_llm.py`](tinker_llm.py) runs models through
[Tinker](https://tinker-docs.thinkingmachines.ai/). Install and log in:

```bash
uv add tinker tinker-cookbook
```

```bash
uv run tinker auth login
```

(or set `TINKER_API_KEY`).

- `TinkerPrefill.create()` is the model for `search`: `Qwen/Qwen3.6-35B-A3B`
  with the `qwen3_5_disable_thinking` renderer, which opens the assistant turn
  with an empty, closed `<think></think>` block, so the model answers without
  hidden reasoning; the reasoning lives in each `<thought>`. The prefill is
  added with the cookbook's `build_generation_prompt(..., prefill=...)`.
- Other Qwen hybrids work the same way: `TinkerPrefill.create(model="Qwen/Qwen3.5-397B-A17B")`.
- `TinkerLLM.create()` is a plain chat call, `llm(system, prompt) -> str`, used
  only by the CoT baseline of the AIME evaluation.
- Each adapter keeps its last call in `.last` (text, tokens, per-token
  log-probs, whether it ended cleanly). `model_path=` samples from your own
  saved weights instead of the base model.

## AIME evaluation

[`eval_aime.py`](eval_aime.py) compares the tree format with plain
chain-of-thought on AIME 2024 + 2025 (60 problems,
[`data/aime_2024_2025.jsonl`](data/aime_2024_2025.jsonl), from
[HuggingFaceH4/aime_2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024)
and [MathArena/aime_2025](https://huggingface.co/datasets/MathArena/aime_2025)).
Both methods use the same model (`Qwen/Qwen3.6-35B-A3B`, thinking off),
temperature 0.7, up to 16,384 output tokens and one sample per problem.

- **CoT**: one chat call. The system prompt asks to think step by step and end
  with `<answer>N</answer>`.
- **Tree**: the tree prompt (`SYSTEM_PROMPT`, assistant prefill) and **one
  depth-first chain** (`search(..., max_calls=1)`). The question ends with
  "(The answer is an integer from 0 to 999, written between `<answer>` and
  `</answer>`.)".

### Strict scoring

A problem counts as correct only when the **format is correct and the answer is
right**. The answer is read from one place only; nothing is searched for
elsewhere in the reply.

- CoT format: exactly one `<answer>...</answer>`, holding only an integer 0-999.
- Tree format, checked on everything the model writes after the pre-started
  `<follow_proposal idx="0"/>`:

| Check | Meaning |
| --- | --- |
| `trace_closed` | The reply ends with `</trace>` and nothing after it |
| `only_nodes` | Inside the trace there is nothing but complete `<node>` elements |
| `valid_chain` | Every node has a non-empty thought; every node after the first has a `<follow_proposal idx>` naming a proposal of the node directly above (empty proposals are ignored, duplicates allowed, no repairs) |
| `final_is_answer` | The last node has no proposals |
| `answer_element` | Exactly one `<answer>` in the trace, as its own element in the last node after its thought |
| `answer_is_integer` | That answer holds only an integer 0-999 |
| `branching` | Every node except the last has at least two proposals (reported only, not required) |

### Results

| Method | Format correct | Format correct AND right | AIME 2024 | AIME 2025 |
| --- | --- | --- | --- | --- |
| CoT | 60/60 = 100% | **45/60 = 75.0%** | 23/30 | 22/30 |
| Tree, one chain | 42/60 = 70% | **36/60 = 60.0%** | 19/30 | 17/30 |

Tree checks passed: `trace_closed` 59, `only_nodes` 56, `valid_chain` 52,
`final_is_answer` 48, `answer_element` 44, `answer_is_integer` 42 (of 60);
`branching` 12. Most of the gap is format: 12 replies stop on a node that still
has proposals, and 8 break the chain. 6 well-formed trees have a wrong answer.
Both methods write about 5,000 output tokens per problem.

Files in [`runs/aime_eval/`](runs/aime_eval/): `cot.jsonl` and `tree.jsonl`
(one line per problem: prediction, format checks, raw reply), `trees/<id>.yaml`
(every tree, for the viewer) and `summary.txt`.

```bash
uv run python eval_aime.py --method both --workers 16          # full run (makes 120 Tinker calls)
uv run python eval_aime.py --method tree --limit 2             # quick check
uv run python eval_aime.py --rescore runs/aime_eval/tree.jsonl # re-apply the checks to saved replies
```

`--model` picks another Tinker model, e.g. `Qwen/Qwen3.5-397B-A17B`.

### Expanding the wrong trees

[`expand_aime.py`](expand_aime.py) grows saved trees: each expansion is one
call that develops the unexplored proposal with the highest `prob` (the
log-prob policy captured during generation; `--score depth` for DFS order),
and new proposals become candidates for the next one.

```bash
uv run python expand_aime.py --wrong            # every well-formed tree with a wrong answer, 5 expansions each
uv run python expand_aime.py --id aime2024_62 --n 3
```

Results for the 6 trees whose format was correct but answer wrong, 5
expansions each (same model, temperature 0.7):

| Problem | Correct | First answer | Answers in the tree after expanding | Correct answer found | Majority |
| --- | --- | --- | --- | --- | --- |
| aime2025_22 | 237 | 59 | 237, 59, 237, 237, 237, 237 | yes | **237** ✓ |
| aime2024_62 | 371 | 365 | 387, 373, 411, 459, 365, 371 | yes | tie |
| aime2025_28 | 248 | 3 | 3, 832, 248 | yes | tie |
| aime2024_63 | 385 | 192 | 192, 96, 192, 96, 24, 96 | no | 96 ✗ |
| aime2025_10 | 81 | 131 | 131 (nothing to expand) | no | 131 ✗ |
| aime2025_15 | 735 | 861 | 861 (nothing to expand) | no | 861 ✗ |

- More expansions reach the correct answer in 3 of the 4 trees that had
  unexplored proposals; a majority vote over the tree's answers fixes 1 of them.
  Picking the right answer among several needs a better selector (a critic's
  `value` or a verifier).
- Two trees are 2-node chains with a single proposal, so there is nothing to
  expand.
- The policy often reaches proposals the model gave very low `prob` (0.002-0.03),
  once the likely ones are explored; two of those branches found the correct answer.
- 2 of the 20 calls returned no usable node and were skipped.

Files: `runs/aime_eval/trees_expanded/<id>.yaml` (the grown trees) and
`runs/aime_eval/expansions.jsonl` (answers, majority, and a log of each expansion).

## Tests

```bash
python3 -m unittest -v test_tree test_llm test_tinker_llm
```

The `example.yaml` tests are skipped if PyYAML is not installed.

## Viewer

```bash
python3 serve.py
```

Then open <http://localhost:8000/ui/>. See [`ui/README.md`](ui/README.md).
