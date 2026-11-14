"""Expand saved AIME trees: develop more unexplored proposals of the trees in
runs/aime_eval/trees/ and record every answer the tree reaches.

    uv run python expand_aime.py --id aime2024_62               # one problem, 5 expansions
    uv run python expand_aime.py --id aime2024_62 --id aime2025_28 --n 3
    uv run python expand_aime.py --wrong                        # every well-formed tree with a wrong answer

Each expansion is one model call (assistant prefill, Qwen3.6-35B-A3B, thinking
off, `continue_from`): it develops the unexplored proposal with the highest
`prob` (the log-prob policy captured during generation). New proposals become
candidates for the next expansion; a proposal whose call adds nothing is not
retried.

Writes runs/aime_eval/trees_expanded/<id>.yaml (the grown tree, for the
viewer) and runs/aime_eval/expansions.jsonl (one line per problem: every
answer in the tree, the majority answer, and a log of each expansion).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import yaml

from llm import answer_of, continue_from
from tree import Node, Proposal

RESULTS = Path("runs/aime_eval/tree.jsonl")
OUT = Path("runs/aime_eval/trees_expanded")
LOG = Path("runs/aime_eval/expansions.jsonl")
MODEL = "Qwen/Qwen3.6-35B-A3B"

def tree_answers(root: Node) -> list[Optional[int]]:
    """The answer of every answer node, in tree order (None if not an integer)."""
    answers = [answer_of(n) for n in root.answer_nodes()]
    return [int(a) if a is not None and a.isascii() and a.isdigit() and len(a) <= 3 else None for a in answers]


def majority_answer(answers: list[Optional[int]]) -> Optional[int]:
    """The most common answer, or None when there is none or a tie for first."""
    top = Counter(a for a in answers if a is not None).most_common(2)
    if not top or (len(top) == 2 and top[0][1] == top[1][1]):
        return None
    return top[0][0]


def by_prob(node: Node, proposal: Proposal) -> float:
    return proposal.prob if proposal.prob is not None else -1.0


def expand_tree(row: dict, base_model, n: int) -> dict:
    model = dataclasses.replace(base_model)  # own `last`, shared client
    root = Node.from_dict(yaml.safe_load(Path(row["tree"]).read_text()))
    log, failed = [], set()
    for step in range(1, n + 1):
        candidates = [(nd, p) for nd, p in root.unexplored() if id(p) not in failed]
        if not candidates:
            log.append({"step": step, "note": "no unexplored proposal left"})
            break
        node, prop = max(candidates, key=lambda c: by_prob(*c))
        before = sum(1 for _ in root.walk())
        answer = continue_from(node, prop, model)
        new_nodes = sum(1 for _ in root.walk()) - before
        if new_nodes == 0:
            failed.add(id(prop))
        log.append({"step": step, "proposal": prop.title, "prob": prop.prob, "new_nodes": new_nodes, "answer": answer})
    OUT.mkdir(parents=True, exist_ok=True)
    tree_path = OUT / f"{row['id']}.yaml"
    tree_path.write_text(yaml.safe_dump(root.to_dict(), sort_keys=False, allow_unicode=True, width=100))
    answers = tree_answers(root)
    majority = majority_answer(answers)
    return {
        "id": row["id"],
        "gold": row["answer"],
        "first_pred": row["pred"],
        "answers": answers,
        "found_gold": row["answer"] in answers,
        "majority": majority,  # None when there is a tie
        "majority_correct": majority == row["answer"],
        "nodes": sum(1 for _ in root.walk()) - 1,
        "tree": str(tree_path),
        "log": log,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", action="append", default=[], help="problem id, e.g. aime2024_62 (repeatable)")
    ap.add_argument("--wrong", action="store_true", help="every tree whose format is correct but answer wrong")
    ap.add_argument("--n", type=int, default=5, help="expansions (model calls) per tree")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    rows = {r["id"]: r for r in map(json.loads, RESULTS.read_text().splitlines())}
    ids = list(args.id)
    if args.wrong:
        ids += sorted(i for i, r in rows.items() if r["format_ok"] and not r["correct"])
    if not ids:
        ap.error("give --id or --wrong")
    unknown = [i for i in ids if i not in rows]
    if unknown:
        ap.error(f"unknown problem id(s): {unknown}")

    from tinker_llm import TinkerPrefill

    base = TinkerPrefill.create(model=args.model, max_tokens=args.max_tokens, temperature=args.temperature)
    with ThreadPoolExecutor(max_workers=len(ids)) as pool:
        results = list(pool.map(lambda i: expand_tree(rows[i], base, args.n), dict.fromkeys(ids)))

    previous = {r["id"]: r for r in map(json.loads, LOG.read_text().splitlines())} if LOG.exists() else {}
    previous.update({r["id"]: r for r in results})
    LOG.write_text("".join(json.dumps(r) + "\n" for r in previous.values()))
    for r in results:
        print(f"{r['id']:<12} gold={r['gold']:<4} first={r['first_pred']!s:<4} nodes={r['nodes']:<3} "
              f"found={'yes' if r['found_gold'] else 'no ':<3} majority={r['majority']!s:<5} answers={r['answers']}")


if __name__ == "__main__":
    main()
