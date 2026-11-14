"""Evaluate Qwen3.6-35B-A3B (thinking off) on AIME 2024 + 2025 (60 problems).

Two methods, same model, temperature and token budget:

- cot:  one chat call with a "think step by step" prompt; the answer goes in
        <answer>N</answer>.
- tree: the tree prompt (SYSTEM_PROMPT + assistant prefill), one depth-first
        chain (search with max_calls=1); the answer goes in <answer>N</answer>
        inside the final node's thought.

    uv run python eval_aime.py --method both --workers 16
    uv run python eval_aime.py --method tree --limit 2      # smoke test

Writes runs/aime_eval/<method>.jsonl (one line per problem) and, for the tree
method, every tree to runs/aime_eval/trees/<id>.yaml for the UI.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import yaml

from llm import search
from tree import Node

DATA = Path("data/aime_2024_2025.jsonl")
OUT = Path("runs/aime_eval")
MODEL = "Qwen/Qwen3.6-35B-A3B"

COT_SYSTEM = """\
You are an expert competition mathematician. Solve the problem. Think step by
step and show your reasoning. The answer is an integer from 0 to 999. End your
response with the final answer in the form <answer>N</answer>."""

# Question text is no longer escaped, so the tags reach the model as written.
TREE_SUFFIX = "\n\n(The answer is an integer from 0 to 999, written between <answer> and </answer>.)"

_ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S | re.I)


def extract_answer(text: str) -> Optional[int]:
    """Integer in the last <answer>...</answer> (tolerates \\boxed{}, commas, spaces)."""
    found = _ANSWER.findall(text or "")
    if not found:
        return None
    m = re.search(r"-?\d+", found[-1].replace(",", ""))
    return int(m.group()) if m else None


# --- strict format checks -------------------------------------------------
# A prediction only counts when the whole reply follows the requested format,
# and it is read from one place only; nothing is searched for elsewhere.
_INT_0_999 = re.compile(r"\d{1,3}")
_NODE = re.compile(r"<node\b[^>]*>(.*?)</node\s*>", re.S | re.I)
# The first call pre-starts the reply with <trace>, the question node and this
# open node; the checks cover this open node and everything the model writes.
TREE_OPENER = '<trace>\n<node>\n  <follow_proposal idx="0"/>\n'
TREE_INFO_CHECKS = {"branching"}  # reported, but not part of "format correct"


def _answer_int(content: str) -> Optional[int]:
    content = content.strip()
    return int(content) if _INT_0_999.fullmatch(content) else None


def strict_cot(text: str) -> tuple[bool, Optional[int], dict]:
    """CoT format: exactly one <answer>...</answer>, holding only an integer 0-999."""
    tags = _ANSWER.findall(text or "")
    checks = {"one_answer_tag": len(tags) == 1}
    pred = _answer_int(tags[0]) if len(tags) == 1 else None
    checks["answer_is_integer"] = pred is not None
    return all(checks.values()), pred if all(checks.values()) else None, checks


def strict_tree(reply: str) -> tuple[bool, Optional[int], dict]:
    """Tree format, checked on the pre-started reply plus the continuation:
    - trace_closed: the reply ends with </trace> and nothing after it
    - only_nodes: inside <trace> there is nothing but complete <node> elements
    - valid_chain: every node has a non-empty <thought>, and every node after the
      first has a <follow_proposal idx="k"/> naming a proposal of the node above (no repairs).
      Empty proposals are ignored (a node with only empty ones has none) and
      duplicate titles or idx values are allowed.
    - branching: every node except the last has at least two proposals
      (reported only; not required for a correct format)
    - final_is_answer: the last node has no proposals
    - answer_element: exactly one <answer> in the whole trace, as its own
      element in the last node, after its <thought> (not inside any thought)
    - answer_is_integer: that answer holds only an integer 0-999
    """
    from llm import NodeSpec, decode_chain, DecodeError

    full = TREE_OPENER + (reply or "")
    checks = {k: False for k in ("trace_closed", "only_nodes", "valid_chain", "branching",
                                 "final_is_answer", "answer_element", "answer_is_integer")}
    m = re.fullmatch(r"\s*<trace>(.*)</trace>\s*", full, re.S)
    checks["trace_closed"] = bool(m) and "</trace>" not in m.group(1)
    body = m.group(1) if m else full.split("<trace>", 1)[1]
    checks["only_nodes"] = _NODE.sub("", body).strip() == ""
    pred = None
    try:
        chain = decode_chain(body)
    except DecodeError:  # a node without <thought>
        chain = None
    if chain:
        chain = [NodeSpec(n.thought, [p for p in n.proposals if p.title], n.follow, n.answer) for n in chain]
        checks["valid_chain"] = all(n.thought for n in chain) and all(
            prev.proposals and n.follow is not None and prev.position_of(n.follow) is not None
            for prev, n in zip(chain, chain[1:])
        )
        if not checks["valid_chain"]:
            chain = None
    if chain:
        checks["branching"] = all(len(n.proposals) >= 2 for n in chain[:-1])
        checks["final_is_answer"] = chain[-1].is_answer
        final = chain[-1]
        checks["answer_element"] = len(_ANSWER.findall(body)) == 1 and final.answer is not None
        if checks["answer_element"]:
            pred = _answer_int(final.answer)
            checks["answer_is_integer"] = pred is not None
    ok = all(v for k, v in checks.items() if k not in TREE_INFO_CHECKS)
    return ok, pred if ok else None, checks


def load(limit: Optional[int]) -> list[dict]:
    rows = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    return rows[:limit] if limit else rows


# --- methods ---------------------------------------------------------------
def run_cot(base_llm, row: dict) -> dict:
    llm = dataclasses.replace(base_llm)  # own `last`, shared client
    text = llm(COT_SYSTEM, row["problem"])
    ok, pred, checks = strict_cot(text)
    return {
        "pred": pred,
        "format_ok": ok,
        "checks": checks,
        "tokens": len(llm.last.tokens),
        "clean": llm.last.clean,
        "text": text,
    }


def run_tree(base_prefill, row: dict) -> dict:
    model = dataclasses.replace(base_prefill)
    replies: list[str] = []

    def recording(system, user, prefill):
        reply = model(system, user, prefill)
        replies.append(reply.text)
        return reply

    root = Node.new_root(row["problem"] + TREE_SUFFIX)
    result = search(root, prefill=recording, max_calls=1)
    reply = replies[-1] if replies else ""
    ok, pred, checks = strict_tree(reply)
    tree_path = OUT / "trees" / f"{row['id']}.yaml"
    tree_path.parent.mkdir(parents=True, exist_ok=True)
    tree_path.write_text(yaml.safe_dump(root.to_dict(), sort_keys=False, allow_unicode=True, width=100))
    return {
        "pred": pred,
        "format_ok": ok,
        "checks": checks,
        "reached_answer_node": result.answer is not None,
        "nodes": sum(1 for _ in root.walk()) - 1,
        "tokens": len(model.last.tokens) if model.last else 0,
        "clean": model.last.clean if model.last else False,
        "errors": result.errors,
        "tree": str(tree_path),
        "text": reply,
    }


# --- driver ----------------------------------------------------------------
def evaluate(method: str, runner, rows: list[dict], workers: int) -> list[dict]:
    out_path = OUT / f"{method}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    results: list[dict] = []
    t0 = time.time()
    with out_path.open("w") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(runner, row): row for row in rows}
        for fut in as_completed(futures):
            row = futures[fut]
            try:
                r = fut.result()
            except Exception as e:  # keep going; record the failure
                r = {"pred": None, "error": f"{type(e).__name__}: {e}"}
            r = {"id": row["id"], "year": row["year"], "answer": int(row["answer"]), **r}
            r["correct"] = bool(r.get("format_ok")) and r.get("pred") == r["answer"]
            with lock:
                results.append(r)
                f.write(json.dumps(r) + "\n")
                f.flush()
                print(f"[{method}] {len(results):>2}/{len(rows)} {row['id']:<12} "
                      f"pred={r.get('pred')!s:>5} gold={r['answer']:>3} format={'ok' if r.get('format_ok') else 'BAD'} "
                      f"{'✓' if r['correct'] else '✗'} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    return results


def summarize(method: str, results: list[dict]) -> str:
    n = max(len(results), 1)
    pct = lambda k: f"{k}/{len(results)} = {100 * k / n:.1f}%"
    fmt = sum(bool(r.get("format_ok")) for r in results)
    cor = sum(r["correct"] for r in results)
    lines = [f"== {method}",
             f"   format correct:              {pct(fmt)}",
             f"   format correct AND right:    {pct(cor)}"]
    for year in sorted({r["year"] for r in results}):
        rs = [r for r in results if r["year"] == year]
        lines.append(f"     AIME {year}: format {sum(bool(r.get('format_ok')) for r in rs)}/{len(rs)}, "
                     f"correct {sum(r['correct'] for r in rs)}/{len(rs)}")
    names = sorted({k for r in results for k in (r.get("checks") or {})})
    if names:
        lines.append("   format checks passed:")
        for k in names:
            note = "  (info, not required)" if k in TREE_INFO_CHECKS else ""
            lines.append(f"     {k:<26} {sum(bool((r.get('checks') or {}).get(k)) for r in results)}/{len(results)}{note}")
    lines.append(f"   cut off (no clean stop): {sum(not r.get('clean', True) for r in results)}")
    lines.append(f"   call errors: {sum('error' in r for r in results)}")
    toks = [r["tokens"] for r in results if r.get("tokens")]
    if toks:
        lines.append(f"   output tokens: mean {sum(toks) / len(toks):.0f}, max {max(toks)}")
    if method == "tree":
        lines.append(f"   mean nodes per tree: {sum(r.get('nodes', 0) for r in results) / n:.1f}")
    return "\n".join(lines)


def rescore(path: Path, method: str) -> list[dict]:
    """Re-apply the strict checks to saved replies (no model calls)."""
    out = []
    for r in map(json.loads, path.read_text().splitlines()):
        ok, pred, checks = (strict_cot if method == "cot" else strict_tree)(r.get("text", ""))
        r.update(format_ok=ok, pred=pred, checks=checks, correct=ok and pred == r["answer"])
        out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["cot", "tree", "both"], default="both")
    ap.add_argument("--limit", type=int, help="only the first N problems")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--model", default=MODEL, help=f"Tinker model name (default {MODEL})")
    ap.add_argument("--rescore", metavar="JSONL",
                    help="re-apply the strict checks to a saved results file and print the summary")
    args = ap.parse_args()

    if args.rescore:
        path = Path(args.rescore)
        method = "cot" if path.name.startswith("cot") else "tree"
        print(summarize(f"{method} (rescored {path})", rescore(path, method)))
        return

    from tinker_llm import TinkerLLM, TinkerPrefill

    rows = load(args.limit)
    summaries = []
    if args.method in ("cot", "both"):
        base = TinkerLLM.create(model=args.model, max_tokens=args.max_tokens, temperature=args.temperature)
        res = evaluate("cot", lambda row: run_cot(base, row), rows, args.workers)
        summaries.append(summarize("cot", res))
    if args.method in ("tree", "both"):
        base = TinkerPrefill.create(model=args.model, max_tokens=args.max_tokens, temperature=args.temperature)
        res = evaluate("tree", lambda row: run_tree(base, row), rows, args.workers)
        summaries.append(summarize("tree", res))
    report = "\n\n".join(summaries)
    print("\n" + report)
    (OUT / f"summary_{args.method}.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
