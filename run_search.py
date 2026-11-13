"""Run a tree search on Tinker and save the tree as YAML for the UI.

New search:
    uv run python run_search.py "What is 12 * 15?" --out runs/12x15.yaml

Expand one unexplored proposal of a saved tree (one call):
    uv run python run_search.py --load runs/12x15.yaml \\
        --expand "Use 12 * 15 = 12 * (10 + 5)." --out runs/12x15_expanded.yaml

Then open http://localhost:8000/ui/?file=../runs/12x15.yaml (serve the repo
with `python3 -m http.server 8000`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from llm import DecodeError, PartialChain, expand, expand_continue, expand_prefill, search
from tree import Node, Proposal


def find_unexplored(root: Node, title: str) -> tuple[Node, Proposal]:
    matches = [(n, p) for n, p in root.unexplored() if p.title == title]
    if not matches:
        open_titles = "\n  ".join(p.title for _, p in root.unexplored()) or "(none)"
        raise SystemExit(f"No unexplored proposal titled {title!r}. Unexplored:\n  {open_titles}")
    if len(matches) > 1:
        raise SystemExit(f"{len(matches)} unexplored proposals are titled {title!r}; titles must be unique.")
    return matches[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="?", help="question for a new search (omit with --load)")
    ap.add_argument("--load", help="start from a saved tree (YAML) instead of a new question")
    ap.add_argument("--expand", metavar="TITLE",
                    help="develop only this unexplored proposal (one call) instead of running a search")
    ap.add_argument("--out", default="runs/latest.yaml", help="where to save the tree (YAML)")
    ap.add_argument("--max-calls", type=int, default=5)
    ap.add_argument("--mode", choices=["prefill", "inkling", "gpt-oss"], default="prefill",
                    help="prefill: Qwen3.6-35B-A3B with assistant prefill; inkling: Inkling-Small in "
                         "continue mode; gpt-oss: gpt-oss-120b in develop mode")
    ap.add_argument("--model", help="override the model name for the chosen mode")
    args = ap.parse_args()

    if args.load:
        root = Node.from_dict(yaml.safe_load(Path(args.load).read_text()))
    elif args.question:
        root = Node.new_root(args.question)
    else:
        ap.error("give a question or --load")
    target = find_unexplored(root, args.expand) if args.expand else None

    from tinker_llm import TinkerLLM, TinkerPrefill

    kw = {"model": args.model} if args.model else {}
    if args.mode == "prefill":
        model = TinkerPrefill.create(**kw)
        run_one, run_all = expand_prefill, lambda: search(root, prefill=model, max_calls=args.max_calls)
    elif args.mode == "inkling":
        model = TinkerLLM.inkling(**kw)
        run_one, run_all = expand_continue, lambda: search(root, model, max_calls=args.max_calls, mode="continue")
    else:
        model = TinkerLLM.create(**kw)
        run_one, run_all = expand, lambda: search(root, model, max_calls=args.max_calls)

    if target:
        try:
            created = run_one(*target, model)
            print(f"expanded {target[1].title!r}: {len(created)} new nodes")
        except PartialChain as e:
            print(f"expanded {target[1].title!r} partly: {e}")
        except DecodeError as e:
            print(f"could not expand {target[1].title!r}: {e}")
    else:
        result = run_all()
        print(f"calls: {result.calls}  errors: {result.errors or 'none'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(root.to_dict(), sort_keys=False, allow_unicode=True, width=100))

    print()
    print(root.render(show_thoughts=True))
    print()
    answers = root.answer_nodes()
    print(f"answers: {len(answers)}" + "".join(f"\n  - {n.title}: {n.thought}" for n in answers))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
