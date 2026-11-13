"""Run a tree search on Tinker and save the tree as YAML for the UI.

    uv run python run_search.py "What is 12 * 15?" --out runs/12x15.yaml

Then open http://localhost:8000/ui/?file=../runs/12x15.yaml (serve the repo
with `python3 -m http.server 8000`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from llm import search
from tree import Node


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--out", default="runs/latest.yaml", help="where to save the tree (YAML)")
    ap.add_argument("--max-calls", type=int, default=5)
    ap.add_argument("--mode", choices=["prefill", "inkling", "gpt-oss"], default="prefill",
                    help="prefill: Qwen3.6-35B-A3B with assistant prefill; inkling: Inkling-Small in "
                         "continue mode; gpt-oss: gpt-oss-120b in develop mode")
    ap.add_argument("--model", help="override the model name for the chosen mode")
    args = ap.parse_args()

    from tinker_llm import TinkerLLM, TinkerPrefill

    root = Node.new_root(args.question)
    if args.mode == "prefill":
        model = TinkerPrefill.create(**({"model": args.model} if args.model else {}))
        result = search(root, prefill=model, max_calls=args.max_calls)
    elif args.mode == "inkling":
        model = TinkerLLM.inkling(**({"model": args.model} if args.model else {}))
        result = search(root, model, max_calls=args.max_calls, mode="continue")
    else:
        model = TinkerLLM.create(**({"model": args.model} if args.model else {}))
        result = search(root, model, max_calls=args.max_calls)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(root.to_dict(), sort_keys=False, allow_unicode=True, width=100))

    print(root.render(show_thoughts=True))
    print()
    print(f"calls: {result.calls}  errors: {result.errors or 'none'}")
    print(f"answer: {result.answer.thought if result.answer else 'none'}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
