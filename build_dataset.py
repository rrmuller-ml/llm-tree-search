"""Build the RL training set: AIME 1983-2023, without the evaluation problems.

    uv run python build_dataset.py

Source: di-zhang-fdu/AIME_1983_2024 on Hugging Face (one CSV, ~0.3 MB).
Excluded:
- every 2024 problem (AIME 2024 is in the eval set, data/aime_2024_2025.jsonl);
- any problem whose text matches an eval problem (safety check);
- answers that are not a single integer 0-999 (e.g. "080 or 081").
No difficulty filtering. Writes data/aime_1983_2023.jsonl in the same shape
as the eval file: {"id", "year", "problem", "answer"} plus "part" and "number".
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

SOURCE = ("di-zhang-fdu/AIME_1983_2024", "AIME_Dataset_1983_2024.csv")
EVAL = Path("data/aime_2024_2025.jsonl")
OUT = Path("data/aime_1983_2023.jsonl")
EXCLUDED_YEARS = {2024}


def normalize(text: str) -> str:
    """Lowercase letters and digits only, to compare problem texts across sources."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def main() -> None:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(SOURCE[0], SOURCE[1], repo_type="dataset")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    eval_texts = {normalize(json.loads(line)["problem"]) for line in EVAL.read_text().splitlines() if line.strip()}

    kept, dropped = [], {"year": 0, "matches_eval": 0, "answer": 0}
    for r in rows:
        year = int(r["Year"])
        if year in EXCLUDED_YEARS:
            dropped["year"] += 1
            continue
        if normalize(r["Question"]) in eval_texts:
            dropped["matches_eval"] += 1
            continue
        answer = r["Answer"].strip()
        if not re.fullmatch(r"\d{1,3}", answer):
            dropped["answer"] += 1
            continue
        part = r["Part"].strip()
        kept.append({
            "id": f"aime{year}{('_' + part) if part else ''}_{r['Problem Number']}",
            "year": year,
            "part": part or None,
            "number": int(r["Problem Number"]),
            "problem": r["Question"].strip(),
            "answer": str(int(answer)),
        })

    ids = [k["id"] for k in kept]
    assert len(ids) == len(set(ids)), "duplicate ids"
    OUT.write_text("".join(json.dumps(k) + "\n" for k in kept))
    print(f"source rows: {len(rows)}  kept: {len(kept)}  dropped: {dropped}")
    print(f"years {min(k['year'] for k in kept)}-{max(k['year'] for k in kept)}  -> {OUT}")


if __name__ == "__main__":
    main()
