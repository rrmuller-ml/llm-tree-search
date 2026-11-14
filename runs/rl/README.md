# RL training log book

Experiments with `rl_train.py`: GRPO on LoRA (Tinker), Qwen/Qwen3.6-35B-A3B,
with assistant-prefill tree traces. The training set is AIME 1983–2023 (918 problems).
The eval set is AIME 2024–2025 (60 problems × 4 samples, temperature 0.7).

Each run directory keeps `config.json`, `metrics.jsonl` (one line per step),
`eval.jsonl` (scores per problem), `checkpoints.jsonl` (Tinker paths) and `train.log`.
The rollout texts (`samples.jsonl`) are not committed because they are too large
(18 MB for run1).

---

## Experiment 1: `run1`, gradient only on decision tokens (2026-11-13)

**Setup**

```bash
uv run python rl_train.py --name run1 --batch 8 --group 8 --eval-every 10
```

| Setting | Value |
| --- | --- |
| lr | 4e-5 |
| LoRA rank | 32 |
| Batch | 8 problems × 8 rollouts |
| Sampling | temperature 1.0, max 16384 tokens |
| Trained tokens | `decisions` |
| Steps | 30 of 114 (stopped by hand) |

- **Reward:** `valid + (valid AND correct)`, which is 0, 1 or 2.
- **Advantage:** GRPO, i.e. reward minus the group mean. Groups with no spread are skipped.
- **Trained tokens:** only the decision tokens get a gradient (about 900–1700 tokens per
  step, out of about 200k sampled). These are the tokens that choose answer vs. proposal,
  another proposal vs. closing the node, another node vs. ending the trace, and which
  proposal to follow. Thought and proposal text is context only.
- **Checkpoints:** step 30 is saved as `.../sampler_weights/000030` (full path in
  `run1/checkpoints.jsonl`).
- **Time:** about 2.5 minutes per step, plus about 2.5 minutes per eval.

**Eval (AIME 2024–2025, 60 × 4)**

| Step | Valid | Valid & correct | Solved 4/4 | Solved 0/4 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.688 | 0.604 | 16 | 8 |
| 10 | 0.767 | 0.650 | 24 | 7 |
| 20 | 0.750 | 0.608 | 24 | 12 |
| 30 | 0.779 | 0.646 | 24 | 9 |

**Training rollouts (averages over 10 steps)**

| Steps | Reward | Valid | Correct | Groups with signal | Nodes per tree | Proposals per tree |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1–10 | 1.52 | 0.79 | 0.73 | 0.79 | 4.4 | 9.3 |
| 11–20 | 1.48 | 0.77 | 0.71 | 0.69 | 3.7 | 10.9 |
| 21–30 | 1.60 | 0.83 | 0.77 | 0.70 | 3.2 | 9.6 |

**Learnings**

- **Better at formatting.** Eval valid went from 0.69 to 0.78 and training valid from
  0.79 to 0.83. Problems are lost less often to a malformed or unfinished trace.
- **Not better at math.** Valid & correct is flat within noise (0.60 → 0.65 → 0.61 →
  0.65). Most of that small gain comes from more traces being valid, not from better
  reasoning. This is expected: the math lives in the thought tokens, which get no
  gradient.
- **Not better at exploration. If anything, worse.**
  - Trees shrink from 4.4 to 3.2 nodes. The model learns that the shortest safe path to
    an `<answer>` is rewarded.
  - In the step-30 trees, almost all the work is in one giant thought (13k–22k
    characters), so the tree structure is a formality.
  - Proposals that aren't followed get tiny probabilities (around 0.01).
  - Expanding one such proposal (AIME 2025 #10) went straight to a short answer node with
    no real work, and that answer was also wrong.
- **lr matters.** In a smoke run with lr 5e-4 (the cookbook's SL `get_lr`), valid
  collapsed from 0.75 to 0 within a few steps, because the model stopped choosing
  `answer`. 4e-5 (the cookbook's RL default) is stable.
- **Signal is fading.** The share of groups where rewards differ drops from about 0.8 to
  about 0.7, as more groups become all-correct (easy problems) or all-wrong.

**Potential ideas**

- **Train all tokens** (`--train-tokens all`), so that the thoughts, and therefore the
  math, get a gradient too.
- **Reward the tree shape, not only the result.** For example, add terms for the number
  of proposals, the depth of the tree, and the depth of each thought (penalize one giant
  thought; reward thoughts that work on their own proposal). These should be capped or
  gated on correctness, so the model can't farm them.
- **Reward exploration directly.** For example, `continue_from` an unfollowed proposal
  during training and reward proposals whose branch reaches a valid answer.
- **Harder or filtered training problems.** Drop problems the base model always solves,
  to keep the groups with signal high.
- **Save eval trees** (texts) at every eval, so the tree shape can be inspected without
  regenerating it.
