# SMART: Implementation Documentation

Technical reference for the SMART pipeline (Skill-aware Mathematical
Adaptation with Replay and Reinforcement Training). Each section states what
a stage does, why it is designed that way, the exact rules and verification
gates it applies, and the concrete implementation, including default
hyperparameters as they appear in the code.

This document describes implementation. For installation and execution
commands, see `README.md`.

---

## Table of Contents

1. [Notation and Data Contract](#1-notation-and-data-contract)
2. [Stage 1 — Skill-Aware Dataset Construction](#2-stage-1-skill-aware-dataset-construction)
3. [Stage 2 — Supervised Fine-Tuning](#3-stage-2-supervised-fine-tuning)
4. [Stage 3 — Diagnosis](#4-stage-3-diagnosis)
5. [Stage 4 — Data Augmentation](#5-stage-4-data-augmentation)
6. [Stage 5 — Replay-Based Continual Fine-Tuning](#6-stage-5-replay-based-continual-fine-tuning)
7. [Stage 6 — GRPO Reinforcement Learning](#7-stage-6-grpo-reinforcement-learning)
8. [Evaluation](#8-evaluation)
9. [Training-Time Diagnostics](#9-training-time-diagnostics)
10. [Statistical Methodology](#10-statistical-methodology)

---

## 1. Notation and Data Contract

Let a problem instance be the tuple

    (q, s*, a*, S*)

where `q` is the problem statement, `s*` the reference solution, `a*` the gold
final answer (the content of `\boxed{...}` in `s*`), and `S*` the reference
*minimum skill set* — the smallest set of named mathematical skills required
to reach `a*`.

A model produces `ŷ`, from which the pipeline parses:

| Symbol | Parsed from | Meaning |
|---|---|---|
| `t̂` | `<think>...</think>` | reasoning trace |
| `ŝ` | `<solution>...</solution>` | final worked solution |
| `â` | `\boxed{...}` within `ŝ` | predicted final answer |
| `Ŝ` | `[SKILL: ...]` tags and the `Relevant skills:` header within `t̂` | predicted skill set |
| `steps` | segmentation of `t̂` on `[SKILL: ...]` markers | list of `(skill, step_text)` pairs |

Parsing is implemented in `evaluator.parse_model_output()`. All four
evaluation metric families are computed from this single parse, so training
and evaluation agree by construction on what counts as a well-formed output.

**Skill-label coverage.** Skill labels (`S*`) exist only for the MATH *train*
split. Metrics depending on `S*` are therefore `None` per-example on the test
split, excluded from aggregates, and reported with an explicit coverage
percentage rather than silently averaged to a misleading value. Final-answer
correctness and the arithmetic-consistency proxy do not depend on `S*` and
remain valid on any split. `label_test_set_v2.py` extends `S*` to the test
split when held-out skill evaluation is required.

---

## 2. Stage 1 — Skill-Aware Dataset Construction

**Module:** `data_pipeline.py --build-sft`

### Motivation

Conventional supervised fine-tuning on `(q, s*)` pairs optimises only for
producing `a*`. It provides no supervisory signal about *which* mathematical
competencies a derivation invokes, and therefore no way to measure whether a
model has learned to identify its own reasoning strategy. Stage 1 introduces
that signal by making the skill decomposition an explicit, parseable part of
the training target.

### Target format

Each training example is rendered as

```
<think>
Relevant skills: <s_1> | <s_2> | ... | <s_k>
[SKILL: <s_i>]
<step text>
[SKILL: <s_j>]
<step text>
...
</think>
<solution>
<reference solution, verbatim>
</solution>
```

Two design decisions are load-bearing:

1. **The trace is preserved near-verbatim rather than reduced to a category
   label.** `build_think_section()` takes the annotated `reasoning_trace`
   with its inline `[SKILL: ...]` markers intact. A single coarse tag per
   problem would make skill prediction trivially easy and would not support
   per-step skill-usage evaluation.

2. **A `Relevant skills:` header states the minimal skill set separately.**
   This makes set-level skill prediction independently checkable at
   evaluation time without needing to re-derive it from the step tags, and
   provides a redundant target that the model can learn to produce even when
   step-level tagging is imperfect.

The header falls back through `minimum_skills` → `_extraction_raw` →
`skills_used_in_steps`, so an example is never silently emitted without skill
information when any is available.

### Model-agnostic storage

`build_sft_dataset()` writes **raw fields** (`problem`, `think`, `solution`,
`gold_boxed`, `subject`, `level`, `skills`), not a pre-rendered prompt string.
Prompt rendering is deferred to training and evaluation time and performed by
`templates.py` using the *target model's own tokenizer*:

- models with a chat template (Qwen-Instruct, NuminaMath) use it;
- tokenizers rejecting a system role (`deepseek-math-7b-rl`) receive a
  system-role-free render;
- base models with no chat template (`deepseek-math-7b-base`) receive
  plain-completion format.

One dataset therefore serves every model under comparison without
regeneration, and switching `--model` never requires editing a template
string — a property that matters directly for the cross-model generalisation
study.

---

## 3. Stage 2 — Supervised Fine-Tuning

**Module:** `sft_train.py`

### Objective

Standard causal-LM cross-entropy over the rendered sequence. With
`--packing` enabled (default), TRL concatenates examples into fixed-length
blocks with EOS-bounded loss masking, so `<think>`/`<solution>` boundaries
remain intact despite packing.

### Adaptation modes

| Mode | Description |
|---|---|
| `--mode full` | all parameters updated; requires FSDP sharding for 7B+ models on multi-GPU |
| `--mode lora` | base weights frozen, low-rank adapters trained; DDP-sufficient |

Both are supported so that the parameter-efficiency comparison is a measured
result rather than an assumption.

### Default hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `--lora_r` | 32 | LoRA rank |
| `--lora_alpha` | 64 | scaling, conventionally `2r` |
| `--lora_dropout` | 0.05 | |
| `--lr` | 1e-5 | |
| `--epochs` | 2.0 | fractional values permitted |
| `--max_seq_len` | 2048 | |
| `--bf16` | on | |
| `--gradient_checkpointing` | on | disable with `--no_gradient_checkpointing` when VRAM permits; costs roughly 30–40% throughput |

Hyperparameters should be selected by `hyperparameter_search.py` (Section 10),
not by hand.

### Fractional-epoch checkpointing

`--save_every_epochs 0.5` writes a checkpoint every half epoch. The step
interval is computed from the **actual post-packing dataloader length**, not
an analytical estimate, because packing changes the number of optimiser steps
per epoch in a way that cannot be predicted from the raw example count. This
matters because empirically the best checkpoint frequently occurs before the
final epoch; evaluating only the final checkpoint systematically misses it.

### Merging

LoRA runs auto-merge to `<output_dir>_merged`, a standalone model loadable by
vLLM and `transformers`. The adapter-only directory (`adapter_config.json`,
no `config.json`) cannot be loaded directly by vLLM. Intermediate checkpoints
are *not* merged during training — merging mid-run would risk OOM from
holding a second model copy — and are merged on demand into temporary
directories by `run_multi_checkpoint_eval.py`.

---

## 4. Stage 3 — Diagnosis

**Module:** `data_pipeline.py --diagnose`

Augmenting uniformly spends generation budget on material the model already
handles. Diagnosis restricts augmentation to measured weakness.

The stage consumes a predictions file, partitions examples by
`(subject, level)`, computes final-answer accuracy per cluster, and emits
those falling below an accuracy threshold (with a minimum cluster size, to
avoid treating small-sample noise as a real weakness) to
`weak_clusters.json`. This file is the sole input controlling which problems
Stage 4 targets.

This ordering — diagnose first, generate second — follows the GSM-Symbolic
observation that models fail on structurally identical variants of problems
they can otherwise solve. Augmentation is aimed at exactly that failure mode
rather than distributed uniformly.

---

## 5. Stage 4 — Data Augmentation

**Module:** `run_augmentation.py` (driver) → `data_pipeline.py` (generators)

Two perturbation families are applied only to weak clusters. They differ in a
way that determines their verification requirements.

### 5.1 Semantic perturbation

**Rule.** Exactly one of three transformations, selected to fit the problem:

- **(a) Paraphrase** — reword while keeping all quantities and relations identical.
- **(b) Entity rename** — substitute different but semantically equivalent names.
- **(c) Distractor insertion** — add one plausible but mathematically irrelevant clause.

The prompt explicitly forbids changing any number, unit, or relationship
between quantities.

**Verification.** None required. Because meaning is preserved by
construction, the gold answer `a*` carries over unchanged. This is why
semantic perturbation is the prioritised family: it is safe by design.

**Parameters.** `--n_per_problem` (default 2), `--batch_size`.

### 5.2 Numeric perturbation

**Rule.** `perturb_numeric_literals()` rescales integer literals in place by a
random factor drawn from `[0.5, 2.0]`, with `max(1, round(n * factor))`.
Decimal literals are **deliberately skipped** — rescaling them too easily
breaks units or introduces spurious precision.

**Verification — this is the critical gate.** Unlike semantic perturbation,
numeric perturbation *changes the correct answer*, and the new answer is
unknown. A perturbed problem is admitted only if a **self-consistency vote**
passes: the solver is sampled `votes` times independently (default 5), and
the modal boxed answer must be produced by at least `agreement_threshold`
(default 0.8) of those samples. Otherwise the item is rejected.

The rationale is that a perturbed problem whose answer the model cannot
reproduce consistently is one for which no trustworthy label exists;
admitting it would inject label noise into training. The accepted/rejected
counts are reported so the gate's strictness is visible.

**Parameters.** `--n_per_problem`, `--votes`, `--agreement_threshold`,
`--batch_size`.

### 5.3 Batching

All generators take a `batch_generate_fn(prompts) -> completions` callable
operating on a *batch*. Per-problem generation calls would pay engine
scheduling overhead thousands of times and forfeit vLLM's continuous
batching. `run_augmentation.py` supplies this callable, auto-sizing the batch
from detected GPU memory when `--batch_size` is unset, with OOM-safe
recursive halving (`safe_generate`) so an over-optimistic batch degrades
gracefully rather than aborting the run.

---

## 6. Stage 5 — Replay-Based Continual Fine-Tuning

**Module:** `sft_train.py --replay_strategy`, implemented in
`build_replay_mixed_dataset()`

### Motivation

Continuing to train on augmented weak-cluster data alone induces catastrophic
forgetting: performance on previously-mastered clusters degrades. Replay
counters this by mixing original data back into the augmented training set.

### Mixing rule

`--replay_ratio` (default 1.0; 0.7 typical) is the fraction of the **final**
training set drawn from the original, unaugmented data. Given
`N = |original|`:

    n_original  = round(N * replay_ratio)
    n_augmented = N - n_original

so the mixed set has the same cardinality as the original. The augmented pool
is sampled with wraparound when `n_augmented` exceeds its size.

### The four strategies

| Strategy | Sampling | Purpose |
|---|---|---|
| `none` | augmented data ignored entirely; original only | explicit no-replay ablation baseline |
| `random` | uniform from both pools at `replay_ratio` | simplest mixing |
| `balanced` | stratified by `(subject, level)` | prevents augmented weak clusters from dominating purely by oversampling |
| `skill` | stratified by primary skill (first entry of the `skills` field) | skill-balanced replay |

`none` exists specifically so the forgetting-mitigation claim is testable: it
is the control against which the other three are compared. The mixed set is
shuffled with a fixed seed, and the realised ratio is printed
(`[replay-mix] strategy=... ratio_actual=...`) so the achieved mix can be
verified against the requested one.

---

## 7. Stage 6 — GRPO Reinforcement Learning

**Module:** `grpo_train.py`

### Algorithm choice

Group Relative Policy Optimization computes advantages by comparing the
rewards of `G` completions sampled for the *same* prompt, normalising within
the group. This removes the need for a learned value function. Combined with
the absence of a resident process reward model — only the policy and
reference model occupy GPU memory — the stage remains feasible on a single
80 GB card, which PPO with a critic would not be.

`--num_generations` is the group size `G` (default 8). It is the first
parameter to reduce under memory pressure: it affects advantage-estimate
variance, not reward correctness.

### Decomposed reward

Four independently-computed, independently-weighted components. Total reward
is their weighted sum.

#### (i) Correctness — `--w_correctness` (default 1.0)

| Condition | Reward |
|---|---|
| boxed answer matches `a*` | 1.0 |
| boxed answer present but wrong | 0.1 |
| no boxed answer | 0.0 |

The non-zero credit for a wrong-but-well-formed answer provides gradient
signal toward producing parseable output before correctness is achievable.
Matching uses symbolic equivalence (`answers_match`), so `1/2` and `0.5` are
equal.

#### (ii) Format fidelity — `--w_format` (default 0.2)

0.5 for a `<think>` section, 0.5 for a resolvable boxed answer. Computed via
`evaluator.parse_model_output()` — the identical parser used at evaluation
time, so "has the model learned the trained format" is scored the same way
during training and during measurement.

#### (iii) Persistence — `--w_persistence` (default 0.15)

Let `d` be the number of *distinct* `\boxed{...}` values in a completion.

| `d` | Reward |
|---|---|
| 0 | 0.0 |
| 1 | 1.0 |
| ≥2 | `max(-1.0, -0.3 · (d − 1))` |

This penalises flip-flopping — abandoning or reversing a derivation and
emitting several different final answers — as a proxy for committing to a
single reasoning path. Note the reward is genuinely negative for multiple
answers, not merely small.

#### (iv) Chain stability — `--w_chain_stability` (default 0.25)

The fraction of extractable `A = B` step assertions that verify symbolically,
computed by reusing `evaluator.arithmetic_consistency_score()`. When a
completion contains **no** checkable assertion, the reward is a neutral
**0.5**, not 0.0 — otherwise correct reasoning expressed verbally rather than
in equations would be penalised for the absence of something to check.

### Reward weighting

Weights are passed to `GRPOConfig.reward_weights` where the installed TRL
version supports it; otherwise equal weighting is applied and a warning is
printed rather than failing silently.

### Other defaults

| Parameter | Default |
|---|---|
| `--lr` | 1e-6 |
| `--beta` (KL coefficient) | 0.0 |
| `--max_new_tokens` | 1024 |
| `--max_prompt_length` | 512 |
| `--lora_r` / `--lora_alpha` | 32 / 64 |

Prompts exceeding `--max_prompt_length` are dropped in `load_grpo_prompts()`
against the tokenizer directly, rather than relying on `GRPOConfig`'s own
truncation, whose behaviour has varied across TRL releases.

---

## 8. Evaluation

**Module:** `evaluator.py --score`

Four metric families are computed from one parse of each prediction.

### Family 1 — Final-answer correctness

Boxed-answer match against `a*` using symbolic equivalence rather than string
comparison. Available on any split. Reported as `final_accuracy`.

### Family 2 — Intermediate reasoning correctness

A rule-based proxy. For each line of each step containing exactly one `=`,
both sides are normalised (`\frac{a}{b}` → `(a)/(b)`, `\cdot` → `*`, `^` →
`**`, LaTeX stripped) and compared with `simplify(lhs - rhs) == 0`.

Two properties are reported deliberately:

- `arith_consistency` — fraction of checkable assertions that verify;
- `arith_coverage` — fraction of examples containing any checkable assertion.

Coverage is reported separately because most step text is verbal reasoning or
multi-equality chains and is not checkable. Without coverage, a high
consistency score computed over very few assertions would be misleading. This
is explicitly a **lower-bound** signal, not a full step-correctness judge; a
stronger LLM-judge is available via `judge_steps_batch` / `run_judge_eval.py`.

### Family 3 — Skill-prediction correctness

Set comparison between `Ŝ` and `S*` after normalisation (lowercased,
punctuation stripped, whitespace collapsed). Both exact and fuzzy variants
are reported, the latter matching with `difflib.SequenceMatcher` ratio ≥
`--fuzzy-threshold` (default 0.8), which tolerates cosmetic naming variation
without accepting genuinely different skills. Precision, recall, F1, and
exact-set-match rate are all recorded.

### Family 4 — Skill-usage validity

For each step where the model emitted a skill tag, whether that skill
plausibly belongs to the problem's known skill vocabulary (fuzzy-matched
against `S*`). This catches fabricated or off-vocabulary tags anywhere in the
trace. It is distinct from Family 3: Family 3 asks whether the *set* is
right; Family 4 asks whether individual tags are drawn from a plausible
vocabulary. Neither verifies that a tag is correct for its specific position
in the derivation — that requires the LLM judge.

Returns `None` (excluded from aggregates) when the model tagged no steps.

### Aggregation

`aggregate_summary()` reports each metric overall and per `(subject, level)`
cluster. `skill_wise_accuracy()` additionally reports per-skill final-answer
accuracy using **many-to-many attribution**: a problem requiring three skills
contributes to all three skills' counts. This is not a partition, and the
per-skill `n` values consequently sum to more than the example count.

---

## 9. Training-Time Diagnostics

The metrics emitted each logging interval during SFT, for example

```
{'loss': '0.4724', 'grad_norm': '0.1695', 'learning_rate': '1.963e-07',
 'entropy': '0.4921', 'num_tokens': '1.204e+07',
 'mean_token_accuracy': '0.8715', 'epoch': '3.929'}
 99% 759/764 [1:59:23<00:47, 9.55s/it]
```

| Field | Meaning | Interpretation |
|---|---|---|
| `loss` | mean cross-entropy over the interval | primary convergence signal; should decrease then plateau |
| `grad_norm` | L2 norm of the gradient before clipping | stability signal; sustained spikes indicate too high a learning rate, collapse toward zero indicates saturation |
| `learning_rate` | current scheduled LR | confirms the decay schedule is progressing; near-zero at run end is expected |
| `entropy` | mean predictive entropy of the output distribution | falling entropy means increasing confidence; a very low value alongside a flat loss suggests overfitting to the training distribution |
| `num_tokens` | cumulative tokens processed | throughput accounting |
| `mean_token_accuracy` | fraction of next-token predictions matching the target | complements loss; insensitive to how wrong an incorrect prediction is |
| `epoch` | fractional epoch position | cross-references checkpoints written by `--save_every_epochs` |
| `s/it` | seconds per optimiser step | the quantity governing wall-clock cost |

**Note on interpretation.** These are *training-set* quantities. Falling loss
and rising `mean_token_accuracy` indicate the model is fitting the training
distribution; they do not establish improved held-out reasoning. Only the
Section 8 metrics, computed on evaluation data, support that claim. It is
routine and expected for training loss to keep improving while held-out
accuracy plateaus or declines — which is precisely why the pipeline evaluates
every intermediate checkpoint rather than only the final one.

**Cost estimation.** Total wall-clock is approximately
`total_steps × s/it`, where `total_steps ≈ (dataset_size / (per_device_batch_size × grad_accum)) × epochs`.
The progress bar's ETA is generally reliable after the first few dozen steps.

---

## 10. Statistical Methodology

### Hyperparameter selection

**Module:** `hyperparameter_search.py`

Hyperparameters are selected by a small grid search on a held-out
**validation** subset carved from the *train* split, never the test split, so
the reported test-set results are never used — even indirectly — to select
configuration. The split is seeded and provably disjoint. Each candidate is
trained briefly and scored by validation accuracy; results are ranked and the
best configuration reported, along with a citable methodology statement.

The search is deliberately small (typically 4–8 configurations from varying
one or two dimensions). This is a documented, reproducible procedure rather
than an exhaustive sweep, and is what should be described in a paper's
Implementation Details section in place of manual tuning.

### Significance testing

**Function:** `evaluator.mcnemar_test()`, surfaced by
`evaluator.py --compare --run-detailed ...`

Because every run in this pipeline is evaluated on the *same* test problems,
comparisons are **paired**. The appropriate test is therefore McNemar's,
which conditions on the discordant pairs:

- `n01` — baseline wrong, comparison run correct
- `n10` — baseline correct, comparison run wrong

Items both runs answer identically carry no information about a difference
between them and are excluded. The two-sided p-value is computed by the
**exact binomial form**, `2 · P(X ≤ min(n01, n10))` with `X ~ Binomial(n01 + n10, 0.5)`,
rather than the chi-squared approximation, which is unreliable for small
discordant counts. Results are written to `mcnemar_significance.json`.

An unpaired standard-error heuristic (`√(p(1−p)/n)`, flagging differences
beyond ~2 SE) is retained as a coarse per-cluster diagnostic, but it discards
the pairing and is less powerful. **The McNemar result is the one to cite.**

### Reproducibility scope

All scripts seed `random`, `numpy`, `torch`, and `transformers` via
`determinism.set_all_seeds()` (default 42). This makes repeated runs on the
*same* machine and library stack closely reproducible. It does **not**
guarantee bit-identical results across different GPUs or driver versions:
floating-point reduction order in matmul and attention kernels differs at
that level regardless of seed, and the divergence compounds over thousands of
steps. Where feasible, 2–3 seeds per configuration should be run and means
compared, particularly for clusters with fewer than ~150 examples.
